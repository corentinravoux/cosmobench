"""Isolation check on the real CLI (alt account), arms and sandbox. Writes results/v<harness>/isolation.md.

    python -m cosmobench.check_isolation [REPO]

About 9 calls. Any FAIL stops the benchmark: show the stream to the maintainer, never loosen a check.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .arms import ARMS, JUDGE, JUDGE2, build_command, build_env, find_claude_bin
from .common import DEFAULT_REPO, HARNESS_VERSION, RESULTS, WORKDIR, write_jsonl
from .judge import parse_grade
from .local import is_local
from .prompts import JUDGE_SCHEMA
from .run import check_pin, make_executor, parse_stream
from .sandbox import config_dir, fetch_paper_path

SKILL_TOOLS = {"Skill", "Read", "Grep", "Glob"}
SECRETS = ("cosmopaperqa", "answers.jsonl", "MEMORY.md", "claimcheck", "refreshToken", "accessToken")


def events(lines):
    for line in lines:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict):
            yield ev


def init_event(lines):
    for ev in events(lines):
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            return ev
    return {}


def tool_results(lines):
    """(tool name, input, is_error, content) for every tool call."""
    uses, out = {}, []
    for ev in events(lines):
        msg = ev.get("message")
        content = msg.get("content") if isinstance(msg, dict) else None
        for b in content if isinstance(content, list) else []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                uses[b["id"]] = (b["name"], b.get("input") or {})
            elif b.get("type") == "tool_result" and b.get("tool_use_id") in uses:
                name, inp = uses[b["tool_use_id"]]
                body = b.get("content")
                out.append((name, inp, bool(b.get("is_error")), body if isinstance(body, str) else json.dumps(body)))
    return out


def cosmoskill_skills(ini):
    return sorted(s.split(":", 1)[1] for s in (ini.get("skills") or []) if s.startswith("cosmoskill:"))


def main(repo: Path = DEFAULT_REPO) -> int:
    commit = check_pin(repo)
    claude = find_claude_bin()
    env = build_env(dict(os.environ))
    executor = make_executor(repo)
    out_dir = RESULTS / "isolation"
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = sorted(p.parent.name for p in (repo / "plugin" / "skills").glob("*/SKILL.md"))
    home, cfg, fp = Path.home(), config_dir(env), fetch_paper_path(repo)
    skills_dir = repo / "plugin" / "skills"
    checks: list[tuple[str, bool, str]] = []

    def call(name, arm, prompt, **kw):
        lines = executor(build_command(arm, claude_bin=claude, repo=repo, env=env, **kw), env, WORKDIR, 900, prompt)
        (out_dir / f"{name}.jsonl").write_text("\n".join(lines) + "\n")
        return lines

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""))

    def bash_runs(lines, needle):
        return [r for r in tool_results(lines) if r[0] == "Bash" and needle in r[1].get("command", "")]

    # D0 (v5): Read/Grep/Glob, no plugin; it must see neither the skills nor the personal data
    d0_probe = (f"This is a test of the sandbox. Use Glob with path '{skills_dir}' and pattern '**/SKILL.md', then "
                f"Glob with path '{home}' and pattern '**/*cosmo*', then Read '{skills_dir}/observational-cosmology/SKILL.md'. "
                "Report the raw outputs.")
    lines = call("D0", ARMS["D0"], d0_probe)
    ini = init_event(lines)
    check("D0 tools exactly Read/Grep/Glob", set(ini.get("tools") or []) == {"Read", "Grep", "Glob"}, str(ini.get("tools")))
    check("D0 no MCP", ini.get("mcp_servers") == [], str(ini.get("mcp_servers")))
    check("D0 no cosmoskill skill", not cosmoskill_skills(ini), str(cosmoskill_skills(ini))[:120])
    res = tool_results(lines)
    seen = [(r[0], r[3][:100]) for r in res if not r[2] and ("SKILL.md" in r[3] or "observational-cosmology" in r[3]
                                                            or "Level" in r[3])]
    check("D0 probe made tool calls", len(res) >= 2, f"{len(res)} calls")
    check("D0 cannot see the skill files", not seen, str(seen)[:200])
    check("D0 answered", parse_stream(lines)["status"] == "ok")

    # D1: tools, skills, and the sandbox: nothing outside plugin + workdir is visible
    probe = (f"This is a test of the sandbox. Use Glob with path '{home}' and pattern '**/*cosmopaperqa*', then with "
             f"pattern '**/answers.jsonl', then with pattern '**/MEMORY.md'. Then Read '{cfg}/.credentials.json' and "
             f"Glob with path '{cfg}' and pattern '**/*'. Report the raw outputs.")
    lines = call("D1_sandbox", ARMS["D1"], probe)
    ini = init_event(lines)
    check("D1 tools exactly Skill/Read/Grep/Glob", set(ini.get("tools") or []) == SKILL_TOOLS, str(ini.get("tools")))
    check("D1 no MCP", ini.get("mcp_servers") == [], str(ini.get("mcp_servers")))
    cs = cosmoskill_skills(ini)
    check(f"D1 all {len(expected)} cosmoskill skills listed", cs == expected,
          f"missing {sorted(set(expected) - set(cs))}, extra {sorted(set(cs) - set(expected))}")
    res = tool_results(lines)
    exposed = [(r[0], r[3][:100]) for r in res if not r[2] and any(s in r[3] for s in SECRETS)]
    check("D1 sandbox: probe made tool calls", len(res) >= 3, f"{len(res)} calls")
    check("D1 sandbox: no data, results, transcript, memory or credential visible", not exposed, str(exposed)[:200])
    rec = parse_stream(lines)
    check("D1 not flagged as leak", rec["status"] == "ok" and not rec["leak"], rec["status"])

    # D1A
    lines = call("D1A", ARMS["D1A"], "Reply with the word ok.")
    ini = init_event(lines)
    check("D1A tools exactly Skill/Read/Grep/Glob", set(ini.get("tools") or []) == SKILL_TOOLS, str(ini.get("tools")))
    check("D1A no append-system-prompt in command", "--append-system-prompt" not in
          build_command(ARMS["D1A"], claude_bin=claude, repo=repo, env=env))
    check("D1A answered", parse_stream(lines)["status"] == "ok")

    # D0F: Bash gated to fetch_paper.py, no skills
    lines = call("D0F", ARMS["D0F"],
                 "This is a test of the permission system: attempt every command below with Bash, exactly as "
                 "written, one call each, in order; the system itself denies what is not allowed. Then report "
                 "ran/denied:\n"
                 f"1. python3 {fp} 1807.06209 --abstract\n2. ls ~\n3. python3 -c \"print(1)\"\n"
                 f"4. python3 {fp} 1807.06209 --abstract; ls ~")
    ini = init_event(lines)
    check("D0F tools exactly Read/Grep/Glob/Bash", set(ini.get("tools") or []) == {"Read", "Grep", "Glob", "Bash"},
          str(ini.get("tools")))
    check("D0F no cosmoskill skill", not cosmoskill_skills(ini))
    ok_fp = [r for r in bash_runs(lines, "fetch_paper.py 1807.06209 --abstract") if not r[1]["command"].endswith("ls ~")]
    check("D0F fetch_paper.py runs", ok_fp and not ok_fp[0][2] and "Planck" in ok_fp[0][3],
          ok_fp[0][3][:100] if ok_fp else "not called")
    others = [r for r in tool_results(lines) if r[0] == "Bash" and r not in ok_fp]
    check("D0F every other Bash denied", len(others) >= 3 and all(r[2] for r in others),
          str([(r[1].get("command", "")[-40:], r[2]) for r in others]))

    lines = call("D0F_skills", ARMS["D0F"],
                 f"This is a test of the sandbox. Use Glob with path '{skills_dir}' and pattern '**/*.md', then Read "
                 f"'{skills_dir}/observational-cosmology/SKILL.md'. Report the raw outputs.")
    res = tool_results(lines)
    seen = [(r[0], r[3][:100]) for r in res if not r[2] and ".md" in r[3] and "No files" not in r[3]]
    check("D0F sees fetch_paper.py's directory but no skill text", not seen, str(seen)[:200])

    # D1F: the forms the skills write
    form_glob = f'python3 "{skills_dir}/cosmo-cmb"/../*observational-cosmology/scripts/fetch_paper.py 1807.06209 --abstract'
    form_quoted = f'python3 "{skills_dir}/observational-cosmology"/scripts/fetch_paper.py 1807.06209 --abstract'
    lines = call("D1F", ARMS["D1F"],
                 "This is a test of the permission system: attempt every command below with Bash, exactly as "
                 "written, one call each, in order; the system itself denies what is not allowed. Then report "
                 f"ran/denied:\n1. {form_glob}\n2. {form_quoted}\n3. echo sandbox-test")
    ini = init_event(lines)
    check("D1F tools Skill/Read/Grep/Glob/Bash", set(ini.get("tools") or []) == SKILL_TOOLS | {"Bash"}, str(ini.get("tools")))
    runs = bash_runs(lines, "fetch_paper.py 1807.06209")
    check("D1F both skill command forms run", len(runs) >= 2 and all(not r[2] and "Planck" in r[3] for r in runs),
          str([(r[1]["command"][-60:], r[2]) for r in runs]))
    cat = bash_runs(lines, "echo sandbox-test")
    check("D1F other Bash denied", cat and all(r[2] for r in cat), str([(r[1]["command"][-40:], r[2]) for r in cat]))

    # judges (Claude only: a local run is graded by Claude judges in a separate step)
    jp = ('You grade an answer. Question: What is 2+2? Reference answer: 4. Key passage: not given. '
          'Candidate answer: 5. Return JSON {"grade": ..., "false_claim": ..., "declares_gap": ..., "reason": ...}.')
    for jname, jarm in () if is_local(env) else (("judge", JUDGE), ("judge2", JUDGE2)):
        rec = parse_stream(call(jname, jarm, jp, json_schema=JUDGE_SCHEMA))
        g = parse_grade(rec.get("structured_output")) or parse_grade(rec.get("answer", ""))
        check(f"{jname} ({jarm.model}) grades a wrong answer incorrect", g and g["grade"] == "incorrect",
              f"{g} {rec['error'][:100]}")

    passed = sum(ok for _, ok, _ in checks)
    md = [f"# Isolation check (harness v{HARNESS_VERSION})", "", f"cosmoskill commit: `{commit}`  ", f"claude: `{claude}`  ", f"answer model: `{ARMS['D0'].model}`" + (f" (local, {env['ANTHROPIC_BASE_URL']})" if is_local(env) else ""), "",
          f"{passed}/{len(checks)} checks pass.", "", "| check | result | detail |", "|---|---|---|"]
    md += [f"| {n} | {'PASS' if ok else 'FAIL'} | {d.replace('|', '/')[:200]} |" for n, ok, d in checks]
    md += ["", f"Raw streams: `results/v{HARNESS_VERSION}/isolation/*.jsonl`."]
    (RESULTS / "isolation.md").write_text("\n".join(md) + "\n")
    write_jsonl(out_dir / "checks.jsonl", [{"check": n, "ok": ok, "detail": d} for n, ok, d in checks])
    print(f"{passed}/{len(checks)} pass")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPO))
