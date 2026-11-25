"""Run answer calls: one headless call per (item, arm, rep), resumable.

    python -m cosmobench.run --pin [--repo PATH]          record the cosmoskill commit to test
    python -m cosmobench.run --bench cosmopaperqa [--arms L0,D1] [--reps N] [--limit N]
                        [--items id,id] [--per-stratum N] [--dry-run]

Every call runs in the hermetic sandbox of cosmobench.sandbox (bwrap; $HOME hidden).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from .arms import (
    ARMS,
    ARMS_BY_BENCH,
    arm_applies,
    build_command,
    build_env,
    find_claude_bin,
)
from .common import (
    CONFIG,
    DEFAULT_REPO,
    HARNESS_VERSION,
    RESULTS,
    WORKDIR,
    append_jsonl,
    read_jsonl,
    write_jsonl,
)
from .data import load_items
from .prompts import answer_prompt
from .sandbox import config_dir, prepare_workdir, wrap

READ_TOOLS = {"Read": "file_path", "Grep": "path", "Glob": "path"}
MAX_CONSECUTIVE_ERRORS = 3

Executor = Callable[[list, dict, Path, float, str], list]


def touches_config(target: str, cfg: Path | None = None) -> bool:
    """True if a tool target names the Claude config dir (transcripts, memory, credentials)."""
    c = str(cfg or config_dir()).rstrip("/")
    home = str(Path.home()).rstrip("/")
    t = target.replace("~/", home + "/").replace("$HOME/", home + "/")
    return c in t or (Path(c).name in t and t.split(" ", 1)[0].rstrip("/") == home)


def touches_personal_layer(target: str, home: Path | None = None) -> bool:
    """True if a tool target reaches the real personal layer ~/.cosmoskill.

    A Glob target is "<path> <pattern>"; a project-layer lookup (".cosmoskill/" under the
    empty work directory) is not a leak."""
    home = str(home or Path.home())
    personal = home.rstrip("/") + "/.cosmoskill"
    t = target.replace("~/", home.rstrip("/") + "/").replace("$HOME/", home.rstrip("/") + "/")
    if personal in t:
        return True
    parts = t.split(" ", 1)
    if len(parts) == 2 and parts[0].rstrip("/") == home.rstrip("/") and ".cosmoskill" in parts[1]:
        return True
    return False


def reclassify(path: Path) -> int:
    """Re-apply the leak rule to stored records flagged "leak" (rule tightened 2026-11-17)."""
    recs = read_jsonl(path)
    n = 0
    for r in recs:
        if r.get("status") == "leak" and not any(
                touches_personal_layer(x) for x in r.get("read_paths", []) + r.get("bash_commands", [])):
            r["status"], r["error"], r["leak"] = "ok", "", False
            r["reclassified"] = "leak rule tightened"
            n += 1
    write_jsonl(path, recs)
    return n


def parse_stream(lines: list[str]) -> dict:
    out = {"answer": "", "status": "error", "error": "", "skills_invoked": [], "tools_used": [],
           "bash_commands": [], "read_paths": [], "denied": 0, "leak": False,
           "usage": {}, "model_usage": {}, "cost_usd": None, "num_turns": None,
           "duration_ms": None, "structured_output": None, "fetch_calls": 0, "fetch_ok": 0,
           "thinking_tokens": None}
    uses: dict[str, tuple[str, str]] = {}
    failed: set[str] = set()
    result = None
    for line in lines:
        try:
            ev = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(ev, dict):
            continue
        t = ev.get("type")
        msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
        if t == "assistant":
            for b in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
                if not isinstance(b, dict) or b.get("type") != "tool_use":
                    continue
                name, inp = b.get("name", ""), b.get("input") or {}
                out["tools_used"].append(name)
                target = ""
                if name == "Skill":
                    out["skills_invoked"].append(inp.get("skill") or inp.get("command") or "")
                elif name == "Bash":
                    target = inp.get("command", "")
                    out["bash_commands"].append(target)
                elif name in READ_TOOLS:
                    target = str(inp.get(READ_TOOLS[name]) or "") + (" " + inp["pattern"] if inp.get("pattern") and name == "Glob" else "")
                    out["read_paths"].append(target)
                uses[b.get("id", "")] = (name, target)
        elif t == "user":
            content = msg.get("content", [])
            for b in content if isinstance(content, list) else []:
                if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                    continue
                body = b.get("content")
                body = body if isinstance(body, str) else json.dumps(body)
                # an error, or an empty listing (the layer is hidden by bwrap), exposes nothing
                if b.get("is_error") or body.strip() in ("", "No files found", "No matches found", "[]"):
                    failed.add(b.get("tool_use_id", ""))
        elif t == "result":
            result = ev
    for uid, (name, target) in uses.items():
        if (touches_personal_layer(target) or touches_config(target)) and uid not in failed:
            out["leak"] = True
        if name == "Bash" and "fetch_paper" in target:
            out["fetch_calls"] += 1
            if uid not in failed:
                out["fetch_ok"] += 1
    if result is None:
        out["error"] = "no result event"
        return out
    out["answer"] = result.get("result") or ""
    out["usage"] = result.get("usage") or {}
    out["model_usage"] = result.get("modelUsage") or result.get("model_usage_since_last_result") or {}
    out["cost_usd"] = result.get("total_cost_usd")
    out["num_turns"] = result.get("num_turns")
    out["duration_ms"] = result.get("duration_ms")
    out["denied"] = len(result.get("permission_denials") or [])
    mu = out["model_usage"] if isinstance(out["model_usage"], dict) else {}
    out["thinking_tokens"] = sum(int((v or {}).get("thinkingTokens") or 0) for v in mu.values()
                                 if isinstance(v, dict)) if mu else None
    out["structured_output"] = result.get("structured_output")
    if result.get("is_error") or result.get("subtype") != "success":
        out["error"] = str(result.get("result") or result.get("subtype") or "error")[:500]
    elif out["leak"]:
        out["status"] = "leak"
        out["error"] = "personal layer or config dir read"
    else:
        out["status"] = "ok"
    return out


def done_keys(path: Path) -> set[tuple[str, str, int]]:
    """Keys not to rerun: answered, or flagged as a layer leak (excluded, never retried)."""
    return {(r["item_id"], r["arm"], r["rep"]) for r in read_jsonl(path)
            if r.get("status") in ("ok", "leak")}


def make_executor(repo: Path, workdir: Path = WORKDIR) -> Executor:
    """Real executor: every call runs inside the hermetic sandbox (cosmobench.sandbox)."""
    prepare_workdir(workdir, repo)

    def run_one(cmd: list, env: dict, cwd: Path, timeout: float, stdin: str) -> list:
        proc = subprocess.run(wrap(cmd, repo=repo, workdir=workdir, env=env), input=stdin,
                              capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout)
        lines = proc.stdout.splitlines()
        if not lines and proc.stderr:
            return [json.dumps({"type": "result", "is_error": True, "subtype": "cli_error",
                                "result": proc.stderr.strip()[:500]})]
        return lines

    return run_one


def repo_commit(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def pinned_commit(config: Path = CONFIG) -> str | None:
    if not Path(config).exists():
        return None
    return json.loads(Path(config).read_text()).get("cosmoskill_commit")


def check_pin(repo: Path, config: Path = CONFIG) -> str:
    pin = pinned_commit(config)
    head = repo_commit(repo)
    if pin is None:
        raise SystemExit("no pinned commit: run `python -m cosmobench.run --pin` first")
    if pin != head:
        raise SystemExit(f"cosmoskill HEAD {head[:10]} differs from pinned {pin[:10]}; refusing")
    return head


def run_loop(items: list[dict], arms: list[str], reps: int, executor: Executor, out_path: Path, *,
             claude_bin: str = "claude", repo: Path = DEFAULT_REPO, commit: str = "", timeout: float = 1800, workdir: Path = WORKDIR,
             base_env: dict | None = None, log=print) -> int:
    done = done_keys(out_path)
    env = build_env(dict(os.environ) if base_env is None else base_env, workdir)
    consecutive = 0
    for item in items:
        for arm_name in arms:
            if not arm_applies(arm_name, item):
                continue
            arm = ARMS[arm_name]
            for rep in range(reps):
                key = (item["id"], arm_name, rep)
                if key in done:
                    continue
                cmd = build_command(arm, claude_bin=claude_bin, repo=repo, env=env)
                started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                try:
                    lines = executor(cmd, env, workdir, timeout, answer_prompt(item))
                    rec = parse_stream(lines)
                    raw = Path(out_path).parent / "raw" / f"{item['id']}__{arm_name}__{rep}.jsonl"
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
                except subprocess.TimeoutExpired:
                    rec = parse_stream([])
                    rec["error"] = "timeout"
                rec.update({"item_id": item["id"], "bench": item["bench"], "arm": arm_name,
                            "rep": rep, "effort": arm.effort, "model": arm.model,
                            "started_at": started, "cosmoskill_commit": commit,
                            "harness": HARNESS_VERSION})
                append_jsonl(out_path, rec)
                log(f"{item['id']} {arm_name} rep{rep}: {rec['status']}"
                    f"{' ' + rec['error'][:120] if rec['status'] != 'ok' else ''}"
                    f" skills={len(rec['skills_invoked'])} cost={rec['cost_usd']}")
                if rec["status"] == "error":
                    consecutive += 1
                    if consecutive >= MAX_CONSECUTIVE_ERRORS:
                        log(f"stopping after {consecutive} consecutive errors; last: {rec['error'][:300]}")
                        return 2
                else:
                    consecutive = 0
    return 0


def select_items(items: list[dict], *, ids: list[str] | None, limit: int | None,
                 per_stratum: int | None) -> list[dict]:
    if ids:
        wanted = set(ids)
        items = [it for it in items if it["id"] in wanted]
    if per_stratum:
        seen: dict[str, int] = {}
        keep = []
        for it in items:
            s = it.get("stratum", "all")
            if seen.get(s, 0) < per_stratum:
                keep.append(it)
                seen[s] = seen.get(s, 0) + 1
        items = keep
    if limit:
        items = items[:limit]
    return items


def _home_relative(path: Path) -> str:
    """`~/...` form of a path under $HOME, so the pin file holds no user name."""
    path, home = Path(path).expanduser().resolve(), Path.home().resolve()
    return "~/" + str(path.relative_to(home)) if path.is_relative_to(home) else str(path)

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=sorted(ARMS_BY_BENCH))
    ap.add_argument("--arms")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--items")
    ap.add_argument("--per-stratum", type=int)
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reclassify", action="store_true")
    ap.add_argument("--built", action="store_true", help="claim check: use the unaudited built set (pilot only)")
    a = ap.parse_args(argv)
    if a.pin:
        commit = repo_commit(a.repo)
        CONFIG.write_text(json.dumps({"cosmoskill_commit": commit, "repo": _home_relative(a.repo),
                                      "pinned_at": dt.date.today().isoformat()}, indent=1) + "\n")
        print(f"pinned {commit}")
        return 0
    if not a.bench:
        ap.error("--bench is required")
    if a.reclassify:
        print(f"{reclassify(RESULTS / a.bench / 'answers.jsonl')} records reclassified")
        return 0
    commit = check_pin(a.repo)
    arms = a.arms.split(",") if a.arms else ARMS_BY_BENCH[a.bench]
    items = select_items(load_items(a.bench, built=a.built), ids=a.items.split(",") if a.items else None,
                         limit=a.limit, per_stratum=a.per_stratum)
    claude_bin = find_claude_bin()
    if a.dry_run:
        for arm in arms:
            print(arm, " ".join(build_command(ARMS[arm], claude_bin=claude_bin, repo=a.repo)))
        n = sum(arm_applies(arm, it) for it in items for arm in arms) * a.reps
        print(f"{len(items)} items, {n} calls ({', '.join(arms)}, {a.reps} rep)")
        return 0
    WORKDIR.mkdir(parents=True, exist_ok=True)
    return run_loop(items, arms, a.reps, make_executor(a.repo), RESULTS / a.bench / "answers.jsonl",
                    claude_bin=claude_bin, repo=a.repo, commit=commit)


if __name__ == "__main__":
    sys.exit(main())
