"""Local-model mode: the v5 harness unchanged, answers from a llama-server on this machine.

Set before any `python -m cosmobench.*` command:

    BENCH_LOCAL_URL=http://127.0.0.1:9931     the llama-server origin (loopback only)
    BENCH_MODEL=<llm-host recipe name>         = the server's --alias; recorded in every answer
    BENCH_RUN=<run name>                       results/v5/<run name>/, apart from the Opus run

Everything else (prompts, arms, tools, sandbox, gate hook, grading) is the v5 harness. What
changes, and why:

- Claude Code talks to the local server through ANTHROPIC_BASE_URL with a dummy bearer token,
  from its own config dir `~/.claude-local` (never ~/.claude or ~/.claude-alt: those hold the
  owner's accounts and are shared with other machines).
- No `--effort`: thinking is off on the server (recipe `enable_thinking: false`) and
  CLAUDE_CODE_DISABLE_THINKING=1, so the answers come from one fixed, tested configuration.
- `--max-budget-usd` is lifted: Claude Code prices unknown models at Opus rates, and the cap
  would stop long local calls for a cost that is not spent.
- Judges (CosmoPaperQA) never run in local mode: they stay Claude (cosmobench.judge refuses).

    python -m cosmobench.local check            server up, alias = BENCH_MODEL, window, env summary
    python -m cosmobench.local pilot-gate DIR   pilot results in DIR pass V2 (every call ok, < 2 % missing)
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path

LOCAL_CONFIG_DIR = Path.home() / ".claude-local"
FORBIDDEN_CONFIG_DIRS = (Path.home() / ".claude", Path.home() / ".claude-alt")
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
DEFAULT_CTX = 131072
DEFAULT_MAX_OUTPUT = 8192
LOCAL_BUDGET_USD = 1000.0
# env vars that would send the calls to Anthropic with a real credential
CREDENTIAL_VARS = ("ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY")


def local_url(env: Mapping[str, str] | None = None) -> str:
    return (os.environ if env is None else env).get("BENCH_LOCAL_URL", "").strip()


def is_local(env: Mapping[str, str] | None = None) -> bool:
    return bool(local_url(env))


def check_url(url: str) -> str:
    u = urllib.parse.urlparse(url)
    if u.scheme != "http" or u.hostname not in LOOPBACK:
        raise SystemExit(f"BENCH_LOCAL_URL must be http on loopback, got {url!r}")
    if u.path not in ("", "/"):
        raise SystemExit(f"BENCH_LOCAL_URL is the server origin, without a path (no /v1), got {url!r}")
    return url.rstrip("/")


def local_env(base: Mapping[str, str]) -> dict:
    """The environment of one local-mode call. Refuses any shared Claude config dir."""
    env = dict(base)
    url = check_url(local_url(env))
    model = env.get("BENCH_MODEL", "").strip()
    if not model:
        raise SystemExit("local mode needs BENCH_MODEL (the llm-host recipe name = server alias)")
    cfg = Path(env.get("BENCH_LOCAL_CONFIG_DIR") or LOCAL_CONFIG_DIR).expanduser().resolve()
    if any(cfg == d.resolve() for d in FORBIDDEN_CONFIG_DIRS):
        raise SystemExit(f"refusing config dir {cfg}: local mode never uses a shared Claude config dir")
    cfg.mkdir(parents=True, exist_ok=True)
    for var in CREDENTIAL_VARS:
        env.pop(var, None)
    ctx = env.get("BENCH_LOCAL_CTX") or str(DEFAULT_CTX)
    out = env.get("BENCH_LOCAL_MAX_OUTPUT") or str(DEFAULT_MAX_OUTPUT)
    env.update({
        "CLAUDE_CONFIG_DIR": str(cfg),
        "ANTHROPIC_BASE_URL": url,
        # bearer token: a bare ANTHROPIC_API_KEY from a fresh config dir gives "Not logged in" (2.1.322)
        "ANTHROPIC_AUTH_TOKEN": "local",
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": ctx,
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": out,
        "CLAUDE_CODE_DISABLE_THINKING": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        "DISABLE_AUTOUPDATER": "1",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1",
        "API_TIMEOUT_MS": "1800000",
        "API_FORCE_IDLE_TIMEOUT": "0",
        "NO_PROXY": "127.0.0.1,localhost",
    })
    return env


def _get(url: str, timeout: float = 5) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - report any failure as text
        return 0, str(exc)


def check(env: Mapping[str, str] | None = None) -> int:
    env = dict(os.environ if env is None else env)
    if not is_local(env):
        print("BENCH_LOCAL_URL is not set: this is not local mode")
        return 1
    e = local_env(env)
    url, model = e["ANTHROPIC_BASE_URL"], e["ANTHROPIC_MODEL"]
    ok = True
    code, body = _get(url + "/health")
    print(f"{'PASS' if code == 200 else 'FAIL'} health {url}/health -> {code} {body[:80]}")
    ok &= code == 200
    code, body = _get(url + "/props")
    alias, n_ctx = None, None
    if code == 200:
        p = json.loads(body)
        alias = p.get("model_alias")
        n_ctx = (p.get("default_generation_settings") or {}).get("n_ctx")
    print(f"{'PASS' if alias == model else 'FAIL'} alias {alias!r} == BENCH_MODEL {model!r}")
    ok &= alias == model
    want = int(e["CLAUDE_CODE_MAX_CONTEXT_TOKENS"])
    print(f"{'PASS' if n_ctx == want else 'FAIL'} per-slot window {n_ctx} == CLAUDE_CODE_MAX_CONTEXT_TOKENS {want}")
    ok &= n_ctx == want
    print(f"config dir {e['CLAUDE_CONFIG_DIR']}; results under BENCH_RUN={env.get('BENCH_RUN', '')!r}")
    if not env.get("BENCH_RUN"):
        print("FAIL BENCH_RUN is empty: local answers would mix with the Opus run in results/v5/")
        ok = False
    return 0 if ok else 1


def pilot_gate(res: Path) -> int:
    """Pilot gate before the full local run: every call ok (no timeout, no leak) and < 2 % missing
    verdicts (V2). Prints per-arm duration, turns, skill and fetch use."""
    import statistics

    from .common import read_jsonl
    res = Path(res)
    ans = read_jsonl(res / "claimcheck" / "answers.jsonl") + read_jsonl(res / "cosmopaperqa" / "answers.jsonl")
    grades = read_jsonl(res / "claimcheck" / "grades.jsonl")
    bad = [a for a in ans if a.get("status") != "ok"]
    missing = sum(g.get("verdict") == "missing" for g in grades) / max(1, len(grades))
    print(f"pilot: {len(ans)} calls, {len(bad)} not ok, missing verdicts {missing:.1%} of {len(grades)} claims")
    for a in bad:
        print(f"  not ok: {a.get('item_id')} {a.get('arm')} {a.get('status')} {str(a.get('error'))[:120]}")
    for arm in sorted({a["arm"] for a in ans}):
        rs = [a for a in ans if a["arm"] == arm and a.get("status") == "ok"]
        if rs:
            secs = [(a.get("duration_ms") or 0) / 1000 for a in rs]
            print(f"  {arm}: {len(rs)} calls, median {statistics.median(secs):.0f} s, max {max(secs):.0f} s, "
                  f"median turns {statistics.median(a.get('num_turns') or 0 for a in rs)}, "
                  f"skill calls {sum(bool(a.get('skills_invoked')) for a in rs)}/{len(rs)}, "
                  f"fetch {sum((a.get('fetch_ok') or 0) > 0 for a in rs)}/{len(rs)}")
    ok = bool(ans) and bool(grades) and not bad and missing < 0.02
    print("PILOT GATE", "PASS" if ok else "FAIL (every call ok and < 2 % missing verdicts, V2)")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        sys.exit(check())
    if len(sys.argv) == 3 and sys.argv[1] == "pilot-gate":
        sys.exit(pilot_gate(Path(sys.argv[2])))
    sys.exit(__doc__)
