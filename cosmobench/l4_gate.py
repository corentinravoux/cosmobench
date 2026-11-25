#!/usr/bin/env python3
"""PreToolUse hook for Bash: allow only `python3 <fetch_paper.py> <args>`, in any path form.

The skills write the command as `python3 "${CLAUDE_SKILL_DIR}"/../*observational-cosmology/
scripts/fetch_paper.py …` (29 places) or `python3 "${CLAUDE_SKILL_DIR}"/scripts/fetch_paper.py …`
(2 places). A `Bash(<prefix>:*)` permission rule matches neither (probe 2026-11-20), so the
benchmark gates Bash with this hook instead: the command is split like the shell would, the
script path is glob-expanded, and it must resolve to the one fetch_paper.py of the plugin.

Standard library only; copied into <workdir>/.claude/ by cosmobench.sandbox.prepare_workdir.

    python3 l4_gate.py <path of fetch_paper.py>      (reads the hook event on stdin)
"""
from __future__ import annotations

import glob
import json
import os
import re
import shlex
import sys

REASON = ("Only fetch_paper.py may run in this session: one command, "
          "`python3 <path>/fetch_paper.py <arXiv id> [options]`, with no pipes or redirections.")
META = re.compile(r"[;&|<>`\n]|\$\(")


def decide(command: str, fetch_paper: str) -> tuple[bool, str]:
    """(allowed, reason). The reason is shown to the model when the call is denied."""
    if not command or META.search(command):
        return False, REASON
    try:
        argv = shlex.split(command)
    except ValueError:
        return False, REASON
    if len(argv) < 2 or os.path.basename(argv[0]) not in ("python3", "python"):
        return False, REASON
    hits = glob.glob(os.path.expandvars(os.path.expanduser(argv[1])))
    if len(hits) != 1 or os.path.realpath(hits[0]) != os.path.realpath(fetch_paper):
        return False, REASON
    return True, ""


def main() -> int:
    fetch_paper = sys.argv[1]
    event = json.load(sys.stdin)
    ok, why = decide((event.get("tool_input") or {}).get("command", ""), fetch_paper)
    out = {"hookEventName": "PreToolUse", "permissionDecision": "allow" if ok else "deny"}
    if not ok:
        out["permissionDecisionReason"] = why
    print(json.dumps({"hookSpecificOutput": out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
