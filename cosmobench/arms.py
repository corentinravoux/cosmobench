"""Arms and the headless `claude` command line for each call.

v5 runs D0, D1A on every claim-check stratum and D0F, D1F on strata A and B only
(ARM_STRATA); D1 is kept for the isolation check. v3 arms (all at effort `high`; the v2 effort axis showed no cost or answer difference and
tests no package claim):

    D0   skills off, no tools                       baseline
    D1A  plugin loaded, no instruction              the package as a user meets it (primary)
    D1   plugin loaded, told to load skills         skill use when triggered
    D0F  skills off, Bash gated to fetch_paper.py   control: a paper reader without the skills
    D1F  as D1, plus Bash gated to fetch_paper.py   the skills with level 4
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .common import CLAUDE_MODEL, JUDGE2_MODEL, MODEL, WORKDIR
from .local import LOCAL_BUDGET_USD, is_local, local_env
from .prompts import SKILLS_SYSTEM_APPEND, fetch_append
from .sandbox import config_dir, fetch_paper_path

BASE_TOOLS = ["Skill", "Read", "Grep", "Glob"]


@dataclass(frozen=True)
class Arm:
    name: str
    skills: bool
    instructed: bool = False
    fetch: bool = False
    effort: str = "high"
    model: str = MODEL
    agent_tools: bool = False   # Read/Grep/Glob without the plugin (v5: the skills-off arms, finding 6)


# v5: D0 and D0F get the same Read/Grep/Glob tools as the skills arms (only the plugin and the Skill
# tool differ), and D1F is not told to use the skills (only the plugin differs from D0F).
ARMS = {
    "D0": Arm("D0", skills=False, agent_tools=True),
    "D1A": Arm("D1A", skills=True),
    "D1": Arm("D1", skills=True, instructed=True),
    "D0F": Arm("D0F", skills=False, fetch=True, agent_tools=True),
    "D1F": Arm("D1F", skills=True, fetch=True),
}
AGENT_TOOLS = ["Read", "Grep", "Glob"]
JUDGE = Arm("judge", skills=False, model=CLAUDE_MODEL)  # judges stay Claude in local mode
JUDGE2 = Arm("judge2", skills=False, model=JUDGE2_MODEL)
DRAFTER = Arm("draft", skills=False, effort="low", model=CLAUDE_MODEL)
VERIFIER = Arm("verify", skills=False, model=JUDGE2_MODEL, effort="medium")  # v5 build: checks both versions, other model
BLIND = Arm("blind", skills=False, model=JUDGE2_MODEL, effort="low")         # v5 build: picks the original without the paper

# v5 (plan 2026-11-21): D1 dropped (tests the instruction, not the package); TPBench dropped.
ARMS_BY_BENCH = {
    "claimcheck": ["D0", "D1A", "D0F", "D1F"],
    "cosmopaperqa": ["D0", "D1A"],
    "tpbench": ["D0", "D1A"],
}
# Arms restricted to some strata (v5: the F arms run where L4 is the only way to the paper).
ARM_STRATA = {"D0F": {"A", "B"}, "D1F": {"A", "B"}}


def arm_applies(arm: str, item: dict) -> bool:
    allowed = ARM_STRATA.get(arm)
    return allowed is None or item.get("bench") != "claimcheck" or item.get("stratum") in allowed


def config_deny_rules(env: dict | None = None) -> list[str]:
    """Second layer behind the sandbox: deny the model any read of the config dir."""
    p = "/" + str(config_dir(env))  # leading "//" = absolute path in permission rules
    return [f"Read({p})", f"Read({p}/**)"]


def system_append(arm: Arm, repo: Path) -> str:
    parts = []
    if arm.instructed:
        parts.append(SKILLS_SYSTEM_APPEND)
    if arm.fetch:
        parts.append(fetch_append(str(fetch_paper_path(repo))))
    return " ".join(parts)


def build_command(arm: Arm, *, claude_bin: str, repo: Path, json_schema: dict | None = None,
                  max_budget_usd: float = 5.0, env: dict | None = None) -> list[str]:
    """Command line for one call. The prompt is sent on stdin (the tool flags are variadic)."""
    local = is_local(env)
    cmd = [claude_bin, "--print", "--model", arm.model]
    if not local:  # local mode: thinking is off on the server, no effort level (cosmobench.local)
        cmd += ["--effort", arm.effort]
    cmd += [
        "--output-format", "stream-json", "--verbose", "--no-session-persistence",
        "--strict-mcp-config", "--setting-sources", "project",
        "--permission-mode", "dontAsk", "--max-budget-usd", str(LOCAL_BUDGET_USD if local else max_budget_usd),
        "--disallowedTools", "mcp__*", *config_deny_rules(env),
    ]
    tools = (BASE_TOOLS if arm.skills else AGENT_TOOLS if arm.agent_tools else []) + (["Bash"] if arm.fetch else [])
    if arm.skills:
        cmd += ["--plugin-dir", str(Path(repo) / "plugin")]
    else:
        cmd += ["--disable-slash-commands"]
    cmd += ["--tools", ",".join(tools)]
    if arm.skills:
        cmd += ["--allowedTools", *BASE_TOOLS]  # Bash is allowed only by the gate hook
    elif arm.agent_tools:
        cmd += ["--allowedTools", *AGENT_TOOLS]
    append = system_append(arm, repo)
    if append:
        cmd += ["--append-system-prompt", append]
    if json_schema is not None:
        cmd += ["--json-schema", json.dumps(json_schema)]
    return cmd


def build_env(base: dict, workdir: Path = WORKDIR) -> dict:
    env = local_env(base) if is_local(base) else dict(base)
    env.setdefault("CLAUDE_CONFIG_DIR", str(Path("~/.claude-alt").expanduser()))
    env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    layer = Path(workdir) / "empty-layer"
    layer.mkdir(parents=True, exist_ok=True)
    env["COSMOSKILL_HOME"] = str(layer)
    return env


def find_claude_bin() -> str:
    env = os.environ.get("CLAUDE_BIN")
    if env:
        return env
    found = shutil.which("claude")
    if found:
        return found
    cands = sorted(Path.home().glob(".vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude"))
    if cands:
        return str(cands[-1])
    raise FileNotFoundError("claude binary not found; set CLAUDE_BIN")
