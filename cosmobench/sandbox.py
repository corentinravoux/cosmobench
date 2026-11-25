"""Hermetic sandbox for every headless call (answers, judges, drafting).

The v2 harness hid only ~/.cosmoskill. Everything else in $HOME stayed readable by the
skills-on arms' Read/Grep/Glob: data/ (the reference answers), results/, other
repositories, and ~/.claude-alt/projects (session transcripts, memory). v3 runs
each call under bubblewrap with an empty tmpfs on $HOME, and binds back only:

- the alt config dir (read-write: credentials refresh), with every entry except
  `.credentials.json` and `.claude.json` masked (tmpfs on directories, /dev/null on files);
- `<repo>/plugin`, read-only;
- the directory of the claude binary, read-only;
- the work directory, read-write (it holds `.claude/settings.json`, the L4 gate hook, the
  empty personal layer and the paper cache).

/tmp is a fresh tmpfs, the PID namespace is private, the network is shared (the API and,
in the F arms, arXiv through fetch_paper.py only).
"""
from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path

from .common import BENCH, WORKDIR

KEEP_IN_CONFIG = {".credentials.json", ".claude.json"}
GATE_SRC = BENCH / "l4_gate.py"


def fetch_paper_path(repo: Path) -> Path:
    return Path(repo) / "plugin" / "skills" / "observational-cosmology" / "scripts" / "fetch_paper.py"


def config_dir(env: Mapping[str, str] | None = None) -> Path:
    src: Mapping[str, str] = os.environ if env is None else env
    return Path(src.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude-alt")


def prepare_workdir(workdir: Path, repo: Path) -> Path:
    """Write <workdir>/.claude/settings.json with the Bash gate hook; create the empty layer."""
    workdir = Path(workdir)
    (workdir / ".claude").mkdir(parents=True, exist_ok=True)
    (workdir / "empty-layer").mkdir(exist_ok=True)
    (workdir / ".cache").mkdir(exist_ok=True)
    gate = workdir / ".claude" / "l4_gate.py"
    shutil.copyfile(GATE_SRC, gate)
    hook = f"/usr/bin/python3 {gate} {fetch_paper_path(repo)}"
    settings = {"hooks": {"PreToolUse": [{"matcher": "Bash",
                                          "hooks": [{"type": "command", "command": hook}]}]}}
    (workdir / ".claude" / "settings.json").write_text(json.dumps(settings, indent=1) + "\n")
    return workdir


def masks(cfg: Path) -> list[str]:
    """bwrap arguments masking every config entry except the credentials and .claude.json."""
    out: list[str] = []
    if not cfg.is_dir():
        return out
    for entry in sorted(cfg.iterdir()):
        if entry.name in KEEP_IN_CONFIG or entry.is_symlink():
            continue  # symlinks into $HOME dangle once $HOME is a tmpfs
        if entry.is_dir():
            out += ["--tmpfs", str(entry)]
        else:
            out += ["--ro-bind", "/dev/null", str(entry)]
    return out


def wrap(cmd: list[str], *, repo: Path, workdir: Path = WORKDIR, env: dict | None = None,
         home: Path | None = None) -> list[str]:
    """Return the bwrap command line that runs `cmd` (cmd[0] = claude binary) in the sandbox."""
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise SystemExit("bwrap not found: the benchmark refuses to run without the sandbox")
    home = Path(home or Path.home())
    cfg = config_dir(env)
    binary = Path(os.path.realpath(cmd[0]))
    workdir = Path(workdir).resolve()
    plugin = (Path(repo) / "plugin").resolve()
    # The plugin is visible only to the arms that load it. A skills-off arm with Bash (D0F) sees only the
    # scripts directory, for fetch_paper.py; a skills-off arm with Read/Grep/Glob must not read the skills.
    tools = cmd[cmd.index("--tools") + 1].split(",") if "--tools" in cmd else []
    if "--plugin-dir" in cmd:
        plugin_bind = ["--ro-bind", str(plugin), str(plugin)]
    elif "Bash" in tools:
        scripts = fetch_paper_path(repo).resolve().parent
        plugin_bind = ["--ro-bind", str(scripts), str(scripts)]
    else:
        plugin_bind = []
    args = [bwrap, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
            "--unshare-pid", "--die-with-parent", "--tmpfs", str(home),
            "--bind", str(cfg), str(cfg), *masks(cfg),
            *plugin_bind,
            "--ro-bind", str(binary.parent), str(binary.parent),
            "--bind", str(workdir), str(workdir), "--chdir", str(workdir),
            # per-call caches: fetch_paper.py caches papers under XDG_CACHE_HOME, and a cache shared across
            # calls let a later no-fetch arm Grep papers fetched by an F arm (pilot, 2026-11-21)
            "--tmpfs", str(workdir / ".cache"),
            "--setenv", "PATH", f"{binary.parent}:/usr/bin:/bin",
            "--setenv", "XDG_CACHE_HOME", "/tmp/cache"]
    return args + [str(binary)] + list(cmd[1:])
