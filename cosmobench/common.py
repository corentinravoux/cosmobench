"""Paths and JSONL helpers shared by the benchmark modules."""
from __future__ import annotations

import json
import os
from pathlib import Path

HARNESS_VERSION = "5"
BENCH = Path(__file__).resolve().parent  # the package: pin files, l4_gate.py
# Data, results and the work directory live outside the package, never in git.
# COSMOBENCH_HOME sets their root; the default is the checkout root (editable install).
ROOT = Path(os.environ.get("COSMOBENCH_HOME") or BENCH.parent).expanduser().resolve()
DATA = ROOT / "data"
RUN = os.environ.get("BENCH_RUN", "")   # "pilot": results/v5/pilot/, kept apart from the full run
CLAIM_SET = os.environ.get("BENCH_SET", "v5")  # claim-check data dir under data/; "v5k0fresh": 1.1.1 re-test K0 set
RESULTS = ROOT / "results" / f"v{HARNESS_VERSION}" / RUN if RUN else ROOT / "results" / f"v{HARNESS_VERSION}"
WORKDIR = ROOT / "workdir"
CONFIG = BENCH / os.environ.get("BENCH_CONFIG", "config.json")  # pin file; config_v111.json: 1.1.1 re-test
# the skill repository under test; COSMOSKILL_REPO or --repo override
DEFAULT_REPO = Path(os.environ.get("COSMOSKILL_REPO") or Path.home() / "Software" / "cosmoskill").expanduser()
CLAUDE_MODEL = "claude-opus-5-5"
# answer model; local mode (cosmobench.local): the llm-host recipe name = the llama-server alias
MODEL = os.environ.get("BENCH_MODEL") or CLAUDE_MODEL
JUDGE2_MODEL = "claude-sonnet-5-5"  # second judge, another model, for judge-judge agreement
SEED = 20261016


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def write_jsonl(path: Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
