"""Blind LLM judges for CosmoPaperQA answers, provenance score, spot-check sheet.

    python -m cosmobench.judge                      both judges (Opus, then Sonnet), resumable
    python -m cosmobench.judge --judge judge2       one judge only
    python -m cosmobench.judge --sheet              write the spot-check sheet only

v3: a second judge from another model (JUDGE2, Sonnet) grades every answer, for a
judge-judge agreement next to the human spot-check. Sentences that talk about the skills
themselves ("the skill files do not cover ...") are removed before judging: they reveal the
arm and carry no answer content. The v2 pattern that removed every "L1"-"L4" token is gone
(it also removed "L2", the Planck orbit).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

from .arms import JUDGE, JUDGE2, build_command, build_env, find_claude_bin
from .common import DEFAULT_REPO, RESULTS, SEED, WORKDIR, append_jsonl, read_jsonl
from .data import load_items
from .local import is_local
from .prompts import JUDGE_SCHEMA, judge_prompt
from .run import make_executor, parse_stream

GRADES = ("correct", "partial", "incorrect", "abstain")
ARXIV_RE = r"(?:arXiv:\s*)?\b\d{4}\.\d{4,5}(?:v\d+)?\b"
MARKER_RES = [
    re.compile(r"\[(?:src|sec|doc|der)\s*:[^\]]*\]", re.I),
    re.compile(r"\[cosmoskill[^\]]*\]", re.I),
    re.compile(r"\bdoi:\s*\S+|https?://(?:dx\.)?doi\.org/\S+", re.I),
    re.compile(r"https?://arxiv\.org/\S+", re.I),
    re.compile(ARXIV_RE, re.I),
    re.compile(r"`?\b(?:cosmoskill:)?(?:cosmo-[a-z0-9-]+|observational-cosmology)\b`?", re.I),
    re.compile(r"\(?\blevel\s*[1-4]\b\)?", re.I),
]
META_TERMS = re.compile(
    r"\b(?:the\s+|its\s+|their\s+)?(?:[\w-]+\s+)?(?:skill files?|skills?|reference files?|"
    r"dossiers?|cosmoskill|SKILL\.md|personal layer)\b"
    r"|`?\bfetch_paper(?:\.py)?\b`?|\bL4\b|\blevel[- ]4\b", re.I)
JUDGES = {"judge": JUDGE, "judge2": JUDGE2}
GRADE_FILES = {"judge": "grades.jsonl", "judge2": "grades_judge2.jsonl"}
SPOT_PER_ARM = {"cosmopaperqa": 15}


def neutralise(text: str) -> str:
    """Replace words naming the skills or tools by "my sources"; keep the sentence."""
    text = META_TERMS.sub("my sources", text or "")
    return re.sub(r"my sources(?:\s*[(,/]?\s*my sources\)?)+", "my sources", text)


def strip_markers(answer: str) -> str:
    text = answer or ""
    for rx in MARKER_RES:
        text = rx.sub("", text)
    text = neutralise(text)
    text = re.sub(r"\(\s*[,;]?\s*\)", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def provenance(answer: str, item: dict) -> dict:
    raw = answer or ""
    ids = {m.split("v")[0] for m in re.findall(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", raw)}
    doi = (item.get("reference", {}).get("doi") or "").lower()
    doi_core = doi.split("doi.org/")[-1] if doi else ""
    cites_any = bool(ids) or bool(re.search(r"doi(?:\.org)?[:/]", raw, re.I))
    src = item.get("source_arxiv", "")
    cites_source = bool(src and src in ids) or bool(doi_core and doi_core in raw.lower())
    return {"cites_any": cites_any, "cites_source": cites_source}


def parse_grade(text) -> dict | None:
    obj = text
    if isinstance(text, str):
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict) or obj.get("grade") not in GRADES:
        return None
    return {"grade": obj["grade"], "false_claim": bool(obj.get("false_claim", False)),
            "declares_gap": bool(obj.get("declares_gap", False)),
            "reason": str(obj.get("reason", ""))[:500]}


def judge_order(records: list[dict], seed: int = SEED) -> list[dict]:
    recs = sorted(records, key=lambda r: (r["item_id"], r["arm"], r["rep"]))
    random.Random(seed).shuffle(recs)
    return recs


def judge_one(item: dict, answer: str, executor, *, claude_bin: str, repo: Path, env: dict,
              workdir: Path, timeout: float = 1800, arm=JUDGE) -> dict:
    cmd = build_command(arm, claude_bin=claude_bin, repo=repo, json_schema=JUDGE_SCHEMA, env=env)
    prompt = judge_prompt(item, strip_markers(answer))
    calls, cost = 0, 0.0
    for _ in range(2):
        calls += 1
        try:
            rec = parse_stream(executor(cmd, env, workdir, timeout, prompt))
        except subprocess.TimeoutExpired:
            continue
        cost += rec.get("cost_usd") or 0.0
        if rec["status"] == "error" and "limit" in rec["error"].lower():
            return {"grade": "error", "error": rec["error"], "judge_calls": calls, "judge_cost_usd": cost}
        g = parse_grade(rec.get("structured_output")) or parse_grade(rec.get("answer", ""))
        if g:
            return {**g, "judge_calls": calls, "judge_cost_usd": cost}
    return {"grade": "unjudged", "false_claim": None, "declares_gap": None, "reason": "",
            "judge_calls": calls, "judge_cost_usd": cost}


def run_judge(bench: str, executor, *, claude_bin: str, repo: Path = DEFAULT_REPO,
              workdir: Path = WORKDIR, base_env: dict | None = None, log=print,
              judge: str = "judge") -> int:
    items = {it["id"]: it for it in load_items(bench)}
    answers = [r for r in read_jsonl(RESULTS / bench / "answers.jsonl") if r.get("status") == "ok"]
    out = RESULTS / bench / GRADE_FILES[judge]
    done = {(g["item_id"], g["arm"], g["rep"]) for g in read_jsonl(out) if g["grade"] != "error"}
    env = build_env(dict(os.environ) if base_env is None else base_env, workdir)
    errors = 0
    for rec in judge_order(answers):
        key = (rec["item_id"], rec["arm"], rec["rep"])
        if key in done or rec["item_id"] not in items:
            continue
        item = items[rec["item_id"]]
        g = judge_one(item, rec["answer"], executor, claude_bin=claude_bin, repo=repo,
                      env=env, workdir=workdir, arm=JUDGES[judge])
        if g["grade"] == "error":
            errors += 1
            log(f"judge error: {g['error'][:200]}")
            if errors >= 3:
                return 2
            continue
        errors = 0
        row = {"item_id": rec["item_id"], "arm": rec["arm"], "rep": rec["rep"],
               "stratum": item.get("stratum", ""), "judge": judge, **g, **provenance(rec["answer"], item)}
        append_jsonl(out, row)
        log(f"{judge} {key}: {g['grade']} false_claim={g.get('false_claim')}")
    return 0


def spotcheck(bench: str, seed: int = SEED) -> list[dict]:
    items = {it["id"]: it for it in load_items(bench)}
    answers = {(r["item_id"], r["arm"], r["rep"]): r for r in read_jsonl(RESULTS / bench / "answers.jsonl")
               if r.get("status") == "ok"}
    grades = [g for g in read_jsonl(RESULTS / bench / "grades.jsonl") if g["grade"] in GRADES]
    rng = random.Random(seed)
    per_arm = SPOT_PER_ARM[bench]
    by_arm: dict[str, list[dict]] = {}
    for g in sorted(grades, key=lambda g: (g["item_id"], g["arm"], g["rep"])):
        by_arm.setdefault(g["arm"], []).append(g)
    rows = []
    for arm in sorted(by_arm):
        n = per_arm
        pool = by_arm[arm]
        wrong = [g for g in pool if g["grade"] in ("incorrect", "partial")]
        abst = [g for g in pool if g["grade"] == "abstain"]
        right = [g for g in pool if g["grade"] == "correct"]
        pick = rng.sample(wrong, min(4, len(wrong))) + rng.sample(abst, min(2, len(abst)))
        rest = [g for g in pool if g not in pick]
        right_left = [g for g in right if g not in pick]
        fill = rng.sample(right_left, min(n - len(pick), len(right_left)))
        pick += fill
        if len(pick) < n:
            others = [g for g in rest if g not in pick]
            pick += rng.sample(others, min(n - len(pick), len(others)))
        for g in pick:
            it = items[g["item_id"]]
            a = answers.get((g["item_id"], g["arm"], g["rep"]), {})
            rows.append({"item_id": g["item_id"], "arm": g["arm"], "stratum": it.get("stratum", ""),
                         "has_key_passage": bool(it["reference"].get("key_passage")),
                         "question": it["prompt_fields"]["question"], "ideal": it["reference"]["ideal"],
                         "candidate_raw": a.get("answer", ""), "judge_grade": g["grade"],
                         "judge_false_claim": g["false_claim"], "human_grade": "",
                         "human_false_claim": ""})
    rng.shuffle(rows)
    return rows


def write_sheet(bench: str) -> Path:
    rows = spotcheck(bench)
    path = RESULTS / bench / "spotcheck.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["item_id"])
        w.writeheader()
        w.writerows(rows)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["cosmopaperqa"], default="cosmopaperqa")
    ap.add_argument("--judge", choices=sorted(JUDGES))
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    a = ap.parse_args(argv)
    if is_local() and not a.sheet:
        sys.exit("judges are Claude (Opus + Sonnet): unset BENCH_LOCAL_URL to grade a local run (keep BENCH_RUN)")
    if not a.sheet:
        executor = make_executor(a.repo)
        for j in ([a.judge] if a.judge else ["judge", "judge2"]):
            code = run_judge(a.bench, executor, claude_bin=find_claude_bin(), repo=a.repo, judge=j)
            if code:
                return code
    print(write_sheet(a.bench))
    return 0


if __name__ == "__main__":
    sys.exit(main())
