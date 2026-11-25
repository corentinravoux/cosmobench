"""Grade claim-check answers mechanically: each claim's verdict against the known truth.

    python -m cosmobench.grade_claims [--built]   results/v5/claimcheck/answers.jsonl -> grades.jsonl

Per claim, the verdict is `correct`, `wrong`, `cannot` (cannot verify) or `missing` (no
parsable line). Outcomes and net-score points (plan v5, "Scoring"):
    right          +1  verdict matches the truth (correct on a true claim, wrong on a perturbed one)
    false_alarm    -1  `wrong` on a true claim
    false_confirm  -1  `correct` on a perturbed claim (the dangerous error for a fact-checker)
    abstain         0  `cannot`
    missing         0  no verdict found (counted; V2 requires < 2 %)
Claims the audit marked as a wrong ground truth (`excluded`) are graded but flagged, and the
analysis leaves them out. `states_true` (perturbed number claims answered `wrong`): the reply
line states the paper's true value, a mechanical sign of knowing rather than guessing.
`--built` grades against the unaudited built set (pilot).
"""
from __future__ import annotations

import re
import sys

from .common import RESULTS, read_jsonl, write_jsonl
from .data import load_items

LINE_RE = re.compile(
    r"^\W{0,4}(?:claim\s*)?(\d+)\s*\**\s*[:.)\]—–-]\s*\**\s*(correct|wrong|incorrect|cannot\s+(?:be\s+)?verif\w*|unverifiable)(.*)$",
    re.I | re.M)
SCORE = {"right": 1, "false_alarm": -1, "false_confirm": -1, "abstain": 0, "missing": 0}


def parse_lines(answer: str, n: int) -> list[tuple[str, str]]:
    """(verdict, rest of the line) per claim; the first line for a claim number wins."""
    found: dict[int, tuple[str, str]] = {}
    for m in LINE_RE.finditer(answer or ""):
        k, word = int(m.group(1)), m.group(2).lower()
        if not 1 <= k <= n or k in found:
            continue
        v = "correct" if word == "correct" else "wrong" if word in ("wrong", "incorrect") else "cannot"
        found[k] = (v, m.group(3) or "")
    return [found.get(k, ("missing", "")) for k in range(1, n + 1)]


def parse_verdicts(answer: str, n: int) -> list[str]:
    return [v for v, _ in parse_lines(answer, n)]


def outcome(verdict: str, perturbed: bool) -> str:
    if verdict in ("cannot", "missing"):
        return "abstain" if verdict == "cannot" else "missing"
    if verdict == "correct":
        return "false_confirm" if perturbed else "right"
    return "right" if perturbed else "false_alarm"


def states_value(rest: str, value: str) -> bool:
    """The line states the value, numerically (0.30 and 0.3 match)."""
    try:
        v = float(value)
    except ValueError:
        return False
    return any(abs(float(x) - v) < 1e-9 for x in re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d])", rest or ""))


def grade_record(rec: dict, item: dict) -> list[dict]:
    truth = item["reference"]["claims"]
    failed = rec.get("status") != "ok"
    lines = [("missing", "")] * len(truth) if failed else parse_lines(rec.get("answer", ""), len(truth))
    rows = []
    for k, (t, (v, rest)) in enumerate(zip(truth, lines)):
        o = outcome(v, t["perturbed"])
        st = (states_value(rest, t["value_true"]) if t.get("type", "number") == "number" and t["perturbed"]
              and v == "wrong" else None)
        rows.append({"item_id": rec["item_id"], "arm": rec["arm"], "rep": rec["rep"], "claim": k + 1,
                     "stratum": item["stratum"], "block": item.get("block", item["id"]),
                     "type": t.get("type", "number"), "mode": t.get("mode"), "paper": t.get("paper"),
                     "perturbed": t["perturbed"], "excluded": bool(t.get("excluded")),
                     "in_skill_text": t.get("in_skill_text"), "verdict": v, "outcome": o, "score": SCORE[o],
                     "states_true": st, "failed": rec.get("status") if failed else ""})
    return rows


def main(argv=None) -> int:
    built = "--built" in (argv if argv is not None else sys.argv[1:])
    items = {it["id"]: it for it in load_items("claimcheck", built=built)}
    # one record per (item, arm, rep): the last `ok` one, else the last failure (error, timeout, leak),
    # graded as missing so that failures stay in the denominator of every arm (finding 13)
    last: dict = {}
    for rec in read_jsonl(RESULTS / "claimcheck" / "answers.jsonl"):
        if rec["item_id"] not in items:
            continue
        key = (rec["item_id"], rec["arm"], rec["rep"])
        if rec.get("status") == "ok" or last.get(key, {}).get("status") != "ok":
            last[key] = rec
    rows = []
    for rec in last.values():
        rows += grade_record(rec, items[rec["item_id"]])
    write_jsonl(RESULTS / "claimcheck" / "grades.jsonl", rows)
    miss = sum(r["verdict"] == "missing" for r in rows)
    fails = sum(r["status"] != "ok" for r in last.values())
    print(f"{len(rows)} claims graded, {miss} without a parsable verdict, {fails} failed calls graded as missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
