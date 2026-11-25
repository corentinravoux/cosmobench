"""Token and cost summary per arm; projection to the full run.

    python -m cosmobench.cost [--project]
"""
from __future__ import annotations

import argparse
import statistics
import sys

from .arms import ARMS_BY_BENCH
from .common import RESULTS, read_jsonl

TOKEN_KEYS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")
# v5 full run: calls per arm (claim check: D0/D1A on 160 items, F arms on the 60 A+B items)
FULL_ITEMS = {"claimcheck": {"D0": 160, "D1A": 160, "D0F": 60, "D1F": 60}, "cosmopaperqa": 40, "tpbench": 6}
N_JUDGES = {"cosmopaperqa": 2}


def tokens(rec: dict) -> int:
    u = rec.get("usage") or {}
    return sum(int(u.get(k) or 0) for k in TOKEN_KEYS)


def summarise(records: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for arm in sorted({r["arm"] for r in records}):
        rs = [r for r in records if r["arm"] == arm and r.get("status") == "ok"]
        if not rs:
            continue
        toks = [tokens(r) for r in rs]
        out[arm] = {"calls": len(rs), "median_tokens": statistics.median(toks), "total_tokens": sum(toks),
                    "total_cost_usd": sum(r.get("cost_usd") or 0.0 for r in rs),
                    "mean_cost_usd": sum(r.get("cost_usd") or 0.0 for r in rs) / len(rs),
                    "median_duration_ms": statistics.median([r.get("duration_ms") or 0 for r in rs])}
    return out


def project(summary: dict, n_calls_per_arm: dict | int) -> dict:
    out = {}
    for arm, s in summary.items():
        n = n_calls_per_arm if isinstance(n_calls_per_arm, int) else n_calls_per_arm.get(arm, 0)
        f = n / s["calls"]
        out[arm] = {"calls": n, "tokens": s["total_tokens"] * f, "cost_usd": s["total_cost_usd"] * f,
                    "hours": s["median_duration_ms"] * n / 3.6e6}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", action="store_true")
    a = ap.parse_args(argv)
    grand = {"cost_usd": 0.0, "hours": 0.0, "calls": 0}
    for bench in ("claimcheck", "cosmopaperqa", "tpbench"):
        recs = read_jsonl(RESULTS / bench / "answers.jsonl")
        if not recs:
            continue
        s = summarise(recs)
        print(f"## {bench}")
        for arm, v in s.items():
            print(f"  {arm:5s} calls={v['calls']:3d} median_tok={v['median_tokens']:>9,.0f} "
                  f"mean_cost=${v['mean_cost_usd']:.3f} median_s={v['median_duration_ms'] / 1000:.0f}")
        if a.project:
            p = project(s, FULL_ITEMS[bench])
            for arm, v in p.items():
                grand["cost_usd"] += v["cost_usd"]
                grand["hours"] += v["hours"]
                grand["calls"] += v["calls"]
            print("  projected: " + ", ".join(f"{arm} ${v['cost_usd']:.0f}/{v['hours']:.1f}h" for arm, v in p.items()))
        judged = read_jsonl(RESULTS / bench / "grades.jsonl")
        jc = [g.get("judge_cost_usd") or 0 for g in judged if "judge_cost_usd" in g]
        if jc:
            mean = sum(jc) / len(jc)
            print(f"  judge: {len(jc)} graded, mean ${mean:.3f}")
            if a.project and bench in N_JUDGES:
                n = FULL_ITEMS[bench] * len(ARMS_BY_BENCH[bench]) * N_JUDGES[bench]
                grand["cost_usd"] += mean * n
                grand["calls"] += n
                print(f"  judge projected: {n} calls ${mean * n:.0f}")
    if a.project:
        print(f"TOTAL projected (API list price, not subscription usage): ${grand['cost_usd']:.0f}, "
              f"{grand['calls']} calls, answer time {grand['hours']:.1f} h sequential")
    return 0


if __name__ == "__main__":
    sys.exit(main())
