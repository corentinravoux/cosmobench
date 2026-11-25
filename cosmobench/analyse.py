"""Tables for the v3 benchmark, with the pre-registered hypotheses first.

    python -m cosmobench.analyse            prints and writes results/v3/analysis.md + CSVs

Primary hypotheses (plan 2026-11-20, section "Hypotheses"); D1A is the package as shipped:
  H1 recency    claim check, stratum A: right rate D1A > D0
  H2 honesty    claim check, stratum B: false-confirm rate on perturbed claims D1A <= D0
  H3 no harm    CosmoPaperQA: correct rate D1A - D0 > -0.05 (non-inferiority, 90 % CI)
  H4 value      claim check, all strata: right rate D1F > D0F (skills beyond a paper reader)
H1, H2 and H4 are tested with an exact McNemar test on claims and Holm-corrected; every
difference carries a 95 % bootstrap CI that resamples items (claims of one paper move together).
"""
from __future__ import annotations

import csv
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

from .common import RESULTS, SEED, read_jsonl

N_BOOT = 10000


# ---------- statistics ----------

def bootstrap_ci(values: list[float], n: int = N_BOOT, seed: int = SEED, level: float = 0.95) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(values[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    lo = means[int((1 - level) / 2 * n)]
    hi = means[min(n - 1, int((1 + level) / 2 * n))]
    return (lo, hi)


def cluster_diff(by_item: dict[str, list[tuple[float, float]]], n: int = N_BOOT, seed: int = SEED,
                 level: float = 0.95) -> tuple[float, float, float]:
    """Mean of (b - a) over all units, with a CI that resamples items (clusters)."""
    keys = sorted(by_item)
    if not keys:
        nan = float("nan")
        return (nan, nan, nan)

    def stat(ks):
        pairs = [p for k in ks for p in by_item[k]]
        return sum(b - a for a, b in pairs) / len(pairs) if pairs else 0.0

    rng = random.Random(seed)
    boots = sorted(stat([keys[rng.randrange(len(keys))] for _ in keys]) for _ in range(n))
    return (stat(keys), boots[int((1 - level) / 2 * n)], boots[min(n - 1, int((1 + level) / 2 * n))])


def mcnemar_exact(a: list[bool], b: list[bool]) -> float:
    """Two-sided exact McNemar p on paired successes."""
    n01 = sum((not x) and y for x, y in zip(a, b))
    n10 = sum(x and (not y) for x, y in zip(a, b))
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=lambda k: ps[k])
    out, running = {}, 0.0
    for k, name in enumerate(order):
        running = max(running, min(1.0, (len(order) - k) * ps[name]))
        out[name] = running
    return out


def cohen_kappa(x: list[str], y: list[str]) -> float:
    if not x:
        return float("nan")
    cats = sorted(set(x) | set(y))
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


# ---------- claim check ----------

def cc_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    pert = [r for r in rows if r["perturbed"]]
    true = [r for r in rows if not r["perturbed"]]
    def frac(rs, o):
        return (sum(r["outcome"] == o for r in rs) / len(rs)) if rs else float("nan")
    return {"claims": n, "right": frac(rows, "right"), "abstain": frac(rows, "abstain"),
            "missing": frac(rows, "missing"), "false_confirm_on_perturbed": frac(pert, "false_confirm"),
            "false_alarm_on_true": frac(true, "false_alarm"),
            "right_given_committed": (sum(r["outcome"] == "right" for r in rows) /
                                      max(1, sum(r["outcome"] in ("right", "false_confirm", "false_alarm") for r in rows)))}


def cc_paired(rows: list[dict], a: str, b: str, metric: str, subset=lambda r: True) -> dict:
    """Paired comparison of arm b against arm a on claims present for both."""
    key = lambda r: (r["item_id"], r["claim"])
    A = {key(r): r for r in rows if r["arm"] == a and subset(r)}
    B = {key(r): r for r in rows if r["arm"] == b and subset(r)}
    common = sorted(set(A) & set(B))
    hit = (lambda r: r["outcome"] == "right") if metric == "right" else (lambda r: r["outcome"] == "false_confirm")
    by_item: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for k in common:
        by_item[k[0]].append((float(hit(A[k])), float(hit(B[k]))))
    d, lo, hi = cluster_diff(by_item)
    p = mcnemar_exact([hit(A[k]) for k in common], [hit(B[k]) for k in common])
    return {"a": a, "b": b, "metric": metric, "n_claims": len(common), "diff": d, "ci_lo": lo, "ci_hi": hi, "p": p}


# ---------- CosmoPaperQA ----------

def qa_metrics(grades: list[dict]) -> dict:
    gs = [g for g in grades if g["grade"] in ("correct", "partial", "incorrect", "abstain")]
    n = len(gs)
    att = [g for g in gs if g["grade"] != "abstain"]
    c = [1.0 if g["grade"] == "correct" else 0.0 for g in gs]
    lo, hi = bootstrap_ci(c)
    return {"n": n, "correct": sum(c) / n if n else float("nan"), "ci_lo": lo, "ci_hi": hi,
            "lenient": (sum(1.0 if g["grade"] == "correct" else 0.5 if g["grade"] == "partial" else 0 for g in gs) / n) if n else float("nan"),
            "abstain": (n - len(att)) / n if n else float("nan"),
            "correct_given_attempted": (sum(g["grade"] == "correct" for g in att) / len(att)) if att else float("nan"),
            "false_claim": (sum(bool(g.get("false_claim")) for g in gs) / n) if n else float("nan"),
            "cites_source": (sum(bool(g.get("cites_source")) for g in gs) / n) if n else float("nan"),
            "unjudged": sum(g["grade"] == "unjudged" for g in grades)}


def qa_paired(grades: list[dict], a: str, b: str, level: float = 0.95) -> dict:
    A = {g["item_id"]: g for g in grades if g["arm"] == a and g["grade"] != "unjudged"}
    B = {g["item_id"]: g for g in grades if g["arm"] == b and g["grade"] != "unjudged"}
    common = sorted(set(A) & set(B))
    ok = lambda g: g["grade"] == "correct"
    by_item = {k: [(float(ok(A[k])), float(ok(B[k])))] for k in common}
    d, lo, hi = cluster_diff(by_item, level=level)
    return {"a": a, "b": b, "n": len(common), "diff": d, "ci_lo": lo, "ci_hi": hi,
            "p": mcnemar_exact([ok(A[k]) for k in common], [ok(B[k]) for k in common])}


# ---------- report ----------

def fmt(x) -> str:
    if isinstance(x, float):
        return "nan" if x != x else f"{x:.3f}"
    return str(x)


def table(rows: list[dict], cols: list[str]) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cost_rows(bench: str) -> list[dict]:
    recs = [r for r in read_jsonl(RESULTS / bench / "answers.jsonl") if r.get("status") == "ok"]
    out = []
    for arm in sorted({r["arm"] for r in recs}):
        rs = [r for r in recs if r["arm"] == arm]
        sk = [r for r in rs if r.get("skills_invoked")]
        out.append({"bench": bench, "arm": arm, "calls": len(rs),
                    "median_cost_usd": statistics.median(r.get("cost_usd") or 0 for r in rs),
                    "total_cost_usd": sum(r.get("cost_usd") or 0 for r in rs),
                    "median_s": statistics.median((r.get("duration_ms") or 0) / 1000 for r in rs),
                    "skill_rate": len(sk) / len(rs),
                    "fetch_calls": sum(r.get("fetch_calls") or 0 for r in rs),
                    "leaks": sum(bool(r.get("leak")) for r in rs)})
    return out


def main() -> int:
    md = ["# cosmoskill benchmark v3 — analysis", ""]
    hyp: list[dict] = []
    cc = read_jsonl(RESULTS / "claimcheck" / "grades.jsonl")
    if cc:
        rows = []
        for arm in sorted({r["arm"] for r in cc}):
            for s in ("K", "A", "B", "all"):
                sub = [r for r in cc if r["arm"] == arm and (s == "all" or r["stratum"] == s)]
                if sub:
                    rows.append({"arm": arm, "stratum": s, **cc_metrics(sub)})
        h1 = cc_paired(cc, "D0", "D1A", "right", lambda r: r["stratum"] == "A")
        h2 = cc_paired(cc, "D0", "D1A", "false_confirm", lambda r: r["stratum"] == "B" and r["perturbed"])
        h4 = cc_paired(cc, "D0F", "D1F", "right")
        hyp += [{"H": "H1 recency (A: right, D1A-D0)", **h1}, {"H": "H2 honesty (B perturbed: false confirm, D1A-D0)", **h2},
                {"H": "H4 value (all: right, D1F-D0F)", **h4}]
        sec_rows = []
        for a, b in (("D0", "D1A"), ("D0", "D1"), ("D1A", "D1"), ("D0", "D0F"), ("D1", "D1F")):
            for st in ("K", "A", "B"):
                sec_rows.append({**cc_paired(cc, a, b, "right", lambda r, st=st: r["stratum"] == st), "stratum": st})
        md += ["## Claim check", ""] + table(rows, ["arm", "stratum", "claims", "right", "abstain", "missing",
                                                   "false_confirm_on_perturbed", "false_alarm_on_true",
                                                   "right_given_committed"]) + [""]
        md += ["Secondary paired contrasts (right rate, b − a):", ""]
        md += table(sec_rows, ["a", "b", "stratum", "n_claims", "diff", "ci_lo", "ci_hi", "p"]) + [""]
        in_sk = [r for r in cc if r["stratum"] == "A"]
        if in_sk:
            md += ["Stratum A split by whether the skill files state the value:", ""]
            md += table([{"arm": arm, "in_skill_text": flag, **cc_metrics(sub)}
                         for arm in sorted({r["arm"] for r in in_sk}) for flag in (True, False)
                         if (sub := [r for r in in_sk if r["arm"] == arm and r["in_skill_text"] == flag])],
                        ["arm", "in_skill_text", "claims", "right", "abstain", "false_confirm_on_perturbed"]) + [""]
        write_csv(RESULTS / "claimcheck_table.csv", rows)
    qa = read_jsonl(RESULTS / "cosmopaperqa" / "grades.jsonl")
    if qa:
        rows = []
        for arm in sorted({g["arm"] for g in qa}):
            for s in ("covered", "thin", "all"):
                sub = [g for g in qa if g["arm"] == arm and (s == "all" or g["stratum"] == s)]
                if sub:
                    rows.append({"arm": arm, "stratum": s, **qa_metrics(sub)})
        h3 = qa_paired(qa, "D0", "D1A", level=0.90)
        h3["noninferior"] = h3["ci_lo"] > -0.05
        hyp.append({"H": "H3 no harm (CPQA: correct, D1A-D0, 90 % CI > -0.05)", **h3})
        md += ["## CosmoPaperQA", ""] + table(rows, ["arm", "stratum", "n", "correct", "ci_lo", "ci_hi", "lenient",
                                                     "abstain", "correct_given_attempted", "false_claim",
                                                     "cites_source", "unjudged"]) + [""]
        md += table([qa_paired(qa, a, b) for a, b in (("D0", "D1"), ("D1A", "D1"))],
                    ["a", "b", "n", "diff", "ci_lo", "ci_hi", "p"]) + [""]
        j2 = {(g["item_id"], g["arm"]): g["grade"] for g in read_jsonl(RESULTS / "cosmopaperqa" / "grades_judge2.jsonl")}
        pairs = [(g["grade"], j2[(g["item_id"], g["arm"])]) for g in qa if (g["item_id"], g["arm"]) in j2]
        if pairs:
            md += [f"Judge–judge agreement (Opus vs Sonnet), {len(pairs)} answers: "
                   f"raw {sum(a == b for a, b in pairs) / len(pairs):.3f}, "
                   f"Cohen's κ {cohen_kappa([a for a, _ in pairs], [b for _, b in pairs]):.3f}.", ""]
        sheet = RESULTS / "cosmopaperqa" / "spotcheck.csv"
        if sheet.exists():
            hs = [r for r in csv.DictReader(sheet.open(encoding="utf-8")) if r.get("human_grade", "").strip()]
            if hs:
                md += [f"Judge–human agreement, {len(hs)} rows: κ "
                       f"{cohen_kappa([r['judge_grade'] for r in hs], [r['human_grade'].strip() for r in hs]):.3f}.", ""]
        write_csv(RESULTS / "cosmopaperqa_table.csv", rows)
    if hyp:
        adj = holm({h["H"]: h["p"] for h in hyp if not h["H"].startswith("H3")})
        for h in hyp:
            h["p_holm"] = adj.get(h["H"], float("nan"))
        md = md[:2] + ["## Pre-registered hypotheses", ""] + table(
            hyp, ["H", "n_claims", "n", "diff", "ci_lo", "ci_hi", "p", "p_holm", "noninferior"]) + [""] + md[2:]
        write_csv(RESULTS / "hypotheses.csv", hyp)
    tp = read_jsonl(RESULTS / "tpbench" / "grades.jsonl")
    if tp:
        md += ["## TPBench smoke", ""] + table(tp, [k for k in ("item_id", "arm", "verdict", "reason") if k in tp[0]]) + [""]
    costs = [r for b in ("claimcheck", "cosmopaperqa", "tpbench") for r in cost_rows(b)]
    if costs:
        md += ["## Cost and tool use", ""] + table(costs, list(costs[0])) + [""]
        write_csv(RESULTS / "cost_table.csv", costs)
    text = "\n".join(md) + "\n"
    (RESULTS / "analysis.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
