"""Analysis of the v5 benchmark, pre-registered hypotheses first (plan 2026-11-21, v5).

    python3 -m cosmobench.analyse_v5        -> results/v5/analysis.md, hypotheses.csv, claimcheck_table.csv
    BENCH_RUN=pilot python3 -m cosmobench.analyse_v5      the pilot's results/v5/pilot/

  H1 recency   A         net score D1A - D0             superiority
  H3 no harm   K u K0    net score D1A - D0             non-inferiority, margin -0.15 (strata pooled)
  H2 honesty   B         confident-error rate D1A - D0  non-inferiority, margin +0.05
  H4 level 4   A u B     net score D1F - D0F            non-inferiority, margin -0.05
Fixed-sequence testing in that order at one-sided alpha = 0.05: a hypothesis is confirmatory only if
every earlier one passed. Unit of inference: the block (4 papers, 2 items; papers and calls nested).
Superiority: sign flips of block sums. Non-inferiority and CIs: stratified cluster-robust t on blocks,
df = sum over strata of (blocks - 1). (Review 2026-11-21: a percentile bootstrap over 15-20 clusters
is too narrow, in the dangerous direction for non-inferiority.)
"""
from __future__ import annotations

import csv
import itertools
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import t as student_t

from .analyse import cohen_kappa, qa_metrics, qa_paired
from .common import RESULTS, read_jsonl

ALPHA = 0.05
N_FLIPS = 20000
SEED_A = 20261122
ORDER = ("H1", "H3", "H2", "H4")
MARGINS = {"H2": 0.05, "H3": 0.15, "H4": 0.05}
TYPES = ("number", "dataset", "conclusion", "attribution")


# ---------------------------------------------------------------- per-claim values
def is_error(r: dict) -> float:
    return 1.0 if r["outcome"] in ("false_confirm", "false_alarm") else 0.0


METRICS = {"net": lambda r: float(r["score"]), "error": is_error,
           "right": lambda r: 1.0 if r["outcome"] == "right" else 0.0,
           "abstain": lambda r: 1.0 if r["outcome"] == "abstain" else 0.0}


def scored(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not r.get("excluded")]


def paired_blocks(rows: list[dict], a: str, b: str, metric: str, subset=lambda r: True):
    """{stratum: {block: [sum of (b - a) over its claims, number of claims]}}, claims answered in both arms
    (failed calls are graded `missing`, so they stay in)."""
    f = METRICS[metric]
    key = lambda r: (r["item_id"], r["claim"])
    A = {key(r): r for r in scored(rows) if r["arm"] == a and subset(r)}
    B = {key(r): r for r in scored(rows) if r["arm"] == b and subset(r)}
    out: dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for k in sorted(set(A) & set(B)):
        r = A[k]
        cell = out[r["stratum"]][r["block"]]
        cell[0] += f(B[k]) - f(A[k])
        cell[1] += 1
    return {s: dict(v) for s, v in out.items()}


def estimate(blocks) -> dict:
    """Pooled mean difference over claims, stratified cluster-robust SE, df."""
    D = sum(d for s in blocks.values() for d, _ in s.values())
    N = sum(n for s in blocks.values() for _, n in s.values())
    if not N:
        return {"diff": float("nan"), "se": float("nan"), "df": 0, "n_claims": 0, "n_blocks": 0}
    var, df = 0.0, 0
    for s in blocks.values():
        Bs = len(s)
        if Bs < 2:
            continue
        ds, ns = sum(d for d, _ in s.values()), sum(n for _, n in s.values())
        m = ds / ns
        var += Bs / (Bs - 1) * sum((d - n * m) ** 2 for d, n in s.values())
        df += Bs - 1
    return {"diff": D / N, "se": math.sqrt(var) / N, "df": df, "n_claims": N,
            "n_blocks": sum(len(s) for s in blocks.values())}


def t_q(df: int, q: float) -> float:
    return float(student_t.ppf(q, max(df, 1)))


def ci(e: dict, level: float = 0.95) -> tuple[float, float]:
    if e["se"] != e["se"]:
        return (float("nan"), float("nan"))
    h = t_q(e["df"], 0.5 + level / 2) * e["se"]
    return (e["diff"] - h, e["diff"] + h)


def noninferiority(e: dict, margin: float, lower_is_better: bool = False) -> dict:
    """One-sided (1 - ALPHA) bound and p for H0: the difference is beyond the margin."""
    if not e["n_claims"]:
        return {"bound": float("nan"), "p": 1.0, "pass": False}
    tq = t_q(e["df"], 1 - ALPHA)
    if lower_is_better:
        bound = e["diff"] + tq * e["se"]
        z = (margin - e["diff"]) / e["se"] if e["se"] > 0 else (math.inf if e["diff"] < margin else -math.inf)
        ok = bound < margin
    else:
        bound = e["diff"] - tq * e["se"]
        z = (e["diff"] + margin) / e["se"] if e["se"] > 0 else (math.inf if e["diff"] > -margin else -math.inf)
        ok = bound > -margin
    p = float(student_t.sf(z, max(e["df"], 1))) if math.isfinite(z) else (0.0 if z > 0 else 1.0)
    return {"bound": bound, "p": p, "pass": ok}


def sign_flip_p(blocks, n: int = N_FLIPS, seed: int = SEED_A) -> float:
    """One-sided p for a positive pooled difference; each block's sum is flipped (exact up to 16 blocks)."""
    sums = [d for s in blocks.values() for d, _ in s.values()]
    if not sums:
        return 1.0
    obs = sum(sums)
    if len(sums) <= 16:
        hits = sum(sum(x * g for x, g in zip(sums, signs)) >= obs - 1e-12
                   for signs in itertools.product((1, -1), repeat=len(sums)))
        return hits / 2 ** len(sums)
    rng = random.Random(seed)
    hits = sum(sum(x if rng.random() < 0.5 else -x for x in sums) >= obs - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def power_if_zero(e: dict, margin: float) -> float:
    """Approximate power of the non-inferiority bound at the observed SE if the true difference is 0."""
    if not e["n_claims"] or e["se"] != e["se"]:
        return float("nan")
    if e["se"] == 0:
        return 1.0
    return float(student_t.cdf(margin / e["se"] - t_q(e["df"], 1 - ALPHA), max(e["df"], 1)))


def contrast(rows, a, b, metric, subset=lambda r: True) -> dict:
    blk = paired_blocks(rows, a, b, metric, subset)
    e = estimate(blk)
    lo, hi = ci(e)
    return {"a": a, "b": b, "metric": metric, **e, "ci_lo": lo, "ci_hi": hi, "_blocks": blk}


def arm_metrics(rows: list[dict]) -> dict:
    rs = scored(rows)
    n = len(rs)
    if not n:
        return {"claims": 0}
    frac = lambda o: sum(r["outcome"] == o for r in rs) / n
    pert = [r for r in rs if r["perturbed"]]
    true = [r for r in rs if not r["perturbed"]]
    return {"claims": n, "net": sum(r["score"] for r in rs) / n, "right": frac("right"),
            "error": sum(is_error(r) for r in rs) / n, "abstain": frac("abstain"), "missing": frac("missing"),
            "failed_calls_claims": sum(bool(r.get("failed")) for r in rs) / n,
            "false_confirm_on_perturbed": (sum(r["outcome"] == "false_confirm" for r in pert) / len(pert)) if pert else float("nan"),
            "false_alarm_on_true": (sum(r["outcome"] == "false_alarm" for r in true) / len(true)) if true else float("nan")}


# ---------------------------------------------------------------- hypotheses and validity
def in_(*ss):
    return lambda r: r["stratum"] in ss


def hypotheses(cc: list[dict], exclude_types: tuple = (), validity: dict | None = None) -> list[dict]:
    rows = [r for r in cc if r["type"] not in exclude_types]
    validity = validity or {}
    hs: dict[str, dict] = {}
    h1 = contrast(rows, "D0", "D1A", "net", in_("A"))
    h1.update({"H": "H1", "claim": "recency: A, net, D1A − D0 > 0", "test": "superiority (block sign flips)",
               "p": sign_flip_p(h1["_blocks"]), "bound": h1["diff"] - t_q(h1["df"], 1 - ALPHA) * h1["se"]
               if h1["n_claims"] else float("nan")})
    h1["pass"] = h1["p"] < ALPHA
    hs["H1"] = h1
    h3 = contrast(rows, "D0", "D1A", "net", in_("K", "K0"))
    h3.update({"H": "H3", "claim": "no harm: K ∪ K0, net, D1A − D0 > −0.15", "test": "non-inferiority",
               **noninferiority(h3, MARGINS["H3"]), "power_if_0": power_if_zero(h3, MARGINS["H3"])})
    per = {s: contrast(rows, "D0", "D1A", "net", in_(s)) for s in ("K", "K0")}
    h3["note"] = "; ".join(f"{s} {fmt(c['diff'])} [{fmt(c['ci_lo'])}, {fmt(c['ci_hi'])}]" for s, c in per.items())
    harm = [s for s, c in per.items() if c["n_claims"] and c["diff"] < -MARGINS["H3"]]
    if harm:
        h3["note"] += f"; point estimate beyond the margin in {', '.join(harm)}"
    uninf = [s for s in ("K", "K0") if validity.get(f"V6 {s}") is False]
    if uninf:
        h3["note"] += f"; not informative in {', '.join(uninf)} (V6)"
    hs["H3"] = h3
    h2 = contrast(rows, "D0", "D1A", "error", in_("B"))
    h2.update({"H": "H2", "claim": "honesty: B, confident errors, D1A − D0 < +0.05", "test": "non-inferiority",
               **noninferiority(h2, MARGINS["H2"], lower_is_better=True), "power_if_0": power_if_zero(h2, MARGINS["H2"])})
    hs["H2"] = h2
    h4 = contrast(rows, "D0F", "D1F", "net", in_("A", "B"))
    h4.update({"H": "H4", "claim": "level 4: A ∪ B, net, D1F − D0F > −0.05", "test": "non-inferiority",
               **noninferiority(h4, MARGINS["H4"]), "power_if_0": power_if_zero(h4, MARGINS["H4"])})
    if validity.get("V4") is False:
        h4["pass"], h4["note"] = False, "not tested: fetch_paper.py ran in < 80 % of F calls (V4)"
    hs["H4"] = h4
    alive = True
    for k in ORDER:
        h = hs[k]
        h["confirmatory"] = alive
        h["verdict"] = ("pass" if h["pass"] else "fail") if alive else ("pass (exploratory)" if h["pass"] else "fail (exploratory)")
        alive = alive and bool(h["pass"])
    return [hs[k] for k in ORDER]



def uses_plugin(rec: dict) -> bool:
    """The call used the package: a Skill call, or a Read/Grep/Glob path inside the plugin's skills
    (D1A reads deep/ files directly without loading a skill)."""
    return bool(rec.get("skills_invoked")) or any("/plugin/skills/" in (p or "") for p in rec.get("read_paths") or [])

def validity(cc: list[dict], answers: list[dict]) -> tuple[list[dict], dict, list[str]]:
    rows, flags, bad_types = [], {}, []
    for t in TYPES:
        m = arm_metrics([r for r in cc if r["arm"] == "D0" and r["stratum"] == "B" and r["type"] == t])
        ok = (m["net"] <= 0.15) if m.get("claims") else None
        rows.append({"V": f"V1 priors, D0 net in B, {t}", "value": m.get("net", float("nan")), "threshold": "≤ 0.15",
                     "ok": "n/a" if ok is None else ok})
        if ok is False:
            bad_types.append(t)
    for arm in sorted({r["arm"] for r in cc}):
        rs = [r for r in scored(cc) if r["arm"] == arm]
        miss = sum(r["verdict"] == "missing" for r in rs) / len(rs) if rs else float("nan")
        rows.append({"V": f"V2 missing verdicts (failed calls included), {arm}", "value": miss, "threshold": "< 0.02",
                     "ok": miss < 0.02})
    d1a = [r for r in answers if r["arm"] == "D1A" and r.get("status") == "ok"]
    if d1a:
        rate = sum(uses_plugin(r) for r in d1a) / len(d1a)
        rows.append({"V": "V3 plugin use rate, D1A (Skill call or a read of a plugin file)", "value": rate,
                     "threshold": "reported; < 0.5 changes the reading", "ok": "reported"})
        rows.append({"V": "V3a Skill-call rate, D1A", "value": sum(bool(r.get("skills_invoked")) for r in d1a) / len(d1a),
                     "threshold": "reported", "ok": "reported"})
    v4 = []
    for arm in ("D0F", "D1F"):
        rs = [r for r in answers if r["arm"] == arm]
        if rs:
            rate = sum((r.get("fetch_ok") or 0) > 0 for r in rs) / len(rs)
            rows.append({"V": f"V4 calls with a fetch_paper.py run, {arm}", "value": rate, "threshold": "≥ 0.80",
                         "ok": rate >= 0.8})
            v4.append(rate >= 0.8)
    if v4:
        flags["V4"] = all(v4)
    for s in ("K", "K0"):
        m = arm_metrics([r for r in cc if r["arm"] == "D0" and r["stratum"] == s])
        if m.get("claims"):
            rows.append({"V": f"V6 D0 net in {s}", "value": m["net"], "threshold": "≥ 0.20", "ok": m["net"] >= 0.2})
            flags[f"V6 {s}"] = m["net"] >= 0.2
    m = arm_metrics([r for r in cc if r["arm"] == "D0F" and r["stratum"] in ("A", "B")])
    if m.get("claims"):
        rows.append({"V": "V7 ceiling, D0F net in A ∪ B", "value": m["net"], "threshold": "reported; ≥ 0.90 = ceiling",
                     "ok": "ceiling" if m["net"] >= 0.9 else "below ceiling"})
    return rows, flags, bad_types


# ---------------------------------------------------------------- report helpers
def fmt(x) -> str:
    if isinstance(x, float):
        return "nan" if x != x else f"{x:.3f}"
    return str(x)


def table(rows: list[dict], cols: list[str]) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(r.get(c, "")).replace("|", "/") for c in cols) + " |" for r in rows]
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


HCOLS = ["H", "claim", "test", "n_claims", "n_blocks", "diff", "ci_lo", "ci_hi", "bound", "p", "verdict",
         "power_if_0", "note"]


def main() -> int:
    cc = read_jsonl(RESULTS / "claimcheck" / "grades.jsonl")
    answers = read_jsonl(RESULTS / "claimcheck" / "answers.jsonl")
    md = ["# cosmoskill benchmark v5 — analysis", ""]
    if cc:
        vrows, vflags, bad_types = validity(cc, answers)
        hyp = hypotheses(cc, validity=vflags)
        md += ["## Pre-registered hypotheses (fixed sequence H1 → H3 → H2 → H4, one-sided α = 0.05)", ""]
        md += table(hyp, HCOLS) + [""]
        if bad_types:
            md += [f"Robustness (V1 flagged {', '.join(bad_types)}): the same tests without those types.", ""]
            md += table(hypotheses(cc, exclude_types=tuple(bad_types), validity=vflags), HCOLS) + [""]
        md += ["## Validity checks", ""] + table(vrows, ["V", "value", "threshold", "ok"]) + [""]
        rows = []
        for arm in sorted({r["arm"] for r in cc}):
            for s in ("K", "K0", "A", "B", "all"):
                sub = [r for r in cc if r["arm"] == arm and (s == "all" or r["stratum"] == s)]
                if sub:
                    rows.append({"arm": arm, "stratum": s, **arm_metrics(sub)})
        cols = ["arm", "stratum", "claims", "net", "right", "error", "abstain", "missing", "failed_calls_claims",
                "false_confirm_on_perturbed", "false_alarm_on_true"]
        md += ["## Claim check by arm and stratum", ""] + table(rows, cols) + [""]
        by_type = [{"arm": arm, "stratum": s, "type": t, **arm_metrics(sub)}
                   for arm in sorted({r["arm"] for r in cc}) for s in ("K", "K0", "A", "B") for t in TYPES
                   if (sub := [r for r in cc if r["arm"] == arm and r["stratum"] == s and r["type"] == t])]
        md += ["## By claim type (descriptive)", ""] + table(by_type, ["arm", "stratum", "type", "claims", "net", "right",
                                                                       "error", "abstain"]) + [""]
        sec = []
        for a, b, metric, ss, note in (
                ("D0", "D1A", "net", ("K",), "H3 stratum"), ("D0", "D1A", "net", ("K0",), "H3 stratum"),
                ("D0", "D1A", "net", ("B",), ""), ("D0", "D1A", "error", ("K0",), "H2b: neighbouring-result errors"),
                ("D0", "D1A", "error", ("A",), ""), ("D1A", "D1F", "net", ("A", "B"), "what L4 adds to the package"),
                ("D0", "D0F", "net", ("A", "B"), "what L4 adds without the package")):
            c = contrast(cc, a, b, metric, in_(*ss))
            c["subset"], c["note"] = "+".join(ss), note
            sec.append(c)
        h2s = contrast(cc, "D1A", "D0", "error", in_("B"))   # superiority of D1A on errors = D0 - D1A > 0
        h2s.update({"subset": "B", "note": f"H2 superiority (secondary), sign-flip p = {fmt(sign_flip_p(h2s['_blocks']))}"})
        h4s = contrast(cc, "D0F", "D1F", "net", in_("A", "B"))
        h4s.update({"subset": "A+B", "note": f"H4 superiority (secondary), sign-flip p = {fmt(sign_flip_p(h4s['_blocks']))}"})
        sec += [h2s, h4s]
        md += ["## Secondary contrasts (b − a, 95 % cluster-robust t CI)", ""]
        md += table(sec, ["a", "b", "metric", "subset", "n_claims", "n_blocks", "diff", "ci_lo", "ci_hi", "note"]) + [""]
        a_rows = [r for r in cc if r["stratum"] == "A" and r["in_skill_text"] is not None]
        if a_rows:
            md += ["Stratum A by whether a skill file states the claim's key token near the paper's ID "
                   "(conclusion claims have no token and are left out):", ""]
            md += table([{"arm": arm, "in_skill_text": f, **arm_metrics(sub)} for arm in sorted({r["arm"] for r in a_rows})
                         for f in (True, False) if (sub := [r for r in a_rows if r["arm"] == arm and r["in_skill_text"] == f])],
                        ["arm", "in_skill_text", "claims", "net", "right", "error", "abstain"]) + [""]
        nums = [r for r in scored(cc) if r["type"] == "number" and r["perturbed"]]
        if nums:
            md += ["Perturbed number claims: how often a `wrong` verdict states the paper's true value:", ""]
            md += table([{"arm": arm, "stratum": s, "claims": len(sub),
                          "right": sum(r["outcome"] == "right" for r in sub) / len(sub),
                          "states_true_given_wrong": (sum(bool(r["states_true"]) for r in sub if r["verdict"] == "wrong") /
                                                      max(1, sum(r["verdict"] == "wrong" for r in sub)))}
                         for arm in sorted({r["arm"] for r in nums}) for s in ("K", "K0", "A", "B")
                         if (sub := [r for r in nums if r["arm"] == arm and r["stratum"] == s])],
                        ["arm", "stratum", "claims", "right", "states_true_given_wrong"]) + [""]
        write_csv(RESULTS / "hypotheses.csv", hyp)
        write_csv(RESULTS / "claimcheck_table.csv", rows)
    qa = read_jsonl(RESULTS / "cosmopaperqa" / "grades.jsonl")
    if qa:
        rows = [{"arm": arm, **qa_metrics([g for g in qa if g["arm"] == arm])} for arm in sorted({g["arm"] for g in qa})]
        md += ["## CosmoPaperQA (secondary)", ""] + table(rows, ["arm", "n", "correct", "ci_lo", "ci_hi", "lenient",
                                                                 "abstain", "false_claim", "unjudged"]) + [""]
        md += table([qa_paired(qa, "D0", "D1A")], ["a", "b", "n", "diff", "ci_lo", "ci_hi", "p"]) + [""]
        j2 = {(g["item_id"], g["arm"]): g["grade"] for g in read_jsonl(RESULTS / "cosmopaperqa" / "grades_judge2.jsonl")}
        pairs = [(g["grade"], j2[(g["item_id"], g["arm"])]) for g in qa if (g["item_id"], g["arm"]) in j2]
        if pairs:
            md += [f"Judge–judge (Opus vs Sonnet), {len(pairs)} answers: raw {sum(a == b for a, b in pairs) / len(pairs):.3f}, "
                   f"κ {cohen_kappa([a for a, _ in pairs], [b for _, b in pairs]):.3f}.", ""]
    cost = {}
    for bench in ("claimcheck", "cosmopaperqa"):
        recs = [r for r in read_jsonl(RESULTS / bench / "answers.jsonl") if r.get("status") == "ok"]
        if recs:
            crow = [{"arm": arm, "calls": len(rs), "total_usd": sum(r.get("cost_usd") or 0 for r in rs),
                     "mean_usd": sum(r.get("cost_usd") or 0 for r in rs) / len(rs),
                     "median_s": statistics.median((r.get("duration_ms") or 0) / 1000 for r in rs),
                     "skill_rate": sum(bool(r.get("skills_invoked")) for r in rs) / len(rs),
                     "plugin_use_rate": sum(uses_plugin(r) for r in rs) / len(rs),
                     "fetch_ok_calls": sum((r.get("fetch_ok") or 0) > 0 for r in rs)}
                    for arm in sorted({r["arm"] for r in recs}) if (rs := [r for r in recs if r["arm"] == arm])]
            if bench == "claimcheck":
                cost = {c["arm"]: c["mean_usd"] for c in crow}
            md += [f"## Cost, {bench}", ""] + table(crow, ["arm", "calls", "total_usd", "mean_usd", "median_s",
                                                         "skill_rate", "fetch_ok_calls"]) + [""]
    if cc and "D0" in cost and "D1A" in cost:
        h1 = contrast(cc, "D0", "D1A", "net", in_("A"))
        if h1["n_claims"] and h1["diff"] > 0:
            md += [f"Cost per net-score point in A (extra $ per call / extra net per call): "
                   f"${(cost['D1A'] - cost['D0']) / (4 * h1['diff']):.3f} per additional right claim.", ""]
    text = "\n".join(md) + "\n"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "analysis.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
