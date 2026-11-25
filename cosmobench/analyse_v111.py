"""Re-test of cosmoskill 1.1.1 (plan 2026-11-22-plugin-fix-uncovered-papers.md, section 5).

    python3 -m cosmobench.analyse_v111     -> results/v5/v111/analysis_v111.md, hypotheses_v111.csv

Merges the graded runs (arm D1A' = the plugin at 1.1.1, run under the arm name D1A):
    results/v5/claimcheck/             v5: D0, D1A (1.1.0) on K, K0, A, B
    results/v5/k0fresh/claimcheck/     fresh K0 (2015-2019, stratum relabelled K0f): D0, D1A (1.1.0)
    results/v5/k0fresh_v111/claimcheck fresh K0: D1A'
    results/v5/v111/claimcheck/        v5 items: D1A'
    results/v5/[v111/]cosmopaperqa/    D0, D1A / D1A' (both judges)

  H5 fix works     K0f     net, D1A' − D1A          superiority (block sign flips), p < 0.05
  H7 honesty kept  B       errors, D1A' − D0        non-inferiority, upper bound < +0.05
  H6 no harm       K0f     net, D1A' − D0           non-inferiority, lower bound > −0.10
  H8 no regression K ∪ A   net, D1A' − D1A          non-inferiority, lower bound > −0.05
Fixed sequence H5 → H7 → H6 → H8 at one-sided α = 0.05.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .analyse import cohen_kappa, qa_metrics, qa_paired
from .analyse_v5 import (
    ALPHA,
    TYPES,
    arm_metrics,
    contrast,
    in_,
    noninferiority,
    power_if_zero,
    sign_flip_p,
    t_q,
    table,
    write_csv,
)
from .common import HARNESS_VERSION, ROOT, read_jsonl

BASE = ROOT / "results" / f"v{HARNESS_VERSION}"
NEW = "D1A'"
ORDER = ("H5", "H7", "H6", "H8")
MARGINS = {"H7": 0.05, "H6": 0.10, "H8": 0.05}


def _load(path: Path, rename: dict | None = None, fresh: bool = False) -> list[dict]:
    rows = read_jsonl(path)
    for r in rows:
        if rename and r["arm"] in rename:
            r["arm"] = rename[r["arm"]]
        if fresh:
            assert r["item_id"].startswith("cc5f-"), r["item_id"]
            r["stratum"], r["block"] = "K0f", "K0f" + r["block"][2:]
    return rows


def claim_rows(base: Path = BASE) -> list[dict]:
    rows = [r for r in _load(base / "claimcheck" / "grades.jsonl") if r["arm"] in ("D0", "D1A")]
    rows += _load(base / "k0fresh" / "claimcheck" / "grades.jsonl", fresh=True)
    rows += _load(base / "k0fresh_v111" / "claimcheck" / "grades.jsonl", {"D1A": NEW}, fresh=True)
    rows += _load(base / "v111" / "claimcheck" / "grades.jsonl", {"D1A": NEW})
    return rows


def hypotheses(rows: list[dict]) -> list[dict]:
    hs = {}
    h5 = contrast(rows, "D1A", NEW, "net", in_("K0f"))
    h5.update({"H": "H5", "claim": "fix works: K0f, net, D1A' − D1A > 0", "test": "superiority (block sign flips)",
               "p": sign_flip_p(h5["_blocks"]),
               "bound": h5["diff"] - t_q(h5["df"], 1 - ALPHA) * h5["se"] if h5["n_claims"] else float("nan")})
    h5["pass"] = h5["n_claims"] > 0 and h5["p"] < ALPHA
    hs["H5"] = h5
    h7 = contrast(rows, "D0", NEW, "error", in_("B"))
    h7.update({"H": "H7", "claim": "honesty kept: B, errors, D1A' − D0 < +0.05", "test": "non-inferiority",
               **noninferiority(h7, MARGINS["H7"], lower_is_better=True), "power_if_0": power_if_zero(h7, MARGINS["H7"])})
    hs["H7"] = h7
    h6 = contrast(rows, "D0", NEW, "net", in_("K0f"))
    h6.update({"H": "H6", "claim": "no harm vs bare: K0f, net, D1A' − D0 > −0.10", "test": "non-inferiority",
               **noninferiority(h6, MARGINS["H6"]), "power_if_0": power_if_zero(h6, MARGINS["H6"])})
    hs["H6"] = h6
    h8 = contrast(rows, "D1A", NEW, "net", in_("K", "A"))
    h8.update({"H": "H8", "claim": "no regression: K ∪ A, net, D1A' − D1A > −0.05", "test": "non-inferiority",
               **noninferiority(h8, MARGINS["H8"]), "power_if_0": power_if_zero(h8, MARGINS["H8"])})
    hs["H8"] = h8
    alive = True
    for k in ORDER:
        h = hs[k]
        h["confirmatory"] = alive
        h["verdict"] = ("pass" if h["pass"] else "fail") if alive else ("pass (exploratory)" if h["pass"] else "fail (exploratory)")
        alive = alive and bool(h["pass"])
    return [hs[k] for k in ORDER]


HCOLS = ["H", "claim", "test", "n_claims", "n_blocks", "diff", "ci_lo", "ci_hi", "bound", "p", "power_if_0", "verdict"]
STRATA = ("K0f", "K0", "K", "A", "B")


def main() -> int:
    rows = claim_rows()
    out = BASE / "v111"
    md = ["# cosmoskill 1.1.1 re-test — analysis", "",
          "D1A' = plugin 1.1.1 (8954fff), D1A = 1.1.0 (e317d82). K0f = fresh K0 items, 2015–2019 (plan §5 deviation).", ""]
    hyp = hypotheses(rows)
    md += ["## Pre-registered hypotheses (fixed sequence H5 → H7 → H6 → H8, one-sided α = 0.05)", ""]
    md += table(hyp, HCOLS) + [""]
    arms = [a for a in ("D0", "D1A", NEW) if any(r["arm"] == a for r in rows)]
    tab = [{"arm": a, "stratum": s, **arm_metrics(sub)} for s in STRATA for a in arms
           if (sub := [r for r in rows if r["arm"] == a and r["stratum"] == s])]
    cols = ["arm", "stratum", "claims", "net", "right", "error", "abstain", "missing", "failed_calls_claims",
            "false_confirm_on_perturbed", "false_alarm_on_true"]
    md += ["## Claim check by arm and stratum", ""] + table(tab, cols) + [""]
    sec = []
    for a, b, metric, ss, note in (("D1A", NEW, "net", ("K0",), "old K0 items (exploratory: the fix was designed on them)"),
                                   ("D0", NEW, "net", ("K0",), "old K0 items"),
                                   ("D0", "D1A", "net", ("K0f",), "v5 K0 finding, replicated on fresh items?"),
                                   ("D1A", NEW, "net", ("B",), ""), ("D1A", NEW, "error", ("B",), ""),
                                   ("D1A", NEW, "error", ("K0f",), ""), ("D0", NEW, "error", ("K0f",), "")):
        c = contrast(rows, a, b, metric, in_(*ss))
        c["subset"], c["note"] = "+".join(ss), note
        sec.append(c)
    md += ["## Secondary contrasts (b − a, 95 % cluster-robust t CI)", ""]
    md += table(sec, ["a", "b", "metric", "subset", "n_claims", "n_blocks", "diff", "ci_lo", "ci_hi", "note"]) + [""]
    att = [{"arm": a, "stratum": s, **arm_metrics(sub)} for s in ("K0f", "K0") for a in arms
           if (sub := [r for r in rows if r["arm"] == a and r["stratum"] == s and r["type"] == "attribution"])]
    md += ["## Attribution claims in K0 / K0f (attribution guard; v5 D0 error on K0 = 0.32)", ""]
    md += table(att, ["arm", "stratum", "claims", "net", "right", "error", "abstain", "false_confirm_on_perturbed"]) + [""]
    by_type = [{"arm": a, "stratum": s, "type": t, **arm_metrics(sub)} for s in STRATA for t in TYPES for a in arms
               if (sub := [r for r in rows if r["arm"] == a and r["stratum"] == s and r["type"] == t])]
    md += ["## By claim type (descriptive)", ""] + table(by_type, ["arm", "stratum", "type", "claims", "net", "right",
                                                                   "error", "abstain"]) + [""]
    for judge, f in (("Opus judge", "grades.jsonl"), ("Sonnet judge", "grades_judge2.jsonl")):
        qa = [g for g in read_jsonl(BASE / "cosmopaperqa" / f) if g["arm"] in ("D0", "D1A")]
        qa += _load(out / "cosmopaperqa" / f, {"D1A": NEW})
        if not any(g["arm"] == NEW for g in qa):
            continue
        qrows = [{"arm": a, **qa_metrics([g for g in qa if g["arm"] == a])} for a in ("D0", "D1A", NEW)]
        md += [f"## CosmoPaperQA, {judge} (secondary)", ""]
        md += table(qrows, ["arm", "n", "correct", "ci_lo", "ci_hi", "abstain", "false_claim", "unjudged"]) + [""]
        md += table([qa_paired(qa, "D0", NEW), qa_paired(qa, "D1A", NEW)], ["a", "b", "n", "diff", "ci_lo", "ci_hi", "p"]) + [""]
    j1 = {(g["item_id"], g["arm"]): g["grade"] for g in read_jsonl(out / "cosmopaperqa" / "grades.jsonl")}
    j2 = {(g["item_id"], g["arm"]): g["grade"] for g in read_jsonl(out / "cosmopaperqa" / "grades_judge2.jsonl")}
    pairs = [(j1[k], j2[k]) for k in j1 if k in j2]
    if pairs:
        md += [f"Judge–judge on D1A', {len(pairs)} answers: raw {sum(a == b for a, b in pairs) / len(pairs):.3f}, "
               f"κ {cohen_kappa([a for a, _ in pairs], [b for _, b in pairs]):.3f}.", ""]
    cost = []
    for label, path in (("v5 items, D1A'", out / "claimcheck"), ("fresh, D0/D1A", BASE / "k0fresh" / "claimcheck"),
                        ("fresh, D1A'", BASE / "k0fresh_v111" / "claimcheck"), ("CPQA, D1A'", out / "cosmopaperqa")):
        recs = read_jsonl(path / "answers.jsonl")
        for a in sorted({r["arm"] for r in recs}):
            rs = [r for r in recs if r["arm"] == a]
            cost.append({"run": label, "arm": a, "calls": len(rs), "ok": sum(r.get("status") == "ok" for r in rs),
                         "total_usd": sum(r.get("cost_usd") or 0 for r in rs)})
    md += ["## Calls and cost", ""] + table(cost, ["run", "arm", "calls", "ok", "total_usd"]) + [""]
    text = "\n".join(md) + "\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis_v111.md").write_text(text)
    write_csv(out / "hypotheses_v111.csv", hyp)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
