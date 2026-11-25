"""Unit tests for the v5 claim check: generator, assembly, grader and analysis.

    pytest test/test_v5.py
"""
from __future__ import annotations

import csv
import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from cosmobench import analyse_v5 as av
from cosmobench import build_v5 as bv
from cosmobench import grade_claims as gc
from cosmobench.arms import ARMS_BY_BENCH, arm_applies
from cosmobench.prompts import answer_prompt

ABS = (r"Using DESI DR2 BAO and Planck 2018 CMB data we find $\Omega_m = 0.297 \pm 0.007$ and "
       r"$H_0 = 68.5 \pm 0.6$ km/s/Mpc. The BAO distances are in tension with the $\Lambda$CDM prediction "
       r"at $3.1\sigma$. We obtain $\sum m_\nu < 0.064$ eV.")
P = {"arxiv": "2608.01234", "title": "Cosmology from DESI DR2", "abstract": ABS, "published": "2026-08-10",
     "stratum": "A", "domain": "cosmo-galaxy-clustering", "surnames": ["Smithers", "Hubble"]}


class TestDatasets(unittest.TestCase):
    def test_detect_and_targets(self):
        self.assertEqual(bv.datasets_in(ABS), {"desi": ["DESI DR2"], "planck": ["Planck 2018"]})
        self.assertEqual(bv.datasets_in("Pantheon+ and Union3")["sn"], ["Pantheon+", "Union3"])
        self.assertEqual(bv.datasets_in("the Pantheon sample")["sn"], ["Pantheon"])
        self.assertEqual(bv.datasets_in("GWTC-2.1 events"), {})
        # only releases before the paper's month, never one named in the abstract
        self.assertEqual(bv.dataset_targets("DESI DR2", ABS, "2026-08-10"), ["DESI DR1"])      # neighbour only
        self.assertEqual(bv.dataset_targets("Planck 2018", "", "2025-01-01"), ["Planck PR4"])  # no stale 2013/2015
        self.assertEqual(bv.dataset_targets("KiDS-1000", "", "2023-01-01"), [])                # 450 replaced in 2020
        self.assertEqual(bv.dataset_targets("Pantheon+", "", "2024-06-01"), ["Union3", "DES-SN5YR"])
        self.assertEqual(bv.dataset_targets("DESI DR1", "DESI DR1 BAO", "2024-06-01"), ["DESI EDR"])
        self.assertNotIn("DESI DR2", bv.dataset_targets("DESI DR1", "DESI DR1 BAO", "2025-02-01"))
        self.assertIn("DESI DR2", bv.dataset_targets("DESI DR1", "DESI DR1 BAO", "2025-04-01"))
        self.assertEqual(bv.dataset_options(ABS, "2026-08-10"), ["DESI DR2", "Planck 2018"])
        # two members of one group in the abstract: that group is not usable
        self.assertEqual(bv.dataset_options("DESI DR1 and DESI DR2 BAO", "2026-01-01"), [])


class TestConclusion(unittest.TestCase):
    def test_cue_side(self):
        self.assertEqual(bv.cue_side("The BAO distances are in tension with LCDM.", "consistency"), "b")
        self.assertEqual(bv.cue_side("Our results are consistent with Planck.", "consistency"), "a")
        self.assertIsNone(bv.cue_side("We find no tension with Planck.", "consistency"))        # negation
        self.assertIsNone(bv.cue_side("Consistent with LCDM, unlike the tension in H0.", "consistency"))
        self.assertEqual(bv.cue_side("This is inconsistent with GR.", "consistency"), "b")
        self.assertEqual(bv.cue_side("The data disfavour w = -1.", "preference"), "b")
        self.assertEqual(bv.cue_side("The data favour dynamical dark energy.", "preference"), "a")
        self.assertEqual(bv.cue_side("We find no detection of the kSZ signal.", "detection"), "b")
        self.assertEqual(bv.cue_side("We report a detection of the kSZ signal.", "detection"), "a")
        self.assertIn("consistency", bv.conclusion_families(ABS))

    def test_phrase_pairs_are_symmetric(self):
        for ph, (fam, side, other) in bv.PHRASES.items():
            self.assertEqual(bv.PHRASES[other][2], ph)
            self.assertNotEqual(bv.PHRASES[other][1], side)
            self.assertEqual(bv.PHRASES[other][0], fam)


class TestNumbers(unittest.TestCase):
    def test_usable(self):
        sp = bv.usable_spans(ABS)
        self.assertEqual([(s["kind"], s["value"]) for s in sp],
                         [("pm", "0.297"), ("pm", "68.5"), ("sigma", "3.1"), ("limit", "0.064")])
        big = bv.numeric_spans(r"the median is $0.3 \pm 1.3$ rad/m$^2$")
        self.assertFalse(bv.number_usable(big[0]))           # relative error > 12.5 %
        cut = bv.numeric_spans(r"a magnitude threshold ($m_r<20$)")
        self.assertFalse(bv.number_usable(cut[0]))           # selection cut

    def test_perturbation_bounds(self):
        rng = random.Random(4)
        sp = bv.usable_spans(ABS)
        for _ in range(300):
            v, mode = bv.perturb_number(sp[0], sp, rng)          # 0.297 ± 0.007
            d = abs(float(v) - 0.297)
            self.assertEqual(mode, "shift")
            self.assertGreaterEqual(d, max(2 * 0.007, 0.03 * 0.297) * 0.98)
            self.assertLessEqual(d, 0.25 * 0.297 * 1.02)
            self.assertEqual(len(v.split(".")[1]), 3)
            s = float(bv.perturb_number(sp[2], sp, rng)[0])      # 3.1 sigma
            self.assertTrue(0.98 <= abs(s - 3.1) <= 2.04)
            lim = float(bv.perturb_number(sp[3], sp, rng)[0])    # < 0.064
            self.assertTrue(1.45 <= max(lim / 0.064, 0.064 / lim) <= 2.3)

    @mock.patch.object(bv, "SWAP", True)
    def test_swap_prefers_plausible_values_of_the_abstract(self):
        text = r"We find $S_8 = 0.776 \pm 0.017$ from shear alone and $S_8 = 0.812 \pm 0.012$ combined, with $\Omega_m = 0.31 \pm 0.01$."
        sp = bv.usable_spans(text)
        v, mode = bv.perturb_number(sp[0], sp, random.Random(0))
        self.assertEqual((v, mode), ("0.812", "swap"))         # same decimals, 2.1σ-25 % away
        v, mode = bv.perturb_number(sp[2], sp, random.Random(0))
        self.assertEqual(mode, "shift")                       # 0.31 has 2 decimals: no swap to 0.776


class TestValidate(unittest.TestCase):
    def test_types(self):
        ok = {"span": 0, "template": "DESI DR2 BAO with Planck 2018 gives Ωm = {V} ± 0.007."}
        self.assertEqual(bv.validate_type("number", ok, P), "")
        self.assertIn("uncertainty", bv.validate_type("number", {"span": 0, "template": "Ωm = {V} ± 0.01."}, P))
        self.assertIn("other number", bv.validate_type("number", {"span": 0, "template": "Ωm = {V} ± 0.007 and H0 = 68.5."}, P))
        self.assertIn("verdict", bv.validate_type("number", {"span": 0, "template": "Ωm = {V} ± 0.007, consistent with Planck."}, P))
        self.assertEqual(bv.validate_type("number", {"span": 0, "template": "At z = 0.5, Ωm = {V} ± 0.007 (68% C.L.)."}, P), "")
        self.assertIn("fit", bv.validate_type("number", {"span": 0, "template": "Ωm is {V} here."}, P))
        self.assertIn("author", bv.validate_type("number", {"span": 0, "template": "Smithers find Ωm = {V} ± 0.007."}, P))
        eponym = {**P, "abstract": ABS + " Hubble-flow distances agree."}   # a surname the abstract uses is not an author cue
        self.assertEqual(bv.validate_type("number", {"span": 0, "template": "Hubble-flow BAO give Ωm = {V} ± 0.007."}, eponym), "")
        ds = {"name": "DESI DR2", "template": "Combining {D} BAO with Planck 2018 CMB data gives a low matter density."}
        self.assertEqual(bv.validate_type("dataset", ds, P), "")
        self.assertIn("member", bv.validate_type("dataset", {"name": "DESI DR2", "template": "{D} and DESI DR1 BAO."}, P))
        self.assertIn("option", bv.validate_type("dataset", {"name": "DESI DR2", "template": "{D} BAO."}, {**P, "published": "2023-01-01"}))
        self.assertIn("release-dating", bv.validate_type("dataset", {"name": "DESI DR2", "template": "The three-year {D} BAO favour dynamical dark energy."}, P))
        self.assertIn("number in", bv.validate_type("dataset", {"name": "DESI DR2", "template": "{D} BAO give Ωm = 0.297."}, P))
        ev = "The BAO distances are in tension with the $\\Lambda$CDM prediction at $3.1\\sigma$."
        cc = {"phrase": "in tension with", "evidence": ev, "template": "The DESI DR2 BAO distances are {C} the ΛCDM prediction."}
        self.assertEqual(bv.validate_type("conclusion", cc, P), "")
        self.assertIn("number", bv.validate_type("conclusion", {**cc, "template": "The BAO distances are {C} ΛCDM at 3σ."}, P))
        self.assertTrue(bv.has_value("at z = 0.5"))
        self.assertFalse(bv.has_value("DESI DR2, KiDS-1000, S8, H0, 3x2pt and GWTC-4.0"))
        self.assertIn("cue", bv.validate_type("conclusion", {**cc, "phrase": "consistent with"}, P))
        self.assertIn("verdict word", bv.validate_type("conclusion", {**cc, "template": "The BAO distances are fully {C} ΛCDM."}, P))
        self.assertIn("verdict word", bv.validate_type("conclusion", {**cc, "template": "The BAO distances are {C} ΛCDM, a tension."}, P))
        self.assertIn("verbatim", bv.validate_type("conclusion", {**cc, "evidence": "The BAO is in tension with LCDM today."}, P))
        at = {"text": "DESI DR2 BAO combined with Planck 2018 CMB data give Ωm = 0.297 ± 0.007."}
        self.assertEqual(bv.validate_type("attribution", at, P), "")
        self.assertIn("not in the abstract", bv.validate_type("attribution", {"text": at["text"].replace("0.297", "0.300")}, P))
        self.assertIn("arXiv", bv.validate_type("attribution", {"text": at["text"] + " (2608.01234)"}, P))


def _pool():
    decoys = [{"arxiv": "2608.05555", "title": "Cosmology from DESI DR2 voids", "abstract": "Voids give Ωm = 0.30.",
               "published": "2026-09-01", "stratum": "A", "domain": "cosmo-galaxy-clustering"},
              {"arxiv": "2608.06666", "title": "Cosmology from DESI DR2 lensing", "abstract": "We find 0.297 for Ωm.",
               "published": "2026-09-01", "stratum": "A", "domain": "cosmo-galaxy-clustering"},
              {"arxiv": "2603.07777", "title": "Cosmology from DESI DR2 BAO", "abstract": "Too early.",
               "published": "2026-03-01", "stratum": "A", "domain": "cosmo-galaxy-clustering"},
              {"arxiv": "2608.08888", "title": "Cosmology from DESI DR2 clusters", "abstract": "Clusters.",
               "published": "2026-08-20", "stratum": "B", "domain": "cosmo-galaxy-clustering"}]
    return [P] + decoys


class TestRender(unittest.TestCase):
    def test_render_all_types(self):
        ev = "The BAO distances are in tension with the $\\Lambda$CDM prediction at $3.1\\sigma$."
        dr = {"draft": {"number": {"span": 0, "template": "DESI DR2 BAO with Planck 2018 gives Ωm = {V} ± 0.007."},
                        "dataset": {"name": "DESI DR2", "template": "Combining {D} BAO with Planck 2018 gives a low Ωm."},
                        "conclusion": {"phrase": "in tension with", "evidence": ev,
                                       "template": "The DESI DR2 BAO distances are {C} the ΛCDM prediction."},
                        "attribution": {"text": "DESI DR2 BAO combined with Planck 2018 data give Ωm = 0.297 ± 0.007."}},
              "problems": {t: "" for t in bv.TYPES}}
        cs = {c["type"]: c for c in bv.render(P, dr, _pool(), random.Random(0))}
        self.assertEqual(set(cs), set(bv.TYPES))
        self.assertIn("0.297 ± 0.007", cs["number"]["true_text"])
        self.assertNotIn("0.297", cs["number"]["pert_text"])
        self.assertIn(cs["dataset"]["value_pert"], ("DESI EDR", "DESI DR1"))
        self.assertIn("consistent with", cs["conclusion"]["pert_text"])
        # decoy: same stratum, within 90 days, abstract without the claim's numbers (not 2608.06666)
        self.assertEqual(cs["attribution"]["cited_pert"], "2608.05555")
        self.assertEqual(cs["attribution"]["true_text"], cs["attribution"]["pert_text"])

    def test_verify_prompt_keys(self):
        claims = [{"uid": "x:number", "type": "number", "true_text": "T", "pert_text": "F"},
                  {"uid": "x:attribution", "type": "attribution", "true_text": "S", "pert_text": "S",
                   "decoy_abstract": "DECOY"}]
        prompt, keys = bv.verify_prompt(P, claims, random.Random(1))
        self.assertIn("Version 1:", prompt)
        self.assertIn("DECOY", prompt)
        for k, c in zip(keys, claims):
            if c["type"] == "number":
                first = prompt.index("Version 1: T") if k["true_is"] == "1" else prompt.index("Version 1: F")
                self.assertGreater(first, 0)
        self.assertEqual(keys[1]["true_is"], "1")                     # attribution: fixed meaning, paper first
        self.assertEqual(prompt.count(P["abstract"]), 1)               # the paper's abstract is shown once
        bp, idx = bv.blind_prompt(claims, keys)
        self.assertEqual(idx, [1])
        self.assertNotIn(P["abstract"], bp)
        self.assertNotIn("DECOY", bp)
        self.assertTrue(bv.blind_hit({"original": "2", "confidence": "high"}, "2"))
        self.assertFalse(bv.blind_hit({"original": "2", "confidence": "medium"}, "2"))
        self.assertFalse(bv.blind_hit({"original": "1", "confidence": "high"}, "2"))
        self.assertTrue(bv.verdict_ok({"supported": "2", "checkable": True}, "2"))
        self.assertFalse(bv.verdict_ok({"supported": "both", "checkable": True}, "2"))
        self.assertFalse(bv.verdict_ok({"supported": "2", "checkable": False}, "2"))
        self.assertFalse(bv.verdict_ok(None, "1"))


def _synthetic_verified(n_papers: int, stratum: str, rng: random.Random) -> list[dict]:
    out = []
    for k in range(n_papers):
        a = f"2609.{k:05d}"
        types = list(bv.TYPES)
        rng.shuffle(types)
        types = types[:rng.choice((2, 3, 4))]
        claims = []
        for t in types:
            c = {"uid": f"{a}:{t}", "paper": a, "stratum": stratum, "type": t, "true_text": f"{a} {t} true",
                 "pert_text": f"{a} {t} false", "cited_pert": a, "mode": "shift", "value_true": "1.0",
                 "value_pert": "1.2", "evidence": "", "verified": True}
            if t == "attribution":
                c["cited_pert"] = f"2609.{(k + 1) % n_papers:05d}"   # a decoy that is another drawn paper
                c["mode"] = "decoy"
            claims.append(c)
        out.append({"paper": a, "stratum": stratum, "claims": claims})
    return out


class TestAssemble(unittest.TestCase):
    def test_choose_pairs_meets_quotas(self):
        rng = random.Random(0)
        papers = [{"paper": f"p{k}", "types": sorted(rng.sample(bv.TYPES, rng.choice((2, 3, 4))), key=bv.TYPES.index)}
                  for k in range(90)]
        pick, dev = bv.choose_pairs(papers, 60, 30, random.Random(1))
        self.assertEqual(dev, 0)
        self.assertEqual(len(pick), 60)
        self.assertEqual(Counter(t for _, ts in pick for t in ts), {t: 30 for t in bv.TYPES})
        self.assertTrue(all(len(set(ts)) == 2 for _, ts in pick))

    def test_fair_quotas(self):
        self.assertEqual(bv.fair_quotas(60, {t: 99 for t in bv.TYPES}), {t: 30 for t in bv.TYPES})
        q = bv.fair_quotas(60, {"number": 99, "dataset": 12, "conclusion": 99, "attribution": 99})
        self.assertEqual(q["dataset"], 12)
        self.assertEqual(sum(q.values()), 120)
        self.assertTrue(all(abs(q[t] - 36) <= 1 for t in ("number", "conclusion", "attribution")))
        pick, dev = bv.choose_pairs([{"paper": f"p{k}", "types": ["number", "conclusion", "attribution"] + (["dataset"] if k < 12 else [])}
                                     for k in range(80)], 60, q, random.Random(0))
        self.assertEqual(dev, 0)

    def test_assemble_blocks_items_flags(self):
        rng = random.Random(3)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "plugin" / "skills" / "cosmo-x").mkdir(parents=True)
            (d / "plugin" / "skills" / "cosmo-x" / "SKILL.md").write_text("x\n")
            ver = _synthetic_verified(90, "A", rng) + _synthetic_verified(30, "B", rng)
            from cosmobench.common import write_jsonl
            write_jsonl(d / "verified.jsonl", ver)
            with mock.patch.object(bv, "V5", d), mock.patch.dict(bv.ITEMS, {"K": 0, "K0": 0, "A": 30, "B": 30}):
                items = bv.assemble(d)
                rep = json.loads((d / "assemble_report.json").read_text())
        a_items = [it for it in items if it["stratum"] == "A"]
        self.assertEqual(len(a_items), 30)
        self.assertLess(rep["B"]["items"], 30)               # 30 papers cannot fill 30 items: shrinks by blocks
        self.assertEqual(rep["B"]["items"] % 2, 0)
        for it in items:
            refs = it["reference"]["claims"]
            self.assertEqual(len(refs), 4)
            self.assertEqual(len({r["paper"] for r in refs}), 4)                 # 4 different papers
            cited = [c["arxiv"] for c in it["prompt_fields"]["claims"]]
            self.assertEqual(len(set(cited)), 4)                                  # no ID twice in a call
        claims = [r for it in a_items for r in it["reference"]["claims"]]
        self.assertEqual(Counter(r["type"] for r in claims), {t: 30 for t in bv.TYPES})
        self.assertEqual(Counter((r["type"], r["perturbed"]) for r in claims), {(t, f): 15 for t in bv.TYPES for f in (True, False)})
        by_block: dict = {}
        for it in a_items:
            by_block.setdefault(it["block"], []).append(it)
        for blk, its in by_block.items():
            self.assertEqual(len(its), 2)
            p1 = {r["paper"] for r in its[0]["reference"]["claims"]}
            p2 = {r["paper"] for r in its[1]["reference"]["claims"]}
            self.assertEqual(p1, p2)                                              # papers nested in blocks
            for it in its:
                for r in it["reference"]["claims"]:
                    if r["type"] == "attribution" and r["perturbed"]:
                        self.assertNotIn(r["cited"], p1)                          # decoy not in its own block
        shown = a_items[0]["prompt_fields"]["claims"][0]
        ref = a_items[0]["reference"]["claims"][0]
        self.assertEqual(shown["text"], ref["pert_text"] if ref["perturbed"] else ref["true_text"])
        self.assertIn("(arXiv:", answer_prompt(a_items[0]))
        self.assertIn("data set or method", answer_prompt(a_items[0]))

    def test_accept_rule(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            from cosmobench.common import write_jsonl
            write_jsonl(d / "claimcheck_built.jsonl", [{"id": "i", "reference": {"claims": [
                {"uid": "u1"}, {"uid": "u2"}, {"uid": "u3"}, {"uid": "u4"}]}}])

            def sheet(oks):
                with (d / "claim_audit.csv").open("w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=["ok", "uid", "type"])
                    w.writeheader()
                    for k, ok in enumerate(oks, 1):
                        w.writerow({"ok": ok, "uid": f"u{k}", "type": "number" if k % 2 else "dataset"})
            with mock.patch.object(bv, "V5", d), mock.patch.object(bv, "RESULTS", d):
                sheet(["y", "", "y", "y"])
                self.assertEqual(bv.accept(), 1)                  # unfilled rows
                sheet(["n", "n", "n", "y"])
                self.assertEqual(bv.accept(), 3)                  # >= 3 wrong: rebuild
                sheet(["n", "y", "n", "y"])
                self.assertEqual(bv.accept(), 3)                  # two wrong in one type: rebuild that type
                sheet(["n", "y", "y", "n"])
                self.assertEqual(bv.accept(), 0)
                refs = json.loads((d / "claimcheck.jsonl").read_text())["reference"]["claims"]
            self.assertEqual([r["excluded"] for r in refs], [True, False, False, True])


class TestArmsV5(unittest.TestCase):
    def test_arm_strata(self):
        self.assertEqual(ARMS_BY_BENCH["claimcheck"], ["D0", "D1A", "D0F", "D1F"])
        self.assertEqual(ARMS_BY_BENCH["cosmopaperqa"], ["D0", "D1A"])
        k = {"bench": "claimcheck", "stratum": "K0"}
        a = {"bench": "claimcheck", "stratum": "A"}
        self.assertFalse(arm_applies("D0F", k))
        self.assertTrue(arm_applies("D1F", a))
        self.assertTrue(arm_applies("D1A", k))
        self.assertTrue(arm_applies("D0F", {"bench": "cosmopaperqa", "stratum": "thin"}))

    def test_run_loop_skips_f_arms_outside_a_b(self):
        from cosmobench.run import run_loop
        calls = []

        def ex(cmd, env, cwd, timeout, stdin):
            calls.append(stdin)
            return [json.dumps({"type": "result", "subtype": "success", "result": "1: correct"})]
        items = [{"id": "x", "bench": "claimcheck", "stratum": "K", "prompt_fields": {"claims": [{"text": "t", "arxiv": "1"}]}},
                 {"id": "y", "bench": "claimcheck", "stratum": "B", "prompt_fields": {"claims": [{"text": "t", "arxiv": "1"}]}}]
        with tempfile.TemporaryDirectory() as d:
            run_loop(items, ["D0", "D1A", "D0F", "D1F"], 1, ex, Path(d) / "a.jsonl", workdir=Path(d), base_env={},
                     log=lambda *a: None)
            arms = Counter((json.loads(line)["item_id"], json.loads(line)["arm"])
                           for line in (Path(d) / "a.jsonl").read_text().splitlines())
        self.assertEqual(len(calls), 6)
        self.assertNotIn(("x", "D0F"), arms)
        self.assertIn(("y", "D1F"), arms)


class TestGradeV5(unittest.TestCase):
    def test_lines_and_states_true(self):
        ans = "1: wrong — the paper gives Ωm = 0.297 ± 0.007\n2: correct\n3: cannot verify — no access\n"
        self.assertEqual(gc.parse_lines(ans, 4)[0][0], "wrong")
        self.assertIn("0.297", gc.parse_lines(ans, 4)[0][1])
        item = {"id": "i", "stratum": "A", "block": "A-00", "reference": {"claims": [
            {"type": "number", "perturbed": True, "value_true": "0.297", "mode": "swap", "paper": "p1"},
            {"type": "dataset", "perturbed": True, "value_true": "DESI DR2", "paper": "p2"},
            {"type": "conclusion", "perturbed": False, "value_true": "x", "paper": "p3", "excluded": True},
            {"type": "attribution", "perturbed": False, "value_true": "p4", "paper": "p4"}]}}
        rows = gc.grade_record({"item_id": "i", "arm": "D0", "rep": 0, "answer": ans, "status": "ok"}, item)
        self.assertEqual([r["outcome"] for r in rows], ["right", "false_confirm", "abstain", "missing"])
        self.assertEqual([r["score"] for r in rows], [1, -1, 0, 0])
        self.assertEqual(rows[0]["states_true"], True)
        self.assertIsNone(rows[1]["states_true"])
        self.assertTrue(rows[2]["excluded"])
        self.assertFalse(gc.states_value("the paper gives 0.2971", "0.297"))
        self.assertTrue(gc.states_value("the paper gives 0.3", "0.30"))
        self.assertEqual(gc.parse_verdicts("1 — correct\n2 - wrong\n3: not correct", 3), ["correct", "wrong", "missing"])
        failed = gc.grade_record({"item_id": "i", "arm": "D1F", "rep": 0, "answer": ans, "status": "error"}, item)
        self.assertEqual([r["verdict"] for r in failed], ["missing"] * 4)
        self.assertEqual(failed[0]["failed"], "error")


def _grades(effects: dict, n_blocks: int = 15, seed: int = 0) -> list[dict]:
    """Synthetic grades: effects[(arm, stratum)] = (p_right, p_error)."""
    rng = random.Random(seed)
    rows = []
    for (arm, s), (pr, pe) in effects.items():
        r2 = random.Random(f"{seed}{s}")  # same perturbation flags for every arm
        for b in range(n_blocks):
            for h in "ab":
                for c in range(1, 5):
                    u = rng.random()
                    o = "right" if u < pr else ("false_alarm" if u < pr + pe else "abstain")
                    rows.append({"item_id": f"{s}-{b}{h}", "arm": arm, "rep": 0, "claim": c, "stratum": s,
                                 "block": f"{s}-{b}", "type": bv.TYPES[c - 1], "mode": "shift",
                                 "perturbed": r2.random() < 0.5, "excluded": False, "in_skill_text": False,
                                 "verdict": {"right": "correct", "false_alarm": "wrong", "abstain": "cannot"}[o],
                                 "outcome": o, "score": gc.SCORE[o], "states_true": None})
    return rows


class TestAnalyseV5(unittest.TestCase):
    def test_sign_flip_and_bootstrap(self):
        rows = _grades({("D0", "A"): (0.1, 0.05), ("D1A", "A"): (0.6, 0.05)})
        c = av.contrast(rows, "D0", "D1A", "net", lambda r: r["stratum"] == "A")
        self.assertGreater(c["diff"], 0.3)
        self.assertLess(av.sign_flip_p(c["_blocks"], n=2000), 0.01)
        self.assertLess(c["ci_lo"], c["diff"])
        self.assertGreater(c["ci_hi"], c["diff"])
        null = _grades({("D0", "A"): (0.3, 0.05), ("D1A", "A"): (0.3, 0.05)}, seed=5)
        cn = av.contrast(null, "D0", "D1A", "net", lambda r: r["stratum"] == "A")
        self.assertGreater(av.sign_flip_p(cn["_blocks"], n=2000), 0.02)

    def test_noninferiority(self):
        rows = _grades({("D0", "K"): (0.5, 0.05), ("D1A", "K"): (0.5, 0.05)}, n_blocks=40, seed=2)
        c = av.contrast(rows, "D0", "D1A", "net", lambda r: r["stratum"] == "K")
        self.assertTrue(av.noninferiority(c, 0.15)["pass"])
        harm = _grades({("D0", "K"): (0.6, 0.05), ("D1A", "K"): (0.2, 0.05)}, n_blocks=40, seed=2)
        ch = av.contrast(harm, "D0", "D1A", "net", lambda r: r["stratum"] == "K")
        self.assertFalse(av.noninferiority(ch, 0.15)["pass"])
        err = _grades({("D0", "B"): (0.0, 0.02), ("D1A", "B"): (0.0, 0.30)}, seed=3)
        ce = av.contrast(err, "D0", "D1A", "error", lambda r: r["stratum"] == "B")
        self.assertFalse(av.noninferiority(ce, 0.05, lower_is_better=True)["pass"])

    def test_hypotheses_end_to_end(self):
        eff = {("D0", "A"): (0.05, 0.05), ("D1A", "A"): (0.6, 0.03), ("D0", "B"): (0.0, 0.08),
               ("D1A", "B"): (0.0, 0.02), ("D0", "K"): (0.5, 0.05), ("D1A", "K"): (0.55, 0.03),
               ("D0", "K0"): (0.5, 0.05), ("D1A", "K0"): (0.5, 0.05), ("D0F", "A"): (0.9, 0.02),
               ("D1F", "A"): (0.97, 0.01), ("D0F", "B"): (0.9, 0.02), ("D1F", "B"): (0.97, 0.01)}
        rows = _grades(eff, n_blocks=20, seed=7)
        hyp = {h["H"][:2]: h for h in av.hypotheses(rows)}
        self.assertEqual(set(hyp), {"H1", "H2", "H3", "H4"})
        self.assertTrue(hyp["H1"]["pass"])
        self.assertTrue(hyp["H2"]["pass"])
        self.assertTrue(hyp["H4"]["pass"])
        self.assertIn("K0", hyp["H3"]["note"])
        self.assertEqual([h["H"] for h in av.hypotheses(rows)], ["H1", "H3", "H2", "H4"])
        self.assertTrue(all(h["confirmatory"] for h in av.hypotheses(rows)))
        vals = {v["V"]: v for v in av.validity(rows, [])[0]}
        self.assertTrue(vals["V6 D0 net in K"]["ok"])
        self.assertEqual(vals["V7 ceiling, D0F net in A ∪ B"]["ok"], "ceiling")


if __name__ == "__main__":
    unittest.main()


class TestAnalyseV5Stats(unittest.TestCase):
    def test_estimate_and_sequence(self):
        blocks = {"K": {"b1": [1.0, 8], "b2": [-1.0, 8], "b3": [0.0, 8]}}
        e = av.estimate(blocks)
        self.assertAlmostEqual(e["diff"], 0.0)
        self.assertEqual((e["df"], e["n_claims"], e["n_blocks"]), (2, 24, 3))
        self.assertGreater(e["se"], 0)
        ni = av.noninferiority(e, 0.15)
        self.assertTrue(ni["bound"] < 0 < 0.15)
        self.assertAlmostEqual(av.sign_flip_p({"A": {"b": [2.0, 8]}}), 0.5)
        self.assertAlmostEqual(av.sign_flip_p({"A": {f"b{k}": [1.0, 8] for k in range(5)}}), 1 / 32)
        # fixed sequence: H1 failing makes the rest exploratory
        eff = {("D0", "A"): (0.3, 0.05), ("D1A", "A"): (0.3, 0.05), ("D0", "B"): (0.0, 0.05), ("D1A", "B"): (0.0, 0.05),
               ("D0", "K"): (0.5, 0.05), ("D1A", "K"): (0.5, 0.05), ("D0", "K0"): (0.5, 0.05), ("D1A", "K0"): (0.5, 0.05),
               ("D0F", "A"): (0.9, 0.02), ("D1F", "A"): (0.9, 0.02), ("D0F", "B"): (0.9, 0.02), ("D1F", "B"): (0.9, 0.02)}
        hyp = av.hypotheses(_grades(eff, n_blocks=20, seed=11))
        if not hyp[0]["pass"]:
            self.assertFalse(any(h["confirmatory"] for h in hyp[1:]))
            self.assertTrue(all("exploratory" in h["verdict"] for h in hyp[1:]))
        # V4 failure forces H4 "not tested"
        hyp = av.hypotheses(_grades(eff, n_blocks=20, seed=11), validity={"V4": False})
        self.assertIn("not tested", hyp[3]["note"])


def test_plain_asymmetric_errors():
    from cosmobench.build_v5 import plain
    assert plain("H0 = 68.02^{+0.49}_{-0.60} km/s") == "H0 = 68.02 +0.49/−0.60 km/s"
    assert plain("x = 1.2_{-0.3}^{+0.4}") == "x = 1.2 +0.4/−0.3"
    assert plain("10^{-3} eV") == "10^-3 eV"


def test_soft_fill_when_structure_blocks_equal_quotas():
    """Papers that are mostly attribution + one type: equal quotas infeasible, the soft fill uses all papers."""
    papers = ([{"paper": f"a{k}", "types": ["number", "attribution"]} for k in range(14)]
              + [{"paper": f"b{k}", "types": ["dataset", "attribution"]} for k in range(14)]
              + [{"paper": f"c{k}", "types": ["conclusion", "attribution"]} for k in range(14)]
              + [{"paper": f"d{k}", "types": ["number", "dataset", "conclusion"]} for k in range(6)])
    pick, dev = bv.choose_pairs(papers, 48, 24, random.Random(0))
    assert len(pick) == 48 and dev > 0
    got = Counter(t for _, ts in pick for t in ts)
    assert min(got.values()) >= bv.SOFT_MIN * 24
