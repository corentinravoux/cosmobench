"""Unit tests of the benchmark harness. Run from the checkout root: pytest"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa

from cosmobench import analyse, arms, cost, data, judge, l4_gate, run, sandbox
from cosmobench import build_claims as bc
from cosmobench import grade_claims as gc
from cosmobench import grade_tpbench as gt
from cosmobench.prompts import SKILLS_SYSTEM_APPEND, answer_prompt, judge_prompt

REPO = Path("/repo")


def cpqa_table():
    return pa.table({
        "author": ["a", "b"], "question": ["q0?", "q1?"], "ideal": ["i0", "i1"],
        "source file": ["https://arxiv.org/abs/1807.06209v4", "https://arxiv.org/abs/2201.02202v1"],
        "doi": ["https://doi.org/10.1051/0004-6361/201833910", "x"],
        "Location": ["p1", None], "key passage": ["kp", None]})


def tpb_table():
    return pa.table({
        "problem_id": ["Bias of a Sampled Halo Field", "A 3-State QM Problem", "Blackbody", "Other"],
        "domain": ["Cosmology", "QM", "Stat Mech", "HET"], "difficulty_level": ["5", "2", "1", "3"],
        "problem": ["p"] * 4, "solution": ["s"] * 4, "answer": ["a"] * 4,
        "code_answer_requirements": ["r"] * 4, "reference_implementation": ["x"] * 4})


class TestData(unittest.TestCase):
    def test_cpqa_ids_strata_nulls(self):
        items = data.normalise_cosmopaperqa(cpqa_table())
        self.assertEqual([i["id"] for i in items], ["cpqa-000", "cpqa-001"])
        self.assertEqual([i["stratum"] for i in items], ["covered", "thin"])
        self.assertEqual(items[0]["source_arxiv"], "1807.06209")
        self.assertEqual(items[1]["reference"]["key_passage"], "")

    def test_tpbench_slug_domain_subset(self):
        items = data.normalise_tpbench(tpb_table())
        self.assertEqual(items[0]["id"], "tpb-bias-of-a-sampled-halo-field")
        self.assertEqual(items[0]["domain"], "Cosmology")
        self.assertEqual(data.tpbench_subset(items),
                         ["tpb-bias-of-a-sampled-halo-field", "tpb-a-3-state-qm-problem", "tpb-blackbody"])

    def test_roundtrip_jsonl(self):
        items = data.normalise_cosmopaperqa(cpqa_table())
        with tempfile.TemporaryDirectory() as d:
            data.write_jsonl(Path(d) / "cosmopaperqa.jsonl", items)
            self.assertEqual(data.load_items("cosmopaperqa", Path(d)), items)


class TestArms(unittest.TestCase):
    def cmd(self, name, **kw):
        return arms.build_command(arms.ARMS[name], claude_bin="claude", repo=REPO, **kw)

    def test_arm_set(self):
        self.assertEqual(sorted(arms.ARMS), ["D0", "D0F", "D1", "D1A", "D1F"])
        for name in arms.ARMS:
            self.assertEqual(arms.ARMS[name].effort, "high")

    def test_skills_off(self):
        c = self.cmd("D0")
        self.assertIn("--disable-slash-commands", c)
        self.assertEqual(c[c.index("--tools") + 1], "Read,Grep,Glob")   # v5: same tools as D1A minus Skill
        self.assertEqual(c[c.index("--allowedTools") + 1:c.index("--allowedTools") + 4], ["Read", "Grep", "Glob"])
        self.assertNotIn("--plugin-dir", c)
        self.assertNotIn("--append-system-prompt", c)

    def test_skills_on_plugin_dir(self):
        c = self.cmd("D1")
        self.assertEqual(c[c.index("--plugin-dir") + 1], "/repo/plugin")
        self.assertEqual(c[c.index("--tools") + 1], "Skill,Read,Grep,Glob")
        self.assertEqual(c[c.index("--append-system-prompt") + 1], SKILLS_SYSTEM_APPEND)

    def test_d1a_has_no_append(self):
        self.assertNotIn("--append-system-prompt", self.cmd("D1A"))
        self.assertIn("--plugin-dir", self.cmd("D1A"))

    def test_fetch_arms_bash_not_allowed_by_rule(self):
        for name, tools in (("D0F", "Read,Grep,Glob,Bash"), ("D1F", "Skill,Read,Grep,Glob,Bash")):
            c = self.cmd(name)
            self.assertEqual(c[c.index("--tools") + 1], tools)
            self.assertFalse([x for x in c if x.startswith("Bash(")])  # the gate hook allows, no rule
            app = c[c.index("--append-system-prompt") + 1]
            self.assertIn("/repo/plugin/skills/observational-cosmology/scripts/fetch_paper.py", app)
            self.assertIn("No other shell command", app)
        self.assertNotIn("--plugin-dir", self.cmd("D0F"))
        self.assertNotIn(SKILLS_SYSTEM_APPEND, self.cmd("D1F")[self.cmd("D1F").index("--append-system-prompt") + 1])

    def test_common_flags_and_config_deny(self):
        for name in arms.ARMS:
            c = self.cmd(name, env={"CLAUDE_CONFIG_DIR": "/h/.claude-alt"})
            for flag in ("--no-session-persistence", "--strict-mcp-config", "--print"):
                self.assertIn(flag, c)
            self.assertEqual(c[c.index("--model") + 1], "claude-opus-5-5")
            self.assertEqual(c[c.index("--setting-sources") + 1], "project")
            self.assertIn("Read(//h/.claude-alt/**)", c)

    def test_judges_models(self):
        c2 = arms.build_command(arms.JUDGE2, claude_bin="c", repo=REPO, json_schema={"a": 1})
        self.assertEqual(c2[c2.index("--model") + 1], "claude-sonnet-5-5")
        self.assertIn("--json-schema", c2)

    def test_env(self):
        with tempfile.TemporaryDirectory() as d:
            env = arms.build_env({}, Path(d))
            self.assertTrue(env["CLAUDE_CONFIG_DIR"].endswith(".claude-alt"))
            self.assertEqual(env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"], "1")
            self.assertEqual(env["COSMOSKILL_HOME"], str(Path(d) / "empty-layer"))
            self.assertTrue(Path(env["COSMOSKILL_HOME"]).is_dir())
            self.assertEqual(arms.build_env({"CLAUDE_CONFIG_DIR": "/x"}, Path(d))["CLAUDE_CONFIG_DIR"], "/x")


class TestGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "skills"
        (root / "observational-cosmology" / "scripts").mkdir(parents=True)
        (root / "cosmo-cmb").mkdir()
        self.fp = root / "observational-cosmology" / "scripts" / "fetch_paper.py"
        self.fp.write_text("")
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def ok(self, cmd):
        return l4_gate.decide(cmd, str(self.fp))[0]

    def test_allowed_forms(self):
        r = self.root
        self.assertTrue(self.ok(f"python3 {self.fp} 1807.06209 --abstract"))
        self.assertTrue(self.ok(f'python3 "{r}/cosmo-cmb"/../*observational-cosmology/scripts/fetch_paper.py 1807.06209 --grep "8.63"'))
        self.assertTrue(self.ok(f'python3 "{r}/observational-cosmology"/scripts/fetch_paper.py 2503.14454 --section "Results"'))

    def test_denied(self):
        for cmd in ("ls ~", "python3 -c 'print(1)'", f"python3 {self.fp} 1; ls ~", f"python3 {self.fp} 1 | head",
                    f"python3 {self.fp} $(cat /etc/passwd)", f"python3 {self.fp} 1 > /tmp/x", f"cat {self.fp}",
                    f"python3 {self.root}/other.py 1", f"python3 {self.fp} 1 && ls", "", "python3"):
            self.assertFalse(self.ok(cmd), cmd)

    def test_hook_output_shape(self):
        out = subprocess.run([sys.executable, "-I", str(l4_gate.__file__), str(self.fp)], capture_output=True, text=True,
                             input=json.dumps({"tool_input": {"command": "ls ~"}}))
        h = json.loads(out.stdout)["hookSpecificOutput"]
        self.assertEqual((h["hookEventName"], h["permissionDecision"]), ("PreToolUse", "deny"))
        self.assertIn("fetch_paper.py", h["permissionDecisionReason"])


class TestSandbox(unittest.TestCase):
    def test_masks_keep_credentials_only(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d)
            for f in (".credentials.json", ".claude.json", ".last-cleanup"):
                (cfg / f).write_text("x")
            for sub in ("projects", "file-history", "skills"):
                (cfg / sub).mkdir()
            (cfg / "settings.json").symlink_to("/nowhere")
            m = sandbox.masks(cfg)
            joined = " ".join(m)
            self.assertNotIn(".credentials.json", joined)
            self.assertNotIn(".claude.json ", joined + " ")
            self.assertIn(f"--tmpfs {cfg}/projects", joined)
            self.assertIn(f"--ro-bind /dev/null {cfg}/.last-cleanup", joined)
            self.assertNotIn("settings.json", joined)

    def test_wrap_hides_home(self):
        if not shutil.which("bwrap"):
            self.skipTest("no bwrap")
        with tempfile.TemporaryDirectory() as d:
            w = sandbox.wrap(["/bin/true", "--x"], repo=Path(d), workdir=Path(d),
                             env={"CLAUDE_CONFIG_DIR": d}, home=Path("/home/u"))
            self.assertEqual(w[w.index("--tmpfs", w.index("--die-with-parent")) + 1], "/home/u")
            self.assertEqual(w[-2:], ["/usr/bin/true", "--x"] if Path("/usr/bin/true").exists() and
                             Path("/bin/true").resolve() == Path("/usr/bin/true") else [str(Path("/bin/true").resolve()), "--x"])
            self.assertIn("--unshare-pid", w)

    def test_plugin_visible_only_to_skills_arms(self):
        if not shutil.which("bwrap"):
            self.skipTest("no bwrap")
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d) / "repo"
            (repo / "plugin" / "skills" / "observational-cosmology" / "scripts").mkdir(parents=True)
            plugin = str((repo / "plugin").resolve())
            scripts = str((repo / "plugin" / "skills" / "observational-cosmology" / "scripts").resolve())

            def binds(arm):
                c = arms.build_command(arms.ARMS[arm], claude_bin="/bin/true", repo=repo)
                w = sandbox.wrap(c, repo=repo, workdir=Path(d), env={"CLAUDE_CONFIG_DIR": d}, home=Path("/home/u"))
                return {w[k + 1] for k, x in enumerate(w) if x == "--ro-bind"}
            self.assertIn(plugin, binds("D1A"))
            self.assertIn(plugin, binds("D1F"))
            self.assertFalse(plugin in binds("D0") or scripts in binds("D0"))
            self.assertNotIn(plugin, binds("D0F"))
            self.assertIn(scripts, binds("D0F"))

    def test_cache_does_not_persist_across_calls(self):
        """Papers fetched in one call (XDG_CACHE_HOME or workdir/.cache) are gone for the next call."""
        if not shutil.which("bwrap"):
            self.skipTest("no bwrap")
        with tempfile.TemporaryDirectory() as d:
            work, cfg = Path(d) / "work", Path(d) / "cfg"
            work.mkdir(); cfg.mkdir()
            sh = ["/bin/sh", "-c", 'mkdir -p "$XDG_CACHE_HOME/p" && echo x > "$XDG_CACHE_HOME/p/a.html" '
                  '&& mkdir -p .cache && echo x > .cache/b.html && ls "$XDG_CACHE_HOME/p" .cache']
            w = sandbox.wrap(sh, repo=Path(d), workdir=work, env={"CLAUDE_CONFIG_DIR": str(cfg)}, home=Path(d) / "h")
            out = subprocess.run(w, capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("a.html", out.stdout)            # the call itself can cache
            self.assertEqual(list((work / ".cache").rglob("*")), [])   # nothing survives on the host
            self.assertFalse(any(Path(d).rglob("a.html")))

    def test_prepare_workdir(self):
        with tempfile.TemporaryDirectory() as d:
            sandbox.prepare_workdir(Path(d), REPO)
            st = json.loads((Path(d) / ".claude" / "settings.json").read_text())
            cmd = st["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
            self.assertIn("l4_gate.py /repo/plugin/skills/observational-cosmology/scripts/fetch_paper.py", cmd)
            self.assertTrue((Path(d) / "empty-layer").is_dir())


def ev(**kw):
    return json.dumps(kw)


def tool_use(uid, name, inp):
    return ev(type="assistant", message={"content": [{"type": "tool_use", "id": uid, "name": name, "input": inp}]})


def tool_result(uid, err=False):
    return ev(type="user", message={"content": [{"type": "tool_result", "tool_use_id": uid, "is_error": err, "content": "x"}]})


def result(text="ans", err=False, **kw):
    return ev(type="result", subtype="error" if err else "success", is_error=err, result=text,
              usage={"input_tokens": 1, "output_tokens": 2}, total_cost_usd=0.1, num_turns=1,
              duration_ms=10, permission_denials=[], **kw)


class TestRun(unittest.TestCase):
    def test_parse_ok_with_skill_and_bash(self):
        r = run.parse_stream([tool_use("1", "Skill", {"skill": "cosmoskill:cosmo-cmb"}), tool_result("1"),
                              tool_use("2", "Bash", {"command": "python3 fetch_paper.py 1 --abstract"}),
                              tool_result("2"), result()])
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["skills_invoked"], ["cosmoskill:cosmo-cmb"])
        self.assertEqual(r["bash_commands"], ["python3 fetch_paper.py 1 --abstract"])
        self.assertEqual(r["usage"]["output_tokens"], 2)

    def test_fetch_counts_and_thinking(self):
        mu = {"claude-opus-5-5": {"thinkingTokens": 7}, "claude-haiku-5-5": {"thinkingTokens": 0}}
        r = run.parse_stream([tool_use("1", "Bash", {"command": "python3 x/fetch_paper.py 1 --abstract"}), tool_result("1"),
                              tool_use("2", "Bash", {"command": "python3 x/fetch_paper.py 2"}), tool_result("2", err=True),
                              result(modelUsage=mu)])
        self.assertEqual((r["fetch_calls"], r["fetch_ok"], r["thinking_tokens"]), (2, 1, 7))

    def test_config_dir_read_is_leak(self):
        p = str(Path.home() / ".claude-alt" / "projects" / "x.jsonl")
        r = run.parse_stream([tool_use("1", "Read", {"file_path": p}), tool_result("1"), result()])
        self.assertEqual(r["status"], "leak")

    def test_parse_error_and_truncated(self):
        self.assertEqual(run.parse_stream([result("limit reached", err=True)])["status"], "error")
        r = run.parse_stream([tool_use("1", "Read", {"file_path": "/a"})])
        self.assertEqual((r["status"], r["error"]), ("error", "no result event"))

    def test_leak_flagged_only_when_read_succeeds(self):
        p = str(Path.home() / ".cosmoskill" / "README.md")
        denied = run.parse_stream([tool_use("1", "Read", {"file_path": p}), tool_result("1", err=True), result()])
        self.assertEqual(denied["status"], "ok")
        leaked = run.parse_stream([tool_use("1", "Read", {"file_path": p}), tool_result("1"), result()])
        self.assertEqual(leaked["status"], "leak")
        nothing = run.parse_stream([tool_use("1", "Glob", {"path": str(Path.home()), "pattern": ".cosmoskill/**"}),
                                    ev(type="user", message={"content": [{"type": "tool_result", "tool_use_id": "1",
                                                                          "content": "No files found"}]}), result()])
        self.assertEqual(nothing["status"], "ok")
        empty = run.parse_stream([tool_use("1", "Glob", {"path": "/w/empty-layer/.cosmoskill"}), tool_result("1"), result()])
        self.assertEqual(empty["status"], "ok")
        project = run.parse_stream([tool_use("1", "Glob", {"path": "/w/workdir", "pattern": ".cosmoskill/**"}),
                                    tool_result("1"), result()])
        self.assertEqual(project["status"], "ok")

    def test_touches_personal_layer(self):
        h = Path("/home/u")
        self.assertTrue(run.touches_personal_layer("/home/u/.cosmoskill/notes.md", h))
        self.assertTrue(run.touches_personal_layer("/home/u .cosmoskill/**", h))
        self.assertTrue(run.touches_personal_layer("cat ~/.cosmoskill/x", h))
        self.assertFalse(run.touches_personal_layer("/w/workdir .cosmoskill/**", h))
        self.assertFalse(run.touches_personal_layer("/w/workdir/empty-layer/README.md", h))

    def test_done_keys_and_stop_after_errors(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "a.jsonl"
            items = [{"id": f"i{k}", "bench": "cosmopaperqa", "prompt_fields": {"question": "q"}} for k in range(5)]
            calls = []
            def ex(cmd, env, cwd, timeout, stdin):
                calls.append(stdin)
                return [result("usage limit", err=True)]
            code = run.run_loop(items, ["D0"], 1, ex, out, workdir=Path(d), base_env={}, log=lambda *_: None)
            self.assertEqual((code, len(calls)), (2, 3))
            self.assertEqual(run.done_keys(out), set())
            ok = lambda *a: [result()]
            self.assertEqual(run.run_loop(items[:2], ["D0", "D1"], 1, ok, out, workdir=Path(d), base_env={},
                                          log=lambda *_: None), 0)
            self.assertEqual(len(run.done_keys(out)), 4)

    def test_commit_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["git", "init", "-q", d], check=True)
            subprocess.run(["git", "-C", d, "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-q",
                            "--allow-empty", "-m", "x"], check=True)
            cfg = Path(d) / "config.json"
            cfg.write_text(json.dumps({"cosmoskill_commit": "deadbeef"}))
            with self.assertRaises(SystemExit):
                run.check_pin(Path(d), cfg)
            cfg.write_text(json.dumps({"cosmoskill_commit": run.repo_commit(Path(d))}))
            self.assertEqual(run.check_pin(Path(d), cfg), run.repo_commit(Path(d)))

    def test_select_per_stratum(self):
        items = [{"id": str(k), "stratum": "covered" if k < 5 else "thin"} for k in range(10)]
        self.assertEqual([i["id"] for i in run.select_items(items, ids=None, limit=None, per_stratum=2)],
                         ["0", "1", "5", "6"])


REF = "def speed(v: float, w: float) -> float:\n    return v * w\n"


class TestGrader(unittest.TestCase):
    def g(self, cand, ref=REF, **kw):
        return gt.grade(cand, ref, **kw)

    def test_pass_and_algebraic(self):
        self.assertEqual(self.g(REF)["verdict"], "pass")
        self.assertEqual(self.g("def speed(v, w):\n    return w * v\n")["verdict"], "pass")

    def test_wrong_and_missing(self):
        self.assertEqual(self.g("def speed(v, w):\n    return v + w\n")["verdict"], "fail")
        self.assertEqual(self.g(None)["reason"], "no code")
        self.assertIn("missing", self.g("def other(v, w):\n    return v * w\n")["reason"])

    def test_extract_last_block(self):
        ans = "x\n```python\ndef speed(v, w):\n    return 0\n```\nfix:\n```python\n" + REF + "```\n"
        self.assertEqual(self.g(gt.extract_code(ans))["verdict"], "pass")
        self.assertIsNone(gt.extract_code("no code here"))

    def test_raising_exit_hanging(self):
        self.assertIn("error", self.g("def speed(v, w):\n    raise ValueError\n")["reason"])
        self.assertEqual(self.g("import sys\nsys.exit(0)\n")["verdict"], "fail")
        self.assertEqual(self.g("def speed(v, w):\n    while True: pass\n", timeout=3)["reason"], "timeout")

    def test_tuple_array_noarg(self):
        ref = "def f(a: float, b: float):\n    return (a, -b)\n"
        self.assertEqual(self.g("def f(a, b):\n    return [a, -b]\n", ref)["verdict"], "pass")
        self.assertEqual(self.g("def f(a, b):\n    return (a, b)\n", ref)["verdict"], "fail")
        aref = "import numpy as np\ndef phi(q: float, t: np.ndarray) -> np.ndarray:\n    return q * t\n"
        self.assertEqual(self.g("def phi(q, t):\n    return t * q\n", aref)["verdict"], "pass")
        nref = "def answer() -> float:\n    return 5\n"
        self.assertEqual(self.g("def answer():\n    return 5.0\n", nref)["verdict"], "pass")
        self.assertEqual(self.g("def answer():\n    return 4\n", nref)["verdict"], "fail")

    def test_strip_reference(self):
        ref = "\\begin{python}\ndef f():\n    return 1\n\\end{python}\n\n\\newpage"
        self.assertEqual(gt.strip_reference(ref), "def f():\n    return 1\n")


class TestJudge(unittest.TestCase):
    def test_strip_markers(self):
        a = ("H0 = 67.4 km/s/Mpc [src: Table 2] (arXiv:1807.06209v4; doi:10.1051/0004-6361/201833910), "
             "see `cosmoskill:cosmo-cmb` L2 and https://arxiv.org/abs/2503.14454.")
        s = judge.strip_markers(a)
        for gone in ("src:", "1807.06209", "doi", "cosmo-cmb", "arxiv.org", "skill"):
            self.assertNotIn(gone, s)
        self.assertIn("L2", s)  # v2 removed it; it is also the Planck orbit
        n = judge.strip_markers("The cosmo-cmb skill does not cover it, and fetch_paper.py (L4) failed; Planck used PCA.")
        self.assertEqual(n, "my sources does not cover it, and my sources failed; Planck used PCA.")
        self.assertIn("67.4", s)
        self.assertIn("0.1%", judge.strip_markers("about 0.1% at l=2000"))

    def test_provenance(self):
        it = {"source_arxiv": "1807.06209", "reference": {"doi": "https://doi.org/10.1051/0004-6361/201833910"}}
        self.assertEqual(judge.provenance("see arXiv:1807.06209v2", it), {"cites_any": True, "cites_source": True})
        self.assertEqual(judge.provenance("see 2503.14454", it), {"cites_any": True, "cites_source": False})
        self.assertEqual(judge.provenance("no source", it), {"cites_any": False, "cites_source": False})

    def test_parse_grade(self):
        g = judge.parse_grade('```json\n{"grade": "abstain", "false_claim": false, "declares_gap": true, "reason": "r"}\n```')
        self.assertEqual((g["grade"], g["declares_gap"]), ("abstain", True))
        self.assertIsNone(judge.parse_grade('{"grade": "maybe"}'))
        self.assertIsNone(judge.parse_grade("garbage"))
        self.assertEqual(judge.parse_grade({"grade": "correct", "false_claim": True})["false_claim"], True)

    def test_judge_prompt_has_no_arm(self):
        it = {"prompt_fields": {"question": "q"}, "reference": {"ideal": "i", "key_passage": ""}}
        p = judge_prompt(it, "a")
        for s in ("L0", "D1", "skill", "cosmo-"):
            self.assertNotIn(s, p)
        self.assertIn("not given", p)

    def test_order_deterministic(self):
        recs = [{"item_id": str(k), "arm": "D0", "rep": 0} for k in range(20)]
        self.assertEqual(judge.judge_order(recs), judge.judge_order(list(reversed(recs))))

    def test_unjudged_after_one_retry(self):
        calls = []
        def ex(*a):
            calls.append(1)
            return [result("garbage")]
        it = {"prompt_fields": {"question": "q"}, "reference": {"ideal": "i"}}
        g = judge.judge_one(it, "a", ex, claude_bin="c", repo=REPO, env={}, workdir=Path("."), arm=arms.JUDGE2)
        self.assertEqual((g["grade"], len(calls)), ("unjudged", 2))


class TestCost(unittest.TestCase):
    def test_summarise_and_project(self):
        recs = [{"arm": "D0", "status": "ok", "cost_usd": 0.5, "duration_ms": 1000,
                 "usage": {"input_tokens": 1, "cache_read_input_tokens": 2,
                           "cache_creation_input_tokens": 3, "output_tokens": 4}}] * 2
        s = cost.summarise(recs)
        self.assertEqual(s["D0"]["total_tokens"], 20)
        p = cost.project(s, 10)
        self.assertAlmostEqual(p["D0"]["cost_usd"], 5.0)
        self.assertAlmostEqual(p["D0"]["tokens"], 100)


class TestClaims(unittest.TestCase):
    ABS = (r"We find $\Omega_m = 0.295 \pm 0.015$ and $H_0=73.0^{+1.2}_{-1.0}$ km/s/Mpc, a 3.1$\sigma$ "
           r"tension, with $\sum m_\nu < 0.064$ eV at 95% confidence.")

    def test_spans(self):
        sp = bc.numeric_spans(self.ABS)
        self.assertEqual([(s["kind"], s["value"], s["err"]) for s in sp],
                         [("pm", "0.295", "0.015"), ("asym", "73.0", "1.2"), ("sigma", "3.1", None),
                          ("limit", "0.064", None)])  # "95% confidence" is not a result
        self.assertFalse(bc.is_numeric_abstract("We present a new method."))
        rng = bc.numeric_spans("galaxies at 0.6 < z < 1.6 give a 12% gain and $S_8 < 0.80$")
        self.assertEqual([(s["kind"], s["value"]) for s in rng], [("percent", "12"), ("limit", "0.80")])

    def test_template_fits(self):
        self.assertTrue(bc.template_fits("Omega_m = {V} ± 0.015.", "pm"))
        self.assertFalse(bc.template_fits("there are {V} clusters", "limit"))
        self.assertTrue(bc.template_fits("Σmν < {V} eV", "limit"))
        self.assertTrue(bc.template_fits("an upper limit of {V} eV", "limit"))
        self.assertTrue(bc.template_fits("at the {V}σ level", "sigma"))
        self.assertTrue(bc.template_fits("improves it by ~{V}%", "percent"))
        self.assertFalse(bc.template_fits("w0 = {V}", "pm"))
        for t in ("S8 = {V}^{+0.019}_{-0.022}.", "Om = {V} +0.023/−0.036", "w = {V} -0.052/+0.055",
                  "S_8 = {V} ± 0.019", "H0 = {V}^{+8.2}_{-6.1} km/s/Mpc", "r = {V} +0.042 −0.018 (95% C.L.)",
                  "w = {V} (+0.329, −0.396).", "f = {V} (−0.62, +0.57)"):
            self.assertTrue(bc.template_fits(t, "asym"), t)
        self.assertFalse(bc.template_fits("w = {V} in wCDM", "asym"))
        self.assertFalse(bc.template_fits("a {V}-20% gain", "asym"))

    def test_perturb_is_far_and_formatted(self):
        import random
        rng = random.Random(1)
        for _ in range(200):
            v = bc.perturb("pm", "0.295", "0.015", rng)
            self.assertEqual(len(v.split(".")[1]), 3)
            self.assertGreaterEqual(abs(float(v) - 0.295), 0.0295 - 1e-9)
            s = float(bc.perturb("sigma", "3.1", None, rng))
            self.assertGreaterEqual(abs(s - 3.1), 1.5 - 1e-9)
            lim = float(bc.perturb("limit", "0.064", None, rng))
            self.assertTrue(lim >= 0.064 * 1.99 or lim <= 0.064 / 1.99)
            self.assertNotEqual(bc.perturb("pm", "7", "0.1", rng), "7")
            self.assertGreater(float(bc.perturb("pm", "0.02", "0.5", rng)), 0)

    def test_balanced_and_strata(self):
        import datetime as dt
        import random
        f = bc.balanced_flags(80, random.Random(3))
        self.assertEqual((len(f), sum(f)), (80, 40))
        snap, end = dt.date(2026, 10, 26), dt.date(2026, 11, 13)
        self.assertEqual(bc.stratum_of(dt.date(2024, 4, 1), True, snap, end), "K")
        self.assertEqual(bc.stratum_of(dt.date(2026, 8, 1), True, snap, end), "A")
        self.assertEqual(bc.stratum_of(dt.date(2026, 11, 2), False, snap, end), "B")
        self.assertIsNone(bc.stratum_of(dt.date(2026, 11, 2), True, snap, end))
        self.assertIsNone(bc.stratum_of(dt.date(2026, 11, 18), False, snap, end))
        self.assertIsNone(bc.stratum_of(dt.date(2019, 1, 1), True, snap, end))

    def test_validate_draft(self):
        sp = bc.numeric_spans(self.ABS)
        good = {"claims": [{"span": 0, "template": "DESI gives Omega_m = {V} ± 0.015."},
                           {"span": 2, "template": "The tension is {V}σ."}]}
        self.assertEqual(bc.validate_draft(good, sp, "2608.01234"), "")
        bad = {"claims": [{"span": 0, "template": "x {V} {V}"}, {"span": 1, "template": "y {V}"}]}
        self.assertIn("placeholder", bc.validate_draft(bad, sp, "2608.01234"))
        same = {"claims": [{"span": 0, "template": "x {V}"}, {"span": 0, "template": "y {V}"}]}
        self.assertEqual(bc.validate_draft(same, sp, "x"), "same span twice")
        idd = {"claims": [{"span": 0, "template": "2608.01234 gives {V}"}, {"span": 1, "template": "y {V}"}]}
        self.assertIn("arXiv", bc.validate_draft(idd, sp, "2608.01234"))
        tex = {"claims": [{"span": 0, "template": "$\\Omega$ = {V}"}, {"span": 1, "template": "y {V}"}]}
        self.assertIn("LaTeX", bc.validate_draft(tex, sp, "x"))

    def test_assemble_and_prompt(self):
        import random
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "plugin" / "skills" / "cosmo-x").mkdir(parents=True)
            (Path(d) / "plugin" / "skills" / "cosmo-x" / "SKILL.md").write_text("- Omega_m 0.295 (2608.01234)\n")
            c = {"arxiv": "2608.01234", "stratum": "A", "domain": "cosmo-x", "published": "2026-08-01", "title": "t",
                 "spans": bc.numeric_spans(self.ABS)}
            dr = {"claims": [{"span": 0, "template": "DESI gives Omega_m = {V} ± 0.015."},
                             {"span": 2, "template": "The tension is {V}σ."}]}
            it = bc.assemble(c, dr, [False, True], Path(d), random.Random(0))
        sh, rf = it["prompt_fields"]["claims"], it["reference"]["claims"]
        self.assertEqual(sh[0]["text"], "DESI gives Omega_m = 0.295 ± 0.015.")
        self.assertNotEqual(sh[1]["text"], "The tension is 3.1σ.")
        self.assertEqual([r["perturbed"] for r in rf], [False, True])
        self.assertEqual([r["in_skill_text"] for r in rf], [True, False])
        p = answer_prompt(it)
        self.assertIn("1. DESI gives Omega_m = 0.295 ± 0.015. (arXiv:2608.01234)", p)
        self.assertIn("cannot verify", p)


class TestGradeClaims(unittest.TestCase):
    def test_parse_verdicts(self):
        ans = "1: correct — matches Table 2\n**2: wrong** — the paper gives 0.31\n"
        self.assertEqual(gc.parse_verdicts(ans, 2), ["correct", "wrong"])
        self.assertEqual(gc.parse_verdicts("1. Cannot verify — no access\n2) incorrect", 2), ["cannot", "wrong"])
        self.assertEqual(gc.parse_verdicts("Claim 2: correct", 2), ["missing", "correct"])
        self.assertEqual(gc.parse_verdicts("I cannot help.", 2), ["missing", "missing"])
        self.assertEqual(gc.parse_verdicts("1: correct\n1: wrong\n2: cannot be verified", 2), ["correct", "cannot"])

    def test_outcomes(self):
        self.assertEqual(gc.outcome("correct", False), "right")
        self.assertEqual(gc.outcome("wrong", True), "right")
        self.assertEqual(gc.outcome("correct", True), "false_confirm")
        self.assertEqual(gc.outcome("wrong", False), "false_alarm")
        self.assertEqual(gc.outcome("cannot", True), "abstain")
        self.assertEqual(gc.outcome("missing", False), "missing")


class TestAnalyse(unittest.TestCase):
    def test_mcnemar(self):
        self.assertEqual(analyse.mcnemar_exact([True, False], [True, False]), 1.0)
        self.assertAlmostEqual(analyse.mcnemar_exact([False] * 10, [True] * 10), 2 * 0.5 ** 10)

    def test_holm_and_kappa(self):
        adj = analyse.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adj["a"], 0.03)
        self.assertAlmostEqual(adj["c"], 0.06)
        self.assertAlmostEqual(adj["b"], 0.06)
        self.assertAlmostEqual(analyse.cohen_kappa(list("aabb"), list("aabb")), 1.0)
        self.assertAlmostEqual(analyse.cohen_kappa(list("ab"), list("ba")), -1.0)

    def test_bootstrap_and_cluster(self):
        lo, hi = analyse.bootstrap_ci([0, 1] * 50, n=2000)
        self.assertTrue(lo < 0.5 < hi)
        d, lo, hi = analyse.cluster_diff({"i1": [(0, 1), (0, 1)], "i2": [(1, 1)]}, n=500)
        self.assertAlmostEqual(d, 2 / 3)
        self.assertTrue(lo <= d <= hi)

    def test_cc_metrics_and_paired(self):
        rows = []
        for k in range(10):
            for arm, out in (("D0", "abstain"), ("D1A", "right")):
                rows.append({"item_id": f"i{k}", "claim": 1, "arm": arm, "stratum": "A", "perturbed": k % 2 == 0,
                             "in_skill_text": True, "outcome": out, "verdict": "x"})
        m = analyse.cc_metrics([r for r in rows if r["arm"] == "D1A"])
        self.assertEqual((m["right"], m["false_confirm_on_perturbed"]), (1.0, 0.0))
        p = analyse.cc_paired(rows, "D0", "D1A", "right")
        self.assertEqual((p["n_claims"], p["diff"]), (10, 1.0))
        self.assertAlmostEqual(p["p"], 2 * 0.5 ** 10)

    def test_qa_metrics(self):
        gs = [{"grade": g, "false_claim": False} for g in ("correct", "partial", "incorrect", "abstain")] + [{"grade": "unjudged"}]
        m = analyse.qa_metrics(gs)
        self.assertEqual((m["n"], m["correct"], m["lenient"], m["abstain"], m["unjudged"]), (4, 0.25, 0.375, 0.25, 1))
        self.assertAlmostEqual(m["correct_given_attempted"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
