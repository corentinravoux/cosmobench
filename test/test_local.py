"""Local-model mode (cosmobench.local): guards, environment, command line, pilot gate, mock server."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from cosmobench import local
from cosmobench.arms import ARMS, DRAFTER, JUDGE, build_command, build_env
from cosmobench.common import CLAUDE_MODEL


class LocalEnvTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Path(self.tmp.name) / "claude-local"
        self.base = {"BENCH_LOCAL_URL": "http://127.0.0.1:9931", "BENCH_MODEL": "recipe-x",
                     "BENCH_LOCAL_CONFIG_DIR": str(self.cfg), "ANTHROPIC_API_KEY": "sk-real",
                     "CLAUDE_CODE_OAUTH_TOKEN": "oauth-real", "PATH": "/usr/bin"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_env_points_to_local_server_without_real_credentials(self):
        env = local.local_env(self.base)
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:9931")
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "local")
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", env)
        self.assertEqual(env["CLAUDE_CONFIG_DIR"], str(self.cfg.resolve()))
        for var in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"):
            self.assertEqual(env[var], "recipe-x")
        self.assertEqual(env["CLAUDE_CODE_DISABLE_THINKING"], "1")
        self.assertEqual(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], str(local.DEFAULT_CTX))

    def test_build_env_keeps_local_config_dir(self):
        env = build_env(self.base, workdir=Path(self.tmp.name) / "wd")
        self.assertEqual(env["CLAUDE_CONFIG_DIR"], str(self.cfg.resolve()))

    def test_refuses_shared_config_dirs(self):
        for d in local.FORBIDDEN_CONFIG_DIRS:
            with self.assertRaises(SystemExit):
                local.local_env(dict(self.base, BENCH_LOCAL_CONFIG_DIR=str(d)))

    def test_refuses_non_loopback_or_path(self):
        for url in ("http://0.0.0.0:9931", "http://192.168.1.5:9931", "https://127.0.0.1:9931",
                    "http://127.0.0.1:9931/v1"):
            with self.assertRaises(SystemExit):
                local.local_env(dict(self.base, BENCH_LOCAL_URL=url))

    def test_needs_model(self):
        with self.assertRaises(SystemExit):
            local.local_env(dict(self.base, BENCH_MODEL=""))

    def test_command_local_has_no_effort_and_lifted_budget(self):
        env = local.local_env(self.base)
        cmd = build_command(ARMS["D1A"], claude_bin="claude", repo=Path("/r"), env=env)
        self.assertNotIn("--effort", cmd)
        self.assertEqual(cmd[cmd.index("--max-budget-usd") + 1], str(local.LOCAL_BUDGET_USD))
        self.assertIn(f"Read(/{self.cfg.resolve()})", cmd)

    def test_command_claude_unchanged(self):
        cmd = build_command(ARMS["D1A"], claude_bin="claude", repo=Path("/r"), env={"CLAUDE_CONFIG_DIR": "/c"})
        self.assertEqual(cmd[cmd.index("--effort") + 1], "high")
        self.assertEqual(cmd[cmd.index("--max-budget-usd") + 1], "5.0")

    def test_judges_and_drafter_stay_claude(self):
        self.assertEqual(JUDGE.model, CLAUDE_MODEL)
        self.assertEqual(DRAFTER.model, CLAUDE_MODEL)

    def test_judge_refuses_local_mode(self):
        from cosmobench import judge
        with mock.patch.dict("os.environ", {"BENCH_LOCAL_URL": "http://127.0.0.1:9931"}):
            with self.assertRaises(SystemExit):
                judge.main([])


class PilotGateTest(unittest.TestCase):
    def write(self, d: Path, answers: list[dict], grades: list[dict]) -> None:
        (d / "claimcheck").mkdir(parents=True)
        (d / "claimcheck" / "answers.jsonl").write_text("".join(json.dumps(a) + "\n" for a in answers))
        (d / "claimcheck" / "grades.jsonl").write_text("".join(json.dumps(g) + "\n" for g in grades))

    def rec(self, status="ok"):
        return {"item_id": "i", "arm": "D0", "status": status, "duration_ms": 1000, "num_turns": 1,
                "skills_invoked": [], "fetch_ok": 0, "error": ""}

    def test_pass_and_fail(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "ok"
            self.write(d, [self.rec()], [{"verdict": "correct"}] * 4)
            self.assertEqual(local.pilot_gate(d), 0)
            d = Path(t) / "timeout"
            self.write(d, [self.rec("error")], [{"verdict": "missing"}] * 4)
            self.assertEqual(local.pilot_gate(d), 1)
            d = Path(t) / "missing"
            self.write(d, [self.rec()], [{"verdict": "correct"}] * 49 + [{"verdict": "missing"}])
            self.assertEqual(local.pilot_gate(d), 1)   # 2 % is not < 2 %


class MockServerTest(unittest.TestCase):
    def test_scripted_replies(self):
        from cosmobench import local_mock
        with tempfile.TemporaryDirectory() as t:
            local_mock.Handler.log_path = Path(t) / "req.jsonl"
            srv = ThreadingHTTPServer(("127.0.0.1", 0), local_mock.Handler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            try:
                url = f"http://127.0.0.1:{srv.server_address[1]}/v1/messages"

                def post(body):
                    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                                 headers={"content-type": "application/json"})
                    with urllib.request.urlopen(req, timeout=5) as r:
                        return json.loads(r.read())

                user = {"role": "user", "content": "1. A (arXiv:1807.06209)\n2. B (arXiv:2404.03002)"}
                skill = {"name": "Skill", "input_schema": {"properties": {"skill": {}}}}
                r = post({"model": "m", "messages": [user], "tools": [skill]})
                self.assertEqual(r["content"][0]["name"], "Skill")
                bash = {"name": "Bash", "input_schema": {}}
                r = post({"model": "m", "messages": [user], "tools": [bash],
                          "system": "run `python3 /p/fetch_paper.py <id>`"})
                self.assertEqual(r["content"][0]["input"]["command"], "python3 /p/fetch_paper.py 1807.06209 --abstract")
                done = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]}
                r = post({"model": "m", "messages": [user, {"role": "assistant", "content": "x"}, done],
                          "tools": [skill]})
                self.assertEqual(r["content"][0]["text"], "1: cannot verify — mock\n2: cannot verify — mock")
                import time
                for _ in range(50):   # the handler logs after it has replied
                    logged = local_mock.Handler.log_path.read_text().splitlines()
                    if len(logged) == 3:
                        break
                    time.sleep(0.05)
                self.assertEqual(len(logged), 3)
            finally:
                srv.shutdown()


if __name__ == "__main__":
    unittest.main()
