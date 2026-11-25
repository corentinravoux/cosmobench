"""Grade TPBench answers by running the answer code against the reference implementation.

    python -m cosmobench.grade_tpbench              grade results/tpbench/answers.jsonl
    python -m cosmobench.grade_tpbench --self-check grade each reference against itself
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .common import BENCH, RESULTS, SEED, read_jsonl, write_jsonl
from .data import load_items

DOMAINS_FILE = BENCH / "tpbench_domains.json"
CODE_RE = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.S)

DRIVER = r'''
import inspect, json, math, random, sys, typing
import numpy as np
seed, trials, domains = int(sys.argv[1]), int(sys.argv[2]), json.loads(sys.argv[3])
ref_ns, cand_ns = {"__name__": "reference"}, {"__name__": "candidate"}
exec(compile(open("reference.py").read(), "reference.py", "exec"), ref_ns)
def emit(**kw):
    print("@@RESULT@@" + json.dumps(kw)); sys.stdout.flush()
fn_name = None
import ast
for node in ast.parse(open("reference.py").read()).body:
    if isinstance(node, ast.FunctionDef):
        fn_name = node.name; break
ref = ref_ns[fn_name]
try:
    exec(compile(open("candidate.py").read(), "candidate.py", "exec"), cand_ns)
except BaseException as e:
    emit(verdict="fail", reason="error: candidate import: " + type(e).__name__, matched=0, valid=0); sys.exit(0)
cand = cand_ns.get(fn_name)
if not callable(cand):
    emit(verdict="fail", reason="missing function " + fn_name, matched=0, valid=0); sys.exit(0)
rng = random.Random(seed)
params = list(inspect.signature(ref).parameters.values())
kinds = []
for p in params:
    a = p.annotation
    s = a if isinstance(a, str) else getattr(a, "__name__", str(a))
    if a is inspect.Parameter.empty or s == "float":
        kinds.append("float")
    elif s == "int":
        kinds.append("int")
    elif s in ("ndarray", "np.ndarray", "numpy.ndarray"):
        kinds.append("array")
    else:
        emit(verdict="ungradable", reason="annotation " + s, matched=0, valid=0); sys.exit(0)
def draw(name, kind):
    lo, hi = domains.get(name, [0.1, 2.0])
    if kind == "int":
        return rng.randint(int(lo) if name in domains else 1, int(hi) if name in domains else 5)
    if kind == "array":
        return np.array(sorted(rng.uniform(lo, hi) for _ in range(5)))
    return rng.uniform(lo, hi)
def finite(x):
    try:
        return bool(np.all(np.isfinite(np.asarray(x, dtype=float))))
    except Exception:
        return False
n = trials if params else 1
matched = valid = 0
for _ in range(n):
    args = [draw(p.name, k) for p, k in zip(params, kinds)]
    try:
        r = ref(*args)
    except Exception:
        continue
    if not finite(r):
        continue
    valid += 1
    try:
        c = cand(*args)
    except BaseException as e:
        emit(verdict="fail", reason="error: candidate raised " + type(e).__name__, matched=matched, valid=valid); sys.exit(0)
    try:
        ra, ca = np.asarray(r, dtype=float), np.asarray(c, dtype=float)
        ok = ra.shape == ca.shape and bool(np.allclose(ca, ra, rtol=1e-2, atol=1e-10))
    except Exception:
        ok = False
    matched += ok
need = 5 if params else 1
if valid < need:
    emit(verdict="ungradable", reason=f"only {valid} valid trials", matched=matched, valid=valid)
elif matched / valid >= 0.9:
    emit(verdict="pass", reason="", matched=matched, valid=valid)
else:
    emit(verdict="fail", reason="mismatch", matched=matched, valid=valid)
'''


def extract_code(answer: str) -> str | None:
    blocks = CODE_RE.findall(answer or "")
    return blocks[-1] if blocks else None


def strip_reference(ref: str) -> str:
    ref = ref.split(r"\end{python}")[0]
    return ref.replace(r"\begin{python}", "").strip() + "\n"


def grade(candidate_code: str | None, reference_code: str, *, seed: int = SEED, trials: int = 20,
          domains: dict | None = None, timeout: float = 60, python: str = sys.executable) -> dict:
    if not candidate_code:
        return {"verdict": "fail", "reason": "no code", "matched": 0, "valid": 0}
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "reference.py").write_text(reference_code)
        (t / "candidate.py").write_text(candidate_code)
        (t / "driver.py").write_text(DRIVER)
        try:
            proc = subprocess.run([python, "-I", "driver.py", str(seed), str(trials),
                                   json.dumps(domains or {})], cwd=t, capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"verdict": "fail", "reason": "timeout", "matched": 0, "valid": 0}
    for line in proc.stdout.splitlines()[::-1]:
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    return {"verdict": "fail", "reason": "error: no result (exit %s)" % proc.returncode,
            "matched": 0, "valid": 0}


def load_domains() -> dict:
    return json.loads(DOMAINS_FILE.read_text()) if DOMAINS_FILE.exists() else {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args(argv)
    items = {it["id"]: it for it in load_items("tpbench")}
    domains = load_domains()
    if a.self_check:
        bad = 0
        for iid, it in items.items():
            ref = strip_reference(it["reference"]["reference_implementation"])
            g = grade(ref, ref, domains=domains.get(iid))
            bad += g["verdict"] != "pass"
            print(iid, g)
        print(f"{len(items) - bad}/{len(items)} pass")
        return 1 if bad else 0
    grades = []
    for rec in read_jsonl(RESULTS / "tpbench" / "answers.jsonl"):
        if rec.get("status") != "ok" or rec["item_id"] not in items:
            continue
        it = items[rec["item_id"]]
        g = grade(extract_code(rec["answer"]),
                  strip_reference(it["reference"]["reference_implementation"]),
                  domains=domains.get(rec["item_id"]))
        grades.append({"item_id": rec["item_id"], "arm": rec["arm"], "rep": rec["rep"],
                       "domain": it["domain"], **g})
        print(rec["item_id"], rec["arm"], rec["rep"], g["verdict"], g["reason"])
    write_jsonl(RESULTS / "tpbench" / "grades.jsonl", grades)
    return 0


if __name__ == "__main__":
    sys.exit(main())
