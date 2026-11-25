"""Build the claim-check set (v3 primary set): two numbered claims per paper, each either the
paper's own number or a mechanically perturbed one, graded without an LLM judge.

    python -m cosmobench.build_claims candidates   arXiv API -> data/claim_candidates.jsonl
    python -m cosmobench.build_claims draft        one headless call per paper (skills off, no tools)
    python -m cosmobench.build_claims sheet        results/v3/claim_review.csv for the maintainer
    python -m cosmobench.build_claims accept [--all]   kept rows -> data/claimcheck.jsonl

Strata (by the paper the claims cite):
  K  in a sources.tsv of the pinned commit, submitted 2020-01-01 .. 2026-06-30: the model and
     the skills may both know it. Tests that the skills do not cost verification accuracy.
  A  in a sources.tsv, submitted 2026-07-01 .. the skills' content snapshot: after the model
     cutoff, so only the skills (or the paper) can settle it. The skills' claimed value.
  B  in no sources.tsv, submitted after the snapshot and >= 7 days before the build: nobody
     holds it; without L4 the right verdict is `cannot verify`. The honesty control.

The drafting model sees the abstract and the numbers found in it by regex, picks two, and
writes each claim with the placeholder {V} for the central value. The script, not the model,
fills in the true or the perturbed value, so the ground truth is mechanical. Half of the
claims in each stratum are perturbed (seeded balanced assignment).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from .arms import DRAFTER, build_command, build_env, find_claude_bin
from .common import DATA, DEFAULT_REPO, RESULTS, WORKDIR, read_jsonl, write_jsonl
from .run import check_pin, make_executor, parse_stream

CUTOFF = dt.date(2026, 6, 30)
K_START = dt.date(2020, 1, 1)
SEED_CC = 20261120
DRAW = 64          # candidates drawn per stratum (48 on the first pass; same seeded prefix); 40 kept
KEEP = 40
DOMAIN_CAP = 4
K_API_SAMPLE = 400  # K candidate IDs sent to the API (the sources hold thousands)
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

NUM = r"-?\d+(?:\.\d+)?"
POS = r"\d+(?:\.\d+)?"
SPAN_RES = [
    ("pm", re.compile(rf"(?<![\w.])({NUM})\s*(?:\\pm|±|\+/-|\$\\pm\$)\s*({POS})")),
    ("asym", re.compile(rf"(?<![\w.])({NUM})\s*(?:\^\{{\s*\+\s*({POS})\s*\}}\s*_\{{\s*-\s*({POS})\s*\}}"
                        rf"|_\{{\s*-\s*({POS})\s*\}}\s*\^\{{\s*\+\s*({POS})\s*\}})")),
    ("sigma", re.compile(rf"(?<![\w.])({POS})\s*\$?\s*(?:\\sigma|σ|sigma)\b", re.I)),
    ("limit", re.compile(rf"(?:<|>|\\lesssim|\\gtrsim|\\leq|\\geq|\\le\b|\\ge\b|≤|≥)\s*\$?\s*({POS})")),
    ("percent", re.compile(rf"(?<![\w.])({POS})\s*(?:\\%|%)")),
]

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "minItems": 2, "maxItems": 2, "items": {
        "type": "object",
        "properties": {"span": {"type": "integer"}, "template": {"type": "string"}},
        "required": ["span", "template"], "additionalProperties": False}}},
    "required": ["claims"], "additionalProperties": False,
}
DRAFT_PROMPT = """Below is the abstract of a cosmology paper and the numbers found in it. Write two claims, as they would appear in another paper's text when citing this result.

Rules:
- Pick two different numbers from the list that are main results of the paper (a measured value, a significance, a limit). Give each claim's number index as "span".
- Each claim is one sentence, in plain words, that names the survey, data set or method so that it is unambiguous. Do not name the authors, the arXiv ID or the title.
- Write the central value of the chosen number as the placeholder {{V}}, exactly once, and copy the rest of the number (uncertainty, units, the < or > sign, "σ", "%") as in the abstract. Example: "DESI DR2 BAO alone gives Omega_m = {{V}} ± 0.0086."
- Use plain text or simple Unicode (±, σ, Ω), not LaTeX.

Title: {title}
Abstract: {abstract}

Numbers:
{spans}"""


# ---------- numbers in abstracts and perturbation ----------

def numeric_spans(text: str) -> list[dict]:
    """Numbers stated as results: value ± error, asymmetric errors, σ levels, limits, percents."""
    out, taken = [], []
    for kind, rx in SPAN_RES:
        for m in rx.finditer(text or ""):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            g = [x for x in m.groups() if x is not None]
            err = None
            if kind == "pm":
                err = g[1]
            elif kind == "asym":
                err = str(max(float(g[1]), float(g[2])))
            span = {"kind": kind, "value": g[0], "err": err, "start": m.start(), "end": m.end(),
                    "context": text[max(0, m.start() - 60):m.end() + 20].replace("\n", " ")}
            taken.append((m.start(), m.end()))
            if not bad_span(span, text):
                out.append(span)
    out.sort(key=lambda s: s["start"])
    return out


CL_AFTER = re.compile(r"^\s*(?:\\%|%)?\s*(?:confiden|C\.?\s?L\b|credib|c\.l\.)", re.I)
RANGE_BEFORE = re.compile(r"(?:\bz|redshift|\$z|z_\{[^}]{0,12}\}|z_\w+)\s*\$?\s*$"          # z < 0.7
                          r"|\d\s*\$?\s*(?:<|≤|\\lesssim|\\leq|\\le)\s*\$?\s*[\w\\{}_^ ]{1,14}\s*\$?\s*$", re.I)  # 15 < r < 24


def bad_span(span: dict, text: str | None = None) -> str:
    """Why a number is not a result to perturb (a confidence level, a redshift range bound)."""
    ctx = span["context"]
    if text is not None:
        after, before = text[span["end"]:span["end"] + 25], text[max(0, span["start"] - 22):span["start"]]
    else:  # stored drafts: the context is 60 chars before and 20 after the span
        k = ctx.find(span["value"])
        before, after = ctx[max(0, k - 22):k], ctx[k + len(span["value"]):]
    if span["kind"] == "percent" and (span["value"] in ("68", "90", "95", "99", "99.7") or CL_AFTER.search(after)):
        return "confidence level"
    if span["kind"] == "limit" and RANGE_BEFORE.search(re.sub(r"(?:<|>|\\lesssim|\\gtrsim|\\leq|\\geq|\\le|\\ge|≤|≥)\s*\$?\s*$", "", before)):
        return "range bound"
    return ""


KIND_AFTER = {"pm": re.compile(r"^\s*(?:±|\+/-|\+-)"), "asym": re.compile(r"^\s*(?:±|\+/-|[\^_(]?\s*\{?\s*[+\-−]\s*[\d.]+\s*\}?\s*[/_,]?\s*\{?\s*[+\-−]\s*\d)"),
              # {V}^{+a}_{-b}, {V} +a/-b, {V} -b/+a, or {V} ± a when the paper's errors are equal
              "sigma": re.compile(r"^\s*(?:σ|sigma)", re.I), "percent": re.compile(r"^\s*(?:%|per ?cent)", re.I)}
LIMIT_BEFORE = re.compile(r"(?:<|>|≤|≥|≲|≳|below|above|less than|greater than|under|exceed\w*|at most|at least|"
                          r"up to|limit of|bound of|smaller than|larger than)\s*~?\s*$", re.I)


def template_fits(template: str, kind: str) -> bool:
    """The words around {V} match the kind of number: ± after a measurement, σ after a level, ..."""
    before, after = template.split("{V}", 1)
    if kind == "limit":
        return bool(LIMIT_BEFORE.search(before))
    return bool(KIND_AFTER[kind].search(after))


def is_numeric_abstract(text: str) -> bool:
    return len(numeric_spans(text)) >= 2


def _decimals(s: str) -> int:
    return len(s.split(".", 1)[1]) if "." in s else 0


def _fmt(x: float, like: str) -> str:
    d = _decimals(like)
    return f"{x:.{d}f}"


def perturb(kind: str, value: str, err: str | None, rng: random.Random) -> str:
    """A wrong value a careful reader of the paper would reject: >= 4 sigma or >= 10 % away
    for measurements, about a factor 2 for limits and percents, +-1.5-2.5 for sigma levels.
    Same number of decimals; never equal to the true value once formatted."""
    v = float(value)
    sign = rng.choice((-1, 1))
    if kind in ("pm", "asym"):
        e = float(err) if err else 0.0
        shift = max(rng.uniform(4, 8) * e, rng.uniform(0.10, 0.25) * abs(v), 10 ** -_decimals(value))
        new = v + sign * shift
        if v > 0 and new <= 0:
            new = v + shift
    elif kind == "sigma":
        new = v + (sign if v >= 2.5 else 1) * rng.uniform(1.5, 2.5)
    elif kind == "limit":
        new = v * rng.uniform(2, 3) if sign > 0 else v / rng.uniform(2, 3)
    else:  # percent
        new = v * rng.uniform(1.6, 2.5) if (sign > 0 and v < 40) else v / rng.uniform(1.6, 2.5)
    out = _fmt(new, value)
    if out == _fmt(v, value) or float(out) == 0.0:
        step = 10 ** -_decimals(value)
        out = _fmt(v + max(abs(new - v), 2 * step) * (1 if new >= v else -1), value)
    return out


def balanced_flags(n: int, rng: random.Random) -> list[bool]:
    """n booleans, half True (perturbed), in seeded random order."""
    flags = [True] * (n // 2) + [False] * (n - n // 2)
    rng.shuffle(flags)
    return flags


# ---------- strata and candidates ----------

def stratum_of(submitted: dt.date, in_sources: bool, snapshot: dt.date, end_b: dt.date) -> str | None:
    if in_sources and K_START <= submitted <= CUTOFF:
        return "K"
    if in_sources and CUTOFF < submitted <= snapshot:
        return "A"
    if not in_sources and snapshot < submitted <= end_b:
        return "B"
    return None


def sources_index(repo: Path) -> dict[str, str]:
    """arXiv ID -> first skill whose sources.tsv lists it."""
    idx: dict[str, str] = {}
    for tsv in sorted((Path(repo) / "plugin" / "skills").glob("*/references/sources.tsv")):
        skill = tsv.parent.parent.name
        for line in tsv.read_text(encoding="utf-8").splitlines()[1:]:
            cols = line.split("\t")
            if len(cols) > 1:
                m = re.match(r"(\d{4}\.\d{4,5})", cols[1].strip())
                if m:
                    idx.setdefault(m.group(1), skill)
    return idx


def snapshot_date(repo: Path) -> dt.date:
    dates = []
    for f in (Path(repo) / "plugin" / "skills").glob("*/SKILL.md"):
        dates += re.findall(r"Snapshot (\d{4}-\d{2}-\d{2})", f.read_text(encoding="utf-8"))
    return max(dt.date.fromisoformat(d) for d in dates)


def skill_keywords(repo: Path) -> dict[str, set[str]]:
    out = {}
    for f in (Path(repo) / "plugin" / "skills").glob("cosmo-*/SKILL.md"):
        m = re.search(r"^description: (.+)$", f.read_text(encoding="utf-8"), re.M)
        words = re.findall(r"[a-z][a-z0-9-]{4,}", (m.group(1) if m else "").lower())
        out[f.parent.name] = set(words) - {"reviewing", "planning", "fact-checking", "work", "results"}
    return out


def guess_domain(text: str, keywords: dict[str, set[str]]) -> str:
    words = set(re.findall(r"[a-z][a-z0-9-]{4,}", text.lower()))
    best = max(keywords, key=lambda k: (len(words & keywords[k]), k))
    return best if words & keywords[best] else "unassigned"


def in_skill_text(repo: Path, arxiv: str, value: str, window: int = 3) -> bool:
    """True if some skill file states `value` within `window` lines of a mention of `arxiv`."""
    pat = re.compile(rf"(?<![\d.]){re.escape(value)}(?![\d])")
    for f in (Path(repo) / "plugin" / "skills").rglob("*.md"):
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        for k, line in enumerate(lines):
            if arxiv in line and any(pat.search(x) for x in lines[max(0, k - window):k + window + 1]):
                return True
    return False


def _get(params: dict) -> ET.Element:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "cosmoskill-bench (research benchmark)"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                root = ET.fromstring(r.read())
            time.sleep(3)
            return root
        except Exception:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("arXiv API unreachable: " + url)


def _entries(root: ET.Element) -> list[dict]:
    out = []
    for e in root.findall("a:entry", NS):
        aid = e.findtext("a:id", "", NS).rsplit("/abs/", 1)[-1]
        m = re.match(r"(\d{4}\.\d{4,5})", aid)
        if not m:
            continue
        prim = e.find("arxiv:primary_category", NS)
        out.append({"arxiv": m.group(1),
                    "title": " ".join(e.findtext("a:title", "", NS).split()),
                    "abstract": " ".join(e.findtext("a:summary", "", NS).split()),
                    "published": e.findtext("a:published", "", NS)[:10],
                    "primary": prim.get("term") if prim is not None else ""})
    return out


def fetch_ids(ids: list[str]) -> list[dict]:
    out = []
    for k in range(0, len(ids), 50):
        out += _entries(_get({"id_list": ",".join(ids[k:k + 50]), "max_results": 50}))
    return out


def fetch_window(start: dt.date, end: dt.date) -> list[dict]:
    q = f"cat:astro-ph.CO AND submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
    out, k = [], 0
    while True:
        batch = _entries(_get({"search_query": q, "start": k, "max_results": 200,
                               "sortBy": "submittedDate", "sortOrder": "ascending"}))
        out += batch
        if len(batch) < 200:
            return out
        k += 200


def draw(pool: list[dict], n: int, rng: random.Random, cap: int = DOMAIN_CAP) -> list[dict]:
    pool = sorted(pool, key=lambda p: p["arxiv"])
    rng.shuffle(pool)
    count: dict[str, int] = {}
    out = []
    for p in pool:
        if count.get(p["domain"], 0) >= cap:
            continue
        out.append(p)
        count[p["domain"]] = count.get(p["domain"], 0) + 1
        if len(out) == n:
            break
    return out


def candidates(repo: Path, today: dt.date | None = None) -> list[dict]:
    today = today or dt.date.today()
    idx = sources_index(repo)
    snap = snapshot_date(repo)
    end_b = today - dt.timedelta(days=7)
    kw = skill_keywords(repo)
    rng = random.Random(SEED_CC)
    k_ids = sorted(i for i in idx if f"{K_START:%y%m}" <= i[:4] <= f"{CUTOFF:%y%m}")
    k_ids = rng.sample(k_ids, min(K_API_SAMPLE, len(k_ids)))
    a_ids = sorted(i for i in idx if i[:4] > f"{CUTOFF:%y%m}")
    pools: dict[str, list[dict]] = {"K": [], "A": [], "B": []}
    for p in fetch_ids(sorted(k_ids)) + fetch_ids(a_ids):
        s = stratum_of(dt.date.fromisoformat(p["published"]), True, snap, end_b)
        if s in ("K", "A") and p["primary"] == "astro-ph.CO" and is_numeric_abstract(p["abstract"]):
            pools[s].append({**p, "stratum": s, "domain": idx[p["arxiv"]]})
    for p in fetch_window(snap + dt.timedelta(days=1), end_b):
        if p["arxiv"] in idx or p["primary"] != "astro-ph.CO" or not is_numeric_abstract(p["abstract"]):
            continue
        if stratum_of(dt.date.fromisoformat(p["published"]), False, snap, end_b) == "B":
            pools["B"].append({**p, "stratum": "B", "domain": guess_domain(p["title"] + " " + p["abstract"], kw)})
    chosen = []
    for s in "KAB":
        chosen += draw(pools[s], DRAW, rng)
    meta = {"snapshot": snap.isoformat(), "window_K": [str(K_START), str(CUTOFF)],
            "window_A": [str(CUTOFF + dt.timedelta(days=1)), str(snap)],
            "window_B": [str(snap + dt.timedelta(days=1)), str(end_b)],
            "pool": {s: len(v) for s, v in pools.items()}, "drawn": {s: sum(c["stratum"] == s for c in chosen) for s in "KAB"},
            "k_ids_checked": len(k_ids), "a_ids_checked": len(a_ids), "seed": SEED_CC}
    (DATA / "claim_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(meta)
    return chosen


# ---------- drafting, validation, assembly ----------

def spans_block(spans: list[dict]) -> str:
    return "\n".join(f"[{k}] {s['context']}" for k, s in enumerate(spans))


def validate_draft(draft: dict, spans: list[dict], arxiv: str) -> str:
    """Empty string if the draft is usable, else the reason."""
    cl = draft.get("claims") or []
    if len(cl) != 2:
        return "need 2 claims"
    if cl[0]["span"] == cl[1]["span"]:
        return "same span twice"
    for c in cl:
        if not (0 <= c["span"] < len(spans)):
            return "span index out of range"
        t = c["template"]
        if t.count("{V}") != 1:
            return "placeholder {V} not exactly once"
        if re.search(r"\b\d{4}\.\d{4,5}\b", t) or arxiv in t:
            return "arXiv ID in claim"
        if "\\" in t or "$" in t:
            return "LaTeX in claim"
        why = bad_span(spans[c["span"]])
        if why:
            return why
        if not template_fits(t, spans[c["span"]]["kind"]):
            return "template does not fit a " + spans[c["span"]]["kind"]
    return ""


def assemble(c: dict, draft: dict, flags: list[bool], repo: Path, rng: random.Random) -> dict:
    """Item with the shown claims (true or perturbed) and the mechanical ground truth."""
    spans = c["spans"]
    shown, ref = [], []
    for cl, pert in zip(draft["claims"], flags):
        s = spans[cl["span"]]
        value = s["value"].lstrip("-") if s["kind"] in ("sigma", "limit", "percent") else s["value"]
        val_shown = perturb(s["kind"], value, s["err"], rng) if pert else value
        shown.append({"text": cl["template"].replace("{V}", val_shown), "arxiv": c["arxiv"]})
        ref.append({"true_text": cl["template"].replace("{V}", value), "perturbed": pert,
                    "kind": s["kind"], "value_true": value, "value_shown": val_shown,
                    "abstract_context": s["context"],
                    "in_skill_text": in_skill_text(repo, c["arxiv"], value)})
    return {"id": f"cc-{c['stratum']}-{c['arxiv']}", "bench": "claimcheck",
            "prompt_fields": {"claims": shown}, "reference": {"claims": ref},
            "domain": c["domain"], "source_arxiv": c["arxiv"], "stratum": c["stratum"],
            "published": c["published"], "title": c["title"]}


def draft(repo: Path) -> int:
    cands = read_jsonl(DATA / "claim_candidates.jsonl")
    out = DATA / "claim_drafts.jsonl"
    past = read_jsonl(out)
    done = {d["arxiv"] for d in past if not validate_draft(d["draft"], d["spans"], d["arxiv"])}
    tries: dict[str, int] = {}
    for d in past:
        tries[d["arxiv"]] = tries.get(d["arxiv"], 0) + 1
    done |= {a for a, n in tries.items() if n >= 2}  # at most two drafting attempts per paper
    claude = find_claude_bin()
    env = build_env(dict(os.environ))
    executor = make_executor(repo)
    errors = 0
    for c in cands:
        if c["arxiv"] in done:
            continue
        spans = numeric_spans(c["abstract"])
        if len(spans) < 2:
            print("skip", c["arxiv"], "fewer than 2 usable numbers", flush=True)
            continue
        cmd = build_command(DRAFTER, claude_bin=claude, repo=repo, json_schema=DRAFT_SCHEMA, env=env)
        prompt = DRAFT_PROMPT.format(title=c["title"], abstract=c["abstract"], spans=spans_block(spans))
        rec = parse_stream(executor(cmd, env, WORKDIR, 900, prompt))
        d = rec.get("structured_output")
        if rec["status"] != "ok" or not isinstance(d, dict):
            errors += 1
            print("draft error", c["arxiv"], rec["error"][:200])
            if errors >= 3:
                return 2
            continue
        errors = 0
        problem = validate_draft(d, spans, c["arxiv"]) if isinstance(d.get("claims"), list) else "no claims"
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**c, "spans": spans, "draft": d, "problem": problem,
                                 "draft_cost_usd": rec.get("cost_usd")}, ensure_ascii=False) + "\n")
        print(c["stratum"], c["arxiv"], problem or "ok", [x["template"][:70] for x in d.get("claims", [])], flush=True)
    return 0


def build_items(repo: Path, drafts: list[dict]) -> list[dict]:
    """Valid drafts -> items, KEEP per stratum at most, with balanced perturbation per stratum."""
    rng = random.Random(SEED_CC)
    items = []
    for s in "KAB":
        valid = {d["arxiv"]: d for d in drafts
                 if d["stratum"] == s and not validate_draft(d["draft"], d["spans"], d["arxiv"])}  # last valid wins
        ok = sorted(valid.values(), key=lambda d: d["arxiv"])
        rng.shuffle(ok)
        ok = sorted(ok[:KEEP], key=lambda d: d["arxiv"])
        flags = balanced_flags(2 * len(ok), rng)
        for k, d in enumerate(ok):
            items.append(assemble(d, d["draft"], flags[2 * k:2 * k + 2], repo, rng))
    return items


SHEET_COLS = ["keep", "id", "stratum", "arxiv", "title", "n", "shown", "perturbed", "true_text",
              "abstract_context", "in_skill_text", "note"]


def sheet(repo: Path) -> Path:
    items = build_items(repo, read_jsonl(DATA / "claim_drafts.jsonl"))
    write_jsonl(DATA / "claimcheck_built.jsonl", items)
    path = RESULTS / "claim_review.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SHEET_COLS)
        w.writeheader()
        for it in items:
            for n, (sh, rf) in enumerate(zip(it["prompt_fields"]["claims"], it["reference"]["claims"]), 1):
                w.writerow({"keep": "", "id": it["id"], "stratum": it["stratum"], "arxiv": it["source_arxiv"],
                            "title": it["title"], "n": n, "shown": sh["text"], "perturbed": rf["perturbed"],
                            "true_text": rf["true_text"], "abstract_context": rf["abstract_context"],
                            "in_skill_text": rf["in_skill_text"], "note": ""})
    print(f"{len(items)} items: " + ", ".join(f"{s}={sum(i['stratum'] == s for i in items)}" for s in "KAB"))
    return path


def accept(keep_all: bool = False) -> Path:
    """An item is kept unless one of its rows says keep = n/no/0/drop (or, without --all,
    unless no row says keep = y)."""
    items = {it["id"]: it for it in read_jsonl(DATA / "claimcheck_built.jsonl")}
    rows = list(csv.DictReader((RESULTS / "claim_review.csv").open(encoding="utf-8")))
    drop = {r["id"] for r in rows if r["keep"].strip().lower() in ("n", "no", "0", "drop")}
    yes = {r["id"] for r in rows if r["keep"].strip().lower() in ("y", "yes", "1", "x", "keep")}
    kept = [it for i, it in items.items() if i not in drop and (keep_all or i in yes)]
    write_jsonl(DATA / "claimcheck.jsonl", kept)
    print(f"{len(kept)} kept: " + ", ".join(f"{s}={sum(i['stratum'] == s for i in kept)}" for s in "KAB"))
    return DATA / "claimcheck.jsonl"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["candidates", "draft", "sheet", "accept"])
    ap.add_argument("--all", action="store_true", help="accept: keep every item not marked drop")
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    a = ap.parse_args(argv)
    if a.step == "candidates":
        check_pin(a.repo)
        write_jsonl(DATA / "claim_candidates.jsonl", candidates(a.repo))
    elif a.step == "draft":
        check_pin(a.repo)
        return draft(a.repo)
    elif a.step == "sheet":
        print(sheet(a.repo))
    else:
        print(accept(a.all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
