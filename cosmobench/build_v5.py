"""Build the v5 claim check (plan 2026-11-21-benchmark-plan-v5.md).

    python3 -m cosmobench.build_v5 candidates   arXiv + Semantic Scholar -> data/v5/{pool,candidates}.jsonl
    python3 -m cosmobench.build_v5 draft        one Opus call per paper (skills off, no tools, effort low)
    python3 -m cosmobench.build_v5 verify       one Sonnet call per paper: blind check of both versions
    python3 -m cosmobench.build_v5 assemble     blocks of 4 papers -> 2 items each -> data/v5/claimcheck_built.jsonl
    python3 -m cosmobench.build_v5 audit        results/v5/claim_audit.csv (48 claims) for the maintainer
    python3 -m cosmobench.build_v5 accept       audit -> data/v5/claimcheck.jsonl (bad claims excluded from scoring)
    python3 -m cosmobench.build_v5 cpqa-subset  data/v5/cpqa_subset.json (8 CosmoPaperQA items per paper)
    python3 -m cosmobench.build_v5 report       counts and losses at every step

Four claim types per paper (number, dataset, conclusion, attribution). The drafter writes a
template with a placeholder; this script renders the true and the perturbed version, so the
ground truth is mechanical. A verifier from another model must pick the true version blind.
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
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from .arms import BLIND, DRAFTER, VERIFIER, build_command, build_env, find_claude_bin
from .build_claims import (
    API,
    NS,
    _decimals,
    _fmt,
    bad_span,
    guess_domain,
    numeric_spans,
    skill_keywords,
    snapshot_date,
    sources_index,
    template_fits,
)
from .common import (
    CLAIM_SET,
    DATA,
    DEFAULT_REPO,
    RESULTS,
    WORKDIR,
    append_jsonl,
    read_jsonl,
    write_jsonl,
)
from .run import check_pin, make_executor, parse_stream

V5 = DATA / CLAIM_SET
FRESH = CLAIM_SET == "v5k0fresh"   # 1.1.1 re-test: K0 only, 2015-2019 (plan 2026-11-22, deviation)
ID_PREFIX = "cc5f" if FRESH else "cc5"
FRESH_START, FRESH_END = dt.date(2015, 1, 1), dt.date(2019, 12, 31)
FRESH_SEED, FRESH_DRAW = 20261122, 200
SEED_V5 = 20261121
K_START, K_END = dt.date(2020, 1, 1), dt.date(2025, 12, 31)
A_START = dt.date(2026, 7, 1)
B_LAG_DAYS = 7
ITEMS = {"K": 50, "K0": 50, "A": 30, "B": 30}       # items per stratum; papers = 2x, claims = 4x
STRATA = ("K", "K0", "A", "B")
DRAW_FACTOR = 1.5
DRAW_ALL = ("K", "K0", "A", "B")  # every eligible paper is drafted; assembly samples among the verified ones
DOMAIN_CAP = 10       # per skill domain and stratum (B, K0: guessed from description keywords, which cluster)
MIN_CITES = 50
DECOY_DAYS = 90
TYPES = ("number", "dataset", "conclusion", "attribution")
S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=citationCount"
UA = {"User-Agent": "cosmobench (research benchmark)"}

# ---------------------------------------------------------------- data-set table
# canonical name, regex on the abstract, release month (conservative: never earlier than the
# public release, so the "released before the paper" rule can only be stricter than needed).
# Written for this benchmark from general knowledge; the maintainer checks it at the audit.
DATASETS: dict[str, list[tuple[str, str, str]]] = {
    "desi": [("DESI EDR", r"\bDESI\s+(?:EDR|Early\s+Data\s+Release)\b", "2023-06"),
             ("DESI DR1", r"\bDESI\s+(?:DR1|Data\s+Release\s+1|Y1|Year[- ]1)\b", "2024-04"),
             ("DESI DR2", r"\bDESI\s+(?:DR2|Data\s+Release\s+2|Y3|Year[- ]3)\b", "2025-03")],
    "planck": [("Planck 2013", r"\bPlanck\s+2013\b", "2013-03"),
               ("Planck 2015", r"\bPlanck\s+2015\b", "2015-02"),
               ("Planck 2018", r"\bPlanck\s+(?:2018|PR3)\b", "2018-07"),
               ("Planck PR4", r"\bPlanck\s+PR4\b|\bNPIPE\b", "2020-07")],
    "act": [("ACT DR4", r"\bACT\s+DR4\b", "2020-07"), ("ACT DR6", r"\bACT\s+DR6\b", "2023-04")],
    "sdss": [("BOSS DR12", r"\bBOSS\s+DR12\b", "2016-07"), ("eBOSS DR16", r"\beBOSS\s+DR16\b", "2020-07")],
    "des": [("DES Y1", r"\bDES\s+(?:Y1|Year[- ]1)\b", "2017-08"),
            ("DES Y3", r"\bDES\s+(?:Y3|Year[- ]3)\b", "2021-05"),
            ("DES Y6", r"\bDES\s+(?:Y6|Year[- ]6)\b", "2026-02")],
    "kids": [("KiDS-450", r"\bKiDS-?450\b", "2016-07"), ("KiDS-1000", r"\bKiDS-?1000\b", "2020-07"),
             ("KiDS-Legacy", r"\bKiDS-?Legacy\b", "2025-03")],
    "hsc": [("HSC Y1", r"\bHSC\s+(?:Y1|Year[- ]1|first[- ]year)\b", "2018-10"),
            ("HSC Y3", r"\bHSC\s+(?:Y3|Year[- ]3|three[- ]year)\b", "2023-04")],
    "sn": [("Pantheon", r"\bPantheon\b(?![+]|\s*Plus|Plus)", "2017-11"),
           ("Pantheon+", r"\bPantheon\s?(?:\+|Plus\b)", "2022-02"),
           ("Union3", r"\bUnion\s?3(?:\.0)?\b", "2023-11"),
           ("DES-SN5YR", r"\bDES[- ]?SN5YR\b|\bDES[- ]?Y5\b|\bDES\s+5[- ]?(?:yr|YR|year)", "2024-01")],
    "gwtc": [("GWTC-1", r"\bGWTC-?1\b(?!\.\d)", "2018-12"), ("GWTC-2", r"\bGWTC-?2\b(?!\.\d)", "2020-11"),
             ("GWTC-3", r"\bGWTC-?3\b(?!\.\d)", "2021-11"), ("GWTC-4.0", r"\bGWTC-?4(?:\.0)?\b", "2025-09")],
    "hydro": [("EAGLE", r"\bEAGLE\b", "2014-07"), ("IllustrisTNG", r"\bIllustris\s?TNG\b|\bTNG(?:50|100|300)\b", "2017-07"),
              ("SIMBA", r"\bSIMBA\b|\bSimba\b", "2019-01"), ("FLAMINGO", r"\bFLAMINGO\b", "2023-06")],
    "nbody": [("Quijote", r"\bQuijote\b", "2019-09"), ("Uchuu", r"\bUchuu\b", "2020-07"),
              ("AbacusSummit", r"\bAbacusSummit\b", "2021-10")],
    "euclid": [("Euclid Q1", r"\bEuclid\s+Q1\b|\bEuclid\s+Quick\s+(?:Data\s+)?Release\b", "2025-03"),
               ("Euclid DR1", r"\bEuclid\s+DR1\b", "2026-11")],
    "gaia": [("Gaia DR2", r"\bGaia\s+DR2\b", "2018-04"), ("Gaia EDR3", r"\bGaia\s+EDR3\b", "2020-12"),
             ("Gaia DR3", r"\bGaia\s+DR3\b", "2022-06")],
    "cf": [("Cosmicflows-3", r"\bCosmicflows-?3\b|\bCF3\b", "2016-05"),
           ("Cosmicflows-4", r"\bCosmicflows-?4\b|\bCF4\b", "2022-11")],
}
_DS_RX = {name: re.compile(rx) for g in DATASETS.values() for name, rx, _ in g}
_DS_GROUP = {name: g for g, members in DATASETS.items() for name, _, _ in members}
_DS_DATE = {name: d for g in DATASETS.values() for name, _, d in g}


def datasets_in(text: str) -> dict[str, list[str]]:
    """group -> canonical names of its members that the text names."""
    out: dict[str, list[str]] = defaultdict(list)
    for name, rx in _DS_RX.items():
        if rx.search(text or ""):
            out[_DS_GROUP[name]].append(name)
    return dict(out)


# Groups whose members are alternatives in concurrent use; every other group is a sequence of
# releases, where only the neighbouring release is a plausible misquote.
ALTERNATIVES = {"hydro", "nbody", "sn"}
SUPERSEDED = {"Pantheon": "2022-02"}   # alternatives that a later product replaced
STALE_MONTHS = 24


def _months(ym: str) -> int:
    return int(ym[:4]) * 12 + int(ym[5:7]) - 1


def superseded_at(name: str) -> str | None:
    """Month the member was replaced: the next release of a release sequence, or SUPERSEDED."""
    g = _DS_GROUP[name]
    if g in ALTERNATIVES:
        return SUPERSEDED.get(name)
    members = [m for m, _, _ in DATASETS[g]]
    k = members.index(name)
    return DATASETS[g][k + 1][2] if k + 1 < len(members) else None


# Groups whose releases share pipelines and products, so a swap between them is not clearly false
# (self-audit 2026-11-21: Euclid Q1 <-> DR1 spectroscopy pipeline and HEALPix masks).
NO_SWAP_GROUPS = {"euclid"}
# 1.1.1 re-test set (2015-2019): Planck 2015 -> 2013 is the neighbouring swap, and for qualitative claims
# the 2013 version is not clearly false (audit 2026-11-22, rows 6 and 8); excluded in that set only.
if CLAIM_SET == "v5k0fresh":
    NO_SWAP_GROUPS = NO_SWAP_GROUPS | {"planck"}


def excluded_class(c: dict) -> bool:
    """A built claim of a class removed after the audit (scored as excluded)."""
    return c.get("type") == "dataset" and _DS_GROUP.get(c.get("value_true")) in NO_SWAP_GROUPS


def dataset_targets(name: str, abstract: str, submitted: str) -> list[str]:
    """Plausible misquotes of `name` for a paper submitted in `submitted` (review 2026-11-21, finding 1).

    Same group, not named in the abstract, released before the paper's month, not replaced more than
    STALE_MONTHS before the paper (no "a 2025 result from Planck 2013"); in a release sequence, only
    the neighbouring release on either side."""
    g = _DS_GROUP[name]
    if g in NO_SWAP_GROUPS:
        return []
    members = [m for m, _, _ in DATASETS[g]]
    present = set(datasets_in(abstract).get(g, []))
    k = members.index(name)
    month = _months(submitted[:7])
    out = []
    for j, (m, _, d) in enumerate(DATASETS[g]):
        if m == name or m in present or _months(d) >= month:
            continue
        if g not in ALTERNATIVES and abs(j - k) != 1:
            continue
        rep = superseded_at(m)
        if rep is not None and _months(rep) < month - STALE_MONTHS:
            continue
        out.append(m)
    return out


def dataset_options(abstract: str, submitted: str) -> list[str]:
    """Names usable for a data-set claim: the only member of its group in the abstract, with a target."""
    return [ms[0] for g, ms in sorted(datasets_in(abstract).items())
            if len(ms) == 1 and dataset_targets(ms[0], abstract, submitted)]


# ---------------------------------------------------------------- conclusion phrases
# family -> (cue regex of side a, cue regex of side b); phrase -> (family, side, its pair)
CUES = {
    "consistency": (r"(?<!in)consisten|\bin agreement\b|\bagree(?:s|ment)?\b|\bcompatible\b",
                    r"\btension\b|\binconsisten|\bdisagree|\bdiscrepan"),
    "preference": (r"(?<!dis)\bfavou?r(?:s|ed|ing)?\b|\bprefer", r"\bdisfavou?r|\brul(?:e|es|ed|ing) out\b"),
    "order": (r"\bhigher\b|\blarger\b|\bgreater\b|\bexceed|\babove\b|\btimes as (?:large|high|great)\b",
              r"\blower\b|\bsmaller\b|\bbelow\b"),
    "trend": (r"\bincreas|\benhanc|\bboost", r"\bdecreas|\breduc|\bsuppress"),
    "quality": (r"\bimprov|\btighten", r"\bdegrad|\bweaken|\bloosen|\bworsen"),
    "detection": (r"(?<!no )(?<!non-)\bdetect(?:ion|ed|s)?\b", r"\bno (?:significant )?detection\b|\bnon-detection\b|\bnot detect"),
}
PAIRS = [("consistency", "consistent with", "in tension with"), ("preference", "favours", "disfavours"),
         ("preference", "favour", "disfavour"), ("order", "higher than", "lower than"),
         ("order", "larger than", "smaller than"), ("trend", "increases", "decreases"),
         ("trend", "increase", "decrease"), ("quality", "improves", "degrades"),
         ("quality", "improve", "degrade"), ("detection", "a detection of", "no detection of")]
PHRASES: dict[str, tuple[str, str, str]] = {}
for fam, a, b in PAIRS:
    PHRASES[a] = (fam, "a", b)
    PHRASES[b] = (fam, "b", a)
NEGATION = re.compile(r"\b(?:no|not|without|neither|nor|cannot|unable)\b|n't\b", re.I)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z$\\(])", text or "") if s.strip()]


def cue_side(sentence: str, family: str) -> str | None:
    """'a' or 'b' if the sentence holds cues of exactly one side and no stray negation."""
    ra, rb = (re.compile(x, re.I) for x in CUES[family])
    has_a, has_b = bool(ra.search(sentence)), bool(rb.search(sentence))
    if has_a == has_b:
        return None
    rest = rb.sub(" ", sentence) if has_b else sentence  # the b cue of "detection" carries its own "no"
    if NEGATION.search(rest):
        return None
    return "a" if has_a else "b"


def conclusion_families(abstract: str) -> list[str]:
    return sorted({f for s in sentences(abstract) for f in CUES if cue_side(s, f)})


# ---------------------------------------------------------------- numbers
def _bounds(kind: str, v: float, err: float | None) -> tuple[float, float] | None:
    """(min, max) of |shown - true| (absolute for pm/asym/sigma, ratio for limit/percent)."""
    if kind in ("pm", "asym"):
        if v == 0 or err is None:
            return None
        lo, hi = max(2 * err, 0.03 * abs(v)), 0.25 * abs(v)
        return (lo, hi) if lo <= hi else None
    if kind == "sigma":
        return (1.0, 2.0)
    if kind == "limit":
        return (1.5, 2.2)
    return (1.4, 2.0)  # percent


def _distance_ok(kind: str, v: float, w: float, b: tuple[float, float]) -> bool:
    if kind in ("pm", "asym", "sigma"):
        return b[0] - 1e-12 <= abs(w - v) <= b[1] + 1e-12 and (w > 0) == (v > 0) and w != 0
    if v <= 0 or w <= 0:
        return False
    r = max(w / v, v / w)
    return b[0] - 1e-12 <= r <= b[1] + 1e-12 and (kind != "percent" or w < 100)


CUT_BEFORE = re.compile(r"(?:\bm_\{?(?!\\nu|ν)[a-zA-Z]|\bmag|\\ell|ℓ|\bk\b|k_\{?\w|\bscales?\b|\\theta|θ|"
                        r"\br_\{?\w|\bR_\{?\w|S/N|SNR|signal-to-noise|Δ|\\Delta|\bR\b|\bthreshold|\bcut\b|\bN_\{?\w|"
                        r"\bM_\{?(?:\*|\\star|\\rm|h|vir|200|500)|\blog)"
                        r"[^.;,]{0,14}$", re.I)


def selection_bound(span: dict) -> bool:
    """Not a result to perturb: one end of a range, or a limit that reads as a cut chosen by the
    authors (magnitude, scale, S/N, mass, Δz)."""
    ctx = span["context"]
    k = ctx.rfind(span["value"])
    after = ctx[k + len(span["value"]):]
    unit = r"\s*\$?\s*(?:σ|\\sigma|%|\\%|\\,?\\%)?\s*\$?\s*"
    if re.match(unit + r"(?:-|–|—|to|and)" + unit + r"\d", after) or \
            re.search(r"\d" + unit + r"(?:-|–|—|to)" + unit + r"$", ctx[:k]) or \
            re.search(r"between" + unit + r"[\d.]+" + unit + r"and" + unit + r"$", ctx[:k]):
        return True  # one end of a range ("15-20 %", "2.4σ - 4.1σ", "between 2.3 and 3.1σ")
    if span["kind"] != "limit":
        return False
    before = re.sub(r"(?:<|>|\\lesssim|\\gtrsim|\\leq|\\geq|\\le|\\ge|≤|≥)\s*\$?\s*$", "", ctx[:k].rstrip(" $"))
    return bool(CUT_BEFORE.search(before[-40:])) or bool(re.match(r"^\s*\$?\s*(?:'|′|\\arcmin|arcmin|deg|mag)", ctx[k + len(span["value"]):]))


def number_usable(span: dict) -> bool:
    try:
        v = float(span["value"])
        err = float(span["err"]) if span.get("err") else None
    except (TypeError, ValueError):
        return False
    if span["kind"] in ("sigma", "limit", "percent") and v <= 0:
        return False
    if bad_span(span) or selection_bound(span) or _bounds(span["kind"], v, err) is None:
        return False
    try:
        perturb_number(span, [span], random.Random(0))
    except ValueError:
        return False
    return True


def usable_spans(abstract: str) -> list[dict]:
    return [s for s in numeric_spans(abstract) if number_usable(s)]


SWAP = False  # review 2026-11-21, finding 3: swaps put other experiments' values (Planck's H0) on a claim


def perturb_number(span: dict, spans: list[dict], rng: random.Random) -> tuple[str, str]:
    """(shown value, mode). Shift within the bounds of _bounds; swap mode is kept but off (SWAP)."""
    kind, value = span["kind"], span["value"]
    v = float(value)
    err = float(span["err"]) if span.get("err") else None
    b = _bounds(kind, v, err)
    if b is None:
        raise ValueError(f"not perturbable: {kind} {value}")
    group = ("pm", "asym") if kind in ("pm", "asym") else (kind,)
    swaps = sorted({s["value"] for s in spans if s["kind"] in group and s["value"] != value
                    and _decimals(s["value"]) == _decimals(value) and _distance_ok(kind, v, float(s["value"]), b)})
    if swaps and SWAP:
        return rng.choice(swaps), "swap"
    for _ in range(200):
        up = rng.random() < 0.5
        if kind in ("pm", "asym"):
            shift = min(max(rng.uniform(3, 5) * (err or 0.0), b[0]), b[1])
            w = v + (shift if up else -shift)
        elif kind == "sigma":
            step = rng.uniform(1, 2)
            w = v + step if (up or v - step < 0.5) else v - step
        else:
            f = rng.uniform(*b)
            w = v * f if up else v / f
        out = _fmt(w, value)
        if out != _fmt(v, value) and _distance_ok(kind, v, float(out), (b[0] * 0.98, b[1] * 1.02) if kind in ("pm", "asym", "sigma") else (b[0] * 0.97, b[1] * 1.03)):
            return out, "shift"
    raise ValueError(f"no perturbation for {kind} {value}")


# ---------------------------------------------------------------- candidates
def plugin_ids(repo: Path) -> set[str]:
    """Every new-style arXiv ID written anywhere in the plugin."""
    ids: set[str] = set()
    for f in (Path(repo) / "plugin").rglob("*"):
        if f.is_file() and f.suffix in (".md", ".tsv", ".txt", ".json", ".yaml", ".yml", ".py", ".csv"):
            ids |= set(re.findall(r"(?<![\d.])(\d{4}\.\d{4,5})(?![\d])", f.read_text(encoding="utf-8", errors="replace")))
    return ids


def plugin_citations(repo: Path) -> set[tuple[str, int]]:
    """(surname, year) of every author-year citation in the plugin's text ("Smith et al. 2021",
    "Smith & Jones (2021)", "Smith 2021"), lower-cased: K0 papers must not be cited this way either."""
    rx = re.compile(r"\b([A-Z][A-Za-z'\-]+(?:\s[A-Z][a-z]+)?)(?:\s+et\s+al\.?|\s*(?:&|and)\s+[A-Z][A-Za-z'\-]+)?,?\s*\(?((?:19|20)\d\d)[a-z]?\b")
    out: set[tuple[str, int]] = set()
    for f in (Path(repo) / "plugin").rglob("*.md"):
        for m in rx.finditer(f.read_text(encoding="utf-8", errors="replace")):
            out.add((m.group(1).split()[-1].lower(), int(m.group(2))))
    for f in (Path(repo) / "plugin" / "skills").glob("*/references/sources.tsv"):
        for line in f.read_text(encoding="utf-8").splitlines()[1:]:
            c = line.split("\t")
            if len(c) > 3 and c[3].strip().isdigit():
                out.add((c[2].strip().split()[-1].lower() if c[2].strip() else "", int(c[3])))
    return out


def cited_by_name(p: dict, cites: set[tuple[str, int]]) -> bool:
    a, y = (p.get("first_author") or "").lower(), int(p["published"][:4])
    return bool(a) and any((a, yy) in cites for yy in (y, y + 1))


def _entries_v5(root: ET.Element) -> list[dict]:
    out = []
    for e in root.findall("a:entry", NS):
        aid = e.findtext("a:id", "", NS).rsplit("/abs/", 1)[-1]
        m = re.match(r"(\d{4}\.\d{4,5})", aid)
        if not m:
            continue
        prim = e.find("arxiv:primary_category", NS)
        authors = [" ".join((a.findtext("a:name", "", NS) or "").split()) for a in e.findall("a:author", NS)]
        out.append({"arxiv": m.group(1), "title": " ".join(e.findtext("a:title", "", NS).split()),
                    "abstract": " ".join(e.findtext("a:summary", "", NS).split()),
                    "published": e.findtext("a:published", "", NS)[:10],
                    "primary": prim.get("term") if prim is not None else "",
                    "surnames": sorted({a.split()[-1] for a in authors[:60] if a.split()}),
                    "first_author": authors[0].split()[-1] if authors and authors[0].split() else ""})
    return out


def _get_cached(params: dict) -> ET.Element:
    """arXiv API call cached on disk (resumable build), with long backoff on throttling."""
    import hashlib
    import urllib.parse
    url = API + "?" + urllib.parse.urlencode(params)
    cache = V5 / "api_cache" / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".xml")
    if cache.exists():
        return ET.fromstring(cache.read_bytes())
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(10):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                body = r.read()
            root = ET.fromstring(body)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(body)
            time.sleep(3.5)
            return root
        except Exception as e:  # noqa: BLE001 - throttling shows as HTTP 429/503 or a reset
            wait = min(300, 20 * (attempt + 1))
            print(f"  arXiv retry {attempt + 1} in {wait}s: {str(e)[:80]}", flush=True)
            time.sleep(wait)
    raise RuntimeError("arXiv API unreachable: " + url[:200])


def fetch_ids_v5(ids: list[str]) -> list[dict]:
    out = []
    for k in range(0, len(ids), 50):
        out += _entries_v5(_get_cached({"id_list": ",".join(ids[k:k + 50]), "max_results": 50}))
    return out


def fetch_window_v5(start: dt.date, end: dt.date) -> list[dict]:
    q = f"cat:astro-ph.CO AND submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
    out, k = [], 0
    while True:
        batch = _entries_v5(_get_cached({"search_query": q, "start": k, "max_results": 200,
                                  "sortBy": "submittedDate", "sortOrder": "ascending"}))
        out += batch
        if len(batch) < 200:
            return out
        k += 200


def citations(ids: list[str]) -> dict[str, int | None]:
    """arXiv ID -> Semantic Scholar citation count (None if unknown), batches of 400 with backoff."""
    out: dict[str, int | None] = {}
    for k in range(0, len(ids), 400):
        chunk = ids[k:k + 400]
        body = json.dumps({"ids": [f"ARXIV:{i}" for i in chunk]}).encode()
        for attempt in range(8):
            req = urllib.request.Request(S2_BATCH, data=body, headers={**UA, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    res = json.loads(r.read())
                break
            except Exception:
                time.sleep(15 * (attempt + 1))
        else:
            raise RuntimeError("Semantic Scholar unreachable")
        for i, rec in zip(chunk, res):
            out[i] = rec.get("citationCount") if isinstance(rec, dict) else None
        time.sleep(3)
    return out


def usable_types(p: dict) -> list[str]:
    t = []
    if usable_spans(p["abstract"]):
        t.append("number")
    if dataset_options(p["abstract"], p["published"]):
        t.append("dataset")
    if conclusion_families(p["abstract"]):
        t.append("conclusion")
    if usable_spans(p["abstract"]) or datasets_in(p["abstract"]):
        t.append("attribution")
    return t


def stratum_v5(p: dict, src: set[str], plug: set[str], snap: dt.date, end_b: dt.date) -> str | None:
    d = dt.date.fromisoformat(p["published"])
    if p["arxiv"] in src and K_START <= d <= K_END:
        return "K"
    if p["arxiv"] not in plug and K_START <= d <= K_END:
        return "K0"
    if p["arxiv"] in src and A_START <= d <= snap:
        return "A"
    if p["arxiv"] not in plug and snap < d <= end_b:
        return "B"
    return None


def draw_v5(pool: list[dict], n: int, rng: random.Random, cap: int = DOMAIN_CAP) -> list[dict]:
    pool = sorted(pool, key=lambda p: p["arxiv"])
    rng.shuffle(pool)
    count: Counter = Counter()
    out = []
    for p in pool:
        if count[p["domain"]] >= cap and p["domain"] != "unassigned":
            continue
        out.append(p)
        count[p["domain"]] += 1
        if len(out) == n:
            break
    return out


INSPIRE = "https://inspirehep.net/api/literature"
K0_FRAME_TOPCITE = 25   # INSPIRE pre-filter (superset); the S2 >= MIN_CITES rule is applied after


def inspire_frame(start: dt.date = K_START, end: dt.date = K_END, topcite: int = K0_FRAME_TOPCITE) -> list[str]:
    """arXiv IDs of primary astro-ph.CO papers in INSPIRE, dated start..end, with >= topcite citations."""
    import urllib.parse
    q = f"primarch astro-ph.CO and topcite {topcite}+ and de {start.isoformat()}->{end.isoformat()}"
    ids, page = [], 1
    while True:
        url = INSPIRE + "?" + urllib.parse.urlencode({"q": q, "size": 250, "page": page, "fields": "arxiv_eprints",
                                                      "sort": "mostrecent"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                    d = json.loads(r.read())
                break
            except Exception:
                time.sleep(15 * (attempt + 1))
        else:
            raise RuntimeError("INSPIRE unreachable")
        hits = d["hits"]["hits"]
        for h in hits:
            for e in h["metadata"].get("arxiv_eprints", [])[:1]:
                if re.match(r"^\d{4}\.\d{4,5}$", e["value"]):
                    ids.append(e["value"])
        if len(hits) < 250 or page * 250 >= 9750:
            return sorted(set(ids))
        page += 1
        time.sleep(1)


def candidates(repo: Path, today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    rng = random.Random(SEED_V5)
    src_idx = sources_index(repo)
    src, plug = set(src_idx), plugin_ids(repo)
    snap = snapshot_date(repo)
    end_b = today - dt.timedelta(days=B_LAG_DAYS)
    kw = skill_keywords(repo)
    raw: list[dict] = []
    k_ids = sorted(i for i in src if f"{K_START:%y%m}" <= i[:4] <= f"{K_END:%y%m}")
    a_ids = sorted(i for i in src if f"{A_START:%y%m}" <= i[:4] <= f"{snap:%y%m}")
    print(f"fetching {len(k_ids)} K and {len(a_ids)} A source IDs", flush=True)
    raw += fetch_ids_v5(k_ids + a_ids)
    frame = [i for i in inspire_frame() if i not in plug]
    print(f"fetching {len(frame)} K0 IDs (INSPIRE frame) and the B window", flush=True)
    raw += fetch_ids_v5(frame)
    raw += fetch_window_v5(snap + dt.timedelta(days=1), end_b)
    cites = plugin_citations(repo)
    seen, pool = set(), []
    for p in raw:
        if p["arxiv"] in seen or p["primary"] != "astro-ph.CO":
            continue
        seen.add(p["arxiv"])
        s = stratum_v5(p, src, plug, snap, end_b)
        if s is None or (s == "K0" and cited_by_name(p, cites)):
            continue  # K0: uncovered also by author-year citation (review 2026-11-21, finding 16)
        dom = src_idx[p["arxiv"]] if p["arxiv"] in src_idx else guess_domain(p["title"] + " " + p["abstract"], kw)
        pool.append({**p, "stratum": s, "domain": dom, "types": usable_types(p)})
    known = [p["arxiv"] for p in pool if p["stratum"] in ("K", "K0")]
    print(f"citations for {len(known)} K/K0 papers", flush=True)
    cites = citations(known)
    for p in pool:
        p["citations"] = cites.get(p["arxiv"])
    write_jsonl(V5 / "pool.jsonl", pool)
    chosen = []
    eligible = {s: [p for p in pool if p["stratum"] == s and len(p["types"]) >= 3
                    and (s not in ("K", "K0") or (p["citations"] or 0) >= MIN_CITES)] for s in STRATA}
    for s in STRATA:
        n = len(eligible[s]) if s in DRAW_ALL else int(round(2 * ITEMS[s] * DRAW_FACTOR))
        chosen += draw_v5(eligible[s], n, rng, cap=10 ** 6 if s in DRAW_ALL else DOMAIN_CAP)
    meta = {"built": today.isoformat(), "snapshot": snap.isoformat(), "seed": SEED_V5,
            "windows": {"K": [str(K_START), str(K_END)], "K0": [str(K_START), str(K_END)],
                        "A": [str(A_START), str(snap)], "B": [str(snap + dt.timedelta(days=1)), str(end_b)]},
            "k0_frame": f"INSPIRE primarch astro-ph.CO, topcite {K0_FRAME_TOPCITE}+, {len(frame)} IDs not in the plugin",
            "min_citations": MIN_CITES,
            "pool": dict(Counter(p["stratum"] for p in pool)),
            "eligible": {s: len(v) for s, v in eligible.items()},
            "drawn": dict(Counter(p["stratum"] for p in chosen)),
            "types_in_eligible": {s: dict(Counter(t for p in v for t in p["types"])) for s, v in eligible.items()}}
    (V5 / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    write_jsonl(V5 / "candidates.jsonl", chosen)
    print(json.dumps(meta, indent=1))
    return meta


def candidates_k0fresh(repo: Path, today: dt.date | None = None) -> dict:
    """1.1.1 re-test: K0 papers of 2015-2019, same eligibility rule as v5, none of the v5 papers."""
    today = today or dt.date.today()
    used = {c["paper"] for it in read_jsonl(DATA / "v5" / "claimcheck.jsonl") for c in it["reference"]["claims"]}
    used |= {p["arxiv"] for p in read_jsonl(DATA / "v5" / "candidates.jsonl")}
    plug, cites_pl, kw = plugin_ids(repo), plugin_citations(repo), skill_keywords(repo)
    frame = [i for i in inspire_frame(FRESH_START, FRESH_END) if i not in plug and i not in used]
    print(f"fetching {len(frame)} K0 IDs (INSPIRE frame {FRESH_START}..{FRESH_END})", flush=True)
    seen, pool = set(), []
    for p in fetch_ids_v5(frame):
        if p["arxiv"] in seen or p["primary"] != "astro-ph.CO" or cited_by_name(p, cites_pl):
            continue
        if not FRESH_START <= dt.date.fromisoformat(p["published"]) <= FRESH_END:
            continue
        seen.add(p["arxiv"])
        pool.append({**p, "stratum": "K0", "domain": guess_domain(p["title"] + " " + p["abstract"], kw),
                     "types": usable_types(p)})
    cites = citations([p["arxiv"] for p in pool])
    for p in pool:
        p["citations"] = cites.get(p["arxiv"])
    write_jsonl(V5 / "pool.jsonl", pool)
    eligible = [p for p in pool if len(p["types"]) >= 3 and (p["citations"] or 0) >= MIN_CITES]
    chosen = draw_v5(eligible, FRESH_DRAW, random.Random(FRESH_SEED), cap=10 ** 6)
    meta = {"built": today.isoformat(), "seed": FRESH_SEED, "window": [str(FRESH_START), str(FRESH_END)],
            "k0_frame": f"INSPIRE primarch astro-ph.CO, topcite {K0_FRAME_TOPCITE}+, {len(frame)} IDs not in the plugin nor v5",
            "min_citations": MIN_CITES, "pool": len(pool), "eligible": len(eligible), "drawn": len(chosen),
            "eligible_by_year": dict(sorted(Counter(p["published"][:4] for p in eligible).items()))}
    (V5 / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    write_jsonl(V5 / "candidates.jsonl", chosen)
    print(json.dumps(meta, indent=1))
    return meta


# ---------------------------------------------------------------- drafting
DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "number": {"type": "object", "properties": {"span": {"type": "integer"}, "template": {"type": "string"}},
                   "required": ["span", "template"], "additionalProperties": False},
        "dataset": {"type": "object", "properties": {"name": {"type": "string"}, "template": {"type": "string"}},
                    "required": ["name", "template"], "additionalProperties": False},
        "conclusion": {"type": "object", "properties": {"phrase": {"type": "string"}, "evidence": {"type": "string"},
                                                        "template": {"type": "string"}},
                       "required": ["phrase", "evidence", "template"], "additionalProperties": False},
        "attribution": {"type": "object", "properties": {"text": {"type": "string"}},
                        "required": ["text"], "additionalProperties": False},
    },
    "required": list(TYPES), "additionalProperties": False,
}
DRAFT_HEAD = """Below is the abstract of a cosmology paper. Write claims about it, as they would appear in another paper's text when citing this result.

Every claim is one sentence in plain words: plain text or simple Unicode (±, σ, Ω, Λ), no LaTeX, no $ signs. It names the survey, data set or method so that it is unambiguous, and it describes the result exactly as the abstract does: the same quantity, the same data combination, the same model. Never name the authors, the title or the arXiv ID. Use only what the abstract says.

Fill every field. For a claim type marked SKIP, or when nothing in the abstract fits that type's rules (for a number claim: no number in the list is a main result), write -1 or empty strings for it. A skipped claim is better than a forced one.
"""
DRAFT_NUMBER = """NUMBER claim. Pick one number from the list below that is a main result of the paper (a measured value, a significance, a limit), not a selection cut, threshold, scale or setting chosen by the authors. Put its index in number.span. Write number.template with the placeholder {{V}} in place of the central value, exactly once, and copy the rest of the number (uncertainty, units, the < or > sign, "σ", "%") exactly as in the abstract. Apart from that number (and a redshift or a confidence level such as 95%), the sentence contains no other number, and no comparison or verdict word (consistent, tension, agree, higher, lower, favour, detection, improve, ...). Example: "DESI DR2 BAO alone gives Ωm = {{V}} ± 0.0086 in ΛCDM."
Numbers:
{spans}
"""
DRAFT_DATASET = """DATASET claim. The abstract uses: {names}. Pick one and copy its name, exactly as written here, into dataset.name. Write dataset.template: a sentence about a main result of the paper with the placeholder {{D}} where that name goes, exactly once. Do not write the name, or any other release or version of the same survey or simulation, anywhere else in the sentence. The sentence contains no number and no word that dates the release (first-year, three-year, final, early, latest, ...). Example: "Cosmic shear from {{D}} gives S8 lower than the Planck value."
"""
DRAFT_CONCLUSION = """CONCLUSION claim. Pick one qualitative conclusion of the paper and copy, verbatim, the abstract sentence that states it into conclusion.evidence. Write conclusion.template with the placeholder {{C}} where the verdict goes, and put in conclusion.phrase the phrase that makes the sentence true, chosen from this list of pairs: {pairs}. The sentence must read correctly, and be equally plausible, with either phrase of its pair in place of {{C}}: apart from {{C}} it contains no comparison or verdict word (tension, consistent, agree, higher, lower, favour, detect, improve, ...) and no word that fits only one phrase (fully, resolving, mild, strong, ...). The claim contains no numbers. Example: phrase "in tension with", template "The DESI DR2 BAO distances are {{C}} the Planck ΛCDM prediction."
"""
DRAFT_ATTRIBUTION = """ATTRIBUTION claim. Write attribution.text: a sentence that states a main result of the paper and includes at least one number or one named data set copied exactly from the abstract. No placeholder.
"""
PAIRS_TEXT = "; ".join(f'"{a}" / "{b}"' for _, a, b in PAIRS)


def spans_block_v5(spans: list[dict]) -> str:
    return "\n".join(f"[{k}] …{s['context']}…" for k, s in enumerate(spans))


def draft_prompt(p: dict) -> str:
    parts = [DRAFT_HEAD]
    spans = usable_spans(p["abstract"])
    parts.append(DRAFT_NUMBER.format(spans=spans_block_v5(spans)) if "number" in p["types"] else "NUMBER claim: SKIP.\n")
    opts = dataset_options(p["abstract"], p["published"])
    parts.append(DRAFT_DATASET.format(names=", ".join(opts)) if "dataset" in p["types"] else "DATASET claim: SKIP.\n")
    parts.append(DRAFT_CONCLUSION.format(pairs=PAIRS_TEXT) if "conclusion" in p["types"] else "CONCLUSION claim: SKIP.\n")
    parts.append(DRAFT_ATTRIBUTION if "attribution" in p["types"] else "ATTRIBUTION claim: SKIP.\n")
    parts.append(f"Title: {p['title']}\nAbstract: {p['abstract']}")
    return "\n".join(parts)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


IDENT = re.compile(r"(?<![\w.])(?:[A-Za-zα-ωΑ-Ω][A-Za-zα-ωΑ-Ω+]*[-_]?\d+(?:\.\d+)?[A-Za-z]*|\d+x\d+\s?pt|\d+pt|\d+D)\b")


def has_value(text: str) -> bool:
    """A digit outside identifiers such as DR2, Y3, KiDS-1000, S8, H0, 3x2pt: a value."""
    for rx in _DS_RX.values():
        text = rx.sub(" ", text)
    return bool(re.search(r"\d", IDENT.sub(" ", text)))


ALLOWED_NUMBERS = re.compile(
    r"\b(?:68|90|95|99|99\.7)\s*(?:%|per\s?cent)\s*(?:C\.?\s?L\.?|confidence|credib\w*)"   # confidence levels
    r"|[\d.]+\s*[<≤]\s*z\s*[<≤]\s*[\d.]+|\bz\s*(?:=|≈|~|<|>|≤|≥|≪|≫|≲|≳|≃|\\sim)\s*[\d.]+", re.I)  # redshifts
RELEASE_WORDS = re.compile(r"\b(?:first|second|third|three|five|six|final|early|initial|latest|newest|"
                           r"year|years|yr|release)\b", re.I)
ONE_SIDED = re.compile(r"\b(?:fully|perfectly|resolv\w*|mild\w*|strong\w*|slight\w*|marginal\w*|"
                       r"significant\w*|clear\w*|robust\w*)\b", re.I)


def verdict_words(text: str) -> bool:
    return any(re.search(x, text, re.I) for pair in CUES.values() for x in pair)


def common_problem(text: str, p: dict) -> str:
    if re.search(r"\b\d{4}\.\d{4,5}\b", text) or p["arxiv"] in text:
        return "arXiv ID in claim"
    if "\\" in text or "$" in text:
        return "LaTeX in claim"
    title_abs = p["title"] + " " + p["abstract"]
    for s in p.get("surnames", []):
        if len(s) >= 4 and re.search(rf"\b{re.escape(s)}\b", text) and not re.search(rf"\b{re.escape(s)}\b", title_abs):
            return f"author name {s}"
    return ""


def validate_type(kind: str, d: dict, p: dict) -> str:
    """Empty string if the drafted claim of this type is usable, else the reason."""
    spans = usable_spans(p["abstract"])
    if kind == "number":
        k, t = d.get("span", -1), d.get("template", "")
        if k == -1:
            return "skipped by the drafter"
        if not (0 <= k < len(spans)):
            return "span index out of range"
        if t.count("{V}") != 1 or re.search(r"\{[A-Z]\}", t.replace("{V}", "")):
            return "placeholder {V} not exactly once"
        s = spans[k]
        if not template_fits(t, s["kind"]):
            return "template does not fit a " + s["kind"]
        if s["kind"] == "pm" and s["err"] not in t.split("{V}", 1)[1][:30]:
            return "uncertainty not copied"
        if re.search(rf"(?<![\d.]){re.escape(s['value'])}(?![\d])", t):
            return "value written in the template"
        own = set(re.findall(r"\d+(?:\.\d+)?", p["abstract"][s["start"]:s["end"]]))
        rest = ALLOWED_NUMBERS.sub(" ", t.replace("{V}", " "))
        rest = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)", lambda m: " " if m.group(1) in own else m.group(0), rest)
        if has_value(rest):
            return "other number in the template"
        if s["kind"] != "sigma" and verdict_words(t):
            return "verdict word in a number claim"
        return common_problem(t, p)
    if kind == "dataset":
        name, t = d.get("name", ""), d.get("template", "")
        if name not in dataset_options(p["abstract"], p["published"]):
            return "name not an option"
        if t.count("{D}") != 1:
            return "placeholder {D} not exactly once"
        if any(_DS_RX[m].search(t) for m, _, _ in DATASETS[_DS_GROUP[name]]):
            return "group member written in the template"
        if has_value(ALLOWED_NUMBERS.sub(" ", t.replace("{D}", " "))):
            return "number in a data-set claim"
        if RELEASE_WORDS.search(t):
            return "release-dating word in a data-set claim"
        return common_problem(t, p)
    if kind == "conclusion":
        ph, ev, t = d.get("phrase", ""), d.get("evidence", ""), d.get("template", "")
        if ph not in PHRASES:
            return "phrase not in the list"
        fam, side, other = PHRASES[ph]
        if t.count("{C}") != 1:
            return "placeholder {C} not exactly once"
        if has_value(ALLOWED_NUMBERS.sub(" ", t)):
            return "number in conclusion claim"
        if _norm(ev) not in _norm(p["abstract"]) or len(_norm(ev)) < 20:
            return "evidence not verbatim in the abstract"
        if cue_side(ev, fam) != side:
            return "evidence cue does not match the phrase"
        if ph in t or other in t:
            return "phrase written in the template"
        if any(re.search(x, t.replace("{C}", " "), re.I) for x in CUES[fam]) or ONE_SIDED.search(t):
            return "verdict word outside the placeholder"
        return common_problem(t, p)
    if kind == "attribution":
        t = d.get("text", "")
        if len(t) < 30 or "{" in t:
            return "no usable sentence"
        if common_problem(t, p):
            return common_problem(t, p)
        nums = re.findall(r"\d+(?:\.\d+)?", t)
        if not nums and not datasets_in(t):
            return "no number or named data set"
        absn = set(re.findall(r"\d+(?:\.\d+)?", p["abstract"]))
        if any(n not in absn for n in nums):
            return "number not in the abstract"
        return common_problem(t, p)
    return "unknown type"


def run_tasks(tasks: list, work, workers: int = 1, max_errors: int = 3) -> int:
    """Run work(task) -> (ok, message) on `workers` threads; stop after max_errors consecutive failures."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    state = {"errors": 0, "stop": False}
    lock = threading.Lock()

    def one(task):
        if state["stop"]:
            return
        ok, msg = work(task)
        with lock:
            print(msg, flush=True)
            state["errors"] = 0 if ok else state["errors"] + 1
            if state["errors"] >= max_errors:
                state["stop"] = True
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(one, tasks))
    return 2 if state["stop"] else 0


_WRITE = __import__("threading").Lock()


def _append(path: Path, rec: dict) -> None:
    with _WRITE:
        append_jsonl(path, rec)


def draft(repo: Path, limit: int | None = None, workers: int = 1) -> int:
    cands = read_jsonl(V5 / "candidates.jsonl")
    out = V5 / "drafts.jsonl"
    done = {d["arxiv"] for d in read_jsonl(out)}
    claude, env = find_claude_bin(), build_env(dict(os.environ))
    executor = make_executor(repo)
    todo = [p for p in cands if p["arxiv"] not in done][:limit]

    def work(p):
        cmd = build_command(DRAFTER, claude_bin=claude, repo=repo, json_schema=DRAFT_SCHEMA, env=env)
        rec = parse_stream(executor(cmd, env, WORKDIR, 900, draft_prompt(p)))
        d = rec.get("structured_output")
        if rec["status"] != "ok" or not isinstance(d, dict):
            return False, f"draft error {p['arxiv']} {rec['error'][:200]}"
        problems = {t: (validate_type(t, d.get(t) or {}, p) if t in p["types"] else "not usable") for t in TYPES}
        _append(out, {"arxiv": p["arxiv"], "stratum": p["stratum"], "draft": d, "problems": problems,
                      "cost_usd": rec.get("cost_usd")})
        return True, f"{p['stratum']} {p['arxiv']} {({t: (v or 'ok') for t, v in problems.items()})}"
    return run_tasks(todo, work, workers)


# ---------------------------------------------------------------- rendering
def choose_decoy(p: dict, claim_text: str, pool: list[dict]) -> dict | None:
    """Same stratum, within DECOY_DAYS, none of the claim's numbers in its abstract, closest title."""
    d0 = dt.date.fromisoformat(p["published"])
    nums = set(re.findall(r"\d+\.\d+|\d{2,}", claim_text))
    named = [n for ms in datasets_in(claim_text).values() for n in ms]
    newest = max((_DS_DATE[n] for n in named), default="0000-00")
    words = set(re.findall(r"[a-z]{4,}", p["title"].lower()))
    best, key = None, None
    for q in pool:
        if q["arxiv"] == p["arxiv"] or q["stratum"] != p["stratum"]:
            continue
        if abs((dt.date.fromisoformat(q["published"]) - d0).days) > DECOY_DAYS:
            continue
        if q["published"][:7] < newest:
            continue  # the decoy must postdate every data set the claim names (finding 14)
        qn = set(re.findall(r"\d+\.\d+|\d{2,}", q["abstract"]))
        qn |= {f"{float(x):.{_decimals(n)}f}" for x in qn for n in nums if "." in x and "." in n}
        if nums & qn:
            continue  # also 0.2975 against 0.297
        qw = set(re.findall(r"[a-z]{4,}", q["title"].lower()))
        jac = len(words & qw) / max(1, len(words | qw))
        k = (q["domain"] == p["domain"], jac, q["arxiv"])
        if key is None or k > key:
            best, key = q, k
    return best


_SUP_SUB = re.compile(r"\^\{\s*\+?\s*([\d.]+)\s*\}\s*_\{\s*[-−]\s*([\d.]+)\s*\}")
_SUB_SUP = re.compile(r"_\{\s*[-−]\s*([\d.]+)\s*\}\s*\^\{\s*\+?\s*([\d.]+)\s*\}")


def plain(text: str) -> str:
    """Asymmetric errors written X^{+a}_{-b} (drafters copy them from LaTeX) as X +a/−b; other ^{..} as ^.."""
    text = _SUP_SUB.sub(r" +\1/−\2", text)
    text = _SUB_SUP.sub(r" +\2/−\1", text)
    return re.sub(r"\^\{([^{}]*)\}", r"^\1", text)


def render(p: dict, dr: dict, pool: list[dict], rng: random.Random) -> list[dict]:
    """Both versions of every valid claim of a paper, in plain text."""
    return [{**c, "true_text": plain(c["true_text"]), "pert_text": plain(c["pert_text"])}
            for c in _render(p, dr, pool, rng)]


def _render(p: dict, dr: dict, pool: list[dict], rng: random.Random) -> list[dict]:
    out = []
    for t in TYPES:
        # validated again here, so validator fixes made after drafting apply to stored drafts
        if t not in p.get("types", TYPES) or validate_type(t, dr["draft"].get(t) or {}, p):
            continue
        d = dr["draft"][t]
        c = {"uid": f"{p['arxiv']}:{t}", "paper": p["arxiv"], "stratum": p["stratum"], "type": t}
        if t == "number":
            spans = usable_spans(p["abstract"])
            s = spans[d["span"]]
            shown, mode = perturb_number(s, spans, rng)
            c.update(true_text=d["template"].replace("{V}", s["value"]), pert_text=d["template"].replace("{V}", shown),
                     cited_pert=p["arxiv"], mode=mode, kind=s["kind"], value_true=s["value"], value_pert=shown,
                     evidence=s["context"])
        elif t == "dataset":
            target = rng.choice(dataset_targets(d["name"], p["abstract"], p["published"]))
            c.update(true_text=d["template"].replace("{D}", d["name"]), pert_text=d["template"].replace("{D}", target),
                     cited_pert=p["arxiv"], mode="swap", value_true=d["name"], value_pert=target,
                     evidence=next((s for s in sentences(p["abstract"]) if _DS_RX[d["name"]].search(s)), ""))
        elif t == "conclusion":
            other = PHRASES[d["phrase"]][2]
            c.update(true_text=d["template"].replace("{C}", d["phrase"]), pert_text=d["template"].replace("{C}", other),
                     cited_pert=p["arxiv"], mode="flip", value_true=d["phrase"], value_pert=other, evidence=d["evidence"])
        else:
            decoy = choose_decoy(p, d["text"], pool)
            if decoy is None:
                continue
            c.update(true_text=d["text"], pert_text=d["text"], cited_pert=decoy["arxiv"], mode="decoy",
                     value_true=p["arxiv"], value_pert=decoy["arxiv"], evidence=decoy["title"],
                     decoy_abstract=decoy["abstract"], decoy_published=decoy["published"])
        out.append(c)
    return out


# ---------------------------------------------------------------- verification
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"answers": {"type": "array", "items": {
        "type": "object",
        "properties": {"claim": {"type": "integer"}, "supported": {"type": "string", "enum": ["1", "2", "both", "neither"]},
                       "checkable": {"type": "boolean"}},
        "required": ["claim", "supported", "checkable"], "additionalProperties": False}}},
    "required": ["answers"], "additionalProperties": False,
}
VERIFY_HEAD = """You check claims written about a cosmology paper against its abstract. Each claim comes in two versions.

A version is supported only if every element of it matches the abstract: the quantity and its value, the uncertainty, the data set or release, the method or model, and the conclusion. A version that misdescribes any element is not supported, even if its number appears in the abstract.

For each claim answer "1", "2", "both" or "neither": which versions the abstract supports. Also say whether the claim is specific enough that a reader of the paper could check it (checkable: true or false).

Abstract of the paper:
{abstract}
"""
VERIFY_ATTR = """
Claim {n} (attribution). Here there is one sentence, and the question is which abstracts report its result. Judge each abstract on its own.
Sentence: {text}
Version 1 means "the abstract of the paper, above, reports this result"; version 2 means "the other abstract, below, reports this result". Answer "both" if both do, "neither" if neither does.
Other abstract: {other}
"""


def verify_prompt(p: dict, claims: list[dict], rng: random.Random) -> tuple[str, list[dict]]:
    """Prompt and, per claim, which displayed version is the true one."""
    parts, keys = [VERIFY_HEAD.format(abstract=p["abstract"])], []
    for n, c in enumerate(claims, 1):
        true_first = rng.random() < 0.5
        keys.append({"uid": c["uid"], "true_is": "1" if true_first else "2"})
        if c["type"] == "attribution":
            # independent questions about two abstracts; the paper's abstract is not shown twice (finding 4)
            keys[-1]["true_is"] = "1"
            parts.append(VERIFY_ATTR.format(n=n, text=c["true_text"], other=c["decoy_abstract"]))
        else:
            v1, v2 = (c["true_text"], c["pert_text"]) if true_first else (c["pert_text"], c["true_text"])
            parts.append(f"\nClaim {n}.\nVersion 1: {v1}\nVersion 2: {v2}\n")
    return "\n".join(parts), keys


BLIND_SCHEMA = {
    "type": "object",
    "properties": {"answers": {"type": "array", "items": {
        "type": "object",
        "properties": {"claim": {"type": "integer"}, "original": {"type": "string", "enum": ["1", "2"]},
                       "confidence": {"type": "string", "enum": ["low", "medium", "high"]}},
        "required": ["claim", "original", "confidence"], "additionalProperties": False}}},
    "required": ["answers"], "additionalProperties": False,
}
BLIND_HEAD = """Each numbered item below is one sentence from a cosmology manuscript, in two versions. One version is what the cited paper says; the other was altered (a value, a data set or a verdict). You do not see the paper.

Using only your general knowledge and the plausibility of each version, say which version is the original. Give your confidence: "high" only if you are confident without the paper, "medium" if one version looks somewhat more plausible, "low" if you cannot tell.
"""


def blind_prompt(claims: list[dict], keys: list[dict]) -> tuple[str, list[int]]:
    """The non-attribution claims, versions in the same order as for the verifier; no abstract, no ID."""
    parts, idx = [BLIND_HEAD], []
    for n, (c, k) in enumerate(zip(claims, keys), 1):
        if c["type"] == "attribution":
            continue
        v1, v2 = (c["true_text"], c["pert_text"]) if k["true_is"] == "1" else (c["pert_text"], c["true_text"])
        parts.append(f"\nItem {n}.\nVersion 1: {v1}\nVersion 2: {v2}\n")
        idx.append(n)
    return "\n".join(parts), idx


def blind_hit(ans: dict | None, true_is: str) -> bool:
    """The blind model names the true version with high confidence: guessable without the paper."""
    return bool(ans) and ans.get("original") == true_is and ans.get("confidence") == "high"


SOFT_MIN = 0.75          # assembly: least share of a type, relative to equal shares, in a soft fill
BLIND_DROP = ("A", "B")  # strata where a blind hit can only be priors (no model knows these papers)


def verdict_ok(ans: dict | None, true_is: str) -> bool:
    return bool(ans) and ans.get("supported") == true_is and bool(ans.get("checkable"))


def verify(repo: Path, limit: int | None = None, workers: int = 1) -> int:
    pool = read_jsonl(V5 / "pool.jsonl")
    cands = {p["arxiv"]: p for p in read_jsonl(V5 / "candidates.jsonl")}
    drafts = {d["arxiv"]: d for d in read_jsonl(V5 / "drafts.jsonl")}
    out = V5 / "verified.jsonl"
    done = {r["paper"] for r in read_jsonl(out)}
    claude, env = find_claude_bin(), build_env(dict(os.environ))
    executor = make_executor(repo)
    todo = [a for a in sorted(drafts) if a not in done and a in cands][:limit]

    def work(a):
        p, rng = cands[a], random.Random(f"{SEED_V5}:{a}")
        claims = render(p, drafts[a], pool, rng)
        if not claims:
            _append(out, {"paper": a, "stratum": p["stratum"], "claims": [], "note": "no valid draft"})
            return True, f"{p['stratum']} {a} no valid draft"
        prompt, keys = verify_prompt(p, claims, rng)
        cmd = build_command(VERIFIER, claude_bin=claude, repo=repo, json_schema=VERIFY_SCHEMA, env=env)
        rec = parse_stream(executor(cmd, env, WORKDIR, 900, prompt))
        res = rec.get("structured_output")
        if rec["status"] != "ok" or not isinstance(res, dict):
            return False, f"verify error {a} {rec['error'][:200]}"
        by_n = {x.get("claim"): x for x in res.get("answers", []) if isinstance(x, dict)}
        cost = rec.get("cost_usd") or 0.0
        bprompt, bidx = blind_prompt(claims, keys)
        blind: dict = {}
        if bidx:
            bcmd = build_command(BLIND, claude_bin=claude, repo=repo, json_schema=BLIND_SCHEMA, env=env)
            brec = parse_stream(executor(bcmd, env, WORKDIR, 900, bprompt))
            bres = brec.get("structured_output")
            if brec["status"] != "ok" or not isinstance(bres, dict):
                return False, f"blind error {a} {brec['error'][:200]}"
            blind = {x.get("claim"): x for x in bres.get("answers", []) if isinstance(x, dict)}
            cost += brec.get("cost_usd") or 0.0
        for k, (c, key) in enumerate(zip(claims, keys), 1):
            c["verifier"] = by_n.get(k)
            c["true_is"] = key["true_is"]
            c["verified"] = verdict_ok(by_n.get(k), key["true_is"])
            c["blind"] = blind.get(k)
            c["blind_hit"] = blind_hit(blind.get(k), key["true_is"]) if c["type"] != "attribution" else None
        _append(out, {"paper": a, "stratum": p["stratum"], "claims": claims, "cost_usd": cost})
        return True, f"{p['stratum']} {a} {({c['type']: c['verified'] for c in claims})}"
    return run_tasks(todo, work, workers)


# ---------------------------------------------------------------- assembly
def fair_quotas(n_papers: int, avail: dict[str, int]) -> dict[str, int]:
    """Claims per type for 2 * n_papers claims: equal shares, a type short of papers capped at what it
    has and the rest spread over the others (water-filling)."""
    total, q = 2 * n_papers, {t: 0 for t in TYPES}
    open_t = [t for t in TYPES if avail.get(t, 0) > 0]
    while total > 0 and open_t:
        share = max(1, total // len(open_t))
        for t in sorted(open_t, key=lambda t: (avail[t] - q[t], TYPES.index(t))):
            add = min(share, avail[t] - q[t], total)
            q[t] += add
            total -= add
        open_t = [t for t in open_t if q[t] < avail[t]]
    return q


def choose_pairs(papers: list[dict], n_papers: int, quota, rng: random.Random):
    """Pick n_papers papers and 2 claim types each so that type t gets quota[t] claims
    (an int means the same quota for every type). Exact: a small integer programme, with random
    tie-breaking costs so that the choice among feasible selections stays random (seeded).

    papers: [{"paper", "types": [verified types]}]. Returns ([(paper, (t1, t2))], deviation), where
    deviation = claims short of quota + 2 per missing paper (0 when the quotas are met exactly)."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    quota = quota if isinstance(quota, dict) else {t: quota for t in TYPES}
    var = [(k, (a, b)) for k, p in enumerate(papers) for i, a in enumerate(p["types"]) for b in p["types"][i + 1:]]
    n_x, n_t = len(var), len(TYPES)
    n_take = min(n_papers, len(papers))
    if not var or n_take == 0:
        return [], sum(quota.values()) + 2 * n_papers
    # variables: x (one per paper and type pair), shortfall s_t and excess o_t per type, and the
    # largest shortfall m (so that a shortfall is spread over the types rather than piled on one)
    pad = 2 * n_t + 1
    rows, lo, hi = [], [], []
    for k in range(len(papers)):                          # a paper is used at most once
        rows.append([1.0 if v[0] == k else 0.0 for v in var] + [0.0] * pad); lo.append(0); hi.append(1)
    rows.append([1.0] * n_x + [0.0] * pad); lo.append(n_take); hi.append(n_take)   # n_take papers
    for ti, t in enumerate(TYPES):                         # count_t + s_t - o_t = quota_t
        r = [1.0 if t in v[1] else 0.0 for v in var] + [0.0] * pad
        r[n_x + ti], r[n_x + n_t + ti] = 1.0, -1.0
        rows.append(r); lo.append(quota.get(t, 0)); hi.append(quota.get(t, 0))
        r = [0.0] * (n_x + pad)                            # s_t <= m
        r[n_x + ti], r[-1] = 1.0, -1.0
        rows.append(r); lo.append(-np.inf); hi.append(0)
    cost = np.array([rng.random() * 1e-3 for _ in var] + [1.0] * 2 * n_t + [0.5])
    integrality = np.array([1] * n_x + [0] * pad)
    res = milp(cost, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=integrality,
               bounds=Bounds(0, [1] * n_x + [np.inf] * pad))
    if res.x is None:
        return [], sum(quota.values()) + 2 * n_papers
    out = [(papers[k]["paper"], pair) for (k, pair), x in zip(var, res.x[:n_x]) if x > 0.5]
    got = Counter(t for _, ts in out for t in ts)
    dev = sum(max(0, quota.get(t, 0) - got[t]) for t in TYPES) + 2 * (n_papers - len(out))
    return out, dev


def balanced_flags(n: int, rng: random.Random) -> list[bool]:
    flags = [True] * (n // 2) + [False] * (n - n // 2)
    rng.shuffle(flags)
    return flags


def in_skill_text(repo: Path, arxiv: str, tokens: list[str], window: int = 3, _cache: dict = {}) -> bool:
    """True if a skill file names one of the tokens within `window` lines of the arXiv ID."""
    key = str(repo)
    if key not in _cache:
        _cache[key] = [f.read_text(encoding="utf-8", errors="replace").splitlines()
                       for f in (Path(repo) / "plugin" / "skills").rglob("*") if f.suffix in (".md", ".tsv")]
    pats = [re.compile(rf"(?<![\w.]){re.escape(t)}(?![\d\w])") for t in tokens if t]
    for lines in _cache[key]:
        for k, line in enumerate(lines):
            if arxiv in line and any(p.search(x) for x in lines[max(0, k - window):k + window + 1] for p in pats):
                return True
    return False


def claim_tokens(c: dict) -> list[str] | None:
    if c["type"] == "number":
        return [c["value_true"]]
    if c["type"] == "dataset":
        return [c["value_true"]]
    if c["type"] == "conclusion":
        return None
    return re.findall(r"\d+\.\d+", c["true_text"])


def make_blocks(papers: list[str], clashes, rng: random.Random, tries: int = 4000) -> list[list[str]]:
    """Blocks of 4 with the fewest clashes: seeded shuffle, then pairwise swaps that lower the count."""
    order = papers[:]
    rng.shuffle(order)
    blocks = [order[k:k + 4] for k in range(0, len(order), 4)]
    total = sum(clashes(b) for b in blocks)
    for _ in range(tries):
        if total == 0 or len(blocks) < 2:
            break
        i, j = rng.sample(range(len(blocks)), 2)
        a, b = rng.randrange(len(blocks[i])), rng.randrange(len(blocks[j]))
        before = clashes(blocks[i]) + clashes(blocks[j])
        blocks[i][a], blocks[j][b] = blocks[j][b], blocks[i][a]
        after = clashes(blocks[i]) + clashes(blocks[j])
        if after <= before:
            total += after - before
        else:
            blocks[i][a], blocks[j][b] = blocks[j][b], blocks[i][a]
    return blocks


def assemble(repo: Path) -> list[dict]:
    rng = random.Random(SEED_V5)
    abstracts = {p["arxiv"]: p["abstract"] for p in read_jsonl(V5 / "pool.jsonl")}
    ver = [v for v in read_jsonl(V5 / "verified.jsonl") if v.get("claims")]
    items, report = [], {}
    for s in [s for s in STRATA if any(v["stratum"] == s for v in ver)]:
        claims = {c["uid"]: c for v in ver if v["stratum"] == s for c in v["claims"] if c.get("verified")
                  and not (s in BLIND_DROP and c.get("blind_hit"))}
        by_paper: dict[str, list[str]] = defaultdict(list)
        for c in claims.values():
            by_paper[c["paper"]].append(c["type"])
        papers = [{"paper": a, "types": sorted(ts, key=TYPES.index)} for a, ts in sorted(by_paper.items()) if len(ts) >= 2]
        avail = Counter(t for p in papers for t in p["types"])
        n_papers, mode = 2 * ITEMS[s], "equal"
        while n_papers >= 4:
            pick, dev = choose_pairs(papers, n_papers, n_papers // 2, rng)
            if dev == 0:
                break
            soft = [(pick, {t: n_papers // 2 for t in TYPES})]
            q = fair_quotas(n_papers, avail)   # a short type (usually dataset) gets what it has
            pick, dev = choose_pairs(papers, n_papers, q, rng)
            if dev == 0:
                mode = "fair " + json.dumps(q)
                break
            soft.append((pick, q))
            # soft: closest to the equal (else fair) quotas (the solver minimises the deviation),
            # accepted when all n_papers papers are used and every type keeps >= SOFT_MIN of its quota
            ok = [(sp, Counter(t for _, ts in sp for t in ts)) for sp, sq in soft
                  if len(sp) == n_papers and all(Counter(t for _, ts in sp for t in ts)[t] >= SOFT_MIN * sq[t]
                                                 for t in TYPES)]
            if ok:
                pick, got = ok[0]
                mode = "soft " + json.dumps({t: got[t] for t in TYPES})
                break
            n_papers -= 4  # shrink by one block
        n_items = n_papers // 2
        report[s] = {"papers_with_2_verified": len(papers), "verified_per_type": dict(avail),
                     "items": n_items, "target_items": ITEMS[s], "type_quotas": mode}
        if n_papers < 4:
            continue
        chosen = {a: ts for a, ts in pick}
        # blocks of 4 papers: no attribution decoy inside its own block, and no two papers naming the
        # same data set (companion papers could reveal each other's values; finding 17)
        names = {a: {n for ms in datasets_in(abstracts.get(a, "")).values() for n in ms} for a in chosen}

        def clashes(blk):
            n = sum(claims[f"{a}:attribution"]["cited_pert"] in blk for a in blk if "attribution" in chosen[a])
            n += sum(bool(names[x] & names[y]) for i, x in enumerate(blk) for y in blk[i + 1:])
            return n
        blocks = make_blocks(sorted(chosen), clashes, rng)
        report[s]["block_clashes"] = sum(clashes(b) for b in blocks)
        flags_by_type = {t: balanced_flags(sum(t in ts for ts in chosen.values()), rng) for t in TYPES}
        for b, blk in enumerate(blocks):
            halves: list[list[dict]] = [[], []]
            for a in blk:
                ts = list(chosen[a])
                rng.shuffle(ts)
                for h, t in enumerate(ts):
                    c = dict(claims[f"{a}:{t}"])
                    c["perturbed"] = flags_by_type[t].pop()
                    halves[h].append(c)
            for h, cl in enumerate(halves):
                rng.shuffle(cl)
                shown, ref = [], []
                for c in cl:
                    pert = c["perturbed"]
                    cited = c["cited_pert"] if pert else c["paper"]
                    text = c["pert_text"] if pert else c["true_text"]
                    shown.append({"text": text, "arxiv": cited})
                    ref.append({k: c[k] for k in ("uid", "paper", "type", "mode", "value_true", "value_pert",
                                                  "true_text", "pert_text", "evidence") if k in c}
                               | {"perturbed": pert, "cited": cited, "kind": c.get("kind"), "excluded": False,
                                  "in_skill_text": (None if claim_tokens(c) is None else
                                                    in_skill_text(repo, c["paper"], claim_tokens(c)))})
                items.append({"id": f"{ID_PREFIX}-{s}-{b:02d}{'ab'[h]}", "bench": "claimcheck", "stratum": s,
                              "block": f"{s}-{b:02d}", "prompt_fields": {"claims": shown},
                              "reference": {"claims": ref}})
    write_jsonl(V5 / "claimcheck_built.jsonl", items)
    (V5 / "assemble_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    print(f"{len(items)} items, {sum(len(i['reference']['claims']) for i in items)} claims")
    return items


# ---------------------------------------------------------------- audit and accept
AUDIT_COLS = ["ok", "uid", "stratum", "type", "perturbed", "shown", "true_text", "pert_text", "paper", "decoy",
              "evidence", "abstract", "decoy_abstract", "note"]
AUDIT_PER_CELL = 3
def audit_csv() -> Path:
    return V5 / "claim_audit.csv" if FRESH else RESULTS / "claim_audit.csv"




def audit() -> Path:
    items = read_jsonl(V5 / "claimcheck_built.jsonl")
    pool = {p["arxiv"]: p for p in read_jsonl(V5 / "pool.jsonl")}
    rng = random.Random(SEED_V5 + 1)
    cells: dict[tuple, list] = defaultdict(list)
    for it in items:
        for sh, rf in zip(it["prompt_fields"]["claims"], it["reference"]["claims"]):
            cells[(it["stratum"], rf["type"])].append((sh, rf))
    rows = []
    for key in sorted(cells, key=lambda k: (STRATA.index(k[0]), TYPES.index(k[1]))):
        for sh, rf in rng.sample(cells[key], min(4 if FRESH else AUDIT_PER_CELL, len(cells[key]))):
            # both versions always, so a perturbed version the paper supports is visible too (finding 5)
            decoy = rf["value_pert"] if rf["type"] == "attribution" else ""
            rows.append({"ok": "", "uid": rf["uid"], "stratum": key[0], "type": key[1], "perturbed": rf["perturbed"],
                         "shown": f"{sh['text']} (arXiv:{sh['arxiv']})", "true_text": rf["true_text"],
                         "pert_text": rf["pert_text"] if rf["type"] != "attribution" else f"{rf['pert_text']} (arXiv:{decoy})",
                         "paper": rf["paper"], "decoy": decoy, "evidence": rf.get("evidence", ""),
                         "abstract": pool.get(rf["paper"], {}).get("abstract", ""),
                         "decoy_abstract": pool.get(decoy, {}).get("abstract", "") if decoy else "", "note": ""})
    path = audit_csv()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=AUDIT_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} audit rows -> {path}")
    return path


def accept() -> int:
    """ok = y: both versions right; ok = n: a wrong ground truth (the true version unsupported, or the
    perturbed one supported). Rule: <= 2 n in all and < 2 in any one type -> accept, the n claims
    excluded from scoring; else fix the generator (of that type) and rebuild."""
    with audit_csv().open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    blank = [r["uid"] for r in rows if r["ok"].strip().lower() not in ("y", "yes", "n", "no")]
    if blank:
        print(f"{len(blank)} audit rows not filled (ok = y or n): {blank[:5]}")
        return 1
    items = read_jsonl(V5 / "claimcheck_built.jsonl")
    # a class removed after the audit (NO_SWAP_GROUPS) is excluded everywhere; its audit rows leave the count
    cls = {rf["uid"] for it in items for rf in it["reference"]["claims"] if excluded_class(rf)}
    rows = [r for r in rows if r["uid"] not in cls]
    bad = {r["uid"] for r in rows if r["ok"].strip().lower() in ("n", "no")}
    per_type = Counter(r["type"] for r in rows if r["uid"] in bad)
    if len(bad) >= 3 or any(v >= 2 for v in per_type.values()):
        print(f"{len(bad)} wrong ground truths in {len(rows)} ({dict(per_type)}): rejected; fix the generator and rebuild")
        return 3
    for it in items:
        for rf in it["reference"]["claims"]:
            rf["excluded"] = rf["uid"] in bad or rf["uid"] in cls
    write_jsonl(V5 / "claimcheck.jsonl", items)
    print(f"accepted; {len(bad)} audited + {len(cls)} class claims excluded from scoring; "
          f"{len(items)} items -> {V5 / 'claimcheck.jsonl'}")
    return 0


def cpqa_subset(per_paper: int = 8) -> Path:
    items = read_jsonl(DATA / "cosmopaperqa.jsonl")
    rng = random.Random(SEED_V5)
    by: dict[str, list[str]] = defaultdict(list)
    for it in items:
        by[it["source_arxiv"]].append(it["id"])
    ids = sorted(i for a in sorted(by) for i in rng.sample(sorted(by[a]), min(per_paper, len(by[a]))))
    path = V5 / "cpqa_subset.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ids": ids, "per_paper": per_paper, "seed": SEED_V5}, indent=1) + "\n")
    print(f"{len(ids)} CosmoPaperQA items -> {path}")
    return path


def report() -> None:
    meta = json.loads((V5 / "meta.json").read_text()) if (V5 / "meta.json").exists() else {}
    print("pool/eligible/drawn:", meta.get("pool"), meta.get("eligible"), meta.get("drawn"))
    drafts = read_jsonl(V5 / "drafts.jsonl")
    cands = {p["arxiv"]: p for p in read_jsonl(V5 / "candidates.jsonl")}
    for d in drafts:   # re-validate with the current rules (render does the same)
        p = cands.get(d["arxiv"])
        if p:
            d["problems"] = {t: (validate_type(t, d["draft"].get(t) or {}, p) if t in p.get("types", TYPES)
                                 else "not usable") for t in TYPES}
    for s in STRATA:
        ds = [d for d in drafts if d["stratum"] == s]
        if not ds:
            continue
        ok = Counter(t for d in ds for t, v in d["problems"].items() if not v)
        why = Counter(v for d in ds for v in d["problems"].values() if v and v != "not usable")
        print(f"{s}: {len(ds)} drafted; valid per type {dict(ok)}; problems {dict(why.most_common(6))}; "
              f"cost ${sum(d.get('cost_usd') or 0 for d in ds):.2f}")
    ver = read_jsonl(V5 / "verified.jsonl")
    for s in STRATA:
        cs = [c for v in ver if v["stratum"] == s for c in v["claims"]]
        if not cs:
            continue
        tot = Counter(c["type"] for c in cs)
        ok = Counter(c["type"] for c in cs if c.get("verified"))
        ans = Counter((c["verifier"] or {}).get("supported") == c["true_is"] for c in cs)
        print(f"{s}: verified {dict(ok)} of {dict(tot)}; picked true {ans[True]}/{len(cs)}; "
              f"cost ${sum(v.get('cost_usd') or 0 for v in ver if v['stratum'] == s):.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["candidates", "draft", "verify", "assemble", "audit", "accept", "cpqa-subset", "report"])
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=1, help="draft/verify: parallel calls (build only, not answers)")
    a = ap.parse_args(argv)
    V5.mkdir(parents=True, exist_ok=True)
    if a.step in ("candidates", "draft", "verify", "assemble"):
        check_pin(a.repo)
    if a.step == "candidates":
        (candidates_k0fresh if FRESH else candidates)(a.repo)
    elif a.step == "draft":
        return draft(a.repo, a.limit, a.workers)
    elif a.step == "verify":
        return verify(a.repo, a.limit, a.workers)
    elif a.step == "assemble":
        assemble(a.repo)
    elif a.step == "audit":
        audit()
    elif a.step == "accept":
        return accept()
    elif a.step == "cpqa-subset":
        cpqa_subset()
    else:
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
