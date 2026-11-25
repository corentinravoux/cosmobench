"""Fetch and normalise the benchmark datasets.

    python -m cosmobench.data fetch      download both parquet files, write data/<bench>.jsonl
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from .common import CLAIM_SET, DATA, read_jsonl, write_jsonl

SOURCES = {
    "cosmopaperqa": ("ASTROANTS/CosmoPaperQA", "CosmoPaperQA.parquet"),
    "tpbench": ("ZhiqiGao/TPBench", "data/public-00000-of-00001.parquet"),
}
COVERED = ("1807.06209", "2503.14454")  # Planck 2018 parameters, ACT DR6: heavily cited in the skills
TPB_OTHERS = 2  # non-cosmology control problems kept, in file order


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def arxiv_id(text: str | None) -> str:
    m = re.search(r"(\d{4}\.\d{4,5})", text or "")
    return m.group(1) if m else ""


def normalise_cosmopaperqa(table) -> list[dict]:
    items = []
    for i, r in enumerate(table.to_pylist()):
        src = arxiv_id(r.get("source file"))
        items.append({
            "id": f"cpqa-{i:03d}",
            "bench": "cosmopaperqa",
            "prompt_fields": {"question": r["question"]},
            "reference": {
                "ideal": r.get("ideal") or "",
                "key_passage": r.get("key passage") or "",
                "source": r.get("source file") or "",
                "location": r.get("Location") or "",
                "doi": r.get("doi") or "",
            },
            "domain": src,
            "source_arxiv": src,
            "stratum": "covered" if src in COVERED else "thin",
        })
    return items


def normalise_tpbench(table) -> list[dict]:
    items = []
    for r in table.to_pylist():
        items.append({
            "id": "tpb-" + slug(r["problem_id"]),
            "bench": "tpbench",
            "prompt_fields": {
                "problem": r["problem"],
                "code_answer_requirements": r["code_answer_requirements"],
            },
            "reference": {
                "answer": r.get("answer") or "",
                "reference_implementation": r["reference_implementation"],
            },
            "domain": r["domain"],
        })
    return items


def tpbench_subset(items: list[dict]) -> list[str]:
    """All cosmology problems plus the first TPB_OTHERS other problems, in file order."""
    cosmo = [it["id"] for it in items if it["domain"] == "Cosmology"]
    others = [it["id"] for it in items if it["domain"] != "Cosmology"][:TPB_OTHERS]
    keep = set(cosmo + others)
    return [it["id"] for it in items if it["id"] in keep]


def load_items(bench: str, data_dir: Path = DATA, subset: bool = True, built: bool = False) -> list[dict]:
    """v5: the claim check lives in data/v5/ (claimcheck.jsonl after the audit, claimcheck_built.jsonl
    before it, for the pilot); CosmoPaperQA is cut to data/v5/cpqa_subset.json when it exists."""
    if bench == "claimcheck":
        return read_jsonl(Path(data_dir) / CLAIM_SET / ("claimcheck_built.jsonl" if built else "claimcheck.jsonl"))
    items = read_jsonl(Path(data_dir) / f"{bench}.jsonl")
    if bench == "cosmopaperqa" and subset:
        sub = Path(data_dir) / "v5" / "cpqa_subset.json"
        if sub.exists():
            keep = set(json.loads(sub.read_text())["ids"])
            items = [it for it in items if it["id"] in keep]
    if bench == "tpbench" and subset:
        sub = Path(data_dir) / "tpbench_subset.json"
        if sub.exists():
            keep = set(json.loads(sub.read_text())["ids"])
            items = [it for it in items if it["id"] in keep]
    return items


def _download(repo: str, file: str, dest: Path) -> None:
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{file}"
    req = urllib.request.Request(url, headers={"User-Agent": "cosmoskill-bench"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())


def fetch(data_dir: Path = DATA) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    sums = []
    for bench, (repo, file) in SOURCES.items():
        dest = data_dir / f"{bench}.parquet"
        if not dest.exists():
            _download(repo, file, dest)
        sums.append(f"{hashlib.sha256(dest.read_bytes()).hexdigest()}  {dest.name}")
        table = pq.read_table(dest)
        items = normalise_cosmopaperqa(table) if bench == "cosmopaperqa" else normalise_tpbench(table)
        write_jsonl(data_dir / f"{bench}.jsonl", items)
        print(f"{bench}: {len(items)} items")
        if bench == "tpbench":
            ids = tpbench_subset(items)
            (data_dir / "tpbench_subset.json").write_text(json.dumps({"ids": ids}, indent=1) + "\n")
            print(f"tpbench subset: {len(ids)} ({', '.join(ids)})")
    (data_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["fetch"]:
        fetch()
    else:
        sys.exit(__doc__)
