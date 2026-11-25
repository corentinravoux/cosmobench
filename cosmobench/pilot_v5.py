"""v5 pilot driver (plan v5, "Execution" step 2). Results go to results/v5/pilot/ (BENCH_RUN=pilot).

    BENCH_RUN=pilot python3 -m cosmobench.pilot_v5 [--dry-run]

D0 on every built item (cheap; V1 and V6 before the gate); D1A on 2 items per stratum (one per
block, blocks 00 and 01); D0F and D1F on the same items of A and B; CosmoPaperQA 4 items x (D0, D1A).
"""
from __future__ import annotations

import os
import subprocess
import sys

from .data import load_items


def pilot_items() -> list[str]:
    items = load_items("claimcheck", built=True)
    out = []
    for s in ("K", "K0", "A", "B"):
        out += [it["id"] for it in items if it["stratum"] == s and it["id"].endswith(("-00a", "-01a"))]
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if os.environ.get("BENCH_RUN") != "pilot":
        raise SystemExit("set BENCH_RUN=pilot so the pilot does not write into the full run's results")
    sel = pilot_items()
    ab = [i for i in sel if i.split("-")[1] in ("A", "B")]
    cpqa = load_items("cosmopaperqa")
    cp = [it["id"] for it in cpqa if it["stratum"] == "covered"][:2] + [it["id"] for it in cpqa if it["stratum"] == "thin"][:2]
    # small steps first, so that a failure of the skills or fetch arms shows before the long D0 pass
    steps = [["--bench", "claimcheck", "--built", "--arms", "D1A", "--items", ",".join(sel)],
             ["--bench", "claimcheck", "--built", "--arms", "D0F,D1F", "--items", ",".join(ab)],
             ["--bench", "cosmopaperqa", "--arms", "D0,D1A", "--items", ",".join(cp)],
             ["--bench", "claimcheck", "--built", "--arms", "D0"]]
    for st in steps:
        cmd = [sys.executable, "-m", "cosmobench.run", *st, *argv]
        print("+", " ".join(cmd[2:]), flush=True)
        rc = subprocess.run(cmd).returncode
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
