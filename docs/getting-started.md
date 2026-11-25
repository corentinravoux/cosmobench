# Getting started

## Install

```bash
git clone https://github.com/corentinravoux/cosmobench
cd cosmobench
pip install -e ".[dev]"
```

Linux with bubblewrap (`bwrap`) and a logged-in Claude Code CLI are required: every model
call runs in a sandbox with `$HOME` hidden.

## A first run

```bash
python -m cosmobench.data fetch                                # public datasets -> data/
python -m cosmobench.run --pin --repo ~/Software/cosmoskill    # skill commit under test
python -m cosmobench.check_isolation ~/Software/cosmoskill     # any FAIL stops the benchmark
python -m cosmobench.run --bench cosmopaperqa --arms D0,D1A --limit 2 --dry-run
```

Drop `--dry-run` to make the calls. Results go to `results/v5/`, outside git.
