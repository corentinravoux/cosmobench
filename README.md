# cosmobench

[![tests](https://github.com/corentinravoux/cosmobench/actions/workflows/tests.yml/badge.svg)](https://github.com/corentinravoux/cosmobench/actions/workflows/tests.yml)
[![docs](https://github.com/corentinravoux/cosmobench/actions/workflows/docs.yml/badge.svg)](https://corentinravoux.github.io/cosmobench/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Benchmark harness measuring whether agent skills (cosmoskill, thcosmoskill) improve LLM answers in cosmology

📖 **Documentation:** <https://corentinravoux.github.io/cosmobench/>

## What it does

cosmobench runs a model headless through Claude Code under several *arms*
(skills off, skills loaded, skills plus a paper-fetching tool, ...) on fixed
question sets, grades the answers (executed code where the answer is computable,
a blind judge otherwise) and reports accuracy and cost per arm. It started as the
cosmoskill benchmark and will be generalised to thcosmoskill. It also runs
against a local model served through `ANTHROPIC_BASE_URL`.

```
question sets ──▶ arms (claude -p, per-arm skills/tools) ──▶ answers
                                                              │
       report (accuracy, cost, isolation checks) ◀── grading ◀┘
```


## Requirements

- Linux with bubblewrap (`bwrap`): every model call runs in a sandbox with `$HOME` hidden,
  and the harness refuses to run without it.
- The Claude Code CLI (`claude`), logged in.
- A checkout of the skill repository under test (cosmoskill).

## Quick start

```bash
python -m cosmobench.data fetch                                # CosmoPaperQA, TPBench -> data/
python -m cosmobench.run --pin --repo ~/Software/cosmoskill    # record the skill commit under test
python -m cosmobench.check_isolation ~/Software/cosmoskill     # any FAIL stops the benchmark
python -m cosmobench.run --bench cosmopaperqa --arms D0,D1A --limit 2 --dry-run
```

The claim-check set is built with
`python -m cosmobench.build_v5 candidates | draft | verify | assemble | audit | accept`.
A local model (llama-server) runs through `scripts/run_local.sh`
(`preflight`, `start`, `check`, `isolation`, `pilot`, `full`, `grade`, `stop`).

## Configuration

| Variable | Default | Sets |
|---|---|---|
| `COSMOBENCH_HOME` | the checkout | root of `data/`, `results/`, `workdir/` |
| `COSMOSKILL_REPO` | `~/Software/cosmoskill` | skill repository under test (or `--repo`) |
| `CLAUDE_CONFIG_DIR` | `~/.claude-alt` | Claude Code config dir of the runs: a second one keeps benchmark sessions out of your main history |
| `CLAUDE_BIN` | `claude` on `PATH` | Claude Code binary |
| `BENCH_RUN`, `BENCH_SET`, `BENCH_CONFIG`, `BENCH_MODEL` | | run name, claim set, pin file, answer model |
| `BENCH_LOCAL_URL`, `BENCH_LOCAL_*` | | local-model mode, see `cosmobench/local.py` |

## Data

`data/`, `results/` and `workdir/` are never committed: they hold reference answers and raw
transcripts. Public datasets are downloaded at run time; claim-check sets are built locally.

## Install

```bash
git clone https://github.com/corentinravoux/cosmobench
cd cosmobench
pip install -e ".[dev]"      # with test dependencies
```

## Tests

```bash
pytest
```

## Release

```bash
uvx bump-my-version bump patch   # or minor / major: commits + tags v<version>
git push --follow-tags
```

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
