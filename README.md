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

> Status: skeleton. The harness code is being moved in from its first home.

## Install

```bash
git clone https://github.com/corentinravoux/cosmobench
cd cosmobench
pip install -e ".[dev]"      # with test dependencies
```

## Quick start

```python
import cosmobench
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
