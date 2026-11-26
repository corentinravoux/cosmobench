#!/usr/bin/env bash
# cosmobench v5 on a local model (llama-server started by llm-host). Mode: cosmobench/local.py.
#
#   scripts/run_local.sh preflight     data, pin, binaries, GPU/RAM, unit tests (no model)
#   scripts/run_local.sh start         llm-host start <recipe> (LOADS THE MODEL)
#   scripts/run_local.sh check         server, alias and window match; llm_probe.py
#   scripts/run_local.sh isolation     v5 isolation check in local mode (any FAIL stops)
#   scripts/run_local.sh pilot         1 item per stratum, every arm, in <run>-pilot/, with gates
#   scripts/run_local.sh full          claim check D0, D1A, D0F, D1F, then CosmoPaperQA D0, D1A (resumable)
#   scripts/run_local.sh grade         mechanical claim grading + analysis.md (no judge)
#   scripts/run_local.sh stop          llm-host stop
#   scripts/run_local.sh all           preflight start check isolation pilot full grade stop
#
# CosmoPaperQA judges are Claude and cost money: run them apart, with the owner's OK
# (see the plan). Env overrides: RECIPE, BENCH_RUN, REPO, CLAUDE_BIN, PY (the Python where
# cosmobench is installed), COSMOBENCH_HOME (data/ and results/ root; default: the checkout).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)          # the cosmobench checkout
cd "$ROOT"
BASE=${COSMOBENCH_HOME:-$ROOT}                   # data/ and results/ (never in git)
PY=${PY:-python3}
LLM_HOST=${LLM_HOST:-$HOME/.local/bin/llm-host}
PROBE=${PROBE:-$HOME/.claude/skills/local-llm/scripts/llm_probe.py}
RECIPE=${RECIPE:-cosmobench-qwen36-128k}
REPO=${REPO:-$HOME/Software/cosmoskill-v5pin}   # detached worktree at the v5 pin
PORT=${PORT:-9931}
export BENCH_LOCAL_URL=http://127.0.0.1:$PORT
export BENCH_MODEL=$RECIPE
export BENCH_RUN=${BENCH_RUN:-local-qwen36}
export CLAUDE_BIN=${CLAUDE_BIN:-$HOME/.npm-global/bin/claude}
RES=$BASE/results/v5/$BENCH_RUN
DATA=$BASE/data
MIN_FREE_VRAM_MIB=10900                          # recipe measured 11.1 GB used incl. ~0.4 GB desktop (2026-11-11)
WARN_FREE_VRAM_MIB=11400
MAX_ATTEMPTS=6                                   # run.py stops after 3 consecutive errors; resume this often

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$RES/run.log"; }
die() { log "STOP: $*"; exit 1; }
mkdir -p "$RES"

preflight() {
  local ok=1
  "$PY" -c 'import cosmobench, pyarrow, scipy' 2>/dev/null && log "ok   $PY imports cosmobench" || { log "FAIL $PY cannot import cosmobench: set PY to its Python"; ok=0; }
  for f in "$DATA/v5/claimcheck.jsonl" "$DATA/v5/cpqa_subset.json" "$DATA/cosmopaperqa.jsonl"; do
    [[ -s $f ]] && log "ok   $f" || { log "MISSING $f (copy data/v5/ from the machine that built it)"; ok=0; }
  done
  local pin head
  pin=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cosmoskill_commit"])' "cosmobench/${BENCH_CONFIG:-config.json}")
  head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo none)
  [[ $head == "$pin" ]] && log "ok   $REPO at pin ${pin:0:10}" || { log "FAIL $REPO HEAD ${head:0:10} != pin ${pin:0:10}"; ok=0; }
  [[ -z $(git -C "$REPO" status --porcelain 2>/dev/null) ]] && log "ok   pinned worktree clean" || { log "FAIL $REPO has local changes"; ok=0; }
  log "ok   claude $("$CLAUDE_BIN" --version 2>/dev/null | head -1) ($CLAUDE_BIN)"
  command -v bwrap >/dev/null && log "ok   bwrap" || { log "FAIL bwrap missing"; ok=0; }
  [[ $("$LLM_HOST" list 2>/dev/null) == *$'\n'"$RECIPE "* || $("$LLM_HOST" list 2>/dev/null) == "$RECIPE "* ]] && log "ok   recipe $RECIPE" || { log "FAIL recipe $RECIPE not found"; ok=0; }
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null; then
    log "note a server already answers on :$PORT ($("$LLM_HOST" status 2>/dev/null | head -1))"
  else
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if (( free < MIN_FREE_VRAM_MIB )); then log "FAIL VRAM free ${free} MiB < ${MIN_FREE_VRAM_MIB}: close GPU apps (browser, ComfyUI)"; ok=0
    elif (( free < WARN_FREE_VRAM_MIB )); then log "warn VRAM free ${free} MiB: tight; closing the browser gives margin"
    else log "ok   VRAM free ${free} MiB"; fi
  fi
  log "info RAM available $(free -g | awk '/^Mem/{print $7}') GiB (model ~18 GB: experts beyond the GPU are mmapped; close large apps)"
  env -u BENCH_LOCAL_URL -u BENCH_MODEL -u BENCH_RUN -u BENCH_SET "$PY" -m pytest -q test >/dev/null 2>&1 && log "ok   unit tests" || { log "FAIL unit tests"; ok=0; }
  (( ok )) || die "preflight failed"
}

start() { log "loading $RECIPE"; "$LLM_HOST" start "$RECIPE" --wait 900 | tee -a "$RES/run.log"; }
stop()  { "$LLM_HOST" stop | tee -a "$RES/run.log" || true; }

check() {
  if ! "$PY" -m cosmobench.local check 2>&1 | tee -a "$RES/run.log"; then die "cosmobench.local check failed"; fi
  if [[ -f $PROBE ]]; then
    python3 -I "$PROBE" --url "http://127.0.0.1:$PORT" 2>&1 | tee "$RES/probe.txt" | tail -15 | tee -a "$RES/run.log"
    # the probe ends with "N FAIL, M WARN": a bare `grep FAIL` also matches "0 FAIL"
    grep -qE "^\[FAIL\]|^[1-9][0-9]* FAIL" "$RES/probe.txt" && die "llm_probe reports FAIL (see $RES/probe.txt)" || true
  fi
}

isolation() {
  "$PY" -m cosmobench.check_isolation "$REPO" 2>&1 | tee -a "$RES/run.log" || die "isolation check failed: show $RES/isolation.md to the owner, never loosen a check"
}

# one arm of one bench, resumed after transient errors (server restart if it died)
run_arm() {  # bench arm [extra run.py args]
  local bench=$1 arm=$2; shift 2
  for ((i = 1; i <= MAX_ATTEMPTS; i++)); do
    if [[ $(curl -s -m 5 "http://127.0.0.1:$PORT/health" || true) != *ok* ]]; then
      log "server down before $bench/$arm: restarting $RECIPE"; "$LLM_HOST" start "$RECIPE" --replace --wait 900 | tee -a "$RES/run.log"
    fi
    set +e
    "$PY" -m cosmobench.run --bench "$bench" --arms "$arm" --repo "$REPO" "$@" 2>&1 | tee -a "$RES/run.log"
    local code=${PIPESTATUS[0]}
    set -e
    [[ $code -eq 0 ]] && return 0
    log "$bench/$arm exited $code (attempt $i/$MAX_ATTEMPTS); waiting 60 s"; sleep 60
  done
  die "$bench/$arm still failing after $MAX_ATTEMPTS attempts"
}

pilot() {
  local main=$BENCH_RUN
  export BENCH_RUN=$main-pilot
  RES=$BASE/results/v5/$BENCH_RUN; mkdir -p "$RES"
  for arm in D0 D1A D0F D1F; do run_arm claimcheck "$arm" --per-stratum 1; done
  run_arm cosmopaperqa D0 --limit 2; run_arm cosmopaperqa D1A --limit 2
  "$PY" -m cosmobench.grade_claims | tee -a "$RES/run.log"
  if ! "$PY" -m cosmobench.local pilot-gate "$RES" 2>&1 | tee -a "$RES/run.log"; then
    die "pilot gate failed: read $RES/claimcheck/answers.jsonl before the full run"
  fi
  export BENCH_RUN=$main; RES=$BASE/results/v5/$BENCH_RUN
}

full() {
  for arm in D0 D1A D0F D1F; do run_arm claimcheck "$arm"; done   # one arm at a time: the prompt prefix stays cached
  for arm in D0 D1A; do run_arm cosmopaperqa "$arm"; done
  log "full run done"
}

grade() {
  "$PY" -m cosmobench.grade_claims | tee -a "$RES/run.log"
  "$PY" -m cosmobench.analyse_v5 > /dev/null && log "analysis: $RES/analysis.md (CosmoPaperQA needs the Claude judges first)"
}

case ${1:-} in
  preflight|start|check|isolation|pilot|full|grade|stop) "$1" ;;
  all) preflight; start; check; isolation; pilot; full; grade; stop ;;
  *) sed -n '2,18p' "$0"; exit 2 ;;
esac
