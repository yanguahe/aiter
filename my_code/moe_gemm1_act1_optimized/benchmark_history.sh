#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
SNAPSHOT_ROOT="$HERE/repo_snapshot"
COMPARE="$HERE/compare_asm_variants.py"
E2E="$HERE/run_e2e_candidate.py"

if [[ ! -f "$SNAPSHOT_ROOT/aiter/__init__.py" || \
      ! -f "$SNAPSHOT_ROOT/op_tests/test_flydsl_grouped_gemm_gfx1250.py" || \
      ! -f "$SNAPSHOT_ROOT/SOURCE_COMMIT" ]]; then
  echo "self-contained HEAD snapshot is incomplete: $SNAPSHOT_ROOT" >&2
  echo "run on the host/local checkout: python $HERE/sync_head_repo_snapshot.py" >&2
  exit 2
fi
export PYTHONPATH="$SNAPSHOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export AITER_META_DIR="$SNAPSHOT_ROOT"

# These three files are the stable comparison chain.  Keep their order fixed so
# compare_asm_variants.py alternates launch order fairly between timing rounds.
BASELINE="$HERE/baseline_act1_independent.s"
OPT_V1="$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_opt.s"
OPT_DOUBLE_LDS="$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_double_lds.s"
OPT_PERSISTENT="$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent.s"
OPT_PERSISTENT_OVERLAP="$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent_overlap.s"

ISA_SET=("$BASELINE" "$OPT_V1" "$OPT_DOUBLE_LDS")
PERSISTENT_ISA_SET=("$OPT_PERSISTENT" "$OPT_PERSISTENT_OVERLAP")
if [[ -n "${AITER_HISTORY_CANDIDATE:-}" ]]; then
  ISA_SET+=("$(realpath "$AITER_HISTORY_CANDIDATE")")
fi
ALL_ISA_SET=("${ISA_SET[@]}" "${PERSISTENT_ISA_SET[@]}")

# The acceptance metric for this task is GEMM1's profiler time inside the full
# MoE const0 pipeline.  Standalone modes remain available for fast diagnosis.
MODE="${1:-e2e-const0}"
SEED="${AITER_HISTORY_SEED:-0}"
WARMUP="${AITER_HISTORY_WARMUP:-5}"
ROUNDS="${AITER_HISTORY_ROUNDS:-9}"
LAUNCHES="${AITER_HISTORY_LAUNCHES_PER_SAMPLE:-5}"
E2E_ITERS="${AITER_HISTORY_E2E_ITERS:-20}"
RUN_REFERENCE_COMMAND="${AITER_HISTORY_RUN_REFERENCE_COMMAND:-1}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="$HERE/history_runs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${STAMP}_${MODE}.log"

for path in "${ALL_ISA_SET[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing ISA: $path" >&2
    exit 2
  fi
done

read_git_head_without_git() {
  if [[ -n "${AITER_HISTORY_GIT_HEAD:-}" ]]; then
    printf '%s\n' "$AITER_HISTORY_GIT_HEAD"
    return
  fi
  cat "$SNAPSHOT_ROOT/SOURCE_COMMIT"
}

print_context() {
  echo "timestamp_utc=$(date -u --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "mode=$MODE"
  echo "canonical_metric=e2e const0 profiler gemm1 device_time_avg"
  echo "seed=$SEED warmup=$WARMUP rounds=$ROUNDS launches_per_sample=$LAUNCHES"
  echo "snapshot_commit=$(read_git_head_without_git)"
  echo "repo_snapshot=$SNAPSHOT_ROOT"
  echo "safe_baseline=$BASELINE"
  echo "historical_515us_reference=$OPT_V1"
  echo "double_lds_baseline=$OPT_DOUBLE_LDS"
  echo "persistent_full_drain=$OPT_PERSISTENT"
  echo "current_candidate=$OPT_PERSISTENT_OVERLAP"
  printf '%s\n' "${ALL_ISA_SET[@]}" | xargs sha256sum
  if command -v rocm-smi >/dev/null 2>&1; then
    timeout 20 rocm-smi --showproductname --showclocks --showuse --showmemuse || true
  fi
}

run_standalone() {
  local const_args=()
  if [[ "$1" == "const0" ]]; then
    const_args=(--const-init 0)
  fi
  python "$COMPARE" "${ISA_SET[@]}" \
    --seed "$SEED" \
    --warmup "$WARMUP" \
    --rounds "$ROUNDS" \
    --launches-per-sample "$LAUNCHES" \
    --validation-repeats 3 \
    "${const_args[@]}"
  echo "===== persistent-grid standalone comparison: $1 ====="
  python "$COMPARE" "${PERSISTENT_ISA_SET[@]}" \
    --grid-x 16 \
    --grid-y 16 \
    --seed "$SEED" \
    --warmup "$WARMUP" \
    --rounds "$ROUNDS" \
    --launches-per-sample "$LAUNCHES" \
    --validation-repeats 3 \
    "${const_args[@]}"
}

run_e2e_one() {
  local isa="$1"
  local data="$2"
  local grid_x="${3:-}"
  local grid_y="${4:-}"
  local const_args=()
  local grid_args=()
  local grid_label=standard
  if [[ "$data" == "const0" ]]; then
    const_args=(--const-init 0)
  fi
  if [[ -n "$grid_x" || -n "$grid_y" ]]; then
    if [[ -z "$grid_x" || -z "$grid_y" ]]; then
      echo "persistent grid requires both X and Y" >&2
      exit 2
    fi
    grid_args=(--grid-x "$grid_x" --grid-y "$grid_y")
    grid_label="${grid_x}x${grid_y}"
  fi
  echo "===== e2e $(basename "$isa") data=$data grid=$grid_label ====="
  AITER_USE_GROUPED_GEMM=1 \
  AITER_GROUPED_DEBUG=0 \
  ENABLE_CK=0 \
  FLYDSL_DUMP_IR=0 \
  AITER_LOG_MORE=1 \
  AITER_MOE_EXPERT_BALANCE=true \
  AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
    python3 -u "$E2E" --isa "$isa" "${grid_args[@]}" -- \
      --scenario bench \
      --data-format a4w4 \
      --experts 96 \
      --tokens 16384 \
      --topk 6 \
      --iters "$E2E_ITERS" \
      --model-dim 7168 \
      --inter-dim 3072 \
      --act silu \
      --no-bias \
      --no-check-aot-cache \
      "${const_args[@]}"
}

run_all() {
  print_context
  case "$MODE" in
    quick-random)
      WARMUP=1 ROUNDS=1 LAUNCHES=1 run_standalone random
      ;;
    perf-random)
      run_standalone random
      ;;
    perf-const0)
      if [[ "$RUN_REFERENCE_COMMAND" == "1" ]]; then
        echo "===== exact historical command: test_optimized.sh perf-const0 ====="
        AITER_OPT_ISA="$OPT_V1" bash "$HERE/test_optimized.sh" perf-const0
      fi
      echo "===== interleaved history comparison: const0 ====="
      run_standalone const0
      ;;
    perf-both)
      echo "===== standalone random ====="
      run_standalone random
      echo "===== standalone const0 ====="
      run_standalone const0
      ;;
    e2e-random|e2e-const0)
      local data="${MODE#e2e-}"
      for isa in "${ISA_SET[@]}"; do
        run_e2e_one "$isa" "$data"
      done
      for isa in "${PERSISTENT_ISA_SET[@]}"; do
        run_e2e_one "$isa" "$data" 16 16
      done
      ;;
    e2e-both)
      for data in random const0; do
        for isa in "${ISA_SET[@]}"; do
          run_e2e_one "$isa" "$data"
        done
        for isa in "${PERSISTENT_ISA_SET[@]}"; do
          run_e2e_one "$isa" "$data" 16 16
        done
      done
      ;;
    *)
      echo "usage: $0 {quick-random|perf-random|perf-const0|perf-both|e2e-random|e2e-const0|e2e-both}" >&2
      exit 2
      ;;
  esac
}

cd "$REPO"
run_all 2>&1 | tee "$LOG"
echo "history log: $LOG"
