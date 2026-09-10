#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
SNAPSHOT_ROOT="$HERE/repo_snapshot"
MODE="${1:-const0}"
ITERS="${AITER_PERSISTENT_BENCH_ITERS:-20}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="$HERE/history_runs"

if [[ ! -f "$SNAPSHOT_ROOT/aiter/__init__.py" ]]; then
  echo "self-contained HEAD snapshot is incomplete: $SNAPSHOT_ROOT" >&2
  exit 2
fi
export PYTHONPATH="$SNAPSHOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export AITER_META_DIR="$SNAPSHOT_ROOT"

case "$MODE" in
  random|e2e-random)
    DATA_LABEL=random
    CONST_ARGS=()
    ;;
  const0|e2e-const0)
    DATA_LABEL=const0
    CONST_ARGS=(--const-init 0)
    ;;
  *) echo "usage: $0 {random|const0|e2e-random|e2e-const0}" >&2; exit 2 ;;
esac
LOG="$LOG_DIR/${STAMP}_persistent_e2e_${DATA_LABEL}.log"

DOUBLE_LDS="$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_double_lds.s"
PERSISTENT="$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent.s"
PERSISTENT_OVERLAP="$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent_overlap.s"
ISA_SET=("$DOUBLE_LDS" "$PERSISTENT" "$PERSISTENT_OVERLAP")

for isa in "${ISA_SET[@]}"; do
  if [[ ! -f "$isa" ]]; then
    echo "missing ISA: $isa" >&2
    exit 2
  fi
done

COMMON_ENV=(
  AITER_USE_GROUPED_GEMM=1
  AITER_GROUPED_DEBUG=0
  ENABLE_CK=0
  FLYDSL_DUMP_IR=0
  AITER_LOG_MORE=1
  AITER_MOE_EXPERT_BALANCE=true
  AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1
  AITER_FLYDSL_GEMM1_A_PRESHUFFLE=1
)

COMMON_ARGS=(
  --scenario bench
  --data-format a4w4
  --experts 96
  --tokens 16384
  --topk 6
  --iters "$ITERS"
  --model-dim 7168
  --inter-dim 3072
  --act silu
  --no-bias
  --no-check-aot-cache
  "${CONST_ARGS[@]}"
)

run_one() {
  local label="$1"
  local isa="$2"
  shift 2
  echo "===== $label ====="
  env "${COMMON_ENV[@]}" python3 -u "$HERE/run_e2e_candidate.py" \
    --isa "$isa" "$@" -- "${COMMON_ARGS[@]}"
}

mkdir -p "$LOG_DIR"
cd "$REPO"
{
  echo "timestamp_utc=$(date -u --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "mode=$DATA_LABEL"
  echo "canonical_metric=MoE e2e profiler gemm1 device_time_avg"
  echo "iterations=$ITERS"
  sha256sum "${ISA_SET[@]}" "$HERE/moe_gemm1_cpp_launcher_persistent.cpp"
  if command -v rocm-smi >/dev/null 2>&1; then
    timeout 20 rocm-smi --showproductname --showclocks --showuse --showmemuse || true
  fi
  run_one double_lds \
    "$DOUBLE_LDS"
  run_one persistent \
    "$PERSISTENT" \
    --grid-x 16 --grid-y 16
  run_one persistent_overlap \
    "$PERSISTENT_OVERLAP" \
    --grid-x 16 --grid-y 16
} 2>&1 | tee "$LOG"

echo "persistent benchmark log: $LOG"
