#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
SNAPSHOT_ROOT="$HERE/repo_snapshot"
RUNNER="$HERE/gemm_batch_isa_runner.py"
ISA="${AITER_OPT_ISA:-$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_opt.s}"

if [[ ! -f "$SNAPSHOT_ROOT/aiter/__init__.py" ]]; then
  echo "self-contained HEAD snapshot is incomplete: $SNAPSHOT_ROOT" >&2
  exit 2
fi
export PYTHONPATH="$SNAPSHOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export AITER_META_DIR="$SNAPSHOT_ROOT"

COMMON=(
  --isa "$ISA"
  --experts 96
  --tokens 16384
  --topk 6
  --model-dim 7168
  --inter-dim 3072
)

cd "$REPO"
case "${1:-quick-random}" in
  quick-random)
    AITER_LOG_MORE=1 python "$RUNNER" "${COMMON[@]}" \
      --iters 1 --timing-method cuda-event
    ;;
  perf-random)
    AITER_LOG_MORE=1 python "$RUNNER" "${COMMON[@]}" --iters 20
    ;;
  perf-const0)
    AITER_LOG_MORE=1 python "$RUNNER" "${COMMON[@]}" \
      --iters 20 --const-init 0
    ;;
  e2e-random)
    AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 ENABLE_CK=0 \
      FLYDSL_DUMP_IR=0 AITER_LOG_MORE=1 AITER_MOE_EXPERT_BALANCE=true \
      AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
      python3 -u "$HERE/run_e2e_candidate.py" --isa "$ISA" -- \
      --scenario bench --data-format a4w4 --experts 96 --tokens 16384 \
      --topk 6 --iters 20 --model-dim 7168 --inter-dim 3072 --act silu \
      --no-bias --no-check-aot-cache
    ;;
  e2e-const0)
    AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 ENABLE_CK=0 \
      FLYDSL_DUMP_IR=0 AITER_LOG_MORE=1 AITER_MOE_EXPERT_BALANCE=true \
      AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
      python3 -u "$HERE/run_e2e_candidate.py" --isa "$ISA" -- \
      --scenario bench --data-format a4w4 --experts 96 --tokens 16384 \
      --topk 6 --iters 20 --model-dim 7168 --inter-dim 3072 --act silu \
      --no-bias --no-check-aot-cache --const-init 0
    ;;
  *)
    echo "usage: $0 {quick-random|perf-random|perf-const0|e2e-random|e2e-const0}" >&2
    exit 2
    ;;
esac
