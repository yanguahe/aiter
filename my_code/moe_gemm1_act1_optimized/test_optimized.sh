#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
RUNNER="$HERE/gemm_batch_isa_runner.py"
ISA="$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_opt.s"

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
  *)
    echo "usage: $0 {quick-random|perf-random|perf-const0}" >&2
    exit 2
    ;;
esac
