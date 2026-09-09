#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
CLANG="${AITER_GFX1250_CLANG:-/data/yanguahe/code/wk_sp1/llvm-project/mlir_install/bin/clang}"
CLANG_RUNTIME_LIB="${AITER_GFX1250_CLANG_RUNTIME_LIB:-/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/rocm_sysdeps/lib}"
SYMBOL=moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1
STAMP="$(date -u +%Y%m%d_%H%M%S)"
TRACE_ROOT_REL="my_code/moe_gemm1_act1_optimized/att_history/$STAMP"
TRACE_ROOT="$REPO/$TRACE_ROOT_REL"
mkdir -p "$TRACE_ROOT/code_objects"

BASELINE="$HERE/baseline_act1_independent.s"
OPT_V1="$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_opt.s"
OPT_DOUBLE_LDS="$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_double_lds.s"

declare -a TAGS=(baseline optimized_v1 double_lds)
declare -a SOURCES=("$BASELINE" "$OPT_V1" "$OPT_DOUBLE_LDS")
if [[ -n "${AITER_HISTORY_CANDIDATE:-}" ]]; then
  TAGS+=(candidate)
  SOURCES+=("$(realpath "$AITER_HISTORY_CANDIDATE")")
fi

compile_code_object() {
  local source="$1"
  local tag="$2"
  local object="$TRACE_ROOT/code_objects/${tag}.o"
  local code_object="$TRACE_ROOT/code_objects/${tag}.co"
  LD_LIBRARY_PATH="$CLANG_RUNTIME_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$CLANG" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 \
      -mcode-object-version=6 -c "$source" -o "$object" || return
  LD_LIBRARY_PATH="$CLANG_RUNTIME_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$CLANG" -target amdgcn-amd-amdhsa -mcpu=gfx1250 \
      -mcode-object-version=6 -nostdlib -Wl,--no-undefined -shared \
      "$object" -o "$code_object" || return
  [[ -s "$code_object" ]] || return 1
  printf '%s' "$code_object"
}

cd "$REPO"
echo "timestamp_utc=$(date -u --iso-8601=seconds)"
echo "host=$(hostname)"
echo "git_head=$(git rev-parse HEAD)"

for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"
  source="${SOURCES[$i]}"
  echo "===== ATT $tag ====="
  sha256sum "$source"
  code_object="$(compile_code_object "$source" "$tag")"
  AITER_ATT_CODE_OBJECT="$code_object" \
  TRACE_ROOT="$TRACE_ROOT_REL" \
  HIP_VISIBLE_DEVICES=0 \
    bash my_code/get_isa_runner_att.sh \
      "$SYMBOL" \
      "$tag" \
      "python my_code/moe_gemm1_act1_optimized/att_launch_opt.py" \
      --ana-att
done

echo "ATT history root: $TRACE_ROOT"
