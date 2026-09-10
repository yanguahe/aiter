#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
SNAPSHOT_ROOT="$HERE/repo_snapshot"
CLANG="${AITER_GFX1250_CLANG:-/data/yanguahe/code/wk_sp1/llvm-project/mlir_install/bin/clang}"
CLANG_RUNTIME_LIB="${AITER_GFX1250_CLANG_RUNTIME_LIB:-/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/rocm_sysdeps/lib}"
SYMBOL=moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1
VALIDATE_ONLY="${AITER_ATT_VALIDATE_ONLY:-0}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
TRACE_ROOT_REL="my_code/moe_gemm1_act1_optimized/att_history/$STAMP"
TRACE_ROOT="$REPO/$TRACE_ROOT_REL"
mkdir -p "$TRACE_ROOT/code_objects"

if [[ ! -f "$SNAPSHOT_ROOT/aiter/__init__.py" || \
      ! -f "$SNAPSHOT_ROOT/SOURCE_COMMIT" ]]; then
  echo "self-contained HEAD snapshot is incomplete: $SNAPSHOT_ROOT" >&2
  echo "run on the host/local checkout: python $HERE/sync_head_repo_snapshot.py" >&2
  exit 2
fi
export PYTHONPATH="$SNAPSHOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export AITER_META_DIR="$SNAPSHOT_ROOT"

BASELINE="$HERE/baseline_act1_independent.s"
OPT_V1="$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_opt.s"
OPT_DOUBLE_LDS="$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_double_lds.s"
OPT_PERSISTENT="$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent.s"
OPT_PERSISTENT_OVERLAP="$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent_overlap.s"

declare -a TAGS=(baseline optimized_v1 double_lds persistent persistent_overlap)
declare -a SOURCES=(
  "$BASELINE"
  "$OPT_V1"
  "$OPT_DOUBLE_LDS"
  "$OPT_PERSISTENT"
  "$OPT_PERSISTENT_OVERLAP"
)
declare -a GRID_X=("" "" "" 16 16)
declare -a GRID_Y=("" "" "" 16 16)
if [[ -n "${AITER_HISTORY_CANDIDATE:-}" ]]; then
  TAGS+=(candidate)
  SOURCES+=("$(realpath "$AITER_HISTORY_CANDIDATE")")
  GRID_X+=("${AITER_HISTORY_CANDIDATE_GRID_X:-}")
  GRID_Y+=("${AITER_HISTORY_CANDIDATE_GRID_Y:-}")
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
echo "snapshot_commit=$(cat "$SNAPSHOT_ROOT/SOURCE_COMMIT")"
echo "repo_snapshot=$SNAPSHOT_ROOT"
echo "validate_only=$VALIDATE_ONLY"

for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"
  source="${SOURCES[$i]}"
  grid_x="${GRID_X[$i]}"
  grid_y="${GRID_Y[$i]}"
  echo "===== ATT $tag ====="
  sha256sum "$source"
  code_object="$(compile_code_object "$source" "$tag")"
  if [[ "$VALIDATE_ONLY" == "1" ]]; then
    AITER_ATT_CODE_OBJECT="$code_object" \
    AITER_ATT_GRID_X="$grid_x" \
    AITER_ATT_GRID_Y="$grid_y" \
    HIP_VISIBLE_DEVICES=0 \
      python my_code/moe_gemm1_act1_optimized/att_launch_opt.py
  else
    AITER_ATT_CODE_OBJECT="$code_object" \
    AITER_ATT_GRID_X="$grid_x" \
    AITER_ATT_GRID_Y="$grid_y" \
    TRACE_ROOT="$TRACE_ROOT_REL" \
    HIP_VISIBLE_DEVICES=0 \
      bash my_code/get_isa_runner_att.sh \
        "$SYMBOL" \
        "$tag" \
        "python my_code/moe_gemm1_act1_optimized/att_launch_opt.py" \
        --ana-att
  fi
done

echo "ATT history root: $TRACE_ROOT"
