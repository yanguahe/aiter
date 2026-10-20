#!/usr/bin/env bash
set -Eeuo pipefail

# Launch this through torchrun --no-python. Keep all EP ranks alive, but attach
# rocprofv3 only to the selected local rank. Every rank runs the same workload.
if [[ -z "${LOCAL_RANK:-}" ]]; then
    echo "Run this wrapper through torchrun --no-python." >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# This untimed runner does not provide the positive median required by the
# optional SMI replay monitor. External collection also needs no concurrent SMI.
export AITER_SMI_MONITOR=0
TRACE_RANK="${AITER_ATT_LOCAL_RANK:-0}"
if [[ ! "$TRACE_RANK" =~ ^[0-9]+$ || -z "${LOCAL_WORLD_SIZE:-}" ]] ||
   ((TRACE_RANK >= LOCAL_WORLD_SIZE)); then
    echo "AITER_ATT_LOCAL_RANK must select a rank in LOCAL_WORLD_SIZE." >&2
    exit 2
fi
PYTHON_BIN="${AITER_ATT_PYTHON:-python3}"
RUNNER="$SCRIPT_DIR/trace_mega_moe_without_profiler.py"

if [[ "$LOCAL_RANK" != "$TRACE_RANK" || "${AITER_ATT_DISABLE:-0}" == 1 ]]; then
    exec "$PYTHON_BIN" -u "$RUNNER" "$@"
fi

: "${AITER_ATT_CONFIG:?Set AITER_ATT_CONFIG to the rocprofv3 YAML path.}"
PROFILER="${AITER_ATT_ROCPROF:-/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/bin/rocprofv3}"
PROFILER_ENV="${AITER_ATT_ENV:-/data/yanguahe/code/wk_sp1/rocprof_env.sh}"
DECODER_DIR="${AITER_ATT_LIBRARY_PATH:-/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/lib}"
if [[ ! -x "$PROFILER" || ! -r "$PROFILER_ENV" || ! -r "$AITER_ATT_CONFIG" ||
      ! -r "$DECODER_DIR/librocprof-trace-decoder.so" ]]; then
    echo "Missing ATT profiler, environment, configuration, or decoder." >&2
    exit 2
fi

source "$PROFILER_ENV"
export ROCPROF_ATT_LIBRARY_PATH="$DECODER_DIR"
# dsr1_ep4 uses the TheRock ROCm 10.1 runtime. Its bundled SDK avoids loading
# the separate ROCm 7.12 profiler SDK alongside PyTorch's runtime copy.
SDK_CORE="${AITER_ATT_SDK_CORE:-$(dirname -- "$(dirname -- "$PROFILER")")}"
if [[ "$SDK_CORE" == */_rocm_sdk_core ]]; then
    export LD_LIBRARY_PATH="/data/yanguahe/code/wk_sp1/comgr_new:$DECODER_DIR:$SDK_CORE/lib:$SDK_CORE/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"
fi
printf '[ATT] local_rank=%s profiler=%s config=%s\n' \
    "$LOCAL_RANK" "$PROFILER" "$AITER_ATT_CONFIG"
if [[ "${AITER_ATT_ENGINE:-dispatch}" == device ]]; then
    : "${AITER_ATT_DEVICE_LIBRARY:?Set the device ATT tool library path.}"
    : "${AITER_ATT_OUTPUT:?Set the device ATT output path.}"
    export ROCPROFILER_LIBRARY_CTOR=1
    export ROCP_TOOL_LIBRARIES="$AITER_ATT_DEVICE_LIBRARY"
    export LD_PRELOAD="$AITER_ATT_DEVICE_LIBRARY:$SDK_CORE/lib/librocprofiler-sdk.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
    exec "$PYTHON_BIN" -u "$RUNNER" "$@"
fi
exec "$PROFILER" -i "$AITER_ATT_CONFIG" -- "$PYTHON_BIN" -u "$RUNNER" "$@"
