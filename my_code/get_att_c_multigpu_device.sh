#!/usr/bin/env bash
set -Eeuo pipefail

# Method C: device-based ATT for multi-GPU torchrun workloads (Mega MoE EP).
# Every rank runs the original workload; only the selected local rank preloads
# the device ATT tool. No kernel-dispatch interception is configured, so the
# Mori/P2P queues are left untouched. trace_mega_moe_without_profiler.py
# disables Kineto and brackets one graph replay (or one eager K2048 GEMM2).

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
TRACE_ROOT="${TRACE_ROOT:-my_code/thread_trace_runs}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/data/yanguahe/code/wk_sp1}"
ATT_TIMEOUT_SECONDS="${AITER_ATT_TIMEOUT_SECONDS:-1800}"
IDLE_VRAM_MB="${AITER_ATT_IDLE_VRAM_MB:-256}"
IDLE_POLL_SECONDS=6

usage() {
    cat >&2 <<EOF
Collect inside the multi-GPU container (for example dsr1_ep4):
  ${SCRIPT_NAME} <KERNEL_NAME> <output-dir-name> "<BENCH_ARGS>" [options]

  KERNEL_NAME  complete kernel symbol, matched exactly and mapped to ISA after
               capture. It must be a function symbol of a code object loaded by the
               traced rank; this is checked right after the first capture, which
               stops the run and lists candidate symbols otherwise. The device trace
               records every kernel that runs on the traced rank inside the region
               (complete names in logs/executed_kernels_simd<N>.txt).
  BENCH_ARGS   single-line arguments for op_tests/multigpu_tests/bench_mega_moe.py.
               Workload environment variables (MORI_*, AITER_*, FLYDSL_*) are
               inherited from the caller.

Options:
  --all-simd       capture SIMD0..3 in four independent runs (default: SIMD0)
  --simd N         capture only SIMD N (0..3)
  --region R       graph: one original graph replay after warmup (default)
                   gemm2: one extra eager layer, bracket only its K2048 GEMM2
  --rank N         local rank that loads the device ATT tool (default: 0)
  --nproc N        torchrun --nproc_per_node (default: 4)
  --se-mask HEX    shader-engine mask (default: 0xf)
  --no-wait-idle   do not wait for all GPUs to be idle before each capture
  --no-analyze     decode only; skip the PC-to-ISA mapping of KERNEL_NAME
  --ana-att        accepted for get_isa_runner_att.sh compatibility (analysis is on by default)
  --archive        also write <TRACE_ROOT>/<output-dir-name>.tar.gz

Environment overrides:
  TRACE_ROOT (my_code/thread_trace_runs), AITER_ATT_TIMEOUT_SECONDS (1800),
  AITER_ATT_IDLE_VRAM_MB (256), AITER_ATT_SDK_CORE, AITER_ATT_SDK_DEVEL,
  AITER_ATT_LIBRARY_PATH, AITER_ATT_COMGR_DIR, AITER_ATT_OBJDUMP, AITER_ATT_TOOL_DIR

Example (dsr1_ep4, /app/aiter_moe_a4w4_pr_refactor_wt):
  AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 MORI_V2_KERNEL_BACKEND=hip \\
  MORI_EP_TOKOFF_EXT=0 MEGA_DISPATCH=mori FLYDSL_DUMP_IR=0 MORI_GPU_ARCHS=gfx1250 \\
  FLYDSL_GPU_ARCH=gfx1250 AITER_FORCE_A8W4=0 AITER_MOE_EXPERT_BALANCE=True \\
  bash my_code/${SCRIPT_NAME} \\
    a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist_epscatter \\
    ep4_gemm2_fused_att \\
    "-e 256 -k 8 -hd 7168 -id 2048 --layers 16 -tpr 1536 -q a4w4_mxfp4 --combine fused --warmup 5 --iters 2 --prof_replays 0 --acc_verify 0 --profile_table 0 --save_trace 0 --dispatch_wire fp4" \\
    --all-simd
EOF
}

die() {
    echo "${SCRIPT_NAME}: $*" >&2
    exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ "$#" -lt 3 ]]; then
    usage
    exit 1
fi

KERNEL_NAME="$1"
OUTPUT_DIR_NAME="$2"
BENCH_ARGS="$3"
shift 3

SIMD_IDS=(0)
REGION=graph
TRACE_RANK=0
NPROC=4
SE_MASK=0xf
WAIT_IDLE=1
ANALYZE=1
ARCHIVE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all-simd) SIMD_IDS=(0 1 2 3); shift ;;
        --simd) [[ "${2:-}" =~ ^[0-3]$ ]] || die "--simd needs 0..3"; SIMD_IDS=("$2"); shift 2 ;;
        --region) [[ "${2:-}" =~ ^(graph|gemm2)$ ]] || die "--region needs graph or gemm2"; REGION="$2"; shift 2 ;;
        --rank) [[ "${2:-}" =~ ^[0-9]+$ ]] || die "--rank needs an integer"; TRACE_RANK="$2"; shift 2 ;;
        --nproc) [[ "${2:-}" =~ ^[1-9][0-9]*$ ]] || die "--nproc needs a positive integer"; NPROC="$2"; shift 2 ;;
        --se-mask) [[ "${2:-}" =~ ^0x[0-9A-Fa-f]+$ ]] || die "--se-mask needs a hexadecimal value"; SE_MASK="$2"; shift 2 ;;
        --no-wait-idle) WAIT_IDLE=0; shift ;;
        --no-analyze) ANALYZE=0; shift ;;
        --ana-att) ANALYZE=1; shift ;;
        --archive) ARCHIVE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; die "unexpected argument: $1" ;;
    esac
done

[[ "$KERNEL_NAME" =~ ^[A-Za-z_.$][A-Za-z0-9_.$@:+-]*$ ]] || die "KERNEL_NAME must be a complete C/AMDGPU-style kernel symbol"
[[ "$OUTPUT_DIR_NAME" =~ ^[A-Za-z0-9_][A-Za-z0-9._-]*$ ]] || die "output-dir-name may only use letters, numbers, dot, underscore, dash"
[[ -n "$BENCH_ARGS" && "$BENCH_ARGS" != *$'\n'* ]] || die "BENCH_ARGS must be a non-empty single line"
((TRACE_RANK < NPROC)) || die "--rank must be smaller than --nproc"
read -r -a BENCH_ARGV <<< "$BENCH_ARGS"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_DIR="$TRACE_ROOT/$OUTPUT_DIR_NAME"
OUTPUT_ARCHIVE="$TRACE_ROOT/$OUTPUT_DIR_NAME.tar.gz"
LOG_DIR="$OUTPUT_DIR/logs"
[[ ! -e "$OUTPUT_DIR" ]] || die "$OUTPUT_DIR already exists; choose a new output-dir-name"
mkdir -p "$LOG_DIR"
SUMMARY_LOG="$OUTPUT_DIR/summary.log"

exec 9>"/tmp/aiter_att_c_${UID}.lock"
flock -n 9 || die "another ${SCRIPT_NAME} capture holds /tmp/aiter_att_c_${UID}.lock"

# ---- toolchain: profiler, decoder and register library from one ROCm SDK ----
find_sdk_dir() {
    local module="$1" override="$2" candidate
    if [[ -n "$override" ]]; then
        printf '%s\n' "$override"
        return 0
    fi
    candidate="$(python3 -c "import importlib.util, os, sys
spec = importlib.util.find_spec(sys.argv[1])
print(os.path.dirname(spec.origin) if spec and spec.origin else '')" "$module" 2>/dev/null || true)"
    if [[ -n "$candidate" && -d "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
    fi
    for candidate in "/usr/local/lib/python3.12/dist-packages/$module" "/opt/venv/lib/python3.12/site-packages/$module"; do
        if [[ -d "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

SDK_CORE="$(find_sdk_dir _rocm_sdk_core "${AITER_ATT_SDK_CORE:-}")" || die "cannot locate _rocm_sdk_core"
SDK_DEVEL="$(find_sdk_dir _rocm_sdk_devel "${AITER_ATT_SDK_DEVEL:-}")" || die "cannot locate _rocm_sdk_devel"
DECODER_DIR="${AITER_ATT_LIBRARY_PATH:-$SDK_DEVEL/lib}"
COMGR_DIR="${AITER_ATT_COMGR_DIR:-$WORKSPACE_DIR/comgr_new}"
OBJDUMP="${AITER_ATT_OBJDUMP:-$WORKSPACE_DIR/llvm-project/mlir_install/bin/llvm-objdump}"
TOOL_DIR="${AITER_ATT_TOOL_DIR:-$REPO_ROOT/my_code/.att_device_tools}"
AGENT_SO="$TOOL_DIR/libtrace_mega_moe_agent.so"
DECODER_BIN="$TOOL_DIR/decode_mega_moe_agent"
SDK_LIB_PATH="$DECODER_DIR:$SDK_CORE/lib:$SDK_CORE/lib/rocm_sysdeps/lib"
[[ -d "$COMGR_DIR" ]] && SDK_LIB_PATH="$COMGR_DIR:$SDK_LIB_PATH"

[[ -x "$SDK_CORE/bin/rocprofv3" ]] || die "missing $SDK_CORE/bin/rocprofv3"
[[ -r "$SDK_CORE/lib/librocprofiler-sdk.so.1" ]] || die "missing $SDK_CORE/lib/librocprofiler-sdk.so.1"
[[ -r "$DECODER_DIR/librocprof-trace-decoder.so" ]] || die "missing $DECODER_DIR/librocprof-trace-decoder.so"
for required in trace_mega_moe_rank.sh trace_mega_moe_without_profiler.py \
    trace_mega_moe_agent.cpp decode_mega_moe_agent.cpp analyze_mega_moe_device_att.py; do
    [[ -r "$SCRIPT_DIR/$required" ]] || die "missing my_code/$required"
done
[[ -x "$OBJDUMP" ]] || die "missing gfx1250 llvm-objdump: $OBJDUMP"

AITER_FILE="$(python3 -c 'import importlib.util; print(importlib.util.find_spec("aiter").origin)' 2>/dev/null || true)"
[[ "$AITER_FILE" == "$REPO_ROOT/"* ]] || die "python resolves aiter to '$AITER_FILE', not $REPO_ROOT"

build_tools() {
    mkdir -p "$TOOL_DIR"
    if [[ ! -x "$AGENT_SO" || "$SCRIPT_DIR/trace_mega_moe_agent.cpp" -nt "$AGENT_SO" ]]; then
        g++ -std=c++17 -O2 -D__HIP_PLATFORM_AMD__ -fPIC -shared \
            "$SCRIPT_DIR/trace_mega_moe_agent.cpp" -I "$SDK_DEVEL/include" \
            "$SDK_CORE/lib/librocprofiler-sdk.so.1" -Wl,-rpath,"$SDK_CORE/lib" -o "$AGENT_SO"
    fi
    if [[ ! -x "$DECODER_BIN" || "$SCRIPT_DIR/decode_mega_moe_agent.cpp" -nt "$DECODER_BIN" ]]; then
        g++ -std=c++17 -O2 -D__HIP_PLATFORM_AMD__ \
            "$SCRIPT_DIR/decode_mega_moe_agent.cpp" -I "$SDK_DEVEL/include" \
            "$SDK_CORE/lib/librocprofiler-sdk.so.1" -Wl,-rpath,"$SDK_CORE/lib" -o "$DECODER_BIN"
    fi
    chmod a+rx "$AGENT_SO" "$DECODER_BIN"
}

gpu_idle_snapshot() {
    python3 - "$IDLE_VRAM_MB" <<'PY'
import json
import sys
from pathlib import Path

limit = int(sys.argv[1]) * 1024 * 1024
kfd = Path("/sys/class/kfd/kfd/proc")
pids = sorted(p.name for p in kfd.iterdir()) if kfd.is_dir() else ["<unreadable>"]
gpus = []
for device in sorted(Path("/sys/class/drm").glob("card*/device")):
    busy, vram = device / "gpu_busy_percent", device / "mem_info_vram_used"
    if busy.is_file() and vram.is_file():
        gpus.append({"card": device.parent.name, "busy": int(busy.read_text()),
                     "vram_mib": int(vram.read_text()) // 2**20})
print(json.dumps({"kfd_pids": pids, "gpus": gpus}))
idle = not pids and gpus and all(g["busy"] == 0 and g["vram_mib"] * 2**20 < limit for g in gpus)
raise SystemExit(0 if idle else 1)
PY
}

wait_gpu_idle() {
    local log_file="$1"
    while true; do
        printf '%s ' "$(date -u +%FT%TZ)" | tee -a "$log_file"
        if gpu_idle_snapshot 2>&1 | tee -a "$log_file"; then
            return 0
        fi
        echo "GPUs busy; rechecking in ${IDLE_POLL_SECONDS}s" | tee -a "$log_file"
        sleep "$IDLE_POLL_SECONDS"
    done
}

write_placeholder_yaml() {
    # trace_mega_moe_rank.sh requires a readable YAML; the device engine ignores it.
    cat >"$1" <<YAML
jobs:
  - kernel_include_regex: '^${KERNEL_NAME}\$'
    advanced_thread_trace: true
    att_target_cu: 1
    att_shader_engine_mask: "${SE_MASK}"
    att_simd_select: "$2"
    att_buffer_size: "0x10000000"
    att_library_path: ["${DECODER_DIR}"]
YAML
}

capture_simd() {
    local simd="$1"
    local capture_dir="$OUTPUT_DIR/thread_trace/simd${simd}/kernel/rpf_v3"
    local log_file="$LOG_DIR/02_capture_simd${simd}.log"
    local status
    mkdir -p "$capture_dir/device"
    write_placeholder_yaml "$capture_dir/att.yaml" "$simd"
    if ((WAIT_IDLE)); then
        wait_gpu_idle "$LOG_DIR/gpu_idle_before_simd${simd}.log"
    fi
    date -u +%FT%TZ >"$capture_dir/started"
    echo "Running device ATT capture SIMD${simd} region=${REGION} rank=${TRACE_RANK}" | tee "$log_file"
    set +e
    env \
        AITER_ATT_CONFIG="$capture_dir/att.yaml" \
        AITER_ATT_ENV="$OUTPUT_DIR/att_env.sh" \
        AITER_ATT_ROCPROF="$SDK_CORE/bin/rocprofv3" \
        AITER_ATT_LIBRARY_PATH="$DECODER_DIR" \
        AITER_ATT_LOCAL_RANK="$TRACE_RANK" \
        AITER_ATT_DISABLE=0 \
        AITER_ATT_ENGINE=device \
        AITER_ATT_DEVICE_LIBRARY="$AGENT_SO" \
        AITER_ATT_OUTPUT="$capture_dir/device" \
        AITER_ATT_GPU_INDEX="${AITER_ATT_GPU_INDEX:-$TRACE_RANK}" \
        AITER_ATT_TARGET_CU="${AITER_ATT_TARGET_CU:-1}" \
        AITER_ATT_SIMD="$simd" \
        AITER_ATT_SE_MASK="$SE_MASK" \
        AITER_ATT_BUFFER_BYTES="${AITER_ATT_BUFFER_BYTES:-0x10000000}" \
        AITER_ATT_REGION="$REGION" \
        PYTHONUNBUFFERED=1 \
        timeout --signal=TERM --kill-after=60s "$ATT_TIMEOUT_SECONDS" \
        torchrun --standalone --nproc_per_node="$NPROC" --no-python \
        ./my_code/trace_mega_moe_rank.sh "${BENCH_ARGV[@]}" 2>&1 | tee -a "$log_file"
    status=${PIPESTATUS[0]}
    set -e
    echo "capture exit status: $status" | tee -a "$log_file"
    printf '%s\n' "$status" >"$capture_dir/exit"
    date -u +%FT%TZ >"$capture_dir/finished"
    { printf '%s ' "$(date -u +%FT%TZ)"; gpu_idle_snapshot || true; } >"$LOG_DIR/gpu_after_simd${simd}.log" 2>&1
    return "$status"
}

decode_and_analyze_simd() {
    local simd="$1"
    local capture_dir="$OUTPUT_DIR/thread_trace/simd${simd}/kernel/rpf_v3"
    local log_file="$LOG_DIR/03_decode_simd${simd}.log"
    local pid_dir capture se file found=0 status
    local -a pid_dirs captures se_args
    mapfile -t pid_dirs < <(find "$capture_dir/device" -name 'capture*_se*.att' -printf '%h\n' | sort -u)
    if [[ "${#pid_dirs[@]}" -ne 1 ]]; then
        echo "expected one traced PID directory, found ${#pid_dirs[@]}" | tee -a "$log_file"
        return 1
    fi
    pid_dir="${pid_dirs[0]}"
    if ! check_kernel_symbol "$pid_dir" 2>&1 | tee -a "$log_file"; then
        return 2
    fi
    mapfile -t captures < <(find "$pid_dir" -name 'capture*_se*.att' -printf '%f\n' |
        sed -n 's/^capture\([0-9]*\)_se[0-9]*\.att$/\1/p' | sort -un)
    for capture in "${captures[@]}"; do
        se_args=()
        for file in "$pid_dir"/capture"${capture}"_se*.att; do
            if [[ ! -s "$file" ]]; then
                echo "empty raw trace: $file" | tee -a "$log_file"
                return 1
            fi
            se="${file##*_se}"
            se_args+=("${se%.att}:$file")
        done
        set +e
        LD_LIBRARY_PATH="$SDK_LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$DECODER_BIN" "$pid_dir" "$pid_dir/decoded_capture${capture}" "$DECODER_DIR" \
            "${se_args[@]}" 2>&1 | tee -a "$log_file"
        status=${PIPESTATUS[0]}
        set -e
        echo "capture${capture} decode status: $status" | tee -a "$log_file"
        [[ "$status" -eq 0 ]] || return 1
        if ((ANALYZE)); then
            set +e
            LD_LIBRARY_PATH="$SDK_LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
                python3 "$SCRIPT_DIR/analyze_mega_moe_device_att.py" "$pid_dir" \
                --kernel "$KERNEL_NAME" --objdump "$OBJDUMP" \
                --decoded "decoded_capture${capture}" \
                --output "$capture_dir/analysis_capture${capture}" \
                >"$LOG_DIR/04_analyze_simd${simd}_capture${capture}.log" 2>&1
            status=$?
            set -e
            if [[ -s "$capture_dir/analysis_capture${capture}/analysis.json" ]]; then
                found=1
                echo "capture${capture} analysis status: $status (1 means warnings to review)" | tee -a "$log_file"
            else
                echo "capture${capture}: $KERNEL_NAME has no analyzable waves (see 04_analyze log)" | tee -a "$log_file"
                rm -rf "$capture_dir/analysis_capture${capture}"
            fi
        fi
    done
    if ((ANALYZE && !found)); then
        echo "no capture contains decoded waves of $KERNEL_NAME" | tee -a "$log_file"
        return 1
    fi
    list_executed_kernels "$pid_dir" >"$LOG_DIR/executed_kernels_simd${simd}.txt" 2>&1 || true
    return 0
}

check_kernel_symbol() {
    LD_LIBRARY_PATH="$SDK_LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        python3 - "$1" "$OBJDUMP" "$KERNEL_NAME" <<'PY'
import os
import subprocess
import sys
from pathlib import Path

pid_dir, objdump, kernel = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
symbols = set()
for code_object in sorted(pid_dir.glob("code_object_*.out")):
    listing = subprocess.run([objdump, "-t", str(code_object)],
                             capture_output=True, text=True, check=False).stdout
    for line in listing.splitlines():
        parts = line.split()
        if ".text" in parts and "F" in parts[1:parts.index(".text")]:
            symbols.add(parts[-1])
if kernel in symbols:
    raise SystemExit(0)
print(f"{kernel} is not a function symbol of any code object loaded by the traced rank "
      f"({len(symbols)} symbols)")
candidates = sorted(name for name in symbols if name.startswith(kernel))
if candidates:
    print("symbols starting with it:")
else:
    shared = {name: len(os.path.commonprefix([name, kernel])) for name in symbols}
    longest = max(shared.values(), default=0)
    candidates = sorted(name for name, length in shared.items() if longest and length == longest)
    print(f"symbols sharing its longest prefix ({longest} characters):")
for name in candidates[:20]:
    print(f"  {name}")
raise SystemExit(1)
PY
}

list_executed_kernels() {
    # Kernel symbols whose waves were recorded on the traced CU, with wave counts.
    # Use these names as KERNEL_NAME to analyze other kernels (for example the
    # dispatch/combine communication kernels) from the same capture.
    LD_LIBRARY_PATH="$SDK_LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        python3 - "$1" "$OBJDUMP" <<'PY'
import bisect
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

pid_dir, objdump = Path(sys.argv[1]), sys.argv[2]
first = re.compile(rb'"instructions":\[\[\d+,\d+,\d+,\d+,(\d+),(\d+)\]')
for decoded in sorted(pid_dir.glob("decoded_capture*")):
    entries = defaultdict(Counter)
    for wave in decoded.glob("se*_wave*.json"):
        with wave.open("rb") as handle:
            match = first.search(handle.read(4096))
        if match:
            entries[int(match.group(2))][int(match.group(1))] += 1
    totals = Counter()
    for obj, pcs in entries.items():
        listing = subprocess.run([objdump, "-t", str(pid_dir / f"code_object_{obj}.out")],
                                 capture_output=True, text=True, check=False).stdout
        funcs = []
        for line in listing.splitlines():
            # "<addr> g F .text <size> [.protected|.hidden] <name>"
            parts = line.split()
            if ".text" not in parts:
                continue
            section = parts.index(".text")
            if "F" in parts[1:section] and len(parts) >= section + 3:
                funcs.append((int(parts[0], 16), int(parts[section + 1], 16), parts[-1]))
        funcs.sort()
        starts = [entry[0] for entry in funcs]
        for pc, count in pcs.items():
            i = bisect.bisect_right(starts, pc) - 1
            inside = i >= 0 and pc < funcs[i][0] + max(funcs[i][1], 1)
            totals[funcs[i][2] if inside else f"<code_object_{obj}+0x{pc:x}>"] += count
    print(f"{decoded.name}: waves per kernel on the traced CU")
    for name, count in totals.most_common():
        print(f"{count:8d}  {name}")
PY
}

summarize() {
    python3 - "$OUTPUT_DIR" "$KERNEL_NAME" <<'PY'
import json
import sys
from pathlib import Path

root, kernel = Path(sys.argv[1]), sys.argv[2]
rows = []
for simd_dir in sorted((root / "thread_trace").glob("simd*")):
    rpf = simd_dir / "kernel" / "rpf_v3"
    row = {"simd": simd_dir.name, "exit": (rpf / "exit").read_text().strip() if (rpf / "exit").exists() else None}
    chunks = []
    for chunk_file in rpf.glob("device/pid*/chunks.jsonl"):
        chunks += [json.loads(line) for line in chunk_file.read_text().splitlines() if line.strip()]
    row["raw_chunks"] = len(chunks)
    row["raw_bytes"] = sum(c["bytes"] for c in chunks)
    row["buffer_full"] = any(c["flags"] & 6 for c in chunks)
    for analysis in sorted(rpf.glob("analysis_capture*/analysis.json")):
        data = json.loads(analysis.read_text())
        row[analysis.parent.name] = {
            "target_waves": data.get("target_wave_count"),
            "active_waves": data.get("active_wave_count"),
            "tasks_per_active_wave": data.get("persistent_tasks_per_active_wave"),
            "active_wave_span_median": (data.get("active_wave_span_cycles") or {}).get("median"),
            "warnings": data.get("warnings"),
            "missing_pcs": len(data.get("missing_pcs") or []),
        }
    rows.append(row)
print(json.dumps({"kernel": kernel, "simds": rows}, indent=2))
PY
}

# ---- run ----
{
    echo "date: $(date -Is)"
    echo "host: $(hostname)"
    echo "repo: $REPO_ROOT"
    echo "aiter: $AITER_FILE"
    python3 -c 'import torch; print("torch:", torch.__version__)' 2>/dev/null || true
    echo "kernel: $KERNEL_NAME"
    echo "bench args: $BENCH_ARGS"
    echo "region: $REGION  rank: $TRACE_RANK  nproc: $NPROC  simds: ${SIMD_IDS[*]}  se_mask: $SE_MASK"
    echo "sdk core: $SDK_CORE"
    echo "sdk devel: $SDK_DEVEL"
    echo "decoder: $DECODER_DIR/librocprof-trace-decoder.so"
    echo "comgr: $COMGR_DIR"
    echo "objdump: $OBJDUMP"
    "$SDK_CORE/bin/rocprofv3" --version 2>&1 | head -5 || true
    echo "workload environment:"
    env | grep -E '^(MORI_|MEGA_|AITER_|FLYDSL_|HIP_|ROCR_|HSA_)' | sort || true
    echo "source sha256:"
    sha256sum aiter/ops/flydsl/*.py aiter/ops/flydsl/kernels/*.py \
        aiter/ops/flydsl/kernels/mega_moe_gfx1250/*.py aiter/configs/tuned_grouped_fmoe.csv \
        "$SCRIPT_DIR"/trace_mega_moe_*.* "$SCRIPT_DIR/analyze_mega_moe_device_att.py" 2>/dev/null || true
} >"$LOG_DIR/environment.log" 2>&1
cat "$LOG_DIR/environment.log"

cat >"$OUTPUT_DIR/att_env.sh" <<EOF
export LD_LIBRARY_PATH="$SDK_LIB_PATH\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF

build_tools 2>&1 | tee "$LOG_DIR/01_build_tools.log"
sha256sum "$AGENT_SO" "$DECODER_BIN" | tee -a "$LOG_DIR/01_build_tools.log"

overall=0
: >"$LOG_DIR/simd_status.log"
for simd in "${SIMD_IDS[@]}"; do
    capture_status=0
    decode_status=0
    capture_simd "$simd" || capture_status=$?
    if [[ "$capture_status" -eq 0 ]]; then
        decode_and_analyze_simd "$simd" || decode_status=$?
    fi
    echo "simd${simd}_capture_status=${capture_status} simd${simd}_decode_status=${decode_status}" |
        tee -a "$LOG_DIR/simd_status.log"
    if [[ "$capture_status" -ne 0 || "$decode_status" -ne 0 ]]; then
        overall=1
    fi
    if [[ "$decode_status" -eq 2 ]]; then
        echo "stopped: $KERNEL_NAME is not a complete kernel symbol of the traced workload" |
            tee -a "$LOG_DIR/simd_status.log"
        break
    fi
done

{
    cat "$LOG_DIR/simd_status.log"
    summarize
    echo "final_output_dir=$OUTPUT_DIR"
} 2>&1 | tee "$SUMMARY_LOG"

if ((ARCHIVE)); then
    tar -C "$TRACE_ROOT" -czf "$OUTPUT_ARCHIVE" "$OUTPUT_DIR_NAME"
    chmod a+rw "$OUTPUT_ARCHIVE"
    ls -lah "$OUTPUT_ARCHIVE"
fi
chmod -R a+rwX "$OUTPUT_DIR"
echo "Device ATT collection finished with status $overall: $OUTPUT_DIR"
exit "$overall"
