#!/usr/bin/env bash
set -Eeuo pipefail

# Method B: dispatch-based ATT for single-GPU kernels in containers whose PyTorch
# ships the TheRock ROCm SDK (_rocm_sdk_core / _rocm_sdk_devel, rocprofv3 1.3.x),
# such as hyg_fyd_e2e or dsr1_ep4. The self-built rocprofv3 1.1.0 would load a
# second rocprofiler-sdk next to PyTorch's copy and abort during registration, so
# the profiler, register library and trace decoder all come from the same SDK.

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
TRACE_ROOT="${TRACE_ROOT:-my_code/thread_trace_runs}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/data/yanguahe/code/wk_sp1}"
ATT_TIMEOUT_SECONDS="${AITER_ATT_TIMEOUT_SECONDS:-900}"
IDLE_VRAM_MB="${AITER_ATT_IDLE_VRAM_MB:-512}"
IDLE_POLL_SECONDS=6

usage() {
    cat >&2 <<EOF
Collect inside the container (for example hyg_fyd_e2e) from the repository root:
  ${SCRIPT_NAME} <KERNEL_NAME> <output-dir-name> "<TEST_CMD>" [options]

  KERNEL_NAME  complete kernel symbol, matched exactly. The preflight kernel trace
               must contain a dispatch of it; otherwise the script stops before any
               ATT run and lists the dispatched kernels that share its prefix.
  TEST_CMD     single-line workload command, run from the repository root

Options:
  --all-simd      capture SIMD0..3 in four independent runs (default: SIMD3 only)
  --simd N        capture only SIMD N (0..3)
  --iteration N   kernel_iteration_range dispatch index of the kernel (default: 1)
  --se-mask HEX   shader-engine mask (default: 0xf; --ana-att needs 0xf)
  --ana-att       run my_code/analyze_att_capture.py after a successful capture
  --no-wait-idle  do not wait for all GPUs to be idle before each profiled run
  --archive       also write <TRACE_ROOT>/<output-dir-name>.tar.gz

Environment overrides:
  TRACE_ROOT (my_code/thread_trace_runs), WORKSPACE_DIR (/data/yanguahe/code/wk_sp1),
  AITER_ATT_TIMEOUT_SECONDS (900), AITER_ATT_IDLE_VRAM_MB (512),
  AITER_ATT_SDK_CORE, AITER_ATT_SDK_DEVEL, AITER_ATT_LIBRARY_PATH (decoder directory),
  AITER_ATT_COMGR_DIR (default \$WORKSPACE_DIR/comgr_new)

Example (hyg_fyd_e2e, E64/T1536 GEMM2, random data):
  bash my_code/${SCRIPT_NAME} a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist e64_t1536_gemm2_att \\
    "env ENABLE_CK=0 AITER_MOE_EXPERT_BALANCE=true AITER_LOG_MORE=1 AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py --scenario bench --data-format a4w4 --act silu --no-bias --no-check-aot-cache --experts 64 --tokens 1536 --topk 8 --model-dim 7168 --inter-dim 2048 --iters 2" \\
    --all-simd --ana-att
EOF
}

die() {
    echo "${SCRIPT_NAME}: $*" >&2
    exit 1
}

# ---- method-specific toolchain ----------------------------------------------
find_sdk_dir() {
    local module="$1" candidate
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

setup_toolchain() {
    SDK_CORE="${AITER_ATT_SDK_CORE:-$(find_sdk_dir _rocm_sdk_core || true)}"
    SDK_DEVEL="${AITER_ATT_SDK_DEVEL:-$(find_sdk_dir _rocm_sdk_devel || true)}"
    [[ -n "$SDK_CORE" && -x "$SDK_CORE/bin/rocprofv3" ]] || die "cannot locate _rocm_sdk_core/bin/rocprofv3"
    [[ -n "$SDK_DEVEL" && -d "$SDK_DEVEL/lib" ]] || die "cannot locate _rocm_sdk_devel/lib"
    PROFILER="$SDK_CORE/bin/rocprofv3"

    if [[ -n "${AITER_ATT_LIBRARY_PATH:-}" ]]; then
        ROCPROF_ATT_LIBRARY_PATH="$AITER_ATT_LIBRARY_PATH"
    elif [[ -s "$SDK_DEVEL/lib/librocprof-trace-decoder.so" ]]; then
        ROCPROF_ATT_LIBRARY_PATH="$SDK_DEVEL/lib"
    elif [[ -s "$SDK_CORE/lib/librocprof-trace-decoder.so.0.2" ]]; then
        # rocprofv3 opens the unversioned name; expose the SDK-aligned 0.2 decoder.
        ROCPROF_ATT_LIBRARY_PATH="$OUTPUT_DIR/.trace_decoder"
        mkdir -p "$ROCPROF_ATT_LIBRARY_PATH"
        ln -sfn "$SDK_CORE/lib/librocprof-trace-decoder.so.0.2" \
            "$ROCPROF_ATT_LIBRARY_PATH/librocprof-trace-decoder.so"
    else
        die "no SDK-aligned librocprof-trace-decoder.so found under $SDK_DEVEL/lib or $SDK_CORE/lib"
    fi
    export ROCPROF_ATT_LIBRARY_PATH
    [[ -e "$ROCPROF_ATT_LIBRARY_PATH/librocprof-trace-decoder.so" ]] ||
        die "missing $ROCPROF_ATT_LIBRARY_PATH/librocprof-trace-decoder.so"

    # rocminfo cannot enumerate agents while ATT owns the device, but AITER runs
    # it during import. Record the real output before tracing and serve it from a
    # shim inside the profiled process tree only.
    local shim="$OUTPUT_DIR/.rocminfo_shim"
    mkdir -p "$shim"
    if ! rocminfo >"$shim/rocminfo.txt" 2>"$LOG_DIR/rocminfo_capture.err" ||
        ! grep -q gfx1250 "$shim/rocminfo.txt"; then
        die "real rocminfo failed before tracing (see $LOG_DIR/rocminfo_capture.err)"
    fi
    printf '#!/usr/bin/env bash\ncat %q\n' "$shim/rocminfo.txt" >"$shim/rocminfo"
    chmod a+rx "$shim/rocminfo"

    local comgr="${AITER_ATT_COMGR_DIR:-$WORKSPACE_DIR/comgr_new}"
    local libs="$SDK_CORE/lib:$SDK_DEVEL/lib:$SDK_DEVEL/lib/rocm_sysdeps/lib:$SDK_CORE/lib/rocm_sysdeps/lib"
    # comgr_new decodes gfx1250 instructions that older COMGR emits as .long.
    [[ -d "$comgr" ]] && libs="$comgr:$libs"
    [[ -d /opt/rocm/lib ]] && libs="$libs:/opt/rocm/lib"
    export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    PROFILED_PREFIX=(env "PATH=$shim:$PATH")
}
# ------------------------------------------------------------------------------

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
TEST_CMD="$3"
shift 3
SIMD_IDS=(3)
ITERATION=1
SE_MASK=0xf
ANA_ATT=0
WAIT_IDLE=1
ARCHIVE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all-simd) SIMD_IDS=(0 1 2 3); shift ;;
        --simd) [[ "${2:-}" =~ ^[0-3]$ ]] || die "--simd needs 0..3"; SIMD_IDS=("$2"); shift 2 ;;
        --iteration) [[ "${2:-}" =~ ^[1-9][0-9]*$ ]] || die "--iteration needs a positive integer"; ITERATION="$2"; shift 2 ;;
        --se-mask) [[ "${2:-}" =~ ^0x[0-9A-Fa-f]+$ ]] || die "--se-mask needs a hexadecimal value"; SE_MASK="$2"; shift 2 ;;
        --ana-att) ANA_ATT=1; shift ;;
        --no-wait-idle) WAIT_IDLE=0; shift ;;
        --archive) ARCHIVE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; die "unexpected argument: $1" ;;
    esac
done

[[ "$KERNEL_NAME" =~ ^[A-Za-z_.$][A-Za-z0-9_.$@:+-]*$ ]] || die "KERNEL_NAME must be a complete C/AMDGPU-style kernel symbol"
[[ "$OUTPUT_DIR_NAME" =~ ^[A-Za-z0-9_][A-Za-z0-9._-]*$ ]] || die "output-dir-name may only use letters, numbers, dot, underscore, dash"
[[ -n "$TEST_CMD" && "$TEST_CMD" != *$'\n'* ]] || die "TEST_CMD must be a non-empty single line"
read -r -a TEST_ARGV <<< "$TEST_CMD"

cd "$REPO_ROOT"
OUTPUT_DIR="$TRACE_ROOT/$OUTPUT_DIR_NAME"
OUTPUT_ARCHIVE="$TRACE_ROOT/$OUTPUT_DIR_NAME.tar.gz"
LOG_DIR="$OUTPUT_DIR/logs"
[[ ! -e "$OUTPUT_DIR" ]] || die "$OUTPUT_DIR already exists; choose a new output-dir-name"
mkdir -p "$LOG_DIR"

exec 9>"/tmp/aiter_att_dispatch_${UID}.lock"
flock -n 9 || die "another dispatch ATT capture holds /tmp/aiter_att_dispatch_${UID}.lock"

setup_toolchain

# A private FlyDSL cache guarantees that the traced kernel is compiled from the
# current sources without deleting the shared ~/.flydsl/cache.
FLYDSL_CACHE="$OUTPUT_DIR/.flydsl_cache"
TEST_ENV=(
    env -u FLYDSL_DUMP_DIR
    PYTORCH_ALLOC_CONF=expandable_segments:True
    GPU_ARCHS=gfx1250
    ENABLE_CK=0
    AITER_FORCE_GFX1250=1
    FLYDSL_DUMP_IR=0
    FLYDSL_RUNTIME_CACHE_DIR="$FLYDSL_CACHE"
)

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
    ((WAIT_IDLE)) || return 0
    while true; do
        printf '%s ' "$(date -u +%FT%TZ)" | tee -a "$log_file"
        if gpu_idle_snapshot 2>&1 | tee -a "$log_file"; then
            return 0
        fi
        echo "GPUs busy; rechecking in ${IDLE_POLL_SECONDS}s" | tee -a "$log_file"
        sleep "$IDLE_POLL_SECONDS"
    done
}

run_logged() {
    local name="$1" log_file="$2" status
    shift 2
    echo "Running ${name}: $*" | tee "$log_file"
    set +e
    timeout --signal=TERM --kill-after=5s "$ATT_TIMEOUT_SECONDS" "$@" 2>&1 | tee -a "$log_file"
    status=${PIPESTATUS[0]}
    set -e
    echo "${name} exit status: ${status}" | tee -a "$log_file"
    return "$status"
}

check_kernel_dispatched() {
    python3 - "$OUTPUT_DIR/kernel_trace" "$KERNEL_NAME" <<'PY'
import csv
import os
import sqlite3
import sys
from pathlib import Path

root, kernel = Path(sys.argv[1]), sys.argv[2]
names = set()
csv_files = sorted(p for p in root.rglob("*.csv") if "kernel" in p.name)
db_files = sorted(root.rglob("*.db"))
for path in csv_files:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            for column in ("Kernel_Name", "Name"):
                if row.get(column):
                    names.add(row[column].strip())
if not names:
    # rocpd SQLite output: the "kernels" view lists dispatches only, whereas the
    # kernel-symbol tables hold every symbol of every loaded code object.
    for path in db_files:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE type = 'view' AND name = 'kernels'").fetchone():
                names.update(r[0].strip() for r in db.execute("SELECT DISTINCT name FROM kernels") if r[0])
if kernel in names:
    raise SystemExit(0)
print(f"{kernel} was not dispatched in the preflight trace ({len(names)} distinct kernels; "
      f"csv={[str(p) for p in csv_files]} db={[str(p) for p in db_files]})", file=sys.stderr)
candidates = sorted(name for name in names if name.startswith(kernel))
if candidates:
    print("dispatched kernels starting with it:", file=sys.stderr)
else:
    shared = {name: len(os.path.commonprefix([name, kernel])) for name in names}
    longest = max(shared.values(), default=0)
    candidates = sorted(name for name, length in shared.items() if longest and length == longest)
    print(f"dispatched kernels sharing its longest prefix ({longest} characters):", file=sys.stderr)
for name in candidates[:20]:
    print(f"  {name}", file=sys.stderr)
raise SystemExit(1)
PY
}

escape_kernel_regex() {
    python3 -c 'import re, sys; print(re.escape(sys.argv[1]))' "$1"
}

validate_capture() {
    python3 - "$OUTPUT_DIR" "$KERNEL_NAME" "$1" <<'PY'
import json
import sys
from pathlib import Path

root, kernel, simd = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
capture = root / "thread_trace" / f"simd{simd}" / "kernel" / "rpf_v3"
att = [p for p in capture.rglob("*.att") if p.stat().st_size > 0]
codes = [p for p in capture.glob("ui_output_agent_*/code.json") if p.stat().st_size > 0]
waves = [p for p in capture.glob("ui_output_agent_*/se*_sm*_sl*_wv*.json") if p.stat().st_size > 0]
row = {"simd": int(simd), "att_files": len(att), "code_json": len(codes), "wave_files": len(waves)}
failures = []
if not att:
    failures.append("no non-empty .att file")
if len(codes) != 1:
    failures.append(f"expected one code.json, found {len(codes)}")
if not waves:
    failures.append("no decoded wave JSON")
for code in codes:
    text = code.read_text(encoding="utf-8", errors="replace")
    row["kernel_hits"] = text.count(kernel)
    row["dot_long"] = text.count(".long")
    row["v_illegal"] = text.count("v_illegal")
    try:
        row["code_entries"] = len(json.loads(text).get("code") or [])
    except json.JSONDecodeError as exc:
        failures.append(f"invalid code.json: {exc}")
    if row["kernel_hits"] < 1:
        failures.append("target kernel absent from code.json")
    if row["dot_long"] or row["v_illegal"]:
        failures.append("code.json contains .long or v_illegal")
    if not row.get("code_entries"):
        failures.append("code.json has no instructions")
row["failures"] = failures
print(json.dumps(row))
raise SystemExit(1 if failures else 0)
PY
}

write_yaml() {
    local yaml="$1" output_directory="$2" simd="$3"
    cat >"$yaml" <<YAML
jobs:
 -
  kernel_include_regex: '^${KERNEL_REGEX}\$'
  kernel_exclude_regex:
  kernel_iteration_range: "[${ITERATION}]"
  output_file: out
  output_directory: ${output_directory}
  output_format: [csv]
  truncate_kernels: false
  sys_trace: false
  advanced_thread_trace: true
  att_target_cu: 1
  att_shader_engine_mask: "${SE_MASK}"
  att_simd_select: "${simd}"
  att_buffer_size: "0x10000000"
  att_library_path: ["${ROCPROF_ATT_LIBRARY_PATH}"]
YAML
}

{
    echo "date: $(date -Is)"
    echo "host: $(hostname)"
    echo "repo: $REPO_ROOT"
    echo "method: ${SCRIPT_NAME}"
    echo "kernel: $KERNEL_NAME"
    echo "simds: ${SIMD_IDS[*]}  iteration: $ITERATION  se_mask: $SE_MASK"
    echo "profiler: $PROFILER"
    "$PROFILER" --version 2>&1 | head -4 || true
    echo "decoder: $ROCPROF_ATT_LIBRARY_PATH/librocprof-trace-decoder.so -> $(readlink -f "$ROCPROF_ATT_LIBRARY_PATH/librocprof-trace-decoder.so")"
    echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"
    python3 -c 'import importlib.util as u; print("aiter:", u.find_spec("aiter").origin)' 2>/dev/null || true
    echo "test command: $TEST_CMD"
} >"$LOG_DIR/environment.log" 2>&1
cat "$LOG_DIR/environment.log"

wait_gpu_idle "$LOG_DIR/gpu_idle_before_preflight.log"
kernel_trace_status=0
run_logged kernel-trace-stats "$LOG_DIR/01_kernel_trace_stats.log" \
    "${PROFILED_PREFIX[@]}" "$PROFILER" --kernel-trace --stats --output-format csv -d "$OUTPUT_DIR/kernel_trace" -- \
    "${TEST_ENV[@]}" "${TEST_ARGV[@]}" || kernel_trace_status=$?
[[ "$kernel_trace_status" -eq 0 ]] || die "preflight kernel trace failed with status $kernel_trace_status"

check_kernel_dispatched 2>&1 | tee -a "$LOG_DIR/01_kernel_trace_stats.log" ||
    die "$KERNEL_NAME is not a complete kernel name dispatched by TEST_CMD (candidates above)"
echo "preflight dispatched kernel: $KERNEL_NAME" | tee -a "$LOG_DIR/environment.log"
KERNEL_REGEX="$(escape_kernel_regex "$KERNEL_NAME")"

att_status=0
: >"$LOG_DIR/att_simd_status.log"
for simd in "${SIMD_IDS[@]}"; do
    capture_dir="$OUTPUT_DIR/thread_trace/simd${simd}/kernel/rpf_v3"
    yaml="$OUTPUT_DIR/input_kernel_simd${simd}.yaml"
    mkdir -p "$capture_dir"
    write_yaml "$yaml" "$capture_dir" "$simd"
    wait_gpu_idle "$LOG_DIR/gpu_idle_before_simd${simd}.log"
    simd_status=0
    run_logged "advanced-thread-trace-simd${simd}" "$LOG_DIR/02_thread_trace_simd${simd}.log" \
        "${PROFILED_PREFIX[@]}" "$PROFILER" -i "$yaml" -- "${TEST_ENV[@]}" "${TEST_ARGV[@]}" || simd_status=$?
    if [[ "$simd_status" -eq 0 ]]; then
        validate_capture "$simd" 2>&1 | tee -a "$LOG_DIR/02_thread_trace_simd${simd}.log" ||
            simd_status=1
    fi
    echo "simd${simd}_att_status=${simd_status}" | tee -a "$LOG_DIR/att_simd_status.log"
    [[ "$simd_status" -eq 0 ]] || att_status=1
done

ana_status=0
if ((ANA_ATT)); then
    if [[ "$att_status" -ne 0 ]]; then
        echo "Skipping --ana-att because at least one SIMD capture failed." | tee "$LOG_DIR/analyze_att_capture.log"
    elif [[ "${SE_MASK,,}" != 0xf ]]; then
        echo "Skipping --ana-att: analyze_att_capture.py needs all four SEs (--se-mask 0xf)." | tee "$LOG_DIR/analyze_att_capture.log"
    else
        run_logged analyze-att "$LOG_DIR/analyze_att_capture.log" \
            python3 -u "$SCRIPT_DIR/analyze_att_capture.py" --dir "$OUTPUT_DIR" --no-plot || ana_status=$?
    fi
fi

rm -rf "$FLYDSL_CACHE"
{
    echo "kernel=$KERNEL_NAME"
    echo "simd_ids=${SIMD_IDS[*]}"
    cat "$LOG_DIR/att_simd_status.log"
    echo "att_status=$att_status"
    echo "analyze_status=$ana_status"
    echo "final_output_dir=$OUTPUT_DIR"
} | tee "$OUTPUT_DIR/summary.log"

if ((ARCHIVE)); then
    tar -C "$TRACE_ROOT" -czf "$OUTPUT_ARCHIVE" "$OUTPUT_DIR_NAME"
    chmod a+rw "$OUTPUT_ARCHIVE"
    ls -lah "$OUTPUT_ARCHIVE"
fi
chmod -R a+rwX "$OUTPUT_DIR"
if [[ "$att_status" -ne 0 || "$ana_status" -ne 0 ]]; then
    exit 1
fi
echo "ATT collection complete: $OUTPUT_DIR"
