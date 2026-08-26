#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

readonly ARCH="gfx1250"
readonly TRIPLE="amdgcn-amd-amdhsa"
readonly CODE_OBJECT_VERSION="6"

LLVM_BIN="${LLVM_BIN:-/data/yanguahe/code/wk_sp1/llvm-project/mlir_install/bin}"
LLVM_RUNTIME_LIBS="${LLVM_RUNTIME_LIBS:-}"
REPO_ROOT="${REPO_ROOT:-/data/yanguahe/code/wk_sp1/aiter}"

usage() {
    cat <<'EOF'
Usage:
  bash my_code/run_validate_f4gemm_64x64_isa.sh [OUT_DIR]

Environment overrides:
  LLVM_BIN    LLVM tool directory
  LLVM_RUNTIME_LIBS
              Colon-separated LLVM runtime library directories, searched first
  REPO_ROOT   Aiter repository root
  TARGET_S    target assembly source
  REFERENCE_S reference assembly source
  OUT_DIR     output directory (the positional argument takes precedence)

This script only assembles, links, disassembles, and statically inspects the
kernel. It does not run the kernel, access a GPU, use the network, or mutate git.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# > 1 )); then
    usage >&2
    echo "ERROR: expected at most one positional OUT_DIR argument" >&2
    exit 2
fi

if command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
else
    echo "ERROR: Python 3 is required for integrity and static analysis." >&2
    exit 2
fi

canonical_path() {
    "${PYTHON}" - "$1" <<'PY'
import os
import sys
print(os.path.realpath(os.path.abspath(os.path.expanduser(sys.argv[1]))))
PY
}

REPO_ROOT="$(canonical_path "${REPO_ROOT}")"
if [[ -n "${TARGET_S:-}" ]]; then
    target_input="${TARGET_S}"
else
    target_input="${REPO_ROOT}/my_code/f4gemm_bf16_mxfp4_ABpreShuffle_64x64_4x4_ps.s"
fi
if [[ -n "${REFERENCE_S:-}" ]]; then
    reference_input="${REFERENCE_S}"
else
    reference_input="${REPO_ROOT}/my_code/f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps.s"
fi
if [[ "${target_input}" != /* ]]; then
    target_input="${REPO_ROOT}/${target_input}"
fi
if [[ "${reference_input}" != /* ]]; then
    reference_input="${REPO_ROOT}/${reference_input}"
fi
TARGET_S="$(canonical_path "${target_input}")"
REFERENCE_S="$(canonical_path "${reference_input}")"

timestamp="$(date +%Y%m%d_%H%M%S)"
default_out="/tmp/f4gemm_64x64_isa_validation_${timestamp}_$$"
if (( $# == 1 )); then
    OUT_DIR="$1"
else
    OUT_DIR="${OUT_DIR:-${default_out}}"
fi
OUT_DIR="$(canonical_path "${OUT_DIR}")"
mkdir -p "${OUT_DIR}"

on_exit() {
    local status=$?
    trap - EXIT
    if [[ -n "${OUT_DIR:-}" && -d "${OUT_DIR}" ]]; then
        set +e
        chmod -R a+rwX "${OUT_DIR}"
        local chmod_status=$?
        set -e
        if (( chmod_status != 0 )); then
            echo "WARNING: chmod -R a+rwX failed for ${OUT_DIR}" >&2
        fi
    fi
    exit "${status}"
}
trap on_exit EXIT

banner() {
    printf '\n================================================================================\n'
    printf '%s\n' "$1"
    printf '================================================================================\n'
}

quote_command() {
    local rendered=""
    local item
    for item in "$@"; do
        printf -v item '%q' "${item}"
        rendered+="${rendered:+ }${item}"
    done
    printf '%s' "${rendered}"
}

COMMAND_STATUS="${OUT_DIR}/command_status.tsv"
printf 'name\tstatus\tcommand\n' > "${COMMAND_STATUS}"
LAST_STATUS=0

record_command_status() {
    local name="$1"
    local status="$2"
    local command="$3"
    command="${command//$'\t'/ }"
    command="${command//$'\n'/ }"
    printf '%s\t%s\t%s\n' "${name}" "${status}" "${command}" >> "${COMMAND_STATUS}"
}

run_logged() {
    local name="$1"
    local log_file="$2"
    shift 2
    local rendered
    rendered="$(quote_command "$@")"
    printf '\nCOMMAND [%s]: %s\n' "${name}" "${rendered}"
    set +e
    "$@" 2>&1 | tee "${log_file}"
    local status=${PIPESTATUS[0]}
    set -e
    printf 'STATUS  [%s]: %s\n' "${name}" "${status}"
    record_command_status "${name}" "${status}" "${rendered}"
    LAST_STATUS="${status}"
}

run_capture_stdout() {
    local name="$1"
    local stdout_file="$2"
    local stderr_file="$3"
    shift 3
    local rendered
    rendered="$(quote_command "$@")"
    printf '\nCOMMAND [%s]: %s\n' "${name}" "${rendered}"
    : > "${stderr_file}"
    set +e
    "$@" 2> >(tee "${stderr_file}" >&2) | tee "${stdout_file}"
    local status=${PIPESTATUS[0]}
    set -e
    printf 'STATUS  [%s]: %s\n' "${name}" "${status}"
    record_command_status "${name}" "${status}" "${rendered}"
    LAST_STATUS="${status}"
}

declare -a EXPERIMENT_NAMES=()
declare -a EXPERIMENT_LEVELS=()
declare -a EXPERIMENT_DETAILS=()

record_experiment() {
    local name="$1"
    local level="$2"
    local detail="$3"
    detail="${detail//$'\t'/ }"
    detail="${detail//$'\n'/ }"
    EXPERIMENT_NAMES+=("${name}")
    EXPERIMENT_LEVELS+=("${level}")
    EXPERIMENT_DETAILS+=("${detail}")
    printf '%-5s %-31s %s\n' "${level}" "${name}" "${detail}"
}

HARD_FAIL=0
TOOLCHAIN_ROOT_CAUSE=""
ZSTD_UNRESOLVED=0

runtime_path_value="${OUT_DIR}/llvm_runtime_path.value"
runtime_discovery_report="${OUT_DIR}/llvm_runtime_paths.txt"
runtime_discovery_command="$(
    quote_command \
        "${PYTHON}" - \
        "${LLVM_RUNTIME_LIBS}" \
        "${LD_LIBRARY_PATH:-}" \
        "${LLVM_BIN}" \
        "${runtime_path_value}"
)"
printf 'COMMAND [discover_llvm_runtime_libraries]: %s <embedded-python>\n' \
    "${runtime_discovery_command}"
set +e
"${PYTHON}" - \
    "${LLVM_RUNTIME_LIBS}" \
    "${LD_LIBRARY_PATH:-}" \
    "${LLVM_BIN}" \
    "${runtime_path_value}" <<'PY' | tee "${runtime_discovery_report}"
import glob
import os
import sys
from pathlib import Path

override, caller_ld, llvm_bin, output_path = sys.argv[1:5]
rows = []
ordered = []


def add_directory(raw, source, *, preserve_missing=False):
    if not raw:
        return
    path = os.path.realpath(os.path.expanduser(raw))
    exists = os.path.isdir(path)
    rows.append((source, path, exists))
    if (exists or preserve_missing) and path not in ordered:
        ordered.append(path)


for entry in override.split(":"):
    add_directory(entry, "LLVM_RUNTIME_LIBS")

patterns = (
    "/opt/venv/lib/python*/site-packages/_rocm_sdk_core/lib",
    "/opt/venv/lib/python*/site-packages/_rocm_sdk_devel/lib/rocm_sysdeps",
    "/opt/venv/lib/python*/site-packages/_rocm_sdk_devel/lib/rocm_sysdeps/lib",
)
for pattern in patterns:
    for match in sorted(glob.glob(pattern)):
        add_directory(match, f"auto:{pattern}")

zstd_patterns = (
    "/opt/venv/lib/python*/site-packages/**/librocm_sysdeps_zstd.so.1",
    "/opt/venv/lib/python*/site-packages/**/librocm_sysdeps_zstd.so",
)
for pattern in zstd_patterns:
    for match in sorted(glob.glob(pattern, recursive=True)):
        add_directory(str(Path(match).parent), f"auto:zstd-parent:{match}")

for candidate in (
    "/opt/rocm/lib",
    "/opt/rocm/lib64",
    str(Path(llvm_bin).resolve().parent.parent / "lib"),
    str(Path(llvm_bin).resolve().parent.parent / "lib64"),
):
    add_directory(candidate, "auto:standard")

# Preserve every non-empty caller entry exactly in search order, even if it is
# temporarily unavailable (for example, a later-mounted container path).
for entry in caller_ld.split(":"):
    add_directory(entry, "caller:LD_LIBRARY_PATH", preserve_missing=True)

print("LLVM runtime library discovery:")
for source, path, exists in rows:
    print(f"  {'FOUND' if exists else 'MISSING'} [{source}] {path}")
final_path = ":".join(ordered)
Path(output_path).write_text(final_path + "\n", encoding="utf-8", newline="\n")
print(f"Final LD_LIBRARY_PATH: {final_path or '<empty>'}")
print("Only existing override/auto-discovered directories were added; caller entries were preserved.")
PY
runtime_discovery_status=${PIPESTATUS[0]}
set -e
record_command_status \
    "discover_llvm_runtime_libraries" \
    "${runtime_discovery_status}" \
    "${runtime_discovery_command} <embedded-python>"
if (( runtime_discovery_status != 0 )) || [[ ! -f "${runtime_path_value}" ]]; then
    echo "WARNING: LLVM runtime library discovery failed; preserving caller LD_LIBRARY_PATH." >&2
else
    IFS= read -r LD_LIBRARY_PATH < "${runtime_path_value}" || true
    export LD_LIBRARY_PATH
fi

banner "0. Invocation context"
{
    printf 'date: %s\n' "$(date -Is)"
    printf 'host: %s\n' "$(hostname 2>/dev/null || printf unknown)"
    printf 'pwd: %s\n' "$(pwd)"
    printf 'script: %s\n' "$0"
    printf 'script invocation:'
    printf ' %q' "$0" "$@"
    printf '\n'
    printf 'architecture: %s\n' "${ARCH}"
    printf 'target triple: %s\n' "${TRIPLE}"
    printf 'code object version: %s\n' "${CODE_OBJECT_VERSION}"
    printf 'LLVM_BIN: %s\n' "${LLVM_BIN}"
    printf 'LLVM_RUNTIME_LIBS: %s\n' "${LLVM_RUNTIME_LIBS:-<unset>}"
    printf 'LD_LIBRARY_PATH: %s\n' "${LD_LIBRARY_PATH:-<empty>}"
    printf 'REPO_ROOT: %s\n' "${REPO_ROOT}"
    printf 'TARGET_S: %s\n' "${TARGET_S}"
    printf 'REFERENCE_S: %s\n' "${REFERENCE_S}"
    printf 'OUT_DIR: %s\n' "${OUT_DIR}"
    printf 'python: %s\n' "${PYTHON}"
    if [[ "${OUT_DIR}" == "${REPO_ROOT}" || "${OUT_DIR}" == "${REPO_ROOT}/"* ]]; then
        printf 'WARNING: OUT_DIR is inside the repository; the default is outside the git tree.\n'
    fi
} | tee "${OUT_DIR}/invocation.txt"

banner "1. LLVM toolchain inventory and versions"

resolve_tool() {
    local name="$1"
    local candidate="${LLVM_BIN%/}/${name}"
    if [[ -f "${candidate}" ]]; then
        printf '%s' "${candidate}"
        return 0
    fi
    command -v "${name}" 2>/dev/null || return 1
}

CLANG=""
CLANGXX=""
LLVM_MC=""
LD_LLD=""
LLVM_OBJDUMP=""
LLVM_READELF=""
LLVM_READOBJ=""
LLVM_NM=""

if path="$(resolve_tool clang)"; then CLANG="${path}"; fi
if path="$(resolve_tool clang++)"; then CLANGXX="${path}"; fi
if path="$(resolve_tool llvm-mc)"; then LLVM_MC="${path}"; fi
if path="$(resolve_tool ld.lld)"; then LD_LLD="${path}"; fi
if path="$(resolve_tool llvm-objdump)"; then LLVM_OBJDUMP="${path}"; fi
if path="$(resolve_tool llvm-readelf)"; then LLVM_READELF="${path}"; fi
if path="$(resolve_tool llvm-readobj)"; then LLVM_READOBJ="${path}"; fi
if path="$(resolve_tool llvm-nm)"; then LLVM_NM="${path}"; fi

TOOL_INVENTORY="${OUT_DIR}/tool_inventory.tsv"
declare -a TOOL_LABELS=()
declare -a TOOL_PATHS=()

inventory_tool() {
    local requested="$1"
    local path="$2"
    if [[ -n "${path}" ]]; then
        printf '%s\tFOUND\t%s\t%s\n' \
            "${requested}" "$(basename "${path}")" "${path}"
        TOOL_LABELS+=("${requested}")
        TOOL_PATHS+=("${path}")
    else
        printf '%s\tMISSING\t-\t-\n' "${requested}"
    fi
}

inventory_tool "clang" "${CLANG}"
inventory_tool "clang++" "${CLANGXX}"
inventory_tool "llvm-mc" "${LLVM_MC}"
inventory_tool "ld.lld" "${LD_LLD}"
inventory_tool "llvm-objdump" "${LLVM_OBJDUMP}"
inventory_tool "llvm-readelf" "${LLVM_READELF}"
inventory_tool "llvm-readobj" "${LLVM_READOBJ}"
inventory_tool "llvm-nm" "${LLVM_NM}"

LLVM_MC_LAUNCHABLE=0
CLANG_LAUNCHABLE=0
CLANGXX_LAUNCHABLE=0
LD_LLD_LAUNCHABLE=0
LLVM_OBJDUMP_LAUNCHABLE=0
LLVM_READELF_LAUNCHABLE=0
LLVM_READOBJ_LAUNCHABLE=0
LLVM_NM_LAUNCHABLE=0
LDD_NOT_FOUND_TOOLS=0
LDD_UNAVAILABLE=0
version_failures=0
printf 'requested_name\texists\texecutable\tlaunchable\tversion_status\tldd_not_found\tpath\n' \
    > "${TOOL_INVENTORY}"

set_tool_launchability() {
    local label="$1"
    local value="$2"
    case "${label}" in
        clang) CLANG_LAUNCHABLE="${value}" ;;
        clang++) CLANGXX_LAUNCHABLE="${value}" ;;
        llvm-mc) LLVM_MC_LAUNCHABLE="${value}" ;;
        ld.lld) LD_LLD_LAUNCHABLE="${value}" ;;
        llvm-objdump) LLVM_OBJDUMP_LAUNCHABLE="${value}" ;;
        llvm-readelf) LLVM_READELF_LAUNCHABLE="${value}" ;;
        llvm-readobj) LLVM_READOBJ_LAUNCHABLE="${value}" ;;
        llvm-nm) LLVM_NM_LAUNCHABLE="${value}" ;;
    esac
}

for requested in clang clang++ llvm-mc ld.lld llvm-objdump llvm-readelf llvm-readobj llvm-nm; do
    path=""
    case "${requested}" in
        clang) path="${CLANG}" ;;
        clang++) path="${CLANGXX}" ;;
        llvm-mc) path="${LLVM_MC}" ;;
        ld.lld) path="${LD_LLD}" ;;
        llvm-objdump) path="${LLVM_OBJDUMP}" ;;
        llvm-readelf) path="${LLVM_READELF}" ;;
        llvm-readobj) path="${LLVM_READOBJ}" ;;
        llvm-nm) path="${LLVM_NM}" ;;
    esac
    if [[ -z "${path}" ]]; then
        printf '%s\tNO\tNO\tNO\tNOT_RUN\tNOT_RUN\t-\n' "${requested}" \
            | tee -a "${TOOL_INVENTORY}"
    fi
done

for index in "${!TOOL_PATHS[@]}"; do
    label="${TOOL_LABELS[${index}]}"
    tool="${TOOL_PATHS[${index}]}"
    safe_label="${label//+/x}"
    safe_label="${safe_label//./_}"
    executable_state="NO"
    launchable_state="NO"
    version_status="NOT_RUN"
    ldd_not_found="NOT_RUN"
    if [[ -x "${tool}" ]]; then
        executable_state="YES"
    fi

    if command -v ldd >/dev/null 2>&1; then
        run_logged \
            "ldd_${safe_label}" \
            "${OUT_DIR}/ldd_${safe_label}.txt" \
            ldd "${tool}"
        ldd_status="${LAST_STATUS}"
        set +e
        "${PYTHON}" - "${OUT_DIR}/ldd_${safe_label}.txt" <<'PY'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").lower()
raise SystemExit(0 if "not found" in text else 1)
PY
        has_not_found=$?
        set -e
        if (( has_not_found == 0 )); then
            ldd_not_found="YES"
            ((LDD_NOT_FOUND_TOOLS += 1))
            if "${PYTHON}" - "${OUT_DIR}/ldd_${safe_label}.txt" <<'PY'
import sys
from pathlib import Path
lines = Path(sys.argv[1]).read_text(
    encoding="utf-8", errors="replace"
).splitlines()
unresolved = any(
    "librocm_sysdeps_zstd.so.1" in line and "not found" in line.lower()
    for line in lines
)
raise SystemExit(0 if unresolved else 1)
PY
            then
                ZSTD_UNRESOLVED=1
            fi
        else
            ldd_not_found="NO"
        fi
        if (( ldd_status != 0 )); then
            echo "WARNING: ldd returned status ${ldd_status} for ${label}."
        fi
    else
        LDD_UNAVAILABLE=1
        ldd_not_found="UNAVAILABLE"
        printf 'SKIP: ldd is unavailable; dependencies were not enumerated for %s.\n' \
            "${tool}" | tee "${OUT_DIR}/ldd_${safe_label}.txt"
    fi

    if [[ -x "${tool}" ]]; then
        run_logged \
            "version_${safe_label}" \
            "${OUT_DIR}/version_${safe_label}.txt" \
            "${tool}" --version
        version_status="${LAST_STATUS}"
        if (( version_status == 0 )); then
            launchable_state="YES"
            set_tool_launchability "${label}" 1
        else
            ((version_failures += 1))
            if "${PYTHON}" - "${OUT_DIR}/version_${safe_label}.txt" <<'PY'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
raise SystemExit(0 if "librocm_sysdeps_zstd.so.1" in text else 1)
PY
            then
                ZSTD_UNRESOLVED=1
            fi
        fi
    else
        ((version_failures += 1))
        printf 'SKIP: file exists but is not executable: %s\n' "${tool}" \
            | tee "${OUT_DIR}/version_${safe_label}.txt"
    fi
    printf '%s\tYES\t%s\t%s\t%s\t%s\t%s\n' \
        "${label}" \
        "${executable_state}" \
        "${launchable_state}" \
        "${version_status}" \
        "${ldd_not_found}" \
        "${tool}" | tee -a "${TOOL_INVENTORY}"
done

toolchain_level="PASS"
toolchain_detail="core tools exist and launch; dependency/version probes passed"
if (( LLVM_MC_LAUNCHABLE == 0 && CLANG_LAUNCHABLE == 0 )); then
    toolchain_level="FAIL"
    toolchain_detail="llvm-mc and clang are both missing, non-executable, or unlaunchable"
    TOOLCHAIN_ROOT_CAUSE="no launchable core assembler"
    HARD_FAIL=1
elif (( LLVM_OBJDUMP_LAUNCHABLE == 0 )); then
    toolchain_level="FAIL"
    toolchain_detail="llvm-objdump is missing, non-executable, or unlaunchable"
    TOOLCHAIN_ROOT_CAUSE="no launchable core disassembler"
    HARD_FAIL=1
elif (( LLVM_MC_LAUNCHABLE == 0 || CLANG_LAUNCHABLE == 0 )); then
    toolchain_level="WARN"
    toolchain_detail="only one independent assembler route exists and launches"
elif (( version_failures != 0 || LDD_NOT_FOUND_TOOLS != 0 )); then
    toolchain_level="WARN"
    toolchain_detail="${version_failures} version/launch failure(s), ${LDD_NOT_FOUND_TOOLS} tool(s) with unresolved ldd dependencies"
elif (( LLVM_READELF_LAUNCHABLE == 0 || LLVM_READOBJ_LAUNCHABLE == 0 || LLVM_NM_LAUNCHABLE == 0 )); then
    toolchain_level="WARN"
    toolchain_detail="one or more optional ELF metadata tools are missing or unlaunchable"
fi
if (( ZSTD_UNRESOLVED != 0 )); then
    TOOLCHAIN_ROOT_CAUSE="dynamic loader cannot resolve librocm_sysdeps_zstd.so.1"
    {
        echo "ROOT CAUSE: librocm_sysdeps_zstd.so.1 remains unresolved after runtime-path discovery."
        echo "Remediation (no installation performed): point LLVM_RUNTIME_LIBS at the existing SDK library directories, for example:"
        echo "  LLVM_RUNTIME_LIBS=/opt/venv/lib/pythonX.Y/site-packages/_rocm_sdk_core/lib:/opt/venv/lib/pythonX.Y/site-packages/_rocm_sdk_devel/lib/rocm_sysdeps bash $0 [OUT_DIR]"
        echo "or prepend those same existing directories to LD_LIBRARY_PATH before rerunning."
        echo "See ${runtime_discovery_report} and ${OUT_DIR}/ldd_*.txt for discovered and unresolved paths."
    } | tee "${OUT_DIR}/runtime_loader_remediation.txt" >&2
fi
record_experiment "1. Toolchain inventory" "${toolchain_level}" "${toolchain_detail}"

LLVM_OBJDUMP_SHOW_RAW_SUPPORTED=0
LLVM_OBJDUMP_SYMBOLIZE_SUPPORTED=0
LLVM_READOBJ_AMDGPU_METADATA_SUPPORTED=0

probe_help_capability() {
    local label="$1"
    local tool="$2"
    local option="$3"
    local stdout_file="${OUT_DIR}/${label}.help.txt"
    local stderr_file="${OUT_DIR}/${label}.help.stderr.txt"
    local rendered
    rendered="$(quote_command "${tool}" --help)"
    printf '\nCOMMAND [%s_help]: %s\n' "${label}" "${rendered}"
    set +e
    "${tool}" --help > "${stdout_file}" 2> "${stderr_file}"
    local status=$?
    set -e
    record_command_status "${label}_help" "${status}" "${rendered}"
    printf 'STATUS  [%s_help]: %s\n' "${label}" "${status}"
    if (( status != 0 )); then
        printf 'WARNING: capability probe failed for %s; optional flag %s will not be used.\n' \
            "${tool}" "${option}" >&2
        return 1
    fi
    "${PYTHON}" - "${stdout_file}" "${option}" <<'PY'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
raise SystemExit(0 if sys.argv[2] in text else 1)
PY
}

if (( LLVM_OBJDUMP_LAUNCHABLE )); then
    if probe_help_capability \
        "probe_llvm_objdump_show_raw" "${LLVM_OBJDUMP}" "--show-raw-insn"; then
        LLVM_OBJDUMP_SHOW_RAW_SUPPORTED=1
    fi
    if probe_help_capability \
        "probe_llvm_objdump_symbolize" "${LLVM_OBJDUMP}" "--symbolize-operands"; then
        LLVM_OBJDUMP_SYMBOLIZE_SUPPORTED=1
    fi
fi
if (( LLVM_READOBJ_LAUNCHABLE )) && probe_help_capability \
    "probe_llvm_readobj_amdgpu_metadata" \
    "${LLVM_READOBJ}" \
    "--amdgpu-metadata"; then
    LLVM_READOBJ_AMDGPU_METADATA_SUPPORTED=1
fi
{
    printf 'llvm-objdump --show-raw-insn supported: %s\n' \
        "${LLVM_OBJDUMP_SHOW_RAW_SUPPORTED}"
    printf 'llvm-objdump --symbolize-operands supported: %s\n' \
        "${LLVM_OBJDUMP_SYMBOLIZE_SUPPORTED}"
    printf 'llvm-readobj --amdgpu-metadata supported: %s\n' \
        "${LLVM_READOBJ_AMDGPU_METADATA_SUPPORTED}"
} | tee "${OUT_DIR}/llvm_capabilities.txt"

banner "2. Input integrity, syntax, directives, and symbols"

target_exists=0
reference_exists=0
if [[ -f "${TARGET_S}" && -r "${TARGET_S}" ]]; then
    target_exists=1
else
    echo "ERROR: target source is missing or unreadable: ${TARGET_S}" >&2
    HARD_FAIL=1
fi
if [[ -f "${REFERENCE_S}" && -r "${REFERENCE_S}" ]]; then
    reference_exists=1
else
    echo "WARNING: reference source is missing or unreadable: ${REFERENCE_S}" >&2
fi

identical_inputs=0
cmp_state="not-run"
if (( target_exists && reference_exists )); then
    if command -v cmp >/dev/null 2>&1; then
        cmp_command="$(quote_command cmp -s -- "${TARGET_S}" "${REFERENCE_S}")"
        printf 'COMMAND [cmp_target_reference]: %s\n' "${cmp_command}"
        set +e
        cmp -s -- "${TARGET_S}" "${REFERENCE_S}"
        cmp_status=$?
        set -e
        record_command_status "cmp_target_reference" "${cmp_status}" "${cmp_command}"
        if (( cmp_status == 0 )); then
            identical_inputs=1
            cmp_state="identical"
            echo "WARNING: target and reference are byte-for-byte identical; the final 128x128-symbol rewrite may not have started."
        elif (( cmp_status == 1 )); then
            cmp_state="different"
            echo "PASS: target and reference differ."
        else
            cmp_state="error"
            echo "WARNING: cmp could not compare the inputs (status ${cmp_status})."
        fi
    else
        cmp_state="cmp-unavailable"
        echo "WARNING: cmp is unavailable; SHA256 values below provide a fallback comparison."
    fi
fi

if (( target_exists )); then
    cp -- "${TARGET_S}" "${OUT_DIR}/target.original.s"
    cp -- "${TARGET_S}" "${OUT_DIR}/target.assembly.s"
    printf 'Assembly input transformation: exact byte-for-byte copy; no instructions, directives, or comments changed.\n' \
        | tee "${OUT_DIR}/target.assembly.transformation.txt"
fi
if (( reference_exists )); then
    cp -- "${REFERENCE_S}" "${OUT_DIR}/reference.original.s"
fi

input_probe_command="$(quote_command "${PYTHON}" - "${TARGET_S}" "${REFERENCE_S}")"
printf 'COMMAND [input_probe]: %s <embedded-python>\n' "${input_probe_command}"
set +e
"${PYTHON}" - "${TARGET_S}" "${REFERENCE_S}" 2> >(tee "${OUT_DIR}/input_integrity.stderr.txt" >&2) <<'PY' \
    | tee "${OUT_DIR}/input_integrity.txt"
import hashlib
import os
import re
import sys
from pathlib import Path

paths = [Path(value) for value in sys.argv[1:]]
for role, path in zip(("TARGET", "REFERENCE"), paths):
    print(f"\n[{role}]")
    print(f"requested/real path: {path.resolve(strict=False)}")
    if not path.is_file():
        print("state: MISSING")
        continue
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    print("state: readable regular file")
    print(f"size_bytes: {len(data)}")
    print(f"line_count: {len(text.splitlines())}")
    print(f"sha256: {hashlib.sha256(data).hexdigest()}")
    print(f"CRLF_count: {data.count(bytes([13, 10]))}")
    print(f"NUL_count: {data.count(bytes([0]))}")

target = paths[0]
if target.is_file():
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    print("\n[TARGET DIRECTIVES / SYMBOLS / RESOURCES]")
    patterns = (
        r"^\s*\.amdgcn_target\b",
        r"^\s*\.amdhsa_code_object_version\b",
        r"^\s*\.(?:protected|globl|global|type|size)\b",
        r"^\s*\.amdhsa_kernel\b",
        r"^\s*\.end_amdhsa_kernel\b",
        r"^\s*\.amdhsa_(?:group_segment_fixed_size|private_segment_fixed_size|"
        r"kernarg_size|user_sgpr_count|wavefront_size32|next_free_vgpr|"
        r"next_free_sgpr|named_barrier_count|reserve_vcc)\b",
        r"^\s*\.set\s+.*(?:num_vgpr|num_agpr|numbered_sgpr|num_named_barrier)\b",
        r"^\s*\.(?:name|symbol|group_segment_fixed_size|sgpr_count|vgpr_count):",
        r"^\s*amdhsa\.target:",
    )
    combined = re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)
    for number, line in enumerate(lines, 1):
        if combined.search(line):
            print(f"{number}: {line}")
    objdump_comments = sum(
        bool(re.search(r";\s*[0-9a-f]{8,16}:\s+[0-9a-f]", line, re.IGNORECASE))
        for line in lines
    )
    instruction_like = sum(
        bool(re.match(r"^\s*[a-z][a-z0-9_]*\b", re.sub(r";.*$", "", line)))
        for line in lines
    )
    print("\n[TARGET SYNTAX SUMMARY]")
    print(f"objdump_address_encoding_comments: {objdump_comments}")
    print(f"instruction_like_lines: {instruction_like}")
    print("comment_syntax: semicolon suffixes are valid AMDGPU assembler comments")
PY
input_probe_status=${PIPESTATUS[0]}
set -e
record_command_status "input_probe" "${input_probe_status}" "${input_probe_command} <embedded-python>"

input_level="PASS"
input_detail="target/reference integrity recorded; cmp=${cmp_state}"
if (( ! target_exists )); then
    input_level="FAIL"
    input_detail="target source is unavailable"
elif (( ! reference_exists )); then
    input_level="WARN"
    input_detail="target is readable but reference is unavailable"
elif (( identical_inputs )); then
    input_level="WARN"
    input_detail="target is byte-for-byte identical to reference"
elif (( input_probe_status != 0 )); then
    input_level="WARN"
    input_detail="input probe failed with status ${input_probe_status}"
fi
record_experiment "2. Input integrity" "${input_level}" "${input_detail}"

has_elf_magic() {
    "${PYTHON}" - "$1" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
raise SystemExit(0 if path.is_file() and path.stat().st_size > 0 and path.read_bytes()[:4] == b"\x7fELF" else 1)
PY
}

ELF_SUCCESS_INVALID=0
LAST_ELF_REASON=""
attempt_elf_command() {
    local name="$1"
    local output="$2"
    local log_file="$3"
    shift 3
    rm -f -- "${output}"
    run_logged "${name}" "${log_file}" "$@"
    if (( LAST_STATUS != 0 )); then
        LAST_ELF_REASON="command failed with status ${LAST_STATUS}"
        return 1
    fi
    if ! has_elf_magic "${output}"; then
        LAST_ELF_REASON="command returned success but output is not a non-empty ELF"
        ((ELF_SUCCESS_INVALID += 1))
        HARD_FAIL=1
        echo "ERROR: ${name} succeeded but did not produce a valid ELF: ${output}" >&2
        return 1
    fi
    LAST_ELF_REASON="valid non-empty ELF"
    printf 'PASS: %s produced %s\n' "${name}" "${output}"
    return 0
}

validate_et_dyn_amdgpu() {
    local file="$1"
    local report="$2"
    set +e
    "${PYTHON}" - "${file}" <<'PY' | tee "${report}"
import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = path.read_bytes() if path.is_file() else b""
if len(data) < 20 or data[:4] != b"\x7fELF":
    print(f"INVALID: not a complete ELF header: {path}")
    raise SystemExit(1)
byte_order = "<" if data[5] == 1 else ">" if data[5] == 2 else None
if byte_order is None:
    print(f"INVALID: unknown ELF byte order: {data[5]}")
    raise SystemExit(1)
elf_type, machine = struct.unpack_from(byte_order + "HH", data, 16)
print(f"path: {path}")
print(f"ELF e_type: {elf_type} ({'ET_DYN' if elf_type == 3 else 'not ET_DYN'})")
print(f"ELF e_machine: {machine} ({'EM_AMDGPU' if machine == 224 else 'not EM_AMDGPU'})")
raise SystemExit(0 if elf_type == 3 and machine == 224 else 1)
PY
    local status=${PIPESTATUS[0]}
    set -e
    return "${status}"
}

banner "3. Independent gfx1250 syntax and assembly routes"

MC_OK=0
CLANG_OK=0
MC_OBJ=""
CLANG_OBJ=""
ASSEMBLY_SOURCE_USED="${OUT_DIR}/target.assembly.s"

if (( target_exists && LLVM_MC_LAUNCHABLE )); then
    mc_candidate="${OUT_DIR}/target.llvm-mc.o"
    if attempt_elf_command \
        "assemble_llvm_mc" \
        "${mc_candidate}" \
        "${OUT_DIR}/assemble_llvm_mc.log" \
        "${LLVM_MC}" \
        -triple="${TRIPLE}" \
        -mcpu="${ARCH}" \
        -filetype=obj \
        "${ASSEMBLY_SOURCE_USED}" \
        -o "${mc_candidate}"; then
        MC_OK=1
        MC_OBJ="${mc_candidate}"
    fi
elif (( LLVM_MC_LAUNCHABLE == 0 )); then
    echo "WARNING: llvm-mc route skipped because llvm-mc is missing or unlaunchable."
fi

if (( target_exists && CLANG_LAUNCHABLE )); then
    clang_candidate="${OUT_DIR}/target.clang.o"
    if attempt_elf_command \
        "assemble_clang" \
        "${clang_candidate}" \
        "${OUT_DIR}/assemble_clang.log" \
        "${CLANG}" \
        -x assembler \
        -target "${TRIPLE}" \
        -mcpu="${ARCH}" \
        -mcode-object-version="${CODE_OBJECT_VERSION}" \
        -c "${ASSEMBLY_SOURCE_USED}" \
        -o "${clang_candidate}"; then
        CLANG_OK=1
        CLANG_OBJ="${clang_candidate}"
    fi
elif (( CLANG_LAUNCHABLE == 0 )); then
    echo "WARNING: independent clang assembler route skipped because clang is missing or unlaunchable."
fi

has_stale_objdump_comments() {
    "${PYTHON}" - "$1" <<'PY'
import re
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
pattern = re.compile(r";\s*[0-9a-f]{8,16}:\s+[0-9a-f]", re.IGNORECASE)
raise SystemExit(0 if pattern.search(text) else 1)
PY
}

diagnostics_suggest_comment_issue() {
    "${PYTHON}" - \
        "${OUT_DIR}/assemble_llvm_mc.log" \
        "${OUT_DIR}/assemble_clang.log" <<'PY'
import re
import sys
from pathlib import Path

diagnostics = "\n".join(
    Path(value).read_text(encoding="utf-8", errors="replace")
    for value in sys.argv[1:]
    if Path(value).is_file()
)
lexical_error = re.compile(
    r"unexpected token|unknown token|invalid token|junk at end of line|"
    r"expected (?:newline|end of statement)|unexpected ['\"]?[;<]",
    re.IGNORECASE,
)
raise SystemExit(0 if lexical_error.search(diagnostics) else 1)
PY
}

if (( target_exists && ! MC_OK && ! CLANG_OK )) \
    && has_stale_objdump_comments "${ASSEMBLY_SOURCE_USED}" \
    && diagnostics_suggest_comment_issue; then
    banner "3a. Comment-only sanitization fallback after original-source failures"
    sanitized_source="${OUT_DIR}/target.sanitized.s"
    transformation_report="${OUT_DIR}/target.sanitization.txt"
    sanitize_command="$(quote_command "${PYTHON}" - "${ASSEMBLY_SOURCE_USED}" "${sanitized_source}" "${transformation_report}")"
    printf 'COMMAND [sanitize_objdump_comments]: %s <embedded-python>\n' "${sanitize_command}"
    set +e
    "${PYTHON}" - "${ASSEMBLY_SOURCE_USED}" "${sanitized_source}" "${transformation_report}" <<'PY' \
        | tee "${OUT_DIR}/sanitize_objdump_comments.stdout.txt"
import re
import sys
from pathlib import Path

source, destination, report_path = map(Path, sys.argv[1:4])
pattern = re.compile(
    r"^(?P<code>.*?)(?:\s*;\s*[0-9a-f]{8,16}:\s+"
    r"(?:[0-9a-f]{8}(?:\s+|$))+"
    r"(?:<[^>]*>)?\s*)$",
    re.IGNORECASE,
)
changed = []
output = []
for number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
    match = pattern.match(raw)
    if match:
        output.append(match.group("code").rstrip())
        changed.append(number)
    else:
        output.append(raw)
destination.write_text("\n".join(output) + "\n", encoding="utf-8", newline="\n")
report = [
    "Sanitization was attempted only because every available original-copy assembly route failed.",
    "Transformation: remove only trailing '; ADDRESS: ENCODING [<symbol+offset>]' objdump comments.",
    "No instruction text, directive, label, metadata line, or other comment was removed.",
    f"source: {source}",
    f"destination: {destination}",
    f"changed_line_count: {len(changed)}",
    "changed_lines: " + (",".join(map(str, changed)) if changed else "none"),
]
Path(report_path).write_text("\n".join(report) + "\n", encoding="utf-8")
print("\n".join(report))
raise SystemExit(0 if changed else 1)
PY
    sanitize_status=${PIPESTATUS[0]}
    set -e
    record_command_status \
        "sanitize_objdump_comments" \
        "${sanitize_status}" \
        "${sanitize_command} <embedded-python>"

    if (( sanitize_status == 0 )); then
        ASSEMBLY_SOURCE_USED="${sanitized_source}"
        if (( LLVM_MC_LAUNCHABLE )); then
            mc_candidate="${OUT_DIR}/target.llvm-mc.sanitized.o"
            if attempt_elf_command \
                "assemble_llvm_mc_sanitized" \
                "${mc_candidate}" \
                "${OUT_DIR}/assemble_llvm_mc_sanitized.log" \
                "${LLVM_MC}" \
                -triple="${TRIPLE}" \
                -mcpu="${ARCH}" \
                -filetype=obj \
                "${ASSEMBLY_SOURCE_USED}" \
                -o "${mc_candidate}"; then
                MC_OK=1
                MC_OBJ="${mc_candidate}"
            fi
        fi
        if (( CLANG_LAUNCHABLE )); then
            clang_candidate="${OUT_DIR}/target.clang.sanitized.o"
            if attempt_elf_command \
                "assemble_clang_sanitized" \
                "${clang_candidate}" \
                "${OUT_DIR}/assemble_clang_sanitized.log" \
                "${CLANG}" \
                -x assembler \
                -target "${TRIPLE}" \
                -mcpu="${ARCH}" \
                -mcode-object-version="${CODE_OBJECT_VERSION}" \
                -c "${ASSEMBLY_SOURCE_USED}" \
                -o "${clang_candidate}"; then
                CLANG_OK=1
                CLANG_OBJ="${clang_candidate}"
            fi
        fi
    fi
elif (( target_exists && ! MC_OK && ! CLANG_OK )) \
    && has_stale_objdump_comments "${ASSEMBLY_SOURCE_USED}"; then
    echo "Sanitization skipped: assembly diagnostics did not implicate trailing objdump comments."
fi

VALID_OBJECT_COUNT=$((MC_OK + CLANG_OK))
assembly_level="PASS"
assembly_detail="llvm-mc and clang each produced a valid object from the exact copied source"
if (( VALID_OBJECT_COUNT == 0 )); then
    assembly_level="FAIL"
    assembly_detail="no assembler route produced a valid object"
    HARD_FAIL=1
elif (( MC_OK == 0 || CLANG_OK == 0 )); then
    assembly_level="WARN"
    assembly_detail="one valid object produced; the second independent route failed or was unavailable"
elif [[ "${ASSEMBLY_SOURCE_USED}" == "${OUT_DIR}/target.sanitized.s" ]]; then
    assembly_level="WARN"
    assembly_detail="both routes succeeded only after the documented comment-only sanitization"
fi
record_experiment "3. Assembly routes" "${assembly_level}" "${assembly_detail}"

PRIMARY_OBJ=""
if (( MC_OK )); then
    PRIMARY_OBJ="${MC_OBJ}"
elif (( CLANG_OK )); then
    PRIMARY_OBJ="${CLANG_OBJ}"
fi

banner "4. AMDHSA shared code-object link experiments"

LLD_LINK_OK=0
CLANG_LINK_OK=0
LLD_CO=""
CLANG_CO=""
PRIMARY_CO=""

if [[ -n "${PRIMARY_OBJ}" ]] && (( LD_LLD_LAUNCHABLE )); then
    lld_candidate="${OUT_DIR}/target.ld_lld.co"
    if attempt_elf_command \
        "link_ld_lld" \
        "${lld_candidate}" \
        "${OUT_DIR}/link_ld_lld.log" \
        "${LD_LLD}" \
        -shared \
        --no-undefined \
        -o "${lld_candidate}" \
        "${PRIMARY_OBJ}" \
        && validate_et_dyn_amdgpu \
            "${lld_candidate}" "${OUT_DIR}/link_ld_lld.elf_type.txt"; then
        LLD_LINK_OK=1
        LLD_CO="${lld_candidate}"
    else
        echo "ERROR: ld.lld output is not an ET_DYN EM_AMDGPU code object." >&2
        HARD_FAIL=1
    fi
elif (( LD_LLD_LAUNCHABLE == 0 )); then
    echo "WARNING: direct ld.lld link route skipped because ld.lld is missing or unlaunchable."
fi

if [[ -n "${PRIMARY_OBJ}" ]] && (( CLANG_LAUNCHABLE )); then
    clang_co_candidate="${OUT_DIR}/target.clang.co"
    if attempt_elf_command \
        "link_clang_driver" \
        "${clang_co_candidate}" \
        "${OUT_DIR}/link_clang_driver.log" \
        "${CLANG}" \
        -target "${TRIPLE}" \
        -mcpu="${ARCH}" \
        -mcode-object-version="${CODE_OBJECT_VERSION}" \
        -nostdlib \
        -Wl,--no-undefined \
        "${PRIMARY_OBJ}" \
        -o "${clang_co_candidate}" \
        && validate_et_dyn_amdgpu \
            "${clang_co_candidate}" "${OUT_DIR}/link_clang_driver.elf_type.txt"; then
        CLANG_LINK_OK=1
        CLANG_CO="${clang_co_candidate}"
    else
        echo "ERROR: clang-driver output is not an ET_DYN EM_AMDGPU code object." >&2
        HARD_FAIL=1
    fi
fi

if (( CLANG_LINK_OK )); then
    PRIMARY_CO="${CLANG_CO}"
elif (( LLD_LINK_OK )); then
    PRIMARY_CO="${LLD_CO}"
fi

link_level="PASS"
link_detail="loadable-style shared AMDHSA code object produced"
if [[ -z "${PRIMARY_OBJ}" ]]; then
    link_level="WARN"
    link_detail="link skipped because no object assembled"
elif [[ -z "${PRIMARY_CO}" ]]; then
    link_level="WARN"
    link_detail="object assembly succeeded, but no link route produced a code object; object checks continue"
elif (( LLD_LINK_OK == 0 || CLANG_LINK_OK == 0 )); then
    link_level="WARN"
    link_detail="one code-object link route succeeded while the other failed or was unavailable"
fi
record_experiment "4. Code-object link" "${link_level}" "${link_detail}"

banner "5. Complete disassembly plus ELF section and symbol tables"

AUDIT_MANIFEST="${OUT_DIR}/audit_inputs.tsv"
printf 'label\tkind\tpath\n' > "${AUDIT_MANIFEST}"
if (( target_exists )); then
    printf 'target_source\tsource\t%s\n' "${OUT_DIR}/target.original.s" >> "${AUDIT_MANIFEST}"
fi
if (( reference_exists )); then
    printf 'reference_source\tsource\t%s\n' "${OUT_DIR}/reference.original.s" >> "${AUDIT_MANIFEST}"
fi

declare -a INSPECT_LABELS=()
declare -a INSPECT_KINDS=()
declare -a INSPECT_FILES=()
if (( MC_OK )); then
    INSPECT_LABELS+=("llvm_mc_object")
    INSPECT_KINDS+=("object")
    INSPECT_FILES+=("${MC_OBJ}")
fi
if (( CLANG_OK )); then
    INSPECT_LABELS+=("clang_object")
    INSPECT_KINDS+=("object")
    INSPECT_FILES+=("${CLANG_OBJ}")
fi
if (( LLD_LINK_OK )); then
    INSPECT_LABELS+=("lld_code_object")
    INSPECT_KINDS+=("code_object")
    INSPECT_FILES+=("${LLD_CO}")
fi
if (( CLANG_LINK_OK )); then
    INSPECT_LABELS+=("clang_code_object")
    INSPECT_KINDS+=("code_object")
    INSPECT_FILES+=("${CLANG_CO}")
fi

DISASM_VALID_COUNT=0
DISASM_INVALID_COUNT=0
ELF_HEADER_INVALID_COUNT=0
METADATA_TOOL_FAILURES=0
METADATA_NOTE_DECODES=0
METADATA_SPECIALIZED_SKIPS=0
UNRESOLVED_SYMBOL_FILES=0

validate_disassembly() {
    local disassembly="$1"
    local report="$2"
    set +e
    "${PYTHON}" - "${disassembly}" <<'PY' | tee "${report}"
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
count = 0
raw_encoding_lines = 0
bad = []
for number, raw in enumerate(text.splitlines(), 1):
    body = re.sub(r"//.*$", "", raw).strip()
    if not body or body.endswith(":") or body.startswith("Disassembly of section"):
        continue
    if re.match(r"^\.[A-Za-z]", body):
        if body.startswith((".long", ".word", ".byte")):
            bad.append((number, raw))
        continue
    match = re.match(r"^([a-z][a-z0-9_.]*)\b", body)
    if match:
        token = match.group(1)
        if token not in {"file", "format"}:
            count += 1
    if re.search(r"//\s*[0-9a-f]+:\s+[0-9a-f]", raw, re.IGNORECASE):
        raw_encoding_lines += 1
    if re.search(r"<unknown>|\bunknown opcode\b|\binvalid instruction\b", raw, re.IGNORECASE):
        bad.append((number, raw))
print(f"path: {path}")
print(f"instruction_like_lines: {count}")
print(f"raw_encoding_lines: {raw_encoding_lines}")
print(f"undecoded_or_invalid_lines: {len(bad)}")
for number, line in bad[:20]:
    print(f"  {number}: {line}")
if count == 0 or bad:
    raise SystemExit(1)
PY
    local status=${PIPESTATUS[0]}
    set -e
    return "${status}"
}

inspect_elf() {
    local label="$1"
    local kind="$2"
    local file="$3"
    printf '\n--- Inspecting %s: %s ---\n' "${label}" "${file}"
    printf '%s\t%s\t%s\n' "${label}" "${kind}" "${file}" >> "${AUDIT_MANIFEST}"

    run_capture_stdout \
        "${label}_file_header" \
        "${OUT_DIR}/${label}.objdump.file_header.txt" \
        "${OUT_DIR}/${label}.objdump.file_header.stderr.txt" \
        "${LLVM_OBJDUMP}" -f "${file}"
    if (( LAST_STATUS != 0 )); then
        echo "ERROR: llvm-objdump could not parse ELF header for ${file}" >&2
        ((ELF_HEADER_INVALID_COUNT += 1))
        HARD_FAIL=1
    fi

    local disassembly="${OUT_DIR}/${label}.disassembly.txt"
    local -a disassembly_args=(
        --disassemble
        --mcpu="${ARCH}"
    )
    if (( LLVM_OBJDUMP_SHOW_RAW_SUPPORTED )); then
        disassembly_args+=(--show-raw-insn)
    fi
    if (( LLVM_OBJDUMP_SYMBOLIZE_SUPPORTED )); then
        disassembly_args+=(--symbolize-operands)
    fi
    disassembly_args+=("${file}")
    run_capture_stdout \
        "${label}_disassemble_capability_selected" \
        "${disassembly}" \
        "${OUT_DIR}/${label}.disassembly.capability_selected.stderr.txt" \
        "${LLVM_OBJDUMP}" \
        "${disassembly_args[@]}"
    if (( LAST_STATUS != 0 )); then
        echo "WARNING: capability-selected objdump command genuinely failed; retrying the portable gfx1250 form." >&2
        run_capture_stdout \
            "${label}_disassemble_fallback" \
            "${disassembly}" \
            "${OUT_DIR}/${label}.disassembly.fallback.stderr.txt" \
            "${LLVM_OBJDUMP}" -d --mcpu="${ARCH}" "${file}"
    fi

    if (( LAST_STATUS == 0 )) && validate_disassembly \
        "${disassembly}" "${OUT_DIR}/${label}.disassembly.validation.txt"; then
        ((DISASM_VALID_COUNT += 1))
        printf '%s\tdisassembly\t%s\n' "${label}" "${disassembly}" >> "${AUDIT_MANIFEST}"
    else
        echo "ERROR: disassembly is empty, undecoded, or invalid for ${file}" >&2
        ((DISASM_INVALID_COUNT += 1))
        HARD_FAIL=1
        printf '%s\tdisassembly_invalid\t%s\n' "${label}" "${disassembly}" >> "${AUDIT_MANIFEST}"
    fi

    run_capture_stdout \
        "${label}_objdump_sections_symbols" \
        "${OUT_DIR}/${label}.objdump.sections_symbols.txt" \
        "${OUT_DIR}/${label}.objdump.sections_symbols.stderr.txt" \
        "${LLVM_OBJDUMP}" --section-headers --syms "${file}"
    if (( LAST_STATUS != 0 )); then
        ((METADATA_TOOL_FAILURES += 1))
    fi

    if (( LLVM_READELF_LAUNCHABLE )); then
        run_capture_stdout \
            "${label}_readelf" \
            "${OUT_DIR}/${label}.readelf.txt" \
            "${OUT_DIR}/${label}.readelf.stderr.txt" \
            "${LLVM_READELF}" -h -S -s -n -W "${file}"
        if (( LAST_STATUS != 0 )); then
            ((METADATA_TOOL_FAILURES += 1))
        else
            ((METADATA_NOTE_DECODES += 1))
        fi
    fi

    if (( LLVM_READOBJ_LAUNCHABLE )); then
        run_capture_stdout \
            "${label}_readobj" \
            "${OUT_DIR}/${label}.readobj.txt" \
            "${OUT_DIR}/${label}.readobj.stderr.txt" \
            "${LLVM_READOBJ}" \
            --file-headers \
            --sections \
            --symbols \
            --notes \
            "${file}"
        if (( LAST_STATUS != 0 )); then
            ((METADATA_TOOL_FAILURES += 1))
        else
            ((METADATA_NOTE_DECODES += 1))
        fi

        if (( LLVM_READOBJ_AMDGPU_METADATA_SUPPORTED )); then
            run_capture_stdout \
                "${label}_readobj_amdgpu_metadata" \
                "${OUT_DIR}/${label}.readobj.amdgpu_metadata.txt" \
                "${OUT_DIR}/${label}.readobj.amdgpu_metadata.stderr.txt" \
                "${LLVM_READOBJ}" --amdgpu-metadata "${file}"
            if (( LAST_STATUS != 0 )); then
                echo "WARNING: supported llvm-readobj --amdgpu-metadata failed for ${label}." >&2
                ((METADATA_TOOL_FAILURES += 1))
            fi
        else
            ((METADATA_SPECIALIZED_SKIPS += 1))
            printf 'SKIP: llvm-readobj --amdgpu-metadata is unsupported; generic --notes/readelf decoding was retained for %s.\n' \
                "${label}" \
                | tee "${OUT_DIR}/${label}.readobj.amdgpu_metadata.SKIP.txt"
        fi

        run_capture_stdout \
            "${label}_readobj_relocations" \
            "${OUT_DIR}/${label}.readobj.relocations.txt" \
            "${OUT_DIR}/${label}.readobj.relocations.stderr.txt" \
            "${LLVM_READOBJ}" --relocations "${file}"
        if (( LAST_STATUS != 0 )); then
            ((METADATA_TOOL_FAILURES += 1))
        fi
    fi

    if (( LLVM_NM_LAUNCHABLE )); then
        run_capture_stdout \
            "${label}_nm_defined" \
            "${OUT_DIR}/${label}.nm.defined.txt" \
            "${OUT_DIR}/${label}.nm.defined.stderr.txt" \
            "${LLVM_NM}" --defined-only --print-size --size-sort "${file}"
        if (( LAST_STATUS != 0 )); then
            run_capture_stdout \
                "${label}_nm_defined_fallback" \
                "${OUT_DIR}/${label}.nm.defined.txt" \
                "${OUT_DIR}/${label}.nm.defined.fallback.stderr.txt" \
                "${LLVM_NM}" --defined-only "${file}"
        fi
        if (( LAST_STATUS != 0 )); then
            ((METADATA_TOOL_FAILURES += 1))
        fi

        run_capture_stdout \
            "${label}_nm_undefined" \
            "${OUT_DIR}/${label}.nm.undefined.txt" \
            "${OUT_DIR}/${label}.nm.undefined.stderr.txt" \
            "${LLVM_NM}" --undefined-only "${file}"
        if (( LAST_STATUS == 0 )) && [[ -s "${OUT_DIR}/${label}.nm.undefined.txt" ]]; then
            echo "WARNING: unresolved symbols were reported for ${label}:"
            tee "${OUT_DIR}/${label}.nm.undefined.warning.txt" \
                < "${OUT_DIR}/${label}.nm.undefined.txt"
            ((UNRESOLVED_SYMBOL_FILES += 1))
        elif (( LAST_STATUS != 0 )); then
            ((METADATA_TOOL_FAILURES += 1))
        fi
    fi
}

if (( LLVM_OBJDUMP_LAUNCHABLE )); then
    for index in "${!INSPECT_FILES[@]}"; do
        inspect_elf \
            "${INSPECT_LABELS[${index}]}" \
            "${INSPECT_KINDS[${index}]}" \
            "${INSPECT_FILES[${index}]}"
    done
else
    echo "ERROR: llvm-objdump is missing or unlaunchable; produced objects cannot be validated." >&2
    if (( VALID_OBJECT_COUNT > 0 )); then
        DISASM_INVALID_COUNT="${VALID_OBJECT_COUNT}"
    fi
    HARD_FAIL=1
fi

disasm_level="PASS"
disasm_detail="${DISASM_VALID_COUNT} produced ELF file(s) fully disassembled"
if (( VALID_OBJECT_COUNT == 0 )); then
    disasm_level="WARN"
    disasm_detail="UNAVAILABLE: no object assembled; no produced disassembly was available to validate"
elif (( DISASM_INVALID_COUNT != 0 || ELF_HEADER_INVALID_COUNT != 0 )); then
    disasm_level="FAIL"
    disasm_detail="invalid ELF header or disassembly detected"
elif [[ -z "${PRIMARY_CO}" ]]; then
    disasm_level="WARN"
    disasm_detail="object disassembly is valid; linked code-object disassembly is unavailable"
fi
record_experiment "5. Disassembly" "${disasm_level}" "${disasm_detail}"

banner "6-8. Metadata, opcode audit, and design-invariant analysis"

AUDIT_PY="${OUT_DIR}/f4gemm_static_audit.py"
cat > "${AUDIT_PY}" <<'PY'
#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

OLD_SYMBOL = "f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps"
NEW_SYMBOL = "f4gemm_bf16_mxfp4_ABpreShuffle_128x128_4x4_ps"

CATEGORY_NAMES = (
    "wmma",
    "ds_load_b128",
    "ds_load_b32",
    "ds_store_or_write",
    "tensor_load_to_lds",
    "tensor_store_from_lds",
    "s_wait_dscnt",
    "s_wait_tensorcnt",
    "s_wait_alu",
    "wg_barrier_signal",
    "wg_barrier_wait",
    "cluster_barrier_signal",
    "cluster_barrier_wait",
    "conditional_branch",
    "unconditional_branch",
    "conversion",
)

NEW_RING_ADDRESSES = (
    0x00000, 0x04000, 0x08000, 0x0C000,
    0x10000, 0x10400, 0x10800, 0x10C00,
    0x11000, 0x11400, 0x11800, 0x11C00,
    0x12000, 0x16000, 0x1A000, 0x1E000,
    0x22000,
)
OLD_ONLY_RING_ADDRESSES = (0x30000, 0x38000, 0x40000, 0x48000, 0x50000)
SELECTED_CONSTANTS = tuple(dict.fromkeys(
    NEW_RING_ADDRESSES
    + OLD_ONLY_RING_ADDRESSES
    + (0x40, 0x80, 0x100, 0x400, 0x1000, 0x2000, 0x8000, 0x26000, 0x2A000)
))


def sha256_path(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_comments(line):
    line = re.sub(r"/\*.*?\*/", "", line)
    line = re.sub(r";.*$", "", line)
    line = re.sub(r"//.*$", "", line)
    return line.rstrip()


def source_labels(text):
    labels = []
    in_metadata = False
    for raw in text.splitlines():
        clean = strip_comments(raw).strip()
        if clean.startswith(".amdgpu_metadata"):
            in_metadata = True
            continue
        if clean.startswith(".end_amdgpu_metadata"):
            in_metadata = False
            continue
        if in_metadata:
            continue
        match = re.match(r"^([.$A-Za-z_][\w.$]*):\s*$", clean)
        if not match:
            match = re.match(r"^[0-9a-fA-F]+\s+<([^>]+)>:\s*$", clean)
        if match:
            labels.append(match.group(1))
    return labels


def instruction_records(text):
    records = []
    in_metadata = False
    current_label = "<entry>"
    for line_number, raw in enumerate(text.splitlines(), 1):
        clean = strip_comments(raw).strip()
        if clean.startswith(".amdgpu_metadata"):
            in_metadata = True
            continue
        if clean.startswith(".end_amdgpu_metadata"):
            in_metadata = False
            continue
        if in_metadata or not clean:
            continue
        label = re.match(r"^([.$A-Za-z_][\w.$]*):\s*$", clean)
        if not label:
            label = re.match(r"^[0-9a-fA-F]+\s+<([^>]+)>:\s*$", clean)
        if label:
            current_label = label.group(1)
            continue
        if clean.startswith(".") or clean.startswith("#"):
            continue
        # Most AMDGPU llvm-objdump output places address/encoding after //.
        # Also accept conventional address/byte prefixes if a tool emits them.
        clean = re.sub(
            r"^[0-9a-fA-F]+:\s+(?:(?:[0-9a-fA-F]{2}|[0-9a-fA-F]{8})\s+)+",
            "",
            clean,
        )
        match = re.match(r"^([a-z][a-z0-9_.]*)\b(?:\s+(.*))?$", clean)
        if not match:
            continue
        mnemonic = match.group(1).lower()
        if mnemonic in {"file", "format"}:
            continue
        records.append(
            {
                "line": line_number,
                "label_block": current_label,
                "mnemonic": mnemonic,
                "operands": (match.group(2) or "").strip(),
                "text": clean,
            }
        )
    return records


def first_operand(record):
    operands = record["operands"].strip()
    return re.split(r"[\s,]+", operands, maxsplit=1)[0].lower() if operands else ""


def categories(records):
    counts = Counter()
    ds_store_variants = Counter()
    conversion_variants = Counter()
    wmma_variants = Counter()
    for record in records:
        mnemonic = record["mnemonic"]
        operand = first_operand(record)
        if mnemonic.startswith("v_wmma"):
            counts["wmma"] += 1
            wmma_variants[mnemonic] += 1
        if mnemonic == "ds_load_b128":
            counts["ds_load_b128"] += 1
        if mnemonic == "ds_load_b32":
            counts["ds_load_b32"] += 1
        if re.match(r"^ds_(?:store|write)", mnemonic):
            counts["ds_store_or_write"] += 1
            ds_store_variants[mnemonic] += 1
        if mnemonic.startswith("tensor_load_to_lds"):
            counts["tensor_load_to_lds"] += 1
        if mnemonic.startswith("tensor_store_from_lds"):
            counts["tensor_store_from_lds"] += 1
        if mnemonic.startswith("s_wait_dscnt"):
            counts["s_wait_dscnt"] += 1
        if mnemonic.startswith("s_wait_tensorcnt"):
            counts["s_wait_tensorcnt"] += 1
        if mnemonic.startswith("s_wait_alu"):
            counts["s_wait_alu"] += 1
        if mnemonic == "s_barrier_signal" and operand in {"-1", "0xffffffff"}:
            counts["wg_barrier_signal"] += 1
        if mnemonic == "s_barrier_wait" and operand in {"0xffff", "65535"}:
            counts["wg_barrier_wait"] += 1
        if mnemonic == "s_barrier_signal" and operand in {"-3", "0xfffffffd"}:
            counts["cluster_barrier_signal"] += 1
        if mnemonic == "s_barrier_wait" and operand in {"0xfffd", "65533"}:
            counts["cluster_barrier_wait"] += 1
        if mnemonic.startswith("s_cbranch"):
            counts["conditional_branch"] += 1
        if mnemonic == "s_branch":
            counts["unconditional_branch"] += 1
        if (
            re.search(r"(?:^|_)(?:cvt|convert|pack)(?:_|$)", mnemonic)
            or mnemonic.startswith("v_cvt")
        ):
            counts["conversion"] += 1
            conversion_variants[mnemonic] += 1
    return (
        {name: counts[name] for name in CATEGORY_NAMES},
        dict(sorted(ds_store_variants.items())),
        dict(sorted(conversion_variants.items())),
        dict(sorted(wmma_variants.items())),
    )


def parse_integer_literal(token):
    token = token.strip()
    sign = -1 if token.startswith("-") else 1
    unsigned = token[1:] if token[:1] in {"+", "-"} else token
    if re.fullmatch(r"0[xX][0-9a-fA-F]+", unsigned):
        return sign * int(unsigned[2:], 16)
    if re.fullmatch(r"\d+", unsigned):
        return sign * int(unsigned, 10)
    raise ValueError(f"unsupported integer literal: {token!r}")


def remove_disassembly_pc_fields(line):
    line = re.sub(
        r"^\s*[0-9a-fA-F]{8,16}\s+<[^>]+>:\s*",
        "",
        line,
    )
    line = re.sub(
        r"^\s*[0-9a-fA-F]{4,16}:\s+(?:[0-9a-fA-F]{2,16}\s+)+",
        "",
        line,
    )
    return line


def literal_counts(text):
    counter = Counter()
    for raw in text.splitlines():
        clean = remove_disassembly_pc_fields(strip_comments(raw))
        clean = re.sub(r"\b(?:s|v|ttmp)\d+\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(
            r"\b[sv]\[\s*\d+\s*:\s*\d+\s*\]",
            " ",
            clean,
            flags=re.IGNORECASE,
        )
        for token in re.findall(
            r"(?<![A-Za-z0-9_])(?:-?0x[0-9a-fA-F]+|-?\d+)(?![A-Za-z0-9_])",
            clean,
        ):
            counter[parse_integer_literal(token)] += 1
    return counter


def run_literal_parser_self_tests():
    cases = {
        "0000000000001900": 1900,
        "172032": 172032,
        "-17": -17,
        "0x2A000": 0x2A000,
        "-0x2a": -42,
    }
    for token, expected in cases.items():
        actual = parse_integer_literal(token)
        assert actual == expected, (token, expected, actual)
    sample = (
        "0000000000001900 <kernel>:\n"
        "  s_mov_b32 s0, 42 // 000000001900: B0804009 00000002\n"
        "  s_add_i32 s1, s1, -7\n"
        "  s_mov_b32 s2, 0x2a\n"
    )
    counts = literal_counts(sample)
    assert counts[1900] == 0, counts
    assert counts[42] == 2, counts
    assert counts[-7] == 1, counts


def kernel_symbols(text):
    symbols = set()
    patterns = (
        r"(?m)^\s*\.amdhsa_kernel\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"(?m)^\s*\.globl\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"(?m)^\s*\.name:\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)",
        r"(?m)^\s*\.symbol:\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)\.kd",
    )
    for pattern in patterns:
        symbols.update(re.findall(pattern, text))
    return sorted(symbols)


def branch_inventory(records, labels, symbols):
    known = {value.lower() for value in labels} | {
        value.lower() for value in symbols
    }
    branches = []
    unresolved = []
    for record in records:
        mnemonic = record["mnemonic"]
        if mnemonic != "s_branch" and not mnemonic.startswith("s_cbranch"):
            continue
        target = first_operand(record)
        symbolic = bool(re.match(r"^[.$A-Za-z_][\w.$]*$", target))
        resolved = (not symbolic) or target in known
        entry = {
            "line": record["line"],
            "mnemonic": mnemonic,
            "target": target,
            "symbolic": symbolic,
            "resolved_in_text": resolved,
        }
        branches.append(entry)
        if symbolic and not resolved:
            unresolved.append(entry)
    return branches, unresolved


def normalized_instruction_sequence(records):
    sequence = []
    for record in records:
        mnemonic = re.sub(r"_e(?:32|64)$", "", record["mnemonic"])
        operands = re.sub(r"\s+", " ", record["operands"].strip().lower())
        sequence.append(f"{mnemonic} {operands}".rstrip())
    payload = "\n".join(sequence).encode()
    return sequence, hashlib.sha256(payload).hexdigest()


def analyze_input(label, kind, path):
    text = path.read_text(encoding="utf-8", errors="replace")
    records = instruction_records(text)
    labels = source_labels(text)
    symbols = kernel_symbols(text)
    category_counts, ds_variants, conversion_variants, wmma_variants = categories(records)
    branches, unresolved = branch_inventory(records, labels, symbols)
    literals = literal_counts(text)
    wmma_by_label = Counter(
        record["label_block"]
        for record in records
        if record["mnemonic"].startswith("v_wmma")
    )
    sequence, sequence_hash = normalized_instruction_sequence(records)
    return {
        "path": str(path),
        "kind": kind,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_path(path),
        "instruction_count": len(records),
        "normalized_instruction_sha256": sequence_hash,
        "counts": category_counts,
        "ds_store_write_variants": ds_variants,
        "conversion_variants": conversion_variants,
        "wmma_variants": wmma_variants,
        "kernel_symbols": symbols,
        "old_symbol_occurrences": text.count(OLD_SYMBOL),
        "new_symbol_occurrences": text.count(NEW_SYMBOL),
        "selected_constant_occurrences": {
            f"0x{value:x}": literals[value] for value in SELECTED_CONSTANTS
        },
        "new_ring_constant_occurrences": {
            f"0x{value:05x}": literals[value] for value in NEW_RING_ADDRESSES
        },
        "old_only_ring_constant_occurrences": {
            f"0x{value:x}": literals[value] for value in OLD_ONLY_RING_ADDRESSES
        },
        "tile_literal_occurrences": {
            "64_or_0x40": literals[64],
            "128_or_0x80": literals[128],
            "256_or_0x100": literals[256],
        },
        "tile_string_occurrences": {
            "64x64": len(re.findall(r"64x64", text, re.IGNORECASE)),
            "128x128": len(re.findall(r"128x128", text, re.IGNORECASE)),
            "256x256": len(re.findall(r"256x256", text, re.IGNORECASE)),
        },
        "label_count": len(labels),
        "labels": labels,
        "branch_count": len(branches),
        "branches": branches,
        "unresolved_symbolic_branches": unresolved,
        "wmma_by_label_block": dict(sorted(wmma_by_label.items())),
        "wmma_16_count_label_blocks": sorted(
            block for block, count in wmma_by_label.items() if count == 16
        ),
        "phase_boundary_proof": False,
        "phase_boundary_note": (
            "Label-block WMMA counts are inventory only. The parser does not "
            "infer semantic K-phase boundaries and therefore makes no "
            "per-phase 16-WMMA claim."
        ),
    }


def barrier_pair(records):
    for left, right in zip(records, records[1:]):
        if (
            left["mnemonic"] == "s_barrier_signal"
            and first_operand(left) in {"-1", "0xffffffff"}
            and right["mnemonic"] == "s_barrier_wait"
            and first_operand(right) in {"0xffff", "65535"}
        ):
            return True
    return False


def design_analysis(target_text, target_report):
    records = instruction_records(target_text)
    literals = literal_counts(target_text)
    counts = target_report["counts"]
    stores = [
        index
        for index, record in enumerate(records)
        if record["mnemonic"].startswith("tensor_store_from_lds")
    ]
    protocols = []
    for index in stores:
        protocols.append(
            {
                "instruction_index": index,
                "line": records[index]["line"],
                "wg_pair_within_32_instructions_before": barrier_pair(
                    records[max(0, index - 32):index]
                ),
                "wg_pair_within_32_instructions_after": barrier_pair(
                    records[index + 1:index + 33]
                ),
                "wait_tensorcnt_within_32_after": any(
                    record["mnemonic"].startswith("s_wait_tensorcnt")
                    for record in records[index + 1:index + 33]
                ),
            }
        )
    first_last_two_rounds = False
    if stores:
        first_last_two_rounds = (
            barrier_pair(records[max(0, stores[0] - 48):stores[0]])
            and barrier_pair(records[stores[-1] + 1:stores[-1] + 49])
        )

    ring_presence = {
        f"0x{value:05x}": literals[value] for value in NEW_RING_ADDRESSES
    }
    old_presence = {
        f"0x{value:x}": literals[value] for value in OLD_ONLY_RING_ADDRESSES
    }
    warnings = []
    missing_ring = [name for name, count in ring_presence.items() if count == 0]
    if missing_ring:
        warnings.append(
            "HEURISTIC WARNING: expected packed-ring literals missing: "
            + ", ".join(missing_ring)
        )
    if literals[0x2A000] == 0 and not re.search(r"\b172032\b", target_text):
        warnings.append(
            "HEURISTIC WARNING: candidate fixed-LDS end/resource literal "
            "0x2A000 (172032) is absent"
        )
    if literals[0x22000] == 0:
        warnings.append(
            "HEURISTIC WARNING: input-ring/output-staging boundary literal "
            "0x22000 is absent"
        )
    if literals[0x80] == 0 and not re.search(r"(?<!\w)128(?!\w)", target_text):
        warnings.append("HEURISTIC WARNING: 128 tile/wave offset literal was not found")
    if literals[0x40] == 0 and not re.search(r"(?<!\w)64(?!\w)", target_text):
        warnings.append("HEURISTIC WARNING: 64 tile/wave offset literal was not found")
    if counts["s_wait_alu"] == 0:
        warnings.append("s_wait_alu is absent")
    if not first_last_two_rounds:
        warnings.append(
            "HEURISTIC WARNING: nearby-barrier search did not find two output "
            "WG rounds (before first and after last output TDM); this is not "
            "semantic proof"
        )
    if counts["wg_barrier_signal"] != counts["wg_barrier_wait"]:
        warnings.append(
            "HEURISTIC WARNING: raw WG barrier signal/wait counts are not "
            f"balanced ({counts['wg_barrier_signal']} vs "
            f"{counts['wg_barrier_wait']}); counts do not prove protocol semantics"
        )
    if counts["cluster_barrier_signal"] == 0 or counts["cluster_barrier_wait"] == 0:
        warnings.append(
            "HEURISTIC WARNING: raw cluster barrier signal/wait presence is incomplete"
        )
    if counts["cluster_barrier_signal"] != counts["cluster_barrier_wait"]:
        warnings.append(
            "HEURISTIC WARNING: raw cluster barrier signal/wait counts are not "
            f"balanced ({counts['cluster_barrier_signal']} vs "
            f"{counts['cluster_barrier_wait']}); counts do not prove protocol semantics"
        )
    present_old = [name for name, count in old_presence.items() if count]
    if present_old:
        warnings.append(
            "HEURISTIC WARNING: old-only ring/LDS literals remain: "
            + ", ".join(present_old)
        )
    if target_report["old_symbol_occurrences"]:
        warnings.append("original 256x256 kernel symbol remains")
    if target_report["new_symbol_occurrences"] == 0:
        warnings.append("new 128x128 contract kernel symbol is absent")
    if re.search(
        r"(?mi)^\s*\.amdhsa_group_segment_fixed_size\s+327680\b",
        target_text,
    ):
        warnings.append("old 327680-byte LDS descriptor remains")
    if re.search(
        r"(?mi)^\s*\.amdhsa_next_free_vgpr\s+1024\b",
        target_text,
    ):
        warnings.append("old next_free_vgpr=1024 descriptor remains")

    return {
        "heuristic_notice": (
            "Literal presence, nearby-barrier windows, and raw barrier counts are "
            "search aids only; they are not semantic proof and never hard failures."
        ),
        "new_ring_address_occurrences": ring_presence,
        "old_only_ring_address_occurrences": old_presence,
        "output_staging_boundary_0x22000_occurrences": literals[0x22000],
        "fixed_lds_end_0x2a000_occurrences": literals[0x2A000],
        "tile_offset_128_literal_occurrences": literals[0x80],
        "wave_offset_64_literal_occurrences": literals[0x40],
        "wait_alu_count": counts["s_wait_alu"],
        "wg_barrier_signal_count": counts["wg_barrier_signal"],
        "wg_barrier_wait_count": counts["wg_barrier_wait"],
        "cluster_barrier_signal_count": counts["cluster_barrier_signal"],
        "cluster_barrier_wait_count": counts["cluster_barrier_wait"],
        "wmma_total": counts["wmma"],
        "wmma_total_divmod_16": list(divmod(counts["wmma"], 16)),
        "wmma_label_blocks_with_exactly_16": target_report[
            "wmma_16_count_label_blocks"
        ],
        "wmma_phase_claim": (
            "None. Exact-16 label blocks and total/divmod are search aids only; "
            "semantic phase boundaries are not proven."
        ),
        "output_tdm_store_count": counts["tensor_store_from_lds"],
        "output_tdm_shape_interpretation": (
            "CONTRACT PASS: exactly one output TDM store"
            if counts["tensor_store_from_lds"] == 1
            else "CONTRACT FAIL: expected exactly one output TDM store"
        ),
        "output_store_protocol_windows": protocols,
        "two_output_wg_rounds_statically_detected": first_last_two_rounds,
        "warnings": warnings,
    }


def metadata_analysis(out_dir, target_path):
    candidates = [target_path]
    for pattern in (
        "*.readelf*.txt",
        "*.readobj*.txt",
        "*.objdump.sections_symbols.txt",
        "*.nm.*.txt",
    ):
        candidates.extend(sorted(out_dir.glob(pattern)))
    old_symbol_hits = []
    new_symbol_hits = []
    old_lds_hits = []
    candidate_lds_hits = []
    old_vgpr_hits = []
    resource_lines = []
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            lower = line.lower()
            entry = {"file": str(path), "line": number, "text": line}
            if OLD_SYMBOL in line:
                old_symbol_hits.append(entry)
            if NEW_SYMBOL in line:
                new_symbol_hits.append(entry)
            resource_context = any(
                token in lower
                for token in (
                    "group_segment",
                    "groupsegment",
                    "lds",
                    "vgpr",
                    "sgpr",
                    "kernarg",
                    "wavefront",
                    "amdhsa",
                )
            )
            if resource_context:
                resource_lines.append(entry)
            if resource_context and (
                re.search(r"\b327680\b", line)
                or re.search(r"\b0x0*50000\b", line, re.IGNORECASE)
            ):
                old_lds_hits.append(entry)
            if resource_context and (
                re.search(r"\b172032\b", line)
                or re.search(r"\b0x0*2a000\b", line, re.IGNORECASE)
            ):
                candidate_lds_hits.append(entry)
            if "vgpr" in lower and re.search(r"\b1024\b", line):
                old_vgpr_hits.append(entry)
    warnings = []
    if old_symbol_hits:
        warnings.append("original 256x256 kernel symbol remains in source/ELF reports")
    if not new_symbol_hits:
        warnings.append("128x128 contract kernel symbol is absent from source/ELF reports")
    if old_lds_hits:
        warnings.append("old 327680-byte (0x50000) LDS resource remains")
    if old_vgpr_hits:
        warnings.append("old 1024-VGPR resource remains")
    if not candidate_lds_hits:
        warnings.append("candidate 172032-byte (0x2A000) LDS resource is absent")
    return {
        "files_scanned": [str(path) for path in dict.fromkeys(candidates) if path.is_file()],
        "old_symbol_hits": old_symbol_hits,
        "new_symbol_hits": new_symbol_hits,
        "old_lds_hits": old_lds_hits,
        "candidate_lds_hits": candidate_lds_hits,
        "old_vgpr_hits": old_vgpr_hits,
        "resource_lines": resource_lines,
        "warnings": warnings,
    }


def descriptor_values(text, directive):
    pattern = re.compile(
        rf"(?mi)^\s*\.{re.escape(directive)}\s*:?\s*(-?(?:0x[0-9a-f]+|\d+))\b"
    )
    return [parse_integer_literal(token) for token in pattern.findall(text)]


def contract_analysis(target_text, reports, metadata):
    target_report = reports.get("target_source")
    reference_report = reports.get("reference_source")
    disassembly_reports = {
        label: report
        for label, report in reports.items()
        if report["kind"] == "disassembly"
    }
    group_values = descriptor_values(
        target_text, "amdhsa_group_segment_fixed_size"
    )
    next_vgpr_values = descriptor_values(target_text, "amdhsa_next_free_vgpr")
    source_store_count = (
        target_report["counts"]["tensor_store_from_lds"] if target_report else None
    )
    disassembly_store_counts = {
        label: report["counts"]["tensor_store_from_lds"]
        for label, report in disassembly_reports.items()
    }
    disassembly_symbol_occurrences = {
        label: report["new_symbol_occurrences"]
        for label, report in disassembly_reports.items()
    }
    unchanged_reference_copy = bool(
        target_report
        and reference_report
        and target_report["sha256"] == reference_report["sha256"]
    )
    gates = {
        "required_symbol_in_target_source": bool(
            target_report and target_report["new_symbol_occurrences"] > 0
        ),
        "required_symbol_in_every_produced_disassembly": bool(
            disassembly_reports
            and all(value > 0 for value in disassembly_symbol_occurrences.values())
        ),
        "target_group_segment_exactly_172032": group_values == [172032],
        "target_source_exactly_one_tensor_store_from_lds": source_store_count == 1,
        "every_produced_disassembly_exactly_one_tensor_store_from_lds": bool(
            disassembly_reports
            and all(value == 1 for value in disassembly_store_counts.values())
        ),
    }
    failures = [
        name for name, passed in gates.items() if not passed
    ]
    old_findings = {
        "old_256_symbol_in_target_source": bool(
            target_report and target_report["old_symbol_occurrences"] > 0
        ),
        "old_256_symbol_in_produced_disassembly": {
            label: report["old_symbol_occurrences"]
            for label, report in disassembly_reports.items()
            if report["old_symbol_occurrences"]
        },
        "old_group_segment_327680_values": [
            value for value in group_values if value == 327680
        ],
        "old_next_free_vgpr_1024_values": [
            value for value in next_vgpr_values if value == 1024
        ],
        "metadata_old_symbol_hit_count": len(metadata["old_symbol_hits"]),
        "metadata_old_lds_hit_count": len(metadata["old_lds_hits"]),
        "metadata_old_vgpr_hit_count": len(metadata["old_vgpr_hits"]),
    }
    return {
        "contract_symbol": NEW_SYMBOL,
        "contract_group_segment_bytes": 172032,
        "contract_tensor_store_from_lds_count": 1,
        "unchanged_reference_copy": unchanged_reference_copy,
        "target_group_segment_values": group_values,
        "target_next_free_vgpr_values": next_vgpr_values,
        "target_source_tensor_store_from_lds_count": source_store_count,
        "produced_disassembly_tensor_store_from_lds_counts": (
            disassembly_store_counts
        ),
        "produced_disassembly_required_symbol_occurrences": (
            disassembly_symbol_occurrences
        ),
        "gates": gates,
        "failure_count": len(failures),
        "failures": failures,
        "old_contract_findings": old_findings,
    }


def parse_nm_symbols(path):
    if not path.is_file():
        return []
    symbols = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if parts:
            symbols.append(parts[-1])
    return sorted(set(symbols))


def route_analysis(manifest, reports):
    objects = {}
    disassemblies = {}
    for label, kind, path in manifest:
        if kind == "object":
            objects[label] = path
        elif kind == "disassembly":
            disassemblies[label] = path
    routes = {}
    for label, path in objects.items():
        entry = {
            "path": str(path),
            "elf_sha256": sha256_path(path),
            "size_bytes": path.stat().st_size,
            "symbols": parse_nm_symbols(path.parent / f"{label}.nm.defined.txt"),
        }
        if label in disassemblies:
            text = disassemblies[label].read_text(encoding="utf-8", errors="replace")
            records = instruction_records(text)
            counts, _, _, _ = categories(records)
            _, sequence_hash = normalized_instruction_sequence(records)
            entry.update(
                {
                    "disassembly_path": str(disassemblies[label]),
                    "instruction_count": len(records),
                    "normalized_instruction_sha256": sequence_hash,
                    "counts": counts,
                }
            )
        routes[label] = entry
    warnings = []
    comparison = {"available": False}
    if "llvm_mc_object" in routes and "clang_object" in routes:
        left = routes["llvm_mc_object"]
        right = routes["clang_object"]
        comparison = {
            "available": True,
            "whole_elf_hash_equal": left["elf_sha256"] == right["elf_sha256"],
            "whole_elf_difference_is_not_alone_a_failure": True,
            "size_equal": left["size_bytes"] == right["size_bytes"],
            "normalized_instruction_hash_equal": (
                left.get("normalized_instruction_sha256")
                == right.get("normalized_instruction_sha256")
            ),
            "instruction_count_equal": (
                left.get("instruction_count") == right.get("instruction_count")
            ),
            "category_count_differences": {
                name: [
                    left.get("counts", {}).get(name, 0),
                    right.get("counts", {}).get(name, 0),
                ]
                for name in CATEGORY_NAMES
                if left.get("counts", {}).get(name, 0)
                != right.get("counts", {}).get(name, 0)
            },
            "symbol_sets_equal": left.get("symbols") == right.get("symbols"),
            "symbols_only_llvm_mc": sorted(
                set(left.get("symbols", ())) - set(right.get("symbols", ()))
            ),
            "symbols_only_clang": sorted(
                set(right.get("symbols", ())) - set(left.get("symbols", ()))
            ),
        }
        if not comparison["normalized_instruction_hash_equal"]:
            warnings.append("llvm-mc and clang normalized disassemblies differ")
        if comparison["category_count_differences"]:
            warnings.append("llvm-mc and clang opcode category counts differ")
        if not comparison["symbol_sets_equal"]:
            warnings.append("llvm-mc and clang defined symbol sets differ")
    else:
        warnings.append("two-route object comparison is unavailable")
    return {"routes": routes, "comparison": comparison, "warnings": warnings}


def print_hits(title, hits, limit=200):
    print(title)
    if not hits:
        print("  none")
        return
    for hit in hits[:limit]:
        print(f"  {hit['file']}:{hit['line']}: {hit['text']}")
    if len(hits) > limit:
        print(f"  ... {len(hits) - limit} additional hits retained in JSON")


def main():
    run_literal_parser_self_tests()
    print("Embedded literal-parser self-tests: PASS")
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    target_path = Path(args.target)

    manifest = []
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        label, kind, raw_path = line.split("\t", 2)
        path = Path(raw_path)
        if path.is_file():
            manifest.append((label, kind, path))

    reports = {}
    static_warnings = []
    for label, kind, path in manifest:
        if kind not in {"source", "disassembly"}:
            continue
        report = analyze_input(label, kind, path)
        reports[label] = report
        if kind == "source" and report["unresolved_symbolic_branches"]:
            static_warnings.append(
                f"{label}: {len(report['unresolved_symbolic_branches'])} "
                "symbolic branch target(s) not defined in the parsed text"
            )

    target_report = reports.get("target_source")
    if target_report:
        for label, report in reports.items():
            if report["kind"] != "disassembly":
                continue
            differences = {
                name: [target_report["counts"][name], report["counts"][name]]
                for name in CATEGORY_NAMES
                if target_report["counts"][name] != report["counts"][name]
            }
            report["target_source_category_differences"] = differences
            if differences:
                static_warnings.append(
                    f"{label}: opcode category counts differ from target source"
                )

    static_payload = {
        "category_definitions": {
            "wmma": "mnemonic starts with v_wmma",
            "ds_store_or_write": "mnemonic starts with ds_store or ds_write",
            "conversion": "mnemonic contains a cvt/convert/pack component",
            "wg_barrier": "s_barrier_signal -1 / s_barrier_wait 0xffff",
            "cluster_barrier": "s_barrier_signal -3 / s_barrier_wait 0xfffd",
        },
        "inputs": reports,
        "warnings": static_warnings,
    }
    (out_dir / "static_count.json").write_text(
        json.dumps(static_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "static_count.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("input\tgroup\tmetric\tvalue\n")
        for label, report in reports.items():
            handle.write(f"{label}\tsummary\tinstruction_count\t{report['instruction_count']}\n")
            for name, value in report["counts"].items():
                handle.write(f"{label}\tcount\t{name}\t{value}\n")
            for name, value in report["ds_store_write_variants"].items():
                handle.write(f"{label}\tds_store_write_variant\t{name}\t{value}\n")
            for name, value in report["conversion_variants"].items():
                handle.write(f"{label}\tconversion_variant\t{name}\t{value}\n")
            for name, value in report["selected_constant_occurrences"].items():
                handle.write(f"{label}\tconstant\t{name}\t{value}\n")
            for name, value in report["tile_literal_occurrences"].items():
                handle.write(f"{label}\ttile_literal\t{name}\t{value}\n")
            for name, value in report["tile_string_occurrences"].items():
                handle.write(f"{label}\ttile_string\t{name}\t{value}\n")

    target_text = target_path.read_text(encoding="utf-8", errors="replace")
    design = design_analysis(target_text, target_report) if target_report else {
        "warnings": ["target source was not available to the audit"]
    }
    (out_dir / "design_invariants.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metadata = metadata_analysis(out_dir, target_path)
    (out_dir / "metadata_resource_analysis.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    contract = contract_analysis(target_text, reports, metadata)
    (out_dir / "target_contract_gates.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    routes = route_analysis(manifest, reports)
    (out_dir / "route_comparison.json").write_text(
        json.dumps(routes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("STATIC OPCODE / CONTROL-FLOW AUDIT")
    print("Per-phase note: this parser does not infer phase boundaries and makes no per-phase 16-WMMA claim.")
    for label, report in reports.items():
        print(f"\n[{label}] {report['path']}")
        print(f"  sha256={report['sha256']}")
        print(f"  instruction_count={report['instruction_count']}")
        for name in CATEGORY_NAMES:
            print(f"  {name}={report['counts'][name]}")
        print(f"  ds_store/write variants={report['ds_store_write_variants']}")
        print(f"  conversion variants={report['conversion_variants']}")
        print(f"  WMMA variants={report['wmma_variants']}")
        print(f"  kernel symbols={report['kernel_symbols']}")
        print(f"  labels={report['label_count']}")
        print(f"  branches={report['branch_count']}")
        print(
            "  unresolved symbolic branches="
            f"{len(report['unresolved_symbolic_branches'])}"
        )
        print(f"  old symbol strings={report['old_symbol_occurrences']}")
        print(f"  new symbol strings={report['new_symbol_occurrences']}")
        print(f"  selected constants={report['selected_constant_occurrences']}")
        print(f"  new ring constants={report['new_ring_constant_occurrences']}")
        print(f"  old-only ring constants={report['old_only_ring_constant_occurrences']}")
        print(f"  tile literals={report['tile_literal_occurrences']}")
        print(f"  tile strings={report['tile_string_occurrences']}")
        print(f"  exact-16 label blocks={report['wmma_16_count_label_blocks']}")
        print("  branch inventory:")
        for branch in report["branches"]:
            print(
                f"    line {branch['line']}: {branch['mnemonic']} "
                f"{branch['target']} resolved_in_text={branch['resolved_in_text']}"
            )

    print("\nSTATIC AUDIT WARNINGS")
    if static_warnings:
        for warning in static_warnings:
            print(f"  WARNING: {warning}")
    else:
        print("  none")

    print("\nHARD TARGET-CONTRACT GATES")
    print(f"  unchanged_reference_copy: {contract['unchanged_reference_copy']}")
    for name, passed in contract["gates"].items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    print(f"  failure_count: {contract['failure_count']}")
    for failure in contract["failures"]:
        print(f"  CONTRACT FAILURE: {failure}")
    print("  PROMINENT OLD-CONTRACT FINDINGS:")
    for name, value in contract["old_contract_findings"].items():
        print(f"    {name}: {value}")

    print("\nDESIGN HEURISTICS (SEARCH AIDS ONLY; NOT SEMANTIC PROOF)")
    for key, value in design.items():
        if key != "warnings":
            print(f"  {key}: {value}")
    if design.get("warnings"):
        for warning in design["warnings"]:
            print(f"  WARNING: {warning}")
    else:
        print("  warnings: none")

    print("\nMETADATA / RESOURCE INSPECTION")
    print_hits("Old 256x256 symbol hits:", metadata["old_symbol_hits"])
    print_hits("New 128x128 contract symbol hits:", metadata["new_symbol_hits"])
    print_hits("Old 327680-byte LDS hits:", metadata["old_lds_hits"])
    print_hits("Candidate 172032-byte LDS hits:", metadata["candidate_lds_hits"])
    print_hits("Old 1024-VGPR hits:", metadata["old_vgpr_hits"])
    for warning in metadata["warnings"]:
        print(f"  WARNING: {warning}")

    print("\nLLVM-MC VS CLANG OBJECT COMPARISON")
    print(json.dumps(routes, indent=2, sort_keys=True))
    print("Whole-ELF hash/size differences alone are informational and do not fail validation.")

    status_values = {
        "STATIC_WARNINGS": len(static_warnings),
        "DESIGN_WARNINGS": len(design.get("warnings", ())),
        "METADATA_WARNINGS": len(metadata["warnings"]),
        "ROUTE_WARNINGS": len(routes["warnings"]),
        "ROUTE_COMPARISON_AVAILABLE": int(
            routes.get("comparison", {}).get("available", False)
        ),
        "CONTRACT_FAILURES": contract["failure_count"],
        "UNCHANGED_REFERENCE_COPY": int(contract["unchanged_reference_copy"]),
        "TARGET_SOURCE_AUDITED": int("target_source" in reports),
        "REFERENCE_SOURCE_AUDITED": int("reference_source" in reports),
        "DISASSEMBLIES_AUDITED": sum(
            1 for report in reports.values() if report["kind"] == "disassembly"
        ),
        "UNRESOLVED_BRANCHES": sum(
            len(report["unresolved_symbolic_branches"])
            for report in reports.values()
            if report["kind"] == "source"
        ),
    }
    with (out_dir / "analysis_status.env").open("w", encoding="utf-8", newline="\n") as handle:
        for key, value in status_values.items():
            handle.write(f"{key}={value}\n")
    print("\nARTIFACTS")
    for name in (
        "static_count.json",
        "static_count.tsv",
        "design_invariants.json",
        "metadata_resource_analysis.json",
        "target_contract_gates.json",
        "route_comparison.json",
        "analysis_status.env",
    ):
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
PY

audit_args=(
    "${PYTHON}"
    "${AUDIT_PY}"
    --manifest "${AUDIT_MANIFEST}"
    --out-dir "${OUT_DIR}"
    --target "${OUT_DIR}/target.original.s"
)
analysis_outputs=(
    "${OUT_DIR}/static_count.json"
    "${OUT_DIR}/static_count.tsv"
    "${OUT_DIR}/design_invariants.json"
    "${OUT_DIR}/metadata_resource_analysis.json"
    "${OUT_DIR}/target_contract_gates.json"
    "${OUT_DIR}/route_comparison.json"
    "${OUT_DIR}/analysis_status.env"
)
rm -f -- "${analysis_outputs[@]}"
static_audit_unavailable=0
if (( target_exists )); then
    run_capture_stdout \
        "static_audit" \
        "${OUT_DIR}/static_count.txt" \
        "${OUT_DIR}/static_count.stderr.txt" \
        "${audit_args[@]}"
    static_audit_status="${LAST_STATUS}"
else
    static_audit_status=1
    static_audit_unavailable=1
    echo "WARNING: static audit skipped because target source is unavailable."
fi

ANALYSIS_OUTPUTS_MISSING=0
for analysis_output in "${analysis_outputs[@]}"; do
    if [[ ! -s "${analysis_output}" ]]; then
        echo "ERROR: required analysis artifact is missing or empty: ${analysis_output}" >&2
        ((ANALYSIS_OUTPUTS_MISSING += 1))
    fi
done
ANALYSIS_INTERNAL_FAILURE=0
if (( static_audit_unavailable == 0 )) \
    && (( static_audit_status != 0 || ANALYSIS_OUTPUTS_MISSING != 0 )); then
    ANALYSIS_INTERNAL_FAILURE=1
    HARD_FAIL=1
    echo "ERROR: required embedded analysis failed internally; validation cannot pass." >&2
fi

STATIC_WARNINGS=0
DESIGN_WARNINGS=0
METADATA_WARNINGS=0
ROUTE_WARNINGS=0
ROUTE_COMPARISON_AVAILABLE=0
CONTRACT_FAILURES=0
UNCHANGED_REFERENCE_COPY=0
UNRESOLVED_BRANCHES=0
TARGET_SOURCE_AUDITED=0
REFERENCE_SOURCE_AUDITED=0
DISASSEMBLIES_AUDITED=0
if [[ -f "${OUT_DIR}/analysis_status.env" ]]; then
    while IFS='=' read -r key value; do
        case "${key}" in
            STATIC_WARNINGS) STATIC_WARNINGS="${value}" ;;
            DESIGN_WARNINGS) DESIGN_WARNINGS="${value}" ;;
            METADATA_WARNINGS) METADATA_WARNINGS="${value}" ;;
            ROUTE_WARNINGS) ROUTE_WARNINGS="${value}" ;;
            ROUTE_COMPARISON_AVAILABLE) ROUTE_COMPARISON_AVAILABLE="${value}" ;;
            CONTRACT_FAILURES) CONTRACT_FAILURES="${value}" ;;
            UNCHANGED_REFERENCE_COPY) UNCHANGED_REFERENCE_COPY="${value}" ;;
            UNRESOLVED_BRANCHES) UNRESOLVED_BRANCHES="${value}" ;;
            TARGET_SOURCE_AUDITED) TARGET_SOURCE_AUDITED="${value}" ;;
            REFERENCE_SOURCE_AUDITED) REFERENCE_SOURCE_AUDITED="${value}" ;;
            DISASSEMBLIES_AUDITED) DISASSEMBLIES_AUDITED="${value}" ;;
        esac
    done < "${OUT_DIR}/analysis_status.env"
fi

metadata_level="PASS"
metadata_detail="${METADATA_NOTE_DECODES} generic ELF note decode(s) succeeded; ${METADATA_SPECIALIZED_SKIPS} unsupported specialized flag invocation(s) skipped"
if (( VALID_OBJECT_COUNT == 0 )); then
    metadata_level="WARN"
    metadata_detail="SKIP: no object/code object exists; ELF metadata/resource tools were not invoked"
elif (( ANALYSIS_INTERNAL_FAILURE != 0 )); then
    metadata_level="FAIL"
    metadata_detail="ANALYSIS FAILURE: Python metadata/resource analysis crashed or required outputs are missing"
elif (( METADATA_TOOL_FAILURES != 0 || METADATA_NOTE_DECODES == 0 )); then
    metadata_level="FAIL"
    metadata_detail="${METADATA_TOOL_FAILURES} required generic metadata command failure(s), ${METADATA_NOTE_DECODES} successful note decode(s)"
    HARD_FAIL=1
fi
record_experiment "6. Metadata/resources" "${metadata_level}" "${metadata_detail}"

audit_level="PASS"
audit_detail="target source, reference source, and ${DISASSEMBLIES_AUDITED} produced disassembly input(s) audited"
if (( ANALYSIS_INTERNAL_FAILURE != 0 )); then
    audit_level="FAIL"
    audit_detail="ANALYSIS FAILURE: embedded opcode parser crashed or required outputs are missing"
elif (( DISASSEMBLIES_AUDITED == 0 )); then
    audit_level="WARN"
    audit_detail="SOURCE-ONLY: no produced disassembly exists; target=${TARGET_SOURCE_AUDITED}, reference=${REFERENCE_SOURCE_AUDITED} source audit(s)"
elif (( TARGET_SOURCE_AUDITED == 0 || REFERENCE_SOURCE_AUDITED == 0 )); then
    audit_level="WARN"
    audit_detail="incomplete intended inputs: target=${TARGET_SOURCE_AUDITED}, reference=${REFERENCE_SOURCE_AUDITED}, disassemblies=${DISASSEMBLIES_AUDITED}"
elif (( STATIC_WARNINGS != 0 || UNRESOLVED_BRANCHES != 0 )); then
    audit_level="WARN"
    audit_detail="${STATIC_WARNINGS} comparison warning(s), ${UNRESOLVED_BRANCHES} unresolved parsed branch target(s)"
fi
record_experiment "7. Static opcode audit" "${audit_level}" "${audit_detail}"

design_level="PASS"
design_detail="all hard target-contract gates passed; heuristic searches are not semantic proof"
if (( ANALYSIS_INTERNAL_FAILURE != 0 )); then
    design_level="FAIL"
    design_detail="ANALYSIS FAILURE: design/contract analysis crashed or required outputs are missing"
elif (( CONTRACT_FAILURES != 0 )); then
    design_level="FAIL"
    if (( UNCHANGED_REFERENCE_COPY != 0 )); then
        design_detail="INCOMPLETE unchanged reference-copy target; ${CONTRACT_FAILURES} hard final-contract gate(s) failed"
    else
        design_detail="${CONTRACT_FAILURES} hard final-target contract gate(s) failed"
    fi
    HARD_FAIL=1
elif (( DESIGN_WARNINGS != 0 )); then
    design_level="WARN"
    design_detail="${DESIGN_WARNINGS} HEURISTIC WARNING(S); not hard failures or semantic proof; see design_invariants.json"
fi
record_experiment "8. Design invariants" "${design_level}" "${design_detail}"

banner "9. Optional existing parser/tooling applicability"
{
    echo "Inspected: my_code/isa_runner/gemm_isa_runner.py"
    echo "  SKIP: its CLI always proceeds from compile/link to HIP/PyTorch GEMM execution;"
    echo "  it has no compile-only or source-static CLI mode for this kernel."
    echo "Inspected: my_code/isa_runner/tdm_adapter.py"
    echo "  SKIP: capture/replay/run all enter the production GPU capture path, and the"
    echo "  adapter targets a different 104-byte TDM GEMM ABI."
    echo "Inspected: my_code/isa_runner/isa_runner.py"
    echo "  SKIP: its nominal no-smoke path still assembles/links into its cache; this"
    echo "  script already performs equivalent checks in OUT_DIR without invoking GPU code."
    echo "No arguments were invented and no existing runner was invoked."
} | tee "${OUT_DIR}/existing_tooling.txt"
record_experiment \
    "9. Existing tooling" \
    "WARN" \
    "SKIP: no applicable read-only/static CLI mode exists for the requested kernel"

banner "10. Round-trip and cross-route sanity"
if [[ -f "${OUT_DIR}/route_comparison.json" ]]; then
    "${PYTHON}" - "${OUT_DIR}/route_comparison.json" <<'PY' \
        | tee "${OUT_DIR}/route_comparison.txt"
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
print(f"route comparison JSON: {path}")
for name, route in sorted(data.get("routes", {}).items()):
    print(f"{name}:")
    for key in (
        "path",
        "size_bytes",
        "elf_sha256",
        "instruction_count",
        "normalized_instruction_sha256",
        "symbols",
        "counts",
    ):
        if key in route:
            print(f"  {key}: {route[key]}")
print("comparison:")
for key, value in data.get("comparison", {}).items():
    print(f"  {key}: {value}")
for warning in data.get("warnings", []):
    print(f"WARNING: {warning}")
print("NOTE: whole-ELF differences alone are non-semantic and are not a failure.")
PY
fi

roundtrip_level="PASS"
roundtrip_detail="ELF/disassembly sanity passed; no unresolved symbols reported"
if (( VALID_OBJECT_COUNT == 0 )); then
    roundtrip_level="FAIL"
    roundtrip_detail="UNAVAILABLE: no valid object assembled; ELF/disassembly validation was not performed"
elif (( DISASM_INVALID_COUNT != 0 || ELF_HEADER_INVALID_COUNT != 0 || ELF_SUCCESS_INVALID != 0 )); then
    roundtrip_level="FAIL"
    roundtrip_detail="INVALID: a produced artifact failed ELF-header or disassembly validation"
elif (( ROUTE_WARNINGS != 0 || UNRESOLVED_SYMBOL_FILES != 0 || UNRESOLVED_BRANCHES != 0 )); then
    roundtrip_level="WARN"
    roundtrip_detail="${ROUTE_WARNINGS} route comparison warning(s), ${UNRESOLVED_SYMBOL_FILES} ELF file(s) with undefined symbols"
elif [[ -z "${PRIMARY_CO}" ]]; then
    roundtrip_level="WARN"
    roundtrip_detail="object sanity passed; code-object validation remains incomplete"
fi
record_experiment "10. Round-trip sanity" "${roundtrip_level}" "${roundtrip_detail}"

banner "11. Final PASS/WARN/FAIL summary"
SUMMARY_TSV="${OUT_DIR}/experiment_summary.tsv"
printf 'experiment\tresult\tdetail\n' > "${SUMMARY_TSV}"
{
    printf 'Output directory: %s\n' "${OUT_DIR}"
    printf 'Assembly source used: %s\n' "${ASSEMBLY_SOURCE_USED}"
    printf 'Valid object count: %s\n' "${VALID_OBJECT_COUNT}"
    printf 'Primary object: %s\n' "${PRIMARY_OBJ:-none}"
    printf 'Primary code object: %s\n' "${PRIMARY_CO:-none}"
    printf 'Valid disassembly count: %s\n' "${DISASM_VALID_COUNT}"
    printf 'Invalid disassembly count: %s\n' "${DISASM_INVALID_COUNT}"
    printf '\n'
    for index in "${!EXPERIMENT_NAMES[@]}"; do
        name="${EXPERIMENT_NAMES[${index}]}"
        level="${EXPERIMENT_LEVELS[${index}]}"
        detail="${EXPERIMENT_DETAILS[${index}]}"
        printf '%-5s %-31s %s\n' "${level}" "${name}" "${detail}"
        printf '%s\t%s\t%s\n' "${name}" "${level}" "${detail}" >> "${SUMMARY_TSV}"
    done
    printf '\nArtifact paths:\n'
    "${PYTHON}" - "${OUT_DIR}" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
for path in sorted(root.rglob("*")):
    if path.is_file():
        print(f"  {path} ({path.stat().st_size} bytes)")
PY
} | tee "${OUT_DIR}/final_summary.txt"

final_status=0
if (( VALID_OBJECT_COUNT == 0 )); then
    final_status=1
fi
if (( DISASM_INVALID_COUNT != 0 || ELF_HEADER_INVALID_COUNT != 0 || ELF_SUCCESS_INVALID != 0 )); then
    final_status=1
fi
if (( HARD_FAIL != 0 )); then
    final_status=1
fi

if (( final_status == 0 )); then
    echo "FINAL RESULT: validation completed; warnings do not change the zero exit status." \
        | tee -a "${OUT_DIR}/final_summary.txt"
elif (( VALID_OBJECT_COUNT == 0 )); then
    if [[ -n "${TOOLCHAIN_ROOT_CAUSE}" ]]; then
        echo "FINAL RESULT: hard failure: no valid object assembled; ELF/disassembly artifacts are unavailable (not observed invalid). Root cause: ${TOOLCHAIN_ROOT_CAUSE}." \
            | tee -a "${OUT_DIR}/final_summary.txt" >&2
    else
        echo "FINAL RESULT: hard failure: no valid object assembled; ELF/disassembly artifacts are unavailable (not observed invalid). See assembler logs." \
            | tee -a "${OUT_DIR}/final_summary.txt" >&2
    fi
else
    echo "FINAL RESULT: hard failure: a produced ELF/disassembly artifact is invalid; see validation reports." \
        | tee -a "${OUT_DIR}/final_summary.txt" >&2
fi
echo "Artifacts remain in: ${OUT_DIR}" | tee -a "${OUT_DIR}/final_summary.txt"
exit "${final_status}"
