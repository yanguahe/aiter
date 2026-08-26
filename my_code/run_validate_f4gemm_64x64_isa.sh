#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

readonly ARCH="gfx1250"
readonly TRIPLE="amdgcn-amd-amdhsa"
readonly CODE_OBJECT_VERSION="6"

LLVM_BIN="${LLVM_BIN:-/data/yanguahe/code/wk_sp1/llvm-project/mlir_install/bin}"
REPO_ROOT="${REPO_ROOT:-/data/yanguahe/code/wk_sp1/aiter}"

usage() {
    cat <<'EOF'
Usage:
  bash my_code/run_validate_f4gemm_64x64_isa.sh [OUT_DIR]

Environment overrides:
  LLVM_BIN    LLVM tool directory
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
    if [[ -x "${candidate}" && -f "${candidate}" ]]; then
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
printf 'requested_name\tstate\texecutable_name\tpath\n' > "${TOOL_INVENTORY}"
declare -a TOOL_LABELS=()
declare -a TOOL_PATHS=()

inventory_tool() {
    local requested="$1"
    local path="$2"
    if [[ -n "${path}" ]]; then
        printf '%s\tFOUND\t%s\t%s\n' \
            "${requested}" "$(basename "${path}")" "${path}" | tee -a "${TOOL_INVENTORY}"
        TOOL_LABELS+=("${requested}")
        TOOL_PATHS+=("${path}")
    else
        printf '%s\tMISSING\t-\t-\n' "${requested}" | tee -a "${TOOL_INVENTORY}"
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

version_failures=0
for index in "${!TOOL_PATHS[@]}"; do
    label="${TOOL_LABELS[${index}]}"
    tool="${TOOL_PATHS[${index}]}"
    safe_label="${label//+/x}"
    safe_label="${safe_label//./_}"
    run_logged \
        "version_${safe_label}" \
        "${OUT_DIR}/version_${safe_label}.txt" \
        "${tool}" --version
    if (( LAST_STATUS != 0 )); then
        ((version_failures += 1))
    fi
done

toolchain_level="PASS"
toolchain_detail="core assembly and disassembly tools found"
if [[ -z "${LLVM_OBJDUMP}" || ( -z "${LLVM_MC}" && -z "${CLANG}" ) ]]; then
    toolchain_level="FAIL"
    toolchain_detail="missing llvm-objdump or every assembler route"
    HARD_FAIL=1
elif [[ -z "${LLVM_MC}" || -z "${CLANG}" ]]; then
    toolchain_level="WARN"
    toolchain_detail="only one independent assembler route is available"
elif (( version_failures != 0 )); then
    toolchain_level="WARN"
    toolchain_detail="${version_failures} tool version command(s) failed"
elif [[ -z "${LLVM_READELF}" || -z "${LLVM_READOBJ}" || -z "${LLVM_NM}" ]]; then
    toolchain_level="WARN"
    toolchain_detail="one or more optional ELF metadata tools are unavailable"
fi
record_experiment "1. Toolchain inventory" "${toolchain_level}" "${toolchain_detail}"

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
            echo "WARNING: target and reference are byte-for-byte identical; the 64x64 rewrite may not have started."
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

banner "3. Independent gfx1250 syntax and assembly routes"

MC_OK=0
CLANG_OK=0
MC_OBJ=""
CLANG_OBJ=""
ASSEMBLY_SOURCE_USED="${OUT_DIR}/target.assembly.s"

if (( target_exists )) && [[ -n "${LLVM_MC}" ]]; then
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
elif [[ -z "${LLVM_MC}" ]]; then
    echo "WARNING: llvm-mc route skipped because llvm-mc was not found."
fi

if (( target_exists )) && [[ -n "${CLANG}" ]]; then
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
elif [[ -z "${CLANG}" ]]; then
    echo "WARNING: independent clang assembler route skipped because clang was not found."
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
        if [[ -n "${LLVM_MC}" ]]; then
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
        if [[ -n "${CLANG}" ]]; then
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

if [[ -n "${PRIMARY_OBJ}" && -n "${LD_LLD}" ]]; then
    lld_candidate="${OUT_DIR}/target.ld_lld.co"
    if attempt_elf_command \
        "link_ld_lld" \
        "${lld_candidate}" \
        "${OUT_DIR}/link_ld_lld.log" \
        "${LD_LLD}" \
        -shared \
        --no-undefined \
        -o "${lld_candidate}" \
        "${PRIMARY_OBJ}"; then
        LLD_LINK_OK=1
        LLD_CO="${lld_candidate}"
    fi
elif [[ -z "${LD_LLD}" ]]; then
    echo "WARNING: direct ld.lld link route skipped because ld.lld was not found."
fi

if [[ -n "${PRIMARY_OBJ}" && -n "${CLANG}" ]]; then
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
        -shared \
        "${PRIMARY_OBJ}" \
        -o "${clang_co_candidate}"; then
        CLANG_LINK_OK=1
        CLANG_CO="${clang_co_candidate}"
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
    run_capture_stdout \
        "${label}_disassemble_rich" \
        "${disassembly}" \
        "${OUT_DIR}/${label}.disassembly.rich.stderr.txt" \
        "${LLVM_OBJDUMP}" \
        --disassemble \
        --mcpu="${ARCH}" \
        --show-raw-insn \
        --symbolize-operands \
        "${file}"
    if (( LAST_STATUS != 0 )); then
        echo "WARNING: rich objdump options failed; retrying the portable gfx1250 form." >&2
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

    if [[ -n "${LLVM_READELF}" ]]; then
        run_capture_stdout \
            "${label}_readelf" \
            "${OUT_DIR}/${label}.readelf.txt" \
            "${OUT_DIR}/${label}.readelf.stderr.txt" \
            "${LLVM_READELF}" -h -S -s -n -W "${file}"
        if (( LAST_STATUS != 0 )); then
            ((METADATA_TOOL_FAILURES += 1))
        fi
    fi

    if [[ -n "${LLVM_READOBJ}" ]]; then
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
        fi

        run_capture_stdout \
            "${label}_readobj_amdgpu_metadata" \
            "${OUT_DIR}/${label}.readobj.amdgpu_metadata.txt" \
            "${OUT_DIR}/${label}.readobj.amdgpu_metadata.stderr.txt" \
            "${LLVM_READOBJ}" --amdgpu-metadata "${file}"
        if (( LAST_STATUS != 0 )); then
            echo "WARNING: llvm-readobj --amdgpu-metadata did not succeed for ${label}." >&2
            ((METADATA_TOOL_FAILURES += 1))
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

    if [[ -n "${LLVM_NM}" ]]; then
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

if [[ -n "${LLVM_OBJDUMP}" ]]; then
    for index in "${!INSPECT_FILES[@]}"; do
        inspect_elf \
            "${INSPECT_LABELS[${index}]}" \
            "${INSPECT_KINDS[${index}]}" \
            "${INSPECT_FILES[${index}]}"
    done
else
    echo "ERROR: llvm-objdump is unavailable; produced objects cannot be validated." >&2
    if (( VALID_OBJECT_COUNT > 0 )); then
        DISASM_INVALID_COUNT="${VALID_OBJECT_COUNT}"
    fi
    HARD_FAIL=1
fi

disasm_level="PASS"
disasm_detail="${DISASM_VALID_COUNT} produced ELF file(s) fully disassembled"
if (( VALID_OBJECT_COUNT == 0 )); then
    disasm_level="FAIL"
    disasm_detail="no valid object was available to disassemble"
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
NEW_SYMBOL = "f4gemm_bf16_mxfp4_ABpreShuffle_64x64_4x4_ps"

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


def literal_counts(text):
    counter = Counter()
    for raw in text.splitlines():
        clean = strip_comments(raw)
        clean = re.sub(r"\b(?:s|v|ttmp)\d+\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(
            r"\b[sv]\[\s*\d+\s*:\s*\d+\s*\]",
            " ",
            clean,
            flags=re.IGNORECASE,
        )
        for token in re.findall(
            r"(?<![A-Za-z0-9_])(?:0x[0-9a-fA-F]+|\d+)(?![A-Za-z0-9_])",
            clean,
        ):
            counter[int(token, 0)] += 1
    return counter


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
            "expected packed-ring literals missing: " + ", ".join(missing_ring)
        )
    if literals[0x2A000] == 0 and not re.search(r"\b172032\b", target_text):
        warnings.append("candidate fixed-LDS end/resource 0x2A000 (172032) is absent")
    if literals[0x22000] == 0:
        warnings.append("input-ring/output-staging boundary 0x22000 is absent")
    if literals[0x80] == 0 and not re.search(r"(?<!\w)128(?!\w)", target_text):
        warnings.append("128 tile/wave offset literal was not found")
    if literals[0x40] == 0 and not re.search(r"(?<!\w)64(?!\w)", target_text):
        warnings.append("64 tile/wave offset literal was not found")
    if counts["s_wait_alu"] == 0:
        warnings.append("s_wait_alu is absent")
    if counts["tensor_store_from_lds"] == 0:
        warnings.append("output tensor_store_from_lds candidate is absent")
    if not first_last_two_rounds:
        warnings.append(
            "two output WG barrier rounds (before first and after last output TDM) "
            "were not statically found"
        )
    if counts["wg_barrier_signal"] != counts["wg_barrier_wait"]:
        warnings.append(
            "WG barrier signal/wait static counts are not balanced "
            f"({counts['wg_barrier_signal']} vs {counts['wg_barrier_wait']})"
        )
    if counts["cluster_barrier_signal"] == 0 or counts["cluster_barrier_wait"] == 0:
        warnings.append("cluster barrier signal/wait presence is incomplete")
    if counts["cluster_barrier_signal"] != counts["cluster_barrier_wait"]:
        warnings.append(
            "cluster barrier signal/wait static counts are not balanced "
            f"({counts['cluster_barrier_signal']} vs "
            f"{counts['cluster_barrier_wait']})"
        )
    present_old = [name for name, count in old_presence.items() if count]
    if present_old:
        warnings.append("old-only ring/LDS literals remain: " + ", ".join(present_old))
    if target_report["old_symbol_occurrences"]:
        warnings.append("original 256x256 kernel symbol remains")
    if target_report["new_symbol_occurrences"] == 0:
        warnings.append("new 64x64 kernel symbol is absent")
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
            "one-store candidate"
            if counts["tensor_store_from_lds"] == 1
            else "two-store fallback-like"
            if counts["tensor_store_from_lds"] == 2
            else "unexpected static store count"
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
        warnings.append("64x64 kernel symbol is absent from source/ELF reports")
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

    print("\nDESIGN INVARIANTS")
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
    print_hits("New 64x64 symbol hits:", metadata["new_symbol_hits"])
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
if (( target_exists )); then
    run_capture_stdout \
        "static_audit" \
        "${OUT_DIR}/static_count.txt" \
        "${OUT_DIR}/static_count.stderr.txt" \
        "${audit_args[@]}"
    static_audit_status="${LAST_STATUS}"
else
    static_audit_status=1
    echo "WARNING: static audit skipped because target source is unavailable."
fi

STATIC_WARNINGS=0
DESIGN_WARNINGS=0
METADATA_WARNINGS=0
ROUTE_WARNINGS=0
UNRESOLVED_BRANCHES=0
if [[ -f "${OUT_DIR}/analysis_status.env" ]]; then
    while IFS='=' read -r key value; do
        case "${key}" in
            STATIC_WARNINGS) STATIC_WARNINGS="${value}" ;;
            DESIGN_WARNINGS) DESIGN_WARNINGS="${value}" ;;
            METADATA_WARNINGS) METADATA_WARNINGS="${value}" ;;
            ROUTE_WARNINGS) ROUTE_WARNINGS="${value}" ;;
            UNRESOLVED_BRANCHES) UNRESOLVED_BRANCHES="${value}" ;;
        esac
    done < "${OUT_DIR}/analysis_status.env"
fi

metadata_level="PASS"
metadata_detail="ELF/source metadata and resource reports collected"
if (( static_audit_status != 0 )); then
    metadata_level="WARN"
    metadata_detail="metadata analysis program failed with status ${static_audit_status}"
elif (( METADATA_TOOL_FAILURES != 0 || METADATA_WARNINGS != 0 )); then
    metadata_level="WARN"
    metadata_detail="${METADATA_TOOL_FAILURES} tool command failure(s), ${METADATA_WARNINGS} resource warning(s)"
fi
record_experiment "6. Metadata/resources" "${metadata_level}" "${metadata_detail}"

audit_level="PASS"
audit_detail="source/disassembly opcode, labels, branches, and constants audited"
if (( static_audit_status != 0 )); then
    audit_level="WARN"
    audit_detail="embedded static audit failed with status ${static_audit_status}"
elif (( STATIC_WARNINGS != 0 || UNRESOLVED_BRANCHES != 0 )); then
    audit_level="WARN"
    audit_detail="${STATIC_WARNINGS} comparison warning(s), ${UNRESOLVED_BRANCHES} unresolved parsed branch target(s)"
fi
record_experiment "7. Static opcode audit" "${audit_level}" "${audit_detail}"

design_level="PASS"
design_detail="candidate design literals and synchronization constructs found"
if (( static_audit_status != 0 )); then
    design_level="WARN"
    design_detail="design-invariant analysis unavailable"
elif (( DESIGN_WARNINGS != 0 )); then
    design_level="WARN"
    design_detail="${DESIGN_WARNINGS} incomplete-rewrite/design warning(s); see design_invariants.json"
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
if (( VALID_OBJECT_COUNT == 0 || DISASM_INVALID_COUNT != 0 || ELF_HEADER_INVALID_COUNT != 0 )); then
    roundtrip_level="FAIL"
    roundtrip_detail="object, ELF, or disassembly validity failed"
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
else
    echo "FINAL RESULT: hard failure (no valid object or invalid ELF/disassembly)." \
        | tee -a "${OUT_DIR}/final_summary.txt" >&2
fi
echo "Artifacts remain in: ${OUT_DIR}" | tee -a "${OUT_DIR}/final_summary.txt"
exit "${final_status}"
