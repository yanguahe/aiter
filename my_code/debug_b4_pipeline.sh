#!/usr/bin/env bash
set -uo pipefail

GENERATE_ONLY=0
if [[ "${1:-}" == "--generate-only" ]]; then
    GENERATE_ONLY=1
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "usage: bash my_code/debug_b4_pipeline.sh [--generate-only]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GROUND="${SCRIPT_DIR}/flydsl_dump/gemm_a8w4_tdm_t64x256x256_w1x4_b4_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1/21_final_isa.s"
CURRENT="${SCRIPT_DIR}/moe_gemm1_a8w4.v2.buf4.s"
ADAPTER="${SCRIPT_DIR}/isa_runner/tdm_adapter.py"
OUT_DIR="${B4_DEBUG_OUT_DIR:-${SCRIPT_DIR}/b4_pipeline_debug_results}"
VARIANT_DIR="${OUT_DIR}/variants"
REPORT_DIR="${OUT_DIR}/reports"

mkdir -p "${VARIANT_DIR}" "${REPORT_DIR}"

python3 - "${GROUND}" "${CURRENT}" "${VARIANT_DIR}" "${OUT_DIR}" <<'PY'
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import sys

ground_path = Path(sys.argv[1])
current_path = Path(sys.argv[2])
variant_dir = Path(sys.argv[3])
out_dir = Path(sys.argv[4])

ground = ground_path.read_text().splitlines()
current = current_path.read_text().splitlines()


def one_index(lines: list[str], text: str, start: int = 0, end: int | None = None) -> int:
    end = len(lines) if end is None else end
    hits = [i for i in range(start, end) if lines[i].strip() == text]
    assert len(hits) == 1, (text, hits)
    return hits[0]


def loop_bounds(lines: list[str]) -> tuple[int, int]:
    return one_index(lines, ".LBB0_35:"), one_index(lines, ".LBB0_36:")


def tail_bounds(lines: list[str]) -> tuple[int, int]:
    _, l36 = loop_bounds(lines)
    barrier = next(i for i in range(l36, len(lines)) if lines[i].strip() == "s_barrier_wait -1")
    begin = next(
        i for i in range(barrier, len(lines))
        if lines[i].strip() == "s_wait_alu depctr_va_vdst(0)"
    ) + 1
    end = next(
        i for i in range(begin, len(lines))
        if lines[i].strip() == "s_and_b32 s10, s72, 3"
    )
    return begin, end


def wmma_indices(lines: list[str]) -> list[int]:
    l35, l36 = loop_bounds(lines)
    return [
        i for i in range(l35, l36)
        if lines[i].strip().startswith("v_wmma_scale_f32_16x16x128_f8f6f4")
    ]


def find_unique(lines: list[str], prefix: str, begin: int, end: int) -> str:
    hits = [lines[i] for i in range(begin, end) if lines[i].strip().startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


def find_exact(lines: list[str], text: str, begin: int, end: int) -> str:
    hits = [lines[i] for i in range(begin, end) if lines[i].strip() == text]
    assert hits, text
    assert len({x for x in hits}) == 1, (text, hits)
    return hits[0]


def ds_lines(lines: list[str], begin: int, end: int) -> list[str]:
    return [lines[i] for i in range(begin, end) if lines[i].strip().startswith("ds_load_")]


gb, ge = tail_bounds(ground)
ground_tail = ground[gb:ge]
ground_ds = ds_lines(ground, gb, ge)
assert len(ground_ds) == 40
assert sum("ds_load_b128" in x for x in ground_ds) == 32
assert sum("ds_load_2addr_b32" in x for x in ground_ds) == 8

a0 = [x for x in ground_ds if ", v124" in x]
b0_data = [x for x in ground_ds if any(f"/*v[{n}:" in x for n in (264, 268, 272, 276, 280, 284, 304, 308))]
b1_data = [x for x in ground_ds if any(f"/*v[{n}:" in x for n in (256, 260, 288, 292, 296, 300, 312, 316))]

scale_by_dst = {}
for line in ground_ds:
    if "ds_load_2addr_b32" not in line:
        continue
    m = re.search(r"v\[(\d+):(\d+)\]", line)
    assert m
    scale_by_dst[int(m.group(1))] = line

assert len(a0) == 16
assert len(b0_data) == 8
assert len(b1_data) == 8
assert set(scale_by_dst) == {136, 138, 140, 142, 144, 146, 148, 150}

set_400 = find_exact(ground, "s_set_vgpr_msb 0x400", gb - 10, ge)
set_4000 = find_exact(ground, "s_set_vgpr_msb 0x4000", gb, ge)
set_64 = find_exact(ground, "s_set_vgpr_msb 64", gb, ge)
prod_67 = find_unique(ground, "v_add_nc_u32_e32 v67,", gb, ge)
prod_68 = find_unique(ground, "v_add_nc_u32_e32 v68,", gb, ge)
prod_64 = find_unique(ground, "v_add_nc_u32_e32 v64,", gb, ge)
prod_66 = find_unique(ground, "v_add_nc_u32_e32 v66,", gb, ge)
wait_vm = find_unique(ground, "s_wait_alu depctr_vm_vsrc(0)", gb, ge)

indent = "\t"
wait_va_1 = indent + "s_wait_alu depctr_va_vdst(1)"
wait_va_0 = indent + "s_wait_alu depctr_va_vdst(0)"
set_zero = indent + "s_set_vgpr_msb 0"


def make_target_tail(modefix: bool) -> list[str]:
    # The high byte records the previous low-byte mode. The two reset points
    # are 0->0 transitions in the reordered schedule, so their valid SIMM16 is 0.
    reset_before_b0 = set_zero if modefix else set_4000
    reset_before_a0 = set_zero if modefix else set_400
    return [
        reset_before_b0,
        prod_67,
        prod_68,
        set_64,
        *b0_data,
        set_4000,
        scale_by_dst[140],
        scale_by_dst[144],
        wait_va_1,
        scale_by_dst[150],
        wait_va_0,
        scale_by_dst[148],
        reset_before_a0,
        *a0,
        set_64,
        *b1_data,
        wait_vm,
        set_4000,
        prod_64,
        prod_66,
        scale_by_dst[136],
        scale_by_dst[142],
        wait_va_1,
        scale_by_dst[146],
        wait_va_0,
        scale_by_dst[138],
    ]


def reorder_tail(lines: list[str], modefix: bool = False) -> list[str]:
    result = list(lines)
    begin, end = tail_bounds(result)
    result[begin:end] = make_target_tail(modefix)
    return result


def move_wait0_after_ksl0(lines: list[str], keep_wait11: bool) -> list[str]:
    result = list(lines)
    l35, l36 = loop_bounds(result)
    wait11 = one_index(result, "s_wait_dscnt 0x11", l35, l36)
    wait0 = one_index(result, "s_wait_dscnt 0x0", l35, l36)
    first_wmma = wmma_indices(result)[0]
    assert wait11 < first_wmma < wait0
    result.pop(wait0)
    if not keep_wait11:
        result.pop(wait11)
    wmmas = wmma_indices(result)
    assert len(wmmas) == 32
    result.insert(wmmas[15] + 1, indent + "s_wait_dscnt 0x0")
    return result


def full_wait_before_ksl0(lines: list[str]) -> list[str]:
    result = list(lines)
    l35, l36 = loop_bounds(result)
    for text in ("s_wait_dscnt 0x11", "s_wait_dscnt 0x0"):
        result.pop(one_index(result, text, l35, l36))
        l35, l36 = loop_bounds(result)
    first_wmma = wmma_indices(result)[0]
    result.insert(first_wmma, indent + "s_wait_dscnt 0x0")
    return result


def set_entry_wait(lines: list[str], count: int) -> list[str]:
    result = list(lines)
    l35, l36 = loop_bounds(result)
    i = one_index(result, "s_wait_dscnt 0xc", l35, l36)
    result[i] = indent + f"s_wait_dscnt 0x{count:x}"
    return result


def set_vm_vsrc1(lines: list[str]) -> list[str]:
    result = list(lines)
    l39 = one_index(result, ".LBB0_39:")
    i = next(
        j for j in range(l39, len(result))
        if result[j].strip().startswith("s_wait_alu depctr_vm_vsrc(")
    )
    assert result[i].strip() == "s_wait_alu depctr_vm_vsrc(3)"
    result[i] = indent + "s_wait_alu depctr_vm_vsrc(1)"
    return result


def nonblank(lines: list[str]) -> list[str]:
    return [x.strip() for x in lines if x.strip()]


def tracked_context(lines: list[str]) -> Counter[tuple[str, int]]:
    begin, end = tail_bounds(lines)
    state = -1
    for line in lines[:begin]:
        m = re.fullmatch(
            r"s_set_vgpr_msb\s+(0x[0-9a-fA-F]+|\d+)",
            line.strip(),
        )
        if m:
            state = int(m.group(1), 0)
    assert state >= 0
    result = []
    for line in lines[begin:end]:
        text = line.strip()
        m = re.fullmatch(r"s_set_vgpr_msb\s+(0x[0-9a-fA-F]+|\d+)", text)
        if m:
            state = int(m.group(1), 0)
            continue
        if text.startswith("ds_load_") or any(
            text.startswith(f"v_add_nc_u32_e32 v{v},") for v in (64, 66, 67, 68)
        ):
            result.append((text, state))
    return Counter(result)


ground_context = tracked_context(ground)
ground_low_context = Counter((inst, state & 0xFF) for (inst, state), count in ground_context.items() for _ in range(count))
ground_ds_multiset = Counter(x.strip() for x in ground_ds)
b0_set = {x.strip() for x in b0_data} | {
    scale_by_dst[n].strip() for n in (140, 144, 150, 148)
}
a0_set = {x.strip() for x in a0}
b1_set = {x.strip() for x in b1_data} | {
    scale_by_dst[n].strip() for n in (136, 142, 146, 138)
}


def mode_chain_analysis(lines: list[str]) -> tuple[bool, list[str]]:
    _, l36 = loop_bounds(lines)
    l39 = one_index(lines, ".LBB0_39:")
    end = next(
        i for i in range(l39, len(lines))
        if lines[i].strip().startswith("s_set_vgpr_msb ")
    ) + 1
    previous = -1
    for line in lines[:l36]:
        m = re.fullmatch(r"s_set_vgpr_msb\s+(0x[0-9a-fA-F]+|\d+)", line.strip())
        if m:
            previous = int(m.group(1), 0) & 0xFF
    assert previous >= 0
    transitions = []
    ok = True
    for i in range(l36, end):
        m = re.fullmatch(r"s_set_vgpr_msb\s+(0x[0-9a-fA-F]+|\d+)", lines[i].strip())
        if not m:
            continue
        simm16 = int(m.group(1), 0)
        old_mode = (simm16 >> 8) & 0xFF
        new_mode = simm16 & 0xFF
        valid = old_mode == previous
        transitions.append(
            f"L{i + 1}:0x{old_mode:02x}->0x{new_mode:02x}"
            f"(expected_old=0x{previous:02x},ok={int(valid)})"
        )
        ok &= valid
        previous = new_mode
    return ok, transitions


def vm_source_analysis(lines: list[str]) -> tuple[bool, int, int, int]:
    begin, _ = tail_bounds(lines)
    l39 = one_index(lines, ".LBB0_39:")
    full_wait = max(
        i for i in range(begin, l39)
        if lines[i].strip() == "s_wait_alu depctr_vm_vsrc(0)"
    )
    pending_ds = [
        lines[i].strip() for i in range(full_wait + 1, l39)
        if lines[i].strip().startswith("ds_load_")
    ]
    v64_positions = [
        n for n, text in enumerate(pending_ds, 1)
        if re.search(r",\s*v64(?:\s|$)", text)
    ]
    assert len(v64_positions) == 1, v64_positions
    wait_i = next(
        i for i in range(l39, len(lines))
        if lines[i].strip().startswith("s_wait_alu depctr_vm_vsrc(")
    )
    m = re.fullmatch(r"s_wait_alu depctr_vm_vsrc\((\d+)\)", lines[wait_i].strip())
    assert m
    threshold = int(m.group(1))
    oldest_consumed = len(pending_ds) - threshold
    safe = oldest_consumed >= v64_positions[0]
    return safe, len(pending_ds), v64_positions[0], threshold


def assert_common(
    name: str,
    lines: list[str],
    reordered: bool,
    expect_mode_chain: bool,
    expect_vm_safe: bool,
) -> tuple[bool, bool, int, int, int]:
    begin, end = tail_bounds(lines)
    tail_ds = ds_lines(lines, begin, end)
    assert len(tail_ds) == 40, name
    assert Counter(x.strip() for x in tail_ds) == ground_ds_multiset, name
    assert len(wmma_indices(lines)) == 32, name
    if reordered:
        groups = []
        for line in tail_ds:
            text = line.strip()
            group = "B0" if text in b0_set else "A0" if text in a0_set else "B1" if text in b1_set else "?"
            if not groups or groups[-1][0] != group:
                groups.append([group, 0])
            groups[-1][1] += 1
        assert groups == [["B0", 12], ["A0", 16], ["B1", 12]], (name, groups)
        low_context = Counter(
            (inst, state & 0xFF)
            for (inst, state), count in tracked_context(lines).items()
            for _ in range(count)
        )
        assert low_context == ground_low_context, name
    mode_ok, _ = mode_chain_analysis(lines)
    vm_safe, pending, v64_position, threshold = vm_source_analysis(lines)
    assert mode_ok == expect_mode_chain, (name, mode_ok, expect_mode_chain)
    assert vm_safe == expect_vm_safe, (
        name, vm_safe, expect_vm_safe, pending, v64_position, threshold
    )
    return mode_ok, vm_safe, pending, v64_position, threshold


def assert_waits(name: str, lines: list[str], mode: str, entry_wait: int = 12) -> None:
    l35, l36 = loop_bounds(lines)
    wmmas = wmma_indices(lines)
    wait0s = [i for i in range(l35, l36) if lines[i].strip() == "s_wait_dscnt 0x0"]
    wait11s = [i for i in range(l35, l36) if lines[i].strip() == "s_wait_dscnt 0x11"]
    entry = l35 + 1
    assert lines[entry].strip() == f"s_wait_dscnt 0x{entry_wait:x}", name
    assert entry < wmmas[0]
    if mode == "ground":
        assert len(wait0s) == 1
        wait0 = wait0s[0]
        assert len(wait11s) == 1 and wait11s[0] < wmmas[0] < wait0 < wmmas[1]
    elif mode == "move_keep11":
        assert len(wait0s) == 1
        wait0 = wait0s[0]
        assert len(wait11s) == 1 and wait11s[0] < wmmas[0]
        assert sum(i < wait0 for i in wmmas) == 16
    elif mode == "full":
        assert len(wait0s) == 1
        wait0 = wait0s[0]
        assert not wait11s and wait0 < wmmas[0]
    elif mode == "target":
        assert not wait11s
        overlap_waits = [i for i in wait0s if wmmas[15] < i < wmmas[16]]
        assert len(overlap_waits) == 1, (name, wait0s)
        wait0 = overlap_waits[0]
        assert sum(i < wait0 for i in wmmas) == 16
        assert sum(i > wait0 for i in wmmas) == 16
    else:
        raise AssertionError(mode)


variants: list[tuple[str, list[str], str, bool, str, int, bool, bool]] = []
variants.append((
    "ground",
    list(ground),
    "known-correct control",
    False,
    "ground",
    12,
    True,
    True,
))
variants.append((
    "wait0_moved_keep_wait11",
    move_wait0_after_ksl0(ground, keep_wait11=True),
    "isolates moving wait0 while retaining original DS order and wait11",
    False,
    "move_keep11",
    12,
    True,
    True,
))
variants.append((
    "delete_wait11_full_wait0",
    full_wait_before_ksl0(ground),
    "isolates deleting wait11 with a conservative full wait before KSL0",
    False,
    "full",
    12,
    True,
    True,
))

reordered_ground = reorder_tail(ground, modefix=False)
variants.append((
    "reorder_full_wait0",
    full_wait_before_ksl0(reordered_ground),
    "isolates reordered operand mapping and address dependencies with full DS completion",
    True,
    "full",
    12,
    False,
    False,
))

for count in (0, 4, 8, 12, 16):
    lines = move_wait0_after_ksl0(reordered_ground, keep_wait11=False)
    lines = set_entry_wait(lines, count)
    variants.append((
        f"target_waitcnt{count}",
        lines,
        f"target reorder and KSL0 overlap with entry DScnt threshold {count}",
        True,
        "target",
        count,
        False,
        False,
    ))

old_target12 = next(lines for name, lines, *_ in variants if name == "target_waitcnt12")
modefixed_reordered = reorder_tail(ground, modefix=True)
modefix_only = set_entry_wait(
    move_wait0_after_ksl0(modefixed_reordered, keep_wait11=False),
    12,
)
variants.append((
    "modefix_only",
    modefix_only,
    "fixes the complete SIMM16 old-to-new mode chain but retains VM_VSRC(3)",
    True,
    "target",
    12,
    True,
    False,
))
variants.append((
    "vm1_only",
    set_vm_vsrc1(old_target12),
    "changes downstream VM_VSRC(3) to VM_VSRC(1) but retains invalid mode resets",
    True,
    "target",
    12,
    False,
    True,
))

modefix_vm1_reordered = set_vm_vsrc1(modefixed_reordered)
variants.append((
    "modefix_vm1_full",
    full_wait_before_ksl0(modefix_vm1_reordered),
    "both fixes with a full A1-side DS wait before KSL0",
    True,
    "full",
    12,
    True,
    True,
))

for count in (0, 4, 8, 12):
    lines = move_wait0_after_ksl0(modefix_vm1_reordered, keep_wait11=False)
    lines = set_entry_wait(lines, count)
    variants.append((
        f"modefix_vm1_wait{count}",
        lines,
        f"both fixes with KSL0 overlap and entry DScnt threshold {count}",
        True,
        "target",
        count,
        True,
        True,
    ))

manifest = []
static_lines = []
for (
    name,
    lines,
    hypothesis,
    reordered,
    wait_mode,
    entry_wait,
    expect_mode_chain,
    expect_vm_safe,
) in variants:
    mode_ok, vm_safe, pending, v64_position, vm_threshold = assert_common(
        name,
        lines,
        reordered,
        expect_mode_chain,
        expect_vm_safe,
    )
    assert_waits(name, lines, wait_mode, entry_wait)
    path = variant_dir / f"{name}.s"
    path.write_text("\n".join(lines) + "\n")
    manifest.append((name, str(path), hypothesis))
    static_lines.append(
        f"{name}: static_ok=1 reordered={int(reordered)} wait_mode={wait_mode} "
        f"entry_wait={entry_wait} mode_chain_ok={int(mode_ok)} "
        f"vm_safe={int(vm_safe)} pending_ds_after_full_vm_wait={pending} "
        f"v64_source_position={v64_position} vm_threshold={vm_threshold} "
        f"instructions={len(nonblank(lines))}"
    )

best_target = next(lines for name, lines, *_ in variants if name == "modefix_vm1_wait12")
assert nonblank(best_target) == nonblank(current), (
    "current candidate drifted from deterministic modefix_vm1_wait12 generation"
)

(out_dir / "manifest.tsv").write_text(
    "\n".join("\t".join(row) for row in manifest) + "\n"
)
(out_dir / "static_summary.txt").write_text(
    "\n".join(static_lines)
    + "\ncurrent_candidate_matches_modefix_vm1_wait12=1\n"
    + "tracked_instruction_low8_mode_multiset_matches_ground=1\n"
    + "complete_SIMM16_old_to_new_mode_chain_checked=1\n"
    + "downstream_VM_VSRC_overwrite_safety_checked=1\n"
)
print(f"Generated and statically verified {len(variants)} variants in {variant_dir}")
PY
generator_rc=$?
if [[ ${generator_rc} -ne 0 ]]; then
    echo "Variant generation/static verification failed (exit ${generator_rc})" >&2
    exit "${generator_rc}"
fi

if [[ ${GENERATE_ONLY} -eq 1 ]]; then
    echo "Generate-only mode: hardware replay skipped."
    echo "Static results: ${OUT_DIR}/static_summary.txt"
    exit 0
fi

export ENABLE_CK=0
export AITER_FORCE_GFX1250=1
export AITER_MOE_EXPERT_BALANCE=true
export AITER_TDM_TILE_M=64
export AITER_TDM_TILE_N=256
export AITER_TDM_TILE_K=256
export AITER_TDM_NUM_BUFFERS=4
export AITER_TDM_WIDE_KSL=1

STATUS_TSV="${OUT_DIR}/run_status.tsv"
: > "${STATUS_TSV}"

while IFS=$'\t' read -r name isa hypothesis; do
    json="${REPORT_DIR}/${name}.json"
    stdout_txt="${REPORT_DIR}/${name}.stdout.txt"
    stderr_txt="${REPORT_DIR}/${name}.stderr.txt"
    rm -f -- "${json}"

    echo "=== ${name}: ${hypothesis} ==="
    python3 "${ADAPTER}" replay \
        --which gemm1 \
        --isa "${isa}" \
        --seed 0 \
        --deterministic-route-map \
        --iters 0 \
        --out "${json}" \
        >"${stdout_txt}" 2>"${stderr_txt}"
    rc=$?

    if [[ ! -s "${json}" ]]; then
        python3 - "${json}" "${name}" "${rc}" "${stderr_txt}" <<'PY'
import json
from pathlib import Path
import sys

out, name, rc, stderr_path = sys.argv[1:]
stderr = Path(stderr_path).read_text(errors="replace")
Path(out).write_text(json.dumps({
    "variant": name,
    "passed": False,
    "debug_wrapper_error": "tdm_adapter did not produce a report",
    "exit_code": int(rc),
    "stderr_tail": stderr[-4000:],
}, indent=2) + "\n")
PY
    fi

    printf '%s\t%s\t%s\n' "${name}" "${rc}" "${json}" >> "${STATUS_TSV}"
    echo "${name}: exit=${rc}, report=${json}"
done < "${OUT_DIR}/manifest.tsv"

python3 - "${OUT_DIR}" <<'PY'
from pathlib import Path
import json
import sys

out_dir = Path(sys.argv[1])
hypotheses = {}
for line in (out_dir / "manifest.tsv").read_text().splitlines():
    name, _, hypothesis = line.split("\t", 2)
    hypotheses[name] = hypothesis

header = [
    "variant", "exit", "passed", "rel_l2", "max_abs_diff",
    "missing", "unexpected", "canonical_match", "physical_match",
    "canonical_production_sha256", "canonical_isa_sha256",
    "physical_production_sha256", "physical_isa_sha256", "hypothesis",
]
rows = []
for line in (out_dir / "run_status.tsv").read_text().splitlines():
    name, rc, report_path = line.split("\t", 2)
    path = Path(report_path)
    try:
        report = json.loads(path.read_text())
        output_hash = report.get("output_hash") or {}
        row = [
            name,
            rc,
            str(report.get("passed", "")),
            str(report.get("rel_l2", "")),
            str(report.get("max_abs_diff", "")),
            str(report.get("missing_writes", "")),
            str(report.get("unexpected_writes", "")),
            str(output_hash.get("match", "")),
            str(output_hash.get("physical_match", "")),
            str(output_hash.get("production_sha256", "")),
            str(output_hash.get("isa_sha256", "")),
            str(output_hash.get("physical_production_sha256", "")),
            str(output_hash.get("physical_isa_sha256", "")),
            hypotheses.get(name, ""),
        ]
    except Exception as exc:
        row = [
            name, rc, "", "", "", "", "", "", "", "", "", "", "",
            f"report parse error: {exc}",
        ]
    rows.append(row)

summary = "\n".join("\t".join(row) for row in [header, *rows]) + "\n"
(out_dir / "summary.txt").write_text(summary)
print()
print(summary, end="")
print(f"Results: {out_dir}")
PY

exit 0
