#!/usr/bin/env python3
"""Build four-wave A+B TDM ownership variants for persistent GEMM1.

The first output combines quarter-A plus quarter-B TDM work on every wave with
direct ScaleA/ScaleB CLUSTER_LOAD_B128 gathers.  It is deliberately generated
from the current stable all-NT_RT kernel and keeps the compute/epilogue order.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt.s"
)
OUTPUT_DIRECT_SCALE = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_ab4_scale_cluster.s"
)
EXPECTED_SOURCE_SHA256 = (
    "8934b9767fb8295b7d1bf3c5df246fbb239a2784c76f11873f1ceb490d9f0169"
)

ROLE_PATHS = (
    (0, ".Lbranch_000000001f08:", ".Lmoe_a_prefetched_descriptor_ready:"),
    (1, ".Lbranch_000000002b48:", ".Lmoe_b_prefetched_descriptor_ready:"),
    (2, ".Lbranch_0000000037ac:", ".Lmoe_scale_a_prefetched_descriptor_ready:"),
    (3, ".Lbranch_0000000043f8:", ".Lmoe_scale_b_prefetched_descriptor_ready:"),
)
TAIL_LABELS = (
    ".Lmoe_next_prefetch_a:",
    ".Lmoe_next_prefetch_b:",
    ".Lmoe_next_prefetch_scale_a:",
    ".Lmoe_next_prefetch_scale_b:",
)
A_RING = (0x00000, 0x08000, 0x12000, 0x1A000)
B_RING = (0x30000, 0x38000, 0x40000, 0x48000)
SA_RING = (0x10000, 0x10800, 0x11000, 0x11800)
SB_RING = (0x22000, 0x22800, 0x23000, 0x23800)
ALL_RING_VALUES = {*A_RING, *B_RING, *SA_RING, *SB_RING}

SELECTOR_RE = re.compile(r"^\s*s_mov_b32\s+s33,\s+([^\s;]+)")
DS_SCALE_RE = re.compile(
    r"^\s*ds_load_b32\s+v(?P<dst>\d+),\s+v(?P<src>80|81)\b"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_setup(lines: list[str], entry: str, ready: str) -> list[str]:
    start = lines.index(entry)
    end = lines.index(ready, start + 1)
    first = next(
        i
        for i in range(start, end)
        if re.match(r"^\s*s_mov_b32\s+s95,", lines[i])
    )
    return lines[first:end]


def find_tail_descriptor(lines: list[str], label: str) -> list[str]:
    start = lines.index(label) + 1
    end = next(
        i
        for i in range(start, len(lines))
        if lines[i].lstrip().startswith("tensor_load_to_lds s[32:35]")
    )
    return lines[start:end]


def replace_ring_bases(
    lines: list[str], old: tuple[int, ...], new: tuple[int, ...]
) -> None:
    mapping = dict(zip(old, new, strict=True))
    pattern = re.compile(
        r"^(?P<head>\s*s_mov_b32\s+s(?:33|95|96|97|98),\s+)"
        r"(?P<value>0x[0-9a-fA-F]+|0)(?P<tail>\s*(?:;.*)?)$"
    )
    hits = 0
    for i, line in enumerate(lines):
        match = pattern.match(line)
        if match is None:
            continue
        value = int(match.group("value"), 0)
        if value not in mapping:
            continue
        lines[i] = f"{match.group('head')}0x{mapping[value]:x}{match.group('tail')}"
        hits += 1
    if hits < 1:
        raise RuntimeError("no LDS ring base was rewritten")


def add_quarter_pointer_and_bound(
    lines: list[str], *, tensor: str, quarter: int, tail: bool
) -> None:
    if tensor == "A":
        pointer_hi = 73
        fallback_hi = 63
    elif tensor == "B":
        pointer_hi = 75
        fallback_hi = 47
    else:
        raise ValueError(tensor)
    if quarter == 0:
        return
    byte_offset = quarter * 0x38000
    outer_offset = quarter * 4

    marker = re.compile(rf"^\s*s_or_b32\s+s35,\s+s{pointer_hi},\s+s35")
    hits = [i for i, line in enumerate(lines) if marker.match(line)]
    if len(hits) != 1:
        raise RuntimeError(f"{tensor} pointer marker count={len(hits)}")
    pos = hits[0] + 1
    lines[pos:pos] = [
        f"\ts_add_co_u32 s34, s34, 0x{byte_offset:x}",
        "\ts_add_co_ci_u32 s35, s35, 0",
    ]

    bound = re.compile(r"^\s*s_lshr_b32\s+s26,\s+s26,\s+4\b")
    hits = [i for i, line in enumerate(lines) if bound.match(line)]
    if len(hits) != 1:
        raise RuntimeError(f"{tensor} main bound count={len(hits)}")
    lines[hits[0] + 1 : hits[0] + 1] = [
        f"\ts_sub_co_u32 s26, s26, {outer_offset}"
    ]

    if tail:
        return
    fallback_marker = re.compile(
        rf"^\s*s_or_b32\s+s{fallback_hi},\s+s{fallback_hi},\s+0x80000000"
    )
    hits = [i for i, line in enumerate(lines) if fallback_marker.match(line)]
    if len(hits) != 1:
        raise RuntimeError(f"{tensor} fallback pointer marker count={len(hits)}")
    pos = hits[0]
    fallback_lo = fallback_hi - 1
    lines[pos:pos] = [
        f"\ts_add_co_u32 s{fallback_lo}, s{fallback_lo}, 0x{byte_offset:x}",
        f"\ts_add_co_ci_u32 s{fallback_hi}, s{fallback_hi}, 0",
    ]
    fallback_bound = re.compile(r"^\s*s_lshr_b32\s+s27,\s+s27,\s+4\b")
    hits = [i for i, line in enumerate(lines) if fallback_bound.match(line)]
    if len(hits) != 1:
        raise RuntimeError(f"{tensor} fallback bound count={len(hits)}")
    lines[hits[0] + 1 : hits[0] + 1] = [
        f"\ts_sub_co_u32 s27, s27, {outer_offset}"
    ]


def rename_b_mask_labels(lines: list[str], tag: str) -> list[str]:
    replacements = {
        ".Lbranch_000000002c70": f".Lmoe_ab4_{tag}_b_mask_loop",
        ".Lmoe_next_prefetch_column_mask_loop_b": (
            f".Lmoe_ab4_{tag}_b_tail_mask_loop"
        ),
    }
    return [
        line.replace(old, new)
        for line in lines
        for old, new in [next(
            ((old, new) for old, new in replacements.items() if old in line),
            ("", ""),
        )]
    ]


def rename_labels(lines: list[str], mapping: dict[str, str]) -> list[str]:
    result = []
    for line in lines:
        for old, new in mapping.items():
            line = line.replace(old, new)
        result.append(line)
    return result


def primary_a_setup(template: list[str], wave: int, *, tail: bool) -> list[str]:
    lines = list(template)
    quarter_offset = wave * 0x2000
    replace_ring_bases(
        lines, A_RING, tuple(base + quarter_offset for base in A_RING)
    )
    extent_hits = 0
    for i, line in enumerate(lines):
        if re.search(r"\bs_or_b32 s40, s40, 16\b", line):
            lines[i] = re.sub(r", 16\b", ", 4", line)
            extent_hits += 1
    if extent_hits != 1:
        raise RuntimeError(f"A quarter extent count={extent_hits}")
    add_quarter_pointer_and_bound(lines, tensor="A", quarter=wave, tail=tail)
    return lines


def remap_primary_descriptor(lines: list[str]) -> list[str]:
    mapping = {register: register + 48 for register in range(32, 44)}

    def replace(match: re.Match[str]) -> str:
        register = int(match.group(1))
        return f"s{mapping.get(register, register)}"

    return [re.sub(r"\bs(\d+)\b", replace, line) for line in lines]


def secondary_b_setup(template: list[str], wave: int) -> list[str]:
    start = next(
        i for i, line in enumerate(template) if re.match(r"^\s*s_mov_b32\s+s32,", line)
    )
    end = next(
        i
        for i in range(start, len(template))
        if re.match(r"^\s*s_bitset1_b32\s+s36,\s+21", template[i])
    ) + 1
    lines = list(template[start:end])
    lines = rename_labels(
        lines,
        {".Lbranch_000000002c70": f".Lmoe_ab4_wave{wave}_b_mask_loop"},
    )
    quarter_offset = wave * 0x2000
    replace_ring_bases(
        lines, B_RING, tuple(base + quarter_offset for base in B_RING)
    )
    extent_hits = 0
    for i, line in enumerate(lines):
        if re.search(r"\bs_or_b32 s40, s40, 16\b", line):
            lines[i] = re.sub(r", 16\b", ", 4", line)
            extent_hits += 1
    if extent_hits != 1:
        raise RuntimeError(f"B quarter extent count={extent_hits}")
    add_quarter_pointer_and_bound(lines, tensor="B", quarter=wave, tail=True)
    lines = remap_primary_descriptor(lines)
    lines.extend(("\ts_mov_b32 s104, 0x800", "\ts_mov_b32 s105, 0"))
    return lines


def selector_ring(selector: str) -> int:
    registers = {"s95": 0, "s96": 1, "s97": 2, "s98": 3}
    if selector in registers:
        return registers[selector]
    value = int(selector, 0)
    for ring, group in enumerate(zip(A_RING, B_RING, SA_RING, SB_RING, strict=True)):
        if value in group:
            return ring
    raise RuntimeError(f"unknown input ring selector: {selector}")


def paired_issue(
    line: str, ring: int, issue_id: int, current_k_offset: int
) -> list[str]:
    done = f".Lmoe_ab4_b_issue_done_{issue_id}"
    k_offset = (
        "s58"
        if current_k_offset == 0
        else f"s58, 0x{current_k_offset:x}"
    )
    k_guard = (
        ["\ts_cmp_lt_u32 s58, s19"]
        if current_k_offset == 0
        else [
            f"\ts_add_co_u32 s24, {k_offset}",
            "\ts_cmp_lt_u32 s24, s19",
        ]
    )
    return [
        "\t; Every wave loads one A quarter and one B quarter.",
        "\ts_mul_i32 s24, s22, 0x2000",
        f"\ts_add_co_u32 s33, s24, 0x{A_RING[ring]:x}",
        line,
        f"\ts_add_co_u32 s81, s24, 0x{B_RING[ring]:x}",
        # The secondary B descriptor advances independently and does not run
        # the source kernel's K-tail fallback updates.  Guard it with the
        # exact K offset represented by this static issue site so no request
        # is emitted after the final K256 tile.
        *k_guard,
        f"\ts_cbranch_scc0 {done}",
        "\ttensor_load_to_lds s[80:83], s[84:91] th:TH_LOAD_NT_RT",
        "\ts_add_co_u32 s82, s82, s104",
        "\ts_add_co_ci_u32 s83, s83, s105",
        done + ":",
    ]


def tail_descriptor(
    template: list[str], *, tensor: str, wave: int, tag: str
) -> list[str]:
    lines = list(template)
    if tensor == "B":
        lines = rename_labels(
            lines,
            {
                ".Lmoe_next_prefetch_column_mask_loop_b": (
                    f".Lmoe_ab4_{tag}_tail_b_mask_loop"
                )
            },
        )
    old_ring = A_RING if tensor == "A" else B_RING
    quarter_offset = wave * 0x2000
    replace_ring_bases(
        lines, old_ring, tuple(base + quarter_offset for base in old_ring)
    )
    extent_hits = 0
    for i, line in enumerate(lines):
        if re.search(r"\bs_or_b32 s40, s40, 16\b", line):
            lines[i] = re.sub(r", 16\b", ", 4", line)
            extent_hits += 1
    if extent_hits != 1:
        raise RuntimeError(f"tail {tensor} quarter extent count={extent_hits}")
    add_quarter_pointer_and_bound(lines, tensor=tensor, quarter=wave, tail=True)
    lds = A_RING[0] + quarter_offset if tensor == "A" else B_RING[0] + quarter_offset
    lines.append(f"\ts_mov_b32 s33, 0x{lds:x}")
    lines.append("\ttensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT_RT")
    return lines


def build_ab4(source: str) -> str:
    lines = source.splitlines()
    a_template = find_setup(lines, ROLE_PATHS[0][1], ROLE_PATHS[0][2])
    b_template = find_setup(lines, ROLE_PATHS[1][1], ROLE_PATHS[1][2])
    a_tail = find_tail_descriptor(lines, TAIL_LABELS[0])
    b_tail = find_tail_descriptor(lines, TAIL_LABELS[1])

    # Every task rebuilds both descriptors.  Prefetched tasks still skip the
    # stage-0 issue in the body; the tail below has already issued both halves.
    for wave, entry, ready in reversed(ROLE_PATHS):
        start = lines.index(entry) + 1
        end = lines.index(ready, start)
        setup = [
            f"\t; Wave {wave}: A M-quarter plus B N-quarter descriptors.",
            *primary_a_setup(a_template, wave, tail=False),
            *secondary_b_setup(b_template, wave),
        ]
        lines[start:end] = setup

    # Record the current K offset associated with every static issue site.
    # The source updates its descriptor using the next K offset immediately
    # after each issue; subtracting one K256 step gives the request just sent.
    k_offsets: list[int] = []
    k_update = re.compile(
        r"^\s*s_add_co_u32\s+s24,\s+s58,\s+(0x[0-9a-fA-F]+|\d+)"
    )
    source_cutoff = lines.index(".Lbranch_00000000ad68:")
    source_tdm_positions = [
        i
        for i, candidate in enumerate(lines[:source_cutoff])
        if candidate.lstrip().startswith(
            "tensor_load_to_lds s[32:35], s[36:43]"
        )
    ]
    for issue_index, position in enumerate(source_tdm_positions):
        next_position = (
            source_tdm_positions[issue_index + 1]
            if issue_index + 1 < len(source_tdm_positions)
            else source_cutoff
        )
        match = next(
            (
                k_update.match(lines[i])
                for i in range(position + 1, next_position)
                if k_update.match(lines[i]) is not None
            ),
            None,
        )
        if match is None:
            # The steady loop's update can occur after substantial interleaved
            # WMMA/DS work but still before the following static issue site.
            match = next(
                (
                    k_update.match(lines[i])
                    for i in range(position + 1, min(position + 256, source_cutoff))
                    if k_update.match(lines[i]) is not None
                ),
                None,
            )
        if match is None:
            raise RuntimeError(f"missing K update after issue {issue_index}")
        next_offset = int(match.group(1), 0)
        if next_offset < 0x100:
            raise RuntimeError(f"invalid K update after issue {issue_index}")
        k_offsets.append(next_offset - 0x100)

    # Replace each primary issue with the paired A+B issue.  The source's
    # selector identifies the four-stage LDS ring even in duplicated cold paths.
    cutoff = lines.index(".Lbranch_00000000ad68:")
    output: list[str] = []
    selector: str | None = None
    issue_sites = 0
    for index, line in enumerate(lines):
        if index >= cutoff:
            output.extend(lines[index:])
            break
        match = SELECTOR_RE.match(line)
        if match:
            selector = match.group(1)
        if line.lstrip().startswith("tensor_load_to_lds s[32:35], s[36:43]"):
            if selector is None:
                raise RuntimeError(f"missing selector before TDM line {index + 1}")
            output.extend(
                paired_issue(
                    line,
                    selector_ring(selector),
                    issue_sites,
                    k_offsets[issue_sites],
                )
            )
            issue_sites += 1
        else:
            output.append(line)
    else:
        raise RuntimeError("output-setup cutoff not found")
    lines = output
    if issue_sites != 56:
        raise RuntimeError(f"expected 56 paired issue sites, got {issue_sites}")

    # Rebuild and issue both stage-0 quarter descriptors in the task tail.
    tail_start = lines.index(TAIL_LABELS[0])
    tail_end = lines.index(".Lmoe_next_prefetch_done:")
    tail_lines: list[str] = []
    for wave, label in enumerate(TAIL_LABELS):
        tail_lines.extend(
            (
                label,
                f"\t; Next task wave {wave}: prefetch A and B quarters.",
                *tail_descriptor(a_tail, tensor="A", wave=wave, tag=f"wave{wave}"),
                *tail_descriptor(b_tail, tensor="B", wave=wave, tag=f"wave{wave}"),
                "\ts_branch .Lmoe_next_prefetch_done",
                "",
            )
        )
    lines[tail_start:tail_end] = tail_lines

    # The secondary descriptor consumes the last two architectural SGPRs.
    lines = [
        line.replace(".amdhsa_next_free_sgpr 104", ".amdhsa_next_free_sgpr 106")
        .replace(".numbered_sgpr: 104", ".numbered_sgpr: 106")
        .replace(".set moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1.numbered_sgpr, 104", ".set moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1.numbered_sgpr, 106")
        for line in lines
    ]
    lines[3] = (
        "\t; Persistent MoE GEMM1 all-NT_RT with four-wave quarter A+B TDM."
    )
    text = "\n".join(lines) + "\n"
    if sum(
        line.lstrip().startswith("tensor_load_to_lds")
        for line in text.splitlines()
    ) != 120:
        raise RuntimeError("A+B quarter TDM site count is not 120")
    return text


def direct_scale_pointer_setup() -> list[str]:
    return [
        "\t; Direct-scale base for the wave-local M128/N128 slice.",
        "\ts_and_b32 s24, s22, 1",
        "\ts_mul_i32 s24, s24, 0x7000",
        "\ts_add_co_u32 s76, s76, s24",
        "\ts_add_co_ci_u32 s77, s77, 0",
        "\ts_lshr_b32 s24, s22, 1",
        "\ts_mul_i32 s24, s24, 0x7000",
        "\ts_add_co_u32 s78, s78, s24",
        "\ts_add_co_ci_u32 s79, s79, 0",
    ]


def gather_block(kind: str, destination: int, restore_msb: str, group_id: int) -> list[str]:
    if kind == "ScaleA":
        pointer, lo, hi = "s[76:77]", "s76", "s77"
    else:
        pointer, lo, hi = "s[78:79]", "s78", "s79"
    first_half = destination in {82, 92}
    update = (
        (f"\ts_add_co_u32 {lo}, {lo}, 0x3800", f"\ts_add_co_ci_u32 {hi}, {hi}, 0")
        if first_half
        else (f"\ts_sub_co_u32 {lo}, {lo}, 0x3700", f"\ts_sub_co_ci_u32 {hi}, {hi}, 0")
    )
    return [
        f"\t; {kind} direct B128 gather {group_id} for A+B quarter TDM.",
        "\ts_set_vgpr_msb 0",
        "\ts_mov_b32 m0, 0x10000",
        f"\tcluster_load_b128 v[228:231], v1, {pointer}",
        f"\tcluster_load_b128 v[232:235], v1, {pointer} offset:128",
        f"\tcluster_load_b128 v[236:239], v1, {pointer} offset:7168",
        f"\tcluster_load_b128 v[240:243], v1, {pointer} offset:7296",
        "\ts_wait_loadcnt 0x0",
        "\tv_cmp_eq_u32_e32 vcc_lo, 0, v244",
        "\tv_cndmask_b32_e32 v228, v229, v228, vcc_lo",
        "\tv_cndmask_b32_e32 v230, v231, v230, vcc_lo",
        "\tv_cndmask_b32_e32 v232, v233, v232, vcc_lo",
        "\tv_cndmask_b32_e32 v234, v235, v234, vcc_lo",
        "\tv_cndmask_b32_e32 v236, v237, v236, vcc_lo",
        "\tv_cndmask_b32_e32 v238, v239, v238, vcc_lo",
        "\tv_cndmask_b32_e32 v240, v241, v240, vcc_lo",
        "\tv_cndmask_b32_e32 v242, v243, v242, vcc_lo",
        "\tv_cmp_eq_u32_e32 vcc_lo, 0, v245",
        f"\tv_cndmask_b32_e32 v{destination}, v230, v228, vcc_lo",
        f"\tv_cndmask_b32_e32 v{destination + 1}, v234, v232, vcc_lo",
        f"\tv_cndmask_b32_e32 v{destination + 2}, v238, v236, vcc_lo",
        f"\tv_cndmask_b32_e32 v{destination + 3}, v242, v240, vcc_lo",
        *update,
        f"\ts_set_vgpr_msb {restore_msb}",
    ]


def add_direct_scales(ab4: str) -> str:
    lines = ab4.splitlines()
    lane_marker = (
        "\tv_and_b32_e32 v0, 31, v0                                   "
        "; 000000001984: 3600009F"
    )
    pos = lines.index(lane_marker) + 1
    lines[pos:pos] = [
        "\t; Direct-scale aligned address and lane-in-block selectors.",
        "\tv_lshrrev_b32_e32 v1, 2, v0",
        "\tv_lshlrev_b32_e32 v1, 4, v1",
        "\tv_and_b32_e32 v244, 1, v0",
        "\tv_lshrrev_b32_e32 v245, 1, v0",
        "\tv_and_b32_e32 v245, 1, v245",
    ]
    pointer_marker = (
        "\ts_add_co_ci_u32 s79, 0, s11                                "
        "; 000000001DEC: 824F0B80"
    )
    pos = lines.index(pointer_marker) + 1
    lines[pos:pos] = direct_scale_pointer_setup()

    output: list[str] = []
    active_msb = "0"
    open_group: tuple[str, int] | None = None
    seen: list[int] = []
    groups = 0
    removed = 0
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("s_set_vgpr_msb"):
            active_msb = stripped.split(";", 1)[0].split(None, 1)[1].strip()
        match = DS_SCALE_RE.match(line)
        if match is None:
            output.append(line)
            continue
        src = int(match.group("src"))
        dst = int(match.group("dst"))
        kind = "ScaleA" if src == 80 else "ScaleB"
        bases = (82, 86) if kind == "ScaleA" else (92, 96)
        base = next((value for value in bases if value <= dst < value + 4), None)
        if base is None:
            raise RuntimeError(f"unexpected scale destination v{dst}")
        key = (kind, base)
        if dst == base:
            if open_group is not None:
                raise RuntimeError(f"new group {key} before closing {open_group}")
            open_group = key
            seen = []
            output.extend(gather_block(kind, base, active_msb, groups))
            groups += 1
        if open_group != key:
            raise RuntimeError(f"scale group order mismatch: {open_group} vs {key}")
        seen.append(dst)
        if seen == list(range(base, base + 4)):
            open_group = None
            seen = []
        elif seen != list(range(base, dst + 1)):
            raise RuntimeError(f"non-consecutive scale group {seen}")
        removed += 1

    if open_group is not None or groups != 48 or removed != 192:
        raise RuntimeError(
            f"direct-scale grouping failed: open={open_group}, groups={groups}, removed={removed}"
        )
    lines = [
        re.sub(r"^(\s*s_wait_dscnt)\s+0x(?:14|8|4)\b", r"\1 0x0", line)
        for line in output
    ]
    lines[3] = (
        "\t; Persistent MoE GEMM1 with four-wave quarter A+B TDM and direct scale gathers."
    )
    text = "\n".join(lines) + "\n"
    expected = {
        "tensor_load_to_lds": 120,
        "cluster_load_b128": 192,
        "s_wait_loadcnt": 48,
        "ds_load_b32": 0,
        "v_wmma_scale_f32_32x16x128_f4": 512,
    }
    for mnemonic, wanted in expected.items():
        actual = sum(
            line.lstrip().startswith(mnemonic) for line in text.splitlines()
        )
        if actual != wanted:
            raise RuntimeError(f"{mnemonic}: expected {wanted}, got {actual}")
    return text


def main() -> None:
    digest = sha256(SOURCE)
    if digest != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"winner checksum changed: expected {EXPECTED_SOURCE_SHA256}, got {digest}"
        )
    source = SOURCE.read_text(encoding="utf-8")
    ab4 = build_ab4(source)
    direct = add_direct_scales(ab4)
    OUTPUT_DIRECT_SCALE.write_text(direct, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT_DIRECT_SCALE}")
    print(f"sha256={sha256(OUTPUT_DIRECT_SCALE)}")


if __name__ == "__main__":
    main()
