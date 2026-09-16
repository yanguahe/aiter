#!/usr/bin/env python3
"""Build the retained resident A+B-quarter and scale-half TDM wait6 kernel.

The transformation starts from the current all-NT_RT winner and preserves the
validated production-style WPT2 scale ownership:

  wave0: A quarter 0 + B quarter 0 + ScaleA half 0
  wave1: A quarter 2 + B quarter 2 + ScaleA half 1
  wave2: B quarter 1 + A quarter 1 + ScaleB half 0
  wave3: B quarter 3 + A quarter 3 + ScaleB half 1

Thus every wave moves 8 KiB A, 8 KiB B, and 1 KiB scale per K256 stage.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import build_ab_quarter_scale_variants as ab4
import build_wpt2_owner_balance_variant as wpt2


HERE = Path(__file__).resolve().parent
SOURCE = ab4.SOURCE
OUTPUT = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_ab4_scale_half_tdm.s"
)
OUTPUT_SAME_TENSOR_PROBE = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_wpt2_split_quarter_probe.s"
)
OUTPUT_SAME_TENSOR_SERIAL = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_wpt2_split_quarter_serial_probe.s"
)
OUTPUT_SAME_TENSOR_SERIAL_FULL_SETUP_LOOP = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_wpt2_split_quarter_serial_full_setup_loop.s"
)
OUTPUT_SAME_TENSOR_JITCOPY = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_wpt2_split_quarter_jitcopy_probe.s"
)
OUTPUT_SAME_TENSOR_JITCOPY_SERIAL = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_wpt2_split_quarter_jitcopy_serial_probe.s"
)
OUTPUT_SAME_TENSOR_JITCOPY_SERIAL_FULL_SETUP_LOOP = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_wpt2_split_quarter_jitcopy_serial_full_setup_loop.s"
)
OUTPUT_SAME_TENSOR_GROUP1_REFRESH_SERIAL_FULL_SETUP_LOOP = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_wpt2_split_quarter_group1_refresh_serial_full_setup_loop.s"
)
OUTPUT_SAME_TENSOR_GROUP0_CLONE_SERIAL_FULL_SETUP_LOOP = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_wpt2_split_quarter_group0_clone_serial_full_setup_loop.s"
)
OUTPUT_SAME_TENSOR_FIXED_POINTER_SERIAL_FULL_SETUP_LOOP = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_wpt2_split_quarter_fixed_pointer_serial_full_setup_loop.s"
)
OUTPUT_SERIAL = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_ab4_scale_half_tdm_serial.s"
)
OUTPUT_SERIAL_FULL_SETUP_LOOP = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_ab4_scale_half_tdm_serial_full_setup_loop.s"
)
OUTPUT_FULL_SETUP_LOOP = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_ab4_scale_half_tdm_full_setup_loop.s"
)
OUTPUT_FULL_SETUP_LOOP_WAIT6 = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_ab4_scale_half_tdm_full_setup_loop_wait6.s"
)
OUTPUT_TAIL_THEN_FULL_SETUP_SERIAL = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_ab4_scale_half_tdm_tail_then_full_setup_serial.s"
)
OUTPUT_LATE_TAIL_SERIAL = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_ab4_scale_half_tdm_late_tail_serial.s"
)
OUTPUT_LATE_TAIL_WAIT6 = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_ab4_scale_half_tdm_late_tail_wait6.s"
)
OUTPUT_STATE_ONLY_TAIL_WAIT6 = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_ab4_scale_half_tdm_state_only_tail_wait6.s"
)
OUTPUT_FULL_TAIL_WG_SYNC = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_ab4_scale_half_tdm_full_tail_wg_sync.s"
)
EXPECTED_SOURCE_SHA256 = ab4.EXPECTED_SOURCE_SHA256
SAFE_WPT23_TAIL_SOURCE = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_wpt23_vgpr_descriptor.s"
)
EXPECTED_SAFE_WPT23_TAIL_SHA256 = (
    "4dbf039cd17601ca3cadde9c8edfe32b7c93936fd2ab2635c2609516237edc8a"
)

ROLE_SPECS = (
    # wave, primary tensor, primary quarter, scale tensor, scale half,
    # extra tensor, extra quarter
    (0, "A", 0, "ScaleA", 0, "B", 0),
    (1, "A", 2, "ScaleA", 1, "B", 2),
    (2, "B", 1, "ScaleB", 0, "A", 1),
    (3, "B", 3, "ScaleB", 1, "A", 3),
)
SAME_TENSOR_QUARTER_SPECS = (
    (0, "A", 0, "ScaleA", 0, "A", 1),
    (1, "A", 2, "ScaleA", 1, "A", 3),
    (2, "B", 0, "ScaleB", 0, "B", 1),
    (3, "B", 2, "ScaleB", 1, "B", 3),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remap_sgpr_range(
    lines: list[str], *, old_first: int, new_first: int, count: int = 12
) -> list[str]:
    mapping = {
        register: new_first + register - old_first
        for register in range(old_first, old_first + count)
    }

    def replace(match: re.Match[str]) -> str:
        register = int(match.group(1))
        return f"s{mapping.get(register, register)}"

    return [re.sub(r"\bs(\d+)\b", replace, line) for line in lines]


def relocate_mask_scratch(lines: list[str]) -> None:
    """Keep the resident s44:s55 third descriptor intact.

    The source kernel uses s53 as temporary storage while constructing the
    multicast mask for its primary and scale descriptors.  In this variant
    s44:s55 is a live third TDM descriptor, so those later source sequences
    would overwrite descriptor word s53.  gfx1250 exposes s0..s105 here and
    the source metadata stops at s103, so move only the mask-building scratch
    sequences to s100 in the body and s101 in the tail.  The tail must retain
    the current output descriptor in s99:s102, while s101 is reset at the next
    task entry.  Keep s104:s105 available for the resident pointer delta or
    next-task coordinates.  Do not rewrite genuine s53 descriptor-field writes.
    """

    starts = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^\s*(?:s_bfe_u32|s_mov_b32)\s+s53,", line)
    ]
    ranges: list[tuple[int, int]] = []
    end_pattern = re.compile(
        r"^\s*s_or_b32\s+s(?P<dst>36|84),\s+s53,\s+s(?P=dst)"
    )
    for start in starts:
        end = next(
            (
                index
                for index in range(start, min(start + 24, len(lines)))
                if end_pattern.match(lines[index])
            ),
            None,
        )
        if end is not None:
            ranges.append((start, end))

    if len(ranges) != 24:
        raise RuntimeError(
            f"expected 24 descriptor-mask scratch ranges, got {len(ranges)}"
        )
    tail_start = lines.index(".Lmoe_next_prefetch_a:")
    for start, end in ranges:
        scratch = "s101" if start >= tail_start else "s100"
        for index in range(start, end + 1):
            lines[index] = re.sub(r"\bs53\b", scratch, lines[index])

    metadata = "\t\t.amdhsa_next_free_sgpr 104"
    if lines.count(metadata) != 1:
        raise RuntimeError("next_free_sgpr metadata marker changed")
    lines[lines.index(metadata)] = "\t\t.amdhsa_next_free_sgpr 106"


def rename_tensor_labels(lines: list[str], tensor: str, tag: str) -> list[str]:
    if tensor != "B":
        return list(lines)
    replacements = {
        ".Lbranch_000000002c70": f".Lmoe_ab4_scale_half_{tag}_b_mask_loop",
        ".Lmoe_next_prefetch_column_mask_loop_b": (
            f".Lmoe_ab4_scale_half_{tag}_b_tail_mask_loop"
        ),
    }
    result = []
    for line in lines:
        for old, new in replacements.items():
            line = line.replace(old, new)
        result.append(line)
    return result


def quarter_setup(
    template: list[str],
    *,
    tensor: str,
    quarter: int,
    tag: str,
    tail: bool,
) -> list[str]:
    """Build a complete primary A/B quarter descriptor and fallback state."""

    lines = rename_tensor_labels(template, tensor, tag)
    ring = wpt2.A_RING if tensor == "A" else wpt2.B_RING
    quarter_lds = quarter * 0x2000
    wpt2.replace_literal_destination_bases(
        lines,
        ring,
        tuple(base + quarter_lds for base in ring),
        min_hits=1 if tail else 4,
    )
    hits = 0
    for index, line in enumerate(lines):
        if re.search(r"\bs_or_b32 s40, s40, 16\b", line):
            lines[index] = re.sub(r", 16\b", ", 4", line)
            hits += 1
    if hits != 1:
        raise RuntimeError(f"{tag}: quarter tile extent count={hits}")
    wpt2.add_half_pointer_offset(
        lines,
        pointer_hi=73 if tensor == "A" else 75,
        offset=quarter * 0x38000,
        bound_shift=4,
        bound_delta=quarter * 4,
    )
    return lines


def descriptor_only(lines: list[str]) -> list[str]:
    start = next(
        i for i, line in enumerate(lines) if re.match(r"^\s*s_mov_b32\s+s32,", line)
    )
    end = next(
        i
        for i in range(start, len(lines))
        if re.match(r"^\s*s_bitset1_b32\s+s36,\s+21", lines[i])
    ) + 1
    return lines[start:end]


def extra_payload_setup(
    template: list[str],
    *,
    tensor: str,
    quarter: int,
    tag: str,
    tile54_reg: int = 92,
    tile55_reg: int = 93,
) -> list[str]:
    lines = quarter_setup(
        template, tensor=tensor, quarter=quarter, tag=tag, tail=False
    )
    lines = descriptor_only(lines)
    # s53 is a scratch register in the original descriptor builder.  After
    # remapping s32:s43 to s44:s55 it would alias descriptor word s53 and
    # corrupt tensor_dim0_stride.  The saved tile coordinate has already been
    # consumed before mask construction, so s100 can then become scratch.
    lines = [
        re.sub(r"\bs55\b", f"s{tile55_reg}", re.sub(r"\bs54\b", f"s{tile54_reg}", re.sub(r"\bs53\b", "s100", line)))
        for line in lines
    ]
    return remap_sgpr_range(lines, old_first=32, new_first=44)


def transform_primary_selectors(
    lines: list[str], ready: str, end_label: str, tensor: str, quarter: int
) -> None:
    start = lines.index(ready) + 1
    end = lines.index(end_label, start)
    ring = wpt2.A_RING if tensor == "A" else wpt2.B_RING
    desired = tuple(base + quarter * 0x2000 for base in ring)
    all_bases = (*wpt2.A_RING, *wpt2.B_RING, *wpt2.SA_RING, *wpt2.SB_RING)
    pattern = re.compile(
        r"^(?P<head>\s*s_mov_b32\s+s33,\s+)"
        r"(?P<value>0x[0-9a-fA-F]+|0)(?P<tail>\s*(?:;.*)?)$"
    )
    rewrites = 0
    for index in range(start, end):
        match = pattern.match(lines[index])
        if match is None:
            continue
        value = int(match.group("value"), 0)
        if value not in all_bases:
            continue
        stage = wpt2.selector_ring(match.group("value"))
        lines[index] = (
            f"{match.group('head')}0x{desired[stage]:x}{match.group('tail')}"
        )
        rewrites += 1
    if rewrites != 4:
        raise RuntimeError(f"{ready}: expected four primary selectors, got {rewrites}")


def current_k_offsets(lines: list[str], positions: list[int], cutoff: int) -> list[int]:
    pattern = re.compile(
        r"^\s*s_add_co_u32\s+s24,\s+s58,\s+(0x[0-9a-fA-F]+|\d+)"
    )
    result = []
    for issue_id, position in enumerate(positions):
        next_position = positions[issue_id + 1] if issue_id + 1 < len(positions) else cutoff
        match = next(
            (
                pattern.match(lines[i])
                for i in range(position + 1, next_position)
                if pattern.match(lines[i]) is not None
            ),
            None,
        )
        if match is None:
            match = next(
                (
                    pattern.match(lines[i])
                    for i in range(position + 1, min(position + 256, cutoff))
                    if pattern.match(lines[i]) is not None
                ),
                None,
            )
        if match is None:
            raise RuntimeError(f"missing K update after source issue {issue_id}")
        result.append(int(match.group(1), 0) - 0x100)
    return result


def extra_payload_issue(
    ring: int,
    issue_id: int,
    k_offset: int,
    *,
    refresh_group1: bool = False,
    clone_adjacent_group0: bool = False,
    fixed_adjacent_pointer: bool = False,
    same_tensor_extra: bool = False,
) -> list[str]:
    done = f".Lmoe_ab4_scale_half_extra_done_{issue_id}"
    guard = (
        ["\ts_cmp_lt_u32 s58, s19"]
        if k_offset == 0
        else [
            f"\ts_add_co_u32 s24, s58, 0x{k_offset:x}",
            "\ts_cmp_lt_u32 s24, s19",
        ]
    )
    lds_negative = f".Lmoe_ab4_scale_half_lds_negative_{issue_id}"
    lds_joined = f".Lmoe_ab4_scale_half_lds_join_{issue_id}"
    opposite_lds_delta = 0x30000 if ring < 2 else 0x2E000
    resident_lds = (
        [
            "\ts_mov_b32 s45, s33",
            "\ts_add_co_u32 s45, s45, 0x2000",
        ]
        if same_tensor_extra
        else [
            "\ts_mov_b32 s45, s33",
            "\ts_cmp_lt_u32 s22, 2",
            f"\ts_cbranch_scc0 {lds_negative}",
            f"\ts_add_co_u32 s45, s45, 0x{opposite_lds_delta:x}",
            f"\ts_branch {lds_joined}",
            lds_negative + ":",
            f"\ts_sub_co_u32 s45, s45, 0x{opposite_lds_delta:x}",
            lds_joined + ":",
        ]
    )
    group0 = (
        [
            "\ts_mov_b32 s44, s32",
            "\ts_add_co_u32 s45, s33, 0x2000",
            "\ts_mov_b32 s46, s34",
            "\ts_mov_b32 s47, s35",
            "\ts_add_co_u32 s46, s46, 0x38000",
            "\ts_add_co_ci_u32 s47, s47, 0",
        ]
        if clone_adjacent_group0
        else [
            f"\ts_add_co_u32 s45, s99, 0x{wpt2.A_RING[ring]:x}",
            "\ts_mov_b32 s46, s34",
            "\ts_mov_b32 s47, s35",
            "\ts_add_co_u32 s46, s46, 0x38000",
            "\ts_add_co_ci_u32 s47, s47, 0",
        ]
        if fixed_adjacent_pointer
        else [
            *resident_lds,
            "\ts_add_nc_u64 s[46:47], s[34:35], s[104:105]",
        ]
    )
    lines = [
        "\t; Opposite payload quarter, completing four-wave A+B ownership.",
        *group0,
        *(
            [f"\ts_mov_b32 s{48 + index}, s{36 + index}" for index in range(8)]
            if refresh_group1
            else []
        ),
        *guard,
        f"\ts_cbranch_scc0 {done}",
        "\ttensor_load_to_lds s[44:47], s[48:55] th:TH_LOAD_NT_RT",
        done + ":",
    ]
    if ring == 2:
        lines.extend(
            (
                "\t; Keep the following fourth A+B+Scale trio at <=10 inflight TDM.",
                "\ts_wait_tensorcnt 0x7",
            )
        )
    return lines


def extra_payload_issue_from_primary() -> list[str]:
    """Issue the adjacent quarter by cloning the current primary D#.

    This diagnostic avoids carrying a third descriptor across the hot loop.
    The target shapes are exact multiples of the tile, so retaining the
    primary descriptor's tensor bounds while moving the 4-row tile forward by
    one quarter is valid.
    """

    return [
        "\t; Clone the current primary descriptor for its adjacent quarter.",
        "\ts_mov_b32 s44, s32",
        "\ts_add_co_u32 s45, s33, 0x2000",
        "\ts_mov_b32 s46, s34",
        "\ts_mov_b32 s47, s35",
        "\ts_add_co_u32 s46, s46, 0x38000",
        "\ts_add_co_ci_u32 s47, s47, 0",
        "\ts_mov_b32 s48, s36",
        "\ts_mov_b32 s49, s37",
        "\ts_mov_b32 s50, s38",
        "\ts_mov_b32 s51, s39",
        "\ts_mov_b32 s52, s40",
        "\ts_mov_b32 s53, s41",
        "\ts_mov_b32 s54, s42",
        "\ts_mov_b32 s55, s43",
        "\ttensor_load_to_lds s[44:47], s[48:55] th:TH_LOAD_NT_RT",
    ]


def scale_issue(selector: str, issue_id: int, ring: int) -> list[str]:
    lines = wpt2.secondary_scale_issue(selector, issue_id)
    lines = [
        line.replace(
            "tensor_load_to_lds s[80:83], s[84:91]",
            "tensor_load_to_lds s[80:83], s[84:91] th:TH_LOAD_NT_RT",
        )
        for line in lines
    ]
    return lines


def patch_task_state(lines: list[str]) -> None:
    state = lines.index(".Lmoe_persistent_state_ready:") + 1
    lines[state:state] = [
        "\t; Recover next-task coordinates after tail prefetch reused s44:s55.",
        "\ts_cmp_eq_u32 s101, 0",
        "\ts_cbranch_scc1 .Lmoe_ab4_scale_half_state_ready",
        "\ts_mov_b32 s54, s104",
        "\ts_mov_b32 s55, s105",
        ".Lmoe_ab4_scale_half_state_ready:",
    ]
    marker = (
        "\ts_add_co_ci_u32 s45, 0, s45                                "
        "; 000000001E38: 822D2D80"
    )
    if lines.count(marker) != 1:
        raise RuntimeError("task C-pointer completion marker changed")
    pos = lines.index(marker) + 1
    lines[pos:pos] = [
        "\t; Save state before the extra payload descriptor occupies s44:s55.",
        "\ts_set_vgpr_msb 0",
        "\tv_mov_b32_e32 v252, s44",
        "\tv_mov_b32_e32 v253, s45",
        "\tv_mov_b32_e32 v254, s54",
        "\tv_mov_b32_e32 v255, s55",
    ]


def restore_epilogue_state(lines: list[str]) -> None:
    label = ".Lbranch_00000000ab74:"
    pos = lines.index(label) + 1
    lines[pos:pos] = [
        "\t; Restore output pointer and current tile after descriptor reuse.",
        "\ts_set_vgpr_msb 0",
        "\tv_readfirstlane_b32 s44, v252",
        "\tv_readfirstlane_b32 s45, v253",
        "\tv_readfirstlane_b32 s54, v254",
        "\tv_readfirstlane_b32 s55, v255",
    ]
    marker = "\ts_mov_b32 s102, s45"
    matches = [i for i, line in enumerate(lines) if line.startswith(marker)]
    if len(matches) != 1:
        raise RuntimeError(f"output pointer save marker count={len(matches)}")
    lines[matches[0] + 1 : matches[0] + 1] = [
        "\t; Keep the current M tile while the tail constructs the next task.",
        "\ts_mov_b32 s92, s55",
    ]
    late = re.compile(
        r"^(?P<head>\s*s_mul_i32\s+s24,\s+)s55"
        r"(?P<tail>,\s+0x100\s+;\s+00000000B7A8:.*)$"
    )
    hits = 0
    for index, line in enumerate(lines):
        match = late.match(line)
        if match:
            lines[index] = f"{match.group('head')}s92{match.group('tail')}"
            hits += 1
    if hits != 1:
        raise RuntimeError(f"late current-M use count={hits}")


def tail_primary(
    template: list[str], *, tensor: str, quarter: int, tag: str
) -> list[str]:
    lines = quarter_setup(
        template, tensor=tensor, quarter=quarter, tag=tag, tail=False
    )
    lines = descriptor_only(lines)
    return [
        *lines,
        "\ttensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT_RT",
    ]


def tail_scale(
    template: list[str], *, tensor: str, half: int, tag: str
) -> list[str]:
    lines = wpt2.tail_scale(template, tensor=tensor, half=half, tag=tag)
    if tensor == "ScaleB":
        lines = [
            line.replace(
                ".Lbranch_000000004520",
                f".Lmoe_ab4_scale_half_{tag}_sb_mask_loop",
            )
            for line in lines
        ]
    return [
        line.replace(
            "tensor_load_to_lds s[32:35], s[36:43]",
            "tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT_RT",
        )
        for line in lines
    ]


def tail_extra(
    template: list[str], *, tensor: str, quarter: int, tag: str
) -> list[str]:
    lines = quarter_setup(
        template, tensor=tensor, quarter=quarter, tag=tag, tail=False
    )
    lines = descriptor_only(lines)
    lines = [
        re.sub(r"\bs55\b", "s105", re.sub(r"\bs54\b", "s104", re.sub(r"\bs53\b", "s101", line)))
        for line in lines
    ]
    lines = remap_sgpr_range(lines, old_first=32, new_first=44)
    return [
        *lines,
        "\ttensor_load_to_lds s[44:47], s[48:55] th:TH_LOAD_NT_RT",
    ]


def build(
    source: str,
    role_specs: tuple[tuple[int, str, int, str, int, str, int], ...],
    *,
    jit_copy_same_tensor: bool = False,
    refresh_extra_group1: bool = False,
    clone_adjacent_group0: bool = False,
    fixed_adjacent_pointer: bool = False,
    use_wpt23_tail: bool = False,
) -> str:
    lines = source.splitlines()
    setups = {
        tensor: wpt2.find_setup(lines, path[1], path[2])
        for tensor, path in zip(("A", "B", "ScaleA", "ScaleB"), wpt2.ROLE_PATHS)
    }
    patch_task_state(lines)
    cutoff = lines.index(".Lbranch_00000000ad68:")
    selector: str | None = None
    selector_re = re.compile(r"^\s*s_mov_b32\s+s33,\s+([^\s;]+)")
    positions = [
        i
        for i, line in enumerate(lines[:cutoff])
        if line.lstrip().startswith("tensor_load_to_lds s[32:35], s[36:43]")
    ]
    offsets = current_k_offsets(lines, positions, cutoff)
    same_tensor_extra = all(
        primary == extra
        for _wave, primary, _quarter, _scale, _half, extra, _extra_quarter
        in role_specs
    )
    output: list[str] = []
    issue_id = 0
    for index, line in enumerate(lines):
        if index >= cutoff:
            output.extend(lines[index:])
            break
        match = selector_re.match(line)
        if match:
            selector = match.group(1)
        if not line.lstrip().startswith("tensor_load_to_lds s[32:35], s[36:43]"):
            output.append(line)
            continue
        if selector is None:
            raise RuntimeError(f"missing selector before issue {issue_id}")
        ring = wpt2.selector_ring(selector)
        if selector.startswith("s"):
            even_path = issue_id in {*range(32, 40), *range(48, 52)}
            selector_arg = selector + (":even" if even_path else ":odd")
        else:
            selector_arg = selector
        scale_lines = scale_issue(selector_arg, issue_id, ring)
        extra_first = f".Lmoe_ab4_scale_half_extra_first_{issue_id}"
        joined = f".Lmoe_ab4_scale_half_issue_join_{issue_id}"
        extra_lo = (
            extra_payload_issue_from_primary()
            if jit_copy_same_tensor
            else extra_payload_issue(
                ring,
                issue_id,
                offsets[issue_id],
                refresh_group1=refresh_extra_group1,
                clone_adjacent_group0=clone_adjacent_group0,
                fixed_adjacent_pointer=fixed_adjacent_pointer,
                same_tensor_extra=same_tensor_extra,
            )
        )
        extra_hi = (
            extra_payload_issue_from_primary()
            if jit_copy_same_tensor
            else extra_payload_issue(
                ring,
                issue_id + 100,
                offsets[issue_id],
                refresh_group1=refresh_extra_group1,
                clone_adjacent_group0=clone_adjacent_group0,
                fixed_adjacent_pointer=fixed_adjacent_pointer,
                same_tensor_extra=same_tensor_extra,
            )
        )
        output.extend(
            (
                "\t; Keep every wave's multicast order A, B, then Scale.",
                "\ts_cmp_lt_u32 s22, 2",
                f"\ts_cbranch_scc0 {extra_first}",
                line,
                *extra_lo,
                f"\ts_branch {joined}",
                extra_first + ":",
                *extra_hi,
                line,
                joined + ":",
                *scale_lines,
            )
        )
        issue_id += 1
    else:
        raise RuntimeError("output setup cutoff not found")
    lines = output
    if issue_id != 56:
        raise RuntimeError(f"expected 56 body issue sites, got {issue_id}")

    path_ends = (
        wpt2.ROLE_PATHS[1][1],
        wpt2.ROLE_PATHS[2][1],
        wpt2.ROLE_PATHS[3][1],
        ".Lbranch_000000005060:",
    )
    for spec, path, end_label in zip(role_specs, wpt2.ROLE_PATHS, path_ends, strict=True):
        _wave, primary, quarter, _scale, _half, _extra, _extra_q = spec
        transform_primary_selectors(lines, path[2], end_label, primary, quarter)

    for spec, path in reversed(tuple(zip(role_specs, wpt2.ROLE_PATHS, strict=True))):
        wave, primary, quarter, scale, half, extra, extra_quarter = spec
        insert_at = lines.index(path[2]) + 1
        extra_ring = wpt2.B_RING if extra == "B" else wpt2.A_RING
        lines[insert_at:insert_at] = [
            f"\t; Wave {wave}: {primary} q{quarter}, {extra} q{extra_quarter}, {scale} h{half}.",
            *quarter_setup(
                setups[primary],
                tensor=primary,
                quarter=quarter,
                tag=f"primary_wave{wave}",
                tail=False,
            ),
            *wpt2.secondary_scale_setup(
                setups[scale], tensor=scale, half=half, tag=f"scale_wave{wave}"
            ),
            "\t; Preserve task tile coordinates while constructing s44:s55.",
            "\ts_mov_b32 s92, s54",
            "\ts_mov_b32 s93, s55",
            *extra_payload_setup(
                setups[extra],
                tensor=extra,
                quarter=extra_quarter,
                tag=f"extra_wave{wave}",
            ),
            "\t; Track the resident pointer from the primary descriptor rather",
            "\t; than accumulating a separate K pointer through the hot loop.",
            "\ts_sub_nc_u64 s[104:105], s[46:47], s[34:35]",
            f"\ts_mov_b32 s99, 0x{extra_ring[0] + extra_quarter * 0x2000:x}",
        ]

    tail_start = lines.index(".Lmoe_next_prefetch_a:")
    tail_end = lines.index(".Lmoe_next_prefetch_done:")
    tail_lines: list[str] = []
    labels = (
        ".Lmoe_next_prefetch_a:",
        ".Lmoe_next_prefetch_b:",
        ".Lmoe_next_prefetch_scale_a:",
        ".Lmoe_next_prefetch_scale_b:",
    )
    for label, spec in zip(labels, role_specs, strict=True):
        wave, primary, quarter, scale, half, extra, extra_quarter = spec
        primary_lines = tail_primary(
            setups[primary],
            tensor=primary,
            quarter=quarter,
            tag=f"tail_primary_wave{wave}",
        )
        scale_lines = tail_scale(
            setups[scale],
            tensor=scale,
            half=half,
            tag=f"tail_scale_wave{wave}",
        )
        extra_lines = (
            extra_payload_issue_from_primary()
            if jit_copy_same_tensor
            else tail_extra(
                setups[extra],
                tensor=extra,
                quarter=extra_quarter,
                tag=f"tail_extra_wave{wave}",
            )
        )
        prefix = [
            label,
            f"\t; Next task wave {wave}: {primary} q{quarter}, {extra} q{extra_quarter}, {scale} h{half}.",
            "\t; Preserve next-task coordinates without clobbering the output",
            "\t; address retained in s99:s102 for the current task epilogue.",
            "\ts_mov_b32 s104, s54",
            "\ts_mov_b32 s105, s55",
        ]
        restore_coords = [
            "\ts_mov_b32 s54, s104",
            "\ts_mov_b32 s55, s105",
        ]
        if wave < 2:
            ordered_tail = [
                *primary_lines,
                *extra_lines,
                *restore_coords,
                *scale_lines,
            ]
        else:
            ordered_tail = [
                *extra_lines,
                *restore_coords,
                *primary_lines,
                *scale_lines,
            ]
        tail_lines.extend(
            (
                *prefix,
                *ordered_tail,
                "\ts_branch .Lmoe_next_prefetch_done",
                "",
            )
        )
    if use_wpt23_tail:
        # Reuse the validated WPT23 diagnostic tail for the historical probes.
        # It loads a primary half plus an overlapping opposite quarter.  The
        # actual resident 2+3 full-tail candidate below deliberately selects
        # the native non-overlapping q0/q2/q1/q3 tail constructed above.
        if sha256(SAFE_WPT23_TAIL_SOURCE) != EXPECTED_SAFE_WPT23_TAIL_SHA256:
            raise RuntimeError("validated WPT23 tail source checksum changed")
        donor = SAFE_WPT23_TAIL_SOURCE.read_text(encoding="utf-8").splitlines()
        donor_start = donor.index(".Lmoe_next_prefetch_a:")
        donor_end = donor.index(".Lmoe_next_prefetch_done:", donor_start)
        tail_lines = [
            re.sub(r"\bs92\b", "s101", line)
            for line in donor[donor_start:donor_end]
        ]
    lines[tail_start:tail_end] = tail_lines

    restore_epilogue_state(lines)
    relocate_mask_scratch(lines)
    lines[3] = (
        "\t; Persistent MoE GEMM1: four-wave A+B quarters plus split Scale TDM."
    )
    text = "\n".join(lines) + "\n"
    expected = {
        "tensor_load_to_lds": 292,
        "cluster_load_b32": 0,
        "cluster_load_b128": 0,
        "ds_load_b32": 192,
        "v_wmma_scale_f32_32x16x128_f4": 512,
        "tensor_store_from_lds": 2,
    }
    for mnemonic, wanted in expected.items():
        actual = sum(
            line.lstrip().startswith(mnemonic) for line in text.splitlines()
        )
        if actual != wanted:
            raise RuntimeError(f"{mnemonic}: expected {wanted}, got {actual}")
    labels_found = re.findall(r"^(\.\w[^:]*):\s*$", text, re.MULTILINE)
    duplicates = [name for name in set(labels_found) if labels_found.count(name) != 1]
    if duplicates:
        raise RuntimeError(f"duplicate labels: {sorted(duplicates)}")
    return text


def force_full_setup_each_task(text: str) -> str:
    """Disable next-task TDM prefetch while retaining the resident body."""

    lines = text.splitlines()
    start = lines.index("\t; Prefetch stage 0 of the next persistent")
    end = lines.index(".Lmoe_next_prefetch_done:", start)
    del lines[start:end]
    old = "\ts_branch .Lmoe_persistent_task_prefetched"
    if lines.count(old) != 1:
        raise RuntimeError(f"persistent prefetched-loop branch count={lines.count(old)}")
    lines[lines.index(old)] = "\ts_branch .Lmoe_persistent_task"
    lines[3] = (
        "\t; Diagnostic serialized resident 2+3 TDM with full setup per task."
    )
    return "\n".join(lines) + "\n"


def force_full_setup_after_tail(text: str) -> str:
    """Issue the tail, drain it, then ignore it and rebuild the next task."""

    lines = text.splitlines()
    old = "\ts_branch .Lmoe_persistent_task_prefetched"
    if lines.count(old) != 1:
        raise RuntimeError(f"persistent prefetched-loop branch count={lines.count(old)}")
    pos = lines.index(old)
    lines[pos:pos + 1] = [
        "\t; Diagnostic: finish the unused tail before rebuilding the task.",
        "\ts_wait_idle",
        "\ts_branch .Lmoe_persistent_task",
    ]
    lines[3] = "\t; Diagnostic resident 2+3 tail followed by full task setup."
    return "\n".join(lines) + "\n"


def move_tail_after_output(text: str) -> str:
    """Run next-task input prefetch only after both output TDM stores finish."""

    lines = text.splitlines()
    start = lines.index("\t; Prefetch stage 0 of the next persistent")
    end = lines.index(".Lmoe_next_prefetch_done:", start)
    tail = [
        line.replace(
            ".Lmoe_next_prefetch_done", ".Lmoe_late_next_prefetch_done"
        )
        for line in lines[start:end]
    ]
    del lines[start:end]
    stores = [
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith("tensor_store_from_lds")
    ]
    if len(stores) != 2:
        raise RuntimeError(f"expected two output TDM stores, got {len(stores)}")
    insert_at = stores[-1] + 1
    lines[insert_at:insert_at] = [
        "\t; Complete every output TDM in the cluster before reusing input LDS.",
        "\ts_wait_tensorcnt 0x0",
        "\ts_barrier_signal -3",
        "\ts_barrier_wait 0xfffd",
        *tail,
        ".Lmoe_late_next_prefetch_done:",
    ]
    lines[3] = "\t; Resident 2+3 TDM with next-task input after output completion."
    return "\n".join(lines) + "\n"


def keep_tail_state_only(text: str) -> str:
    """Advance the next task's scalar state but issue no tail input TDM."""

    lines = text.splitlines()
    tail_start = lines.index(".Lmoe_next_prefetch_a:")
    dispatch = max(
        index
        for index in range(0, tail_start)
        if lines[index] == "\ts_cmp_eq_u32 s22, 0"
    )
    tail_end = lines.index(".Lmoe_next_prefetch_done:", tail_start)
    # Keep the common next-task index/base calculations before dispatch, then
    # skip every role-specific descriptor build and TDM issue.
    del lines[dispatch:tail_end]
    old = "\ts_branch .Lmoe_persistent_task_prefetched"
    if lines.count(old) != 1:
        raise RuntimeError(f"persistent prefetched-loop branch count={lines.count(old)}")
    pos = lines.index(old)
    lines[pos:pos + 1] = [
        "\t; State was advanced, but stage 0 was not prefetched.",
        "\ts_mov_b32 s101, 0",
        "\ts_branch .Lmoe_persistent_state_ready",
    ]
    lines[3] = "\t; Resident 2+3 TDM with scalar-only next-task preparation."
    return "\n".join(lines) + "\n"


def retime_three_tdm_pipeline(text: str) -> str:
    """Scale WPT1 tensor waits for three TDM requests per K stage."""

    lines = text.splitlines()
    output: list[str] = []
    skip_next_wait7 = False
    removed_wait7 = 0
    for line in lines:
        if line == "\t; Keep the following fourth A+B+Scale trio at <=10 inflight TDM.":
            skip_next_wait7 = True
            continue
        if skip_next_wait7:
            if line.strip() != "s_wait_tensorcnt 0x7":
                raise RuntimeError("TENSORcnt cap marker is not followed by wait 0x7")
            removed_wait7 += 1
            skip_next_wait7 = False
            continue
        line = re.sub(r"(s_wait_tensorcnt\s+)0x2\b", r"\g<1>0x6", line)
        line = re.sub(r"(s_wait_tensorcnt\s+)0x1\b", r"\g<1>0x3", line)
        output.append(line)
    if skip_next_wait7:
        raise RuntimeError("unterminated TENSORcnt cap marker")
    if removed_wait7 != 28:
        raise RuntimeError(f"expected 28 static wait-0x7 removals, got {removed_wait7}")
    output[3] = "\t; Three-TDM resident pipeline with WPT23-style tensor waits."
    return "\n".join(output) + "\n"


def split_tail_payload_after_output(text: str) -> str:
    """Issue Scale early, but defer the overlapping A/B stage-0 payload.

    The current output allocation can occupy the stage-0 A/B rings.  Preserve
    the native q0/q2/q1/q3 descriptors in dead s56:s79, issue the disjoint
    Scale tail before SiLU, then retire both output stores workgroup-wide and
    issue the saved A and B descriptors immediately before the persistent
    boundary.
    """

    lines = text.splitlines()
    tail_start = lines.index(".Lmoe_next_prefetch_a:")
    tail_end = lines.index(".Lmoe_next_prefetch_done:", tail_start)
    loads = [
        index
        for index in range(tail_start, tail_end)
        if lines[index].lstrip().startswith("tensor_load_to_lds")
    ]
    if len(loads) != 12:
        raise RuntimeError(f"native full tail expected 12 loads, got {len(loads)}")

    replacements: dict[int, list[str]] = {}
    for role in range(4):
        first_payload, second_payload, scale = loads[role * 3 : role * 3 + 3]
        save_primary = [
            "\t; Preserve the role's primary payload descriptor for the late tail.",
            *[
                f"\ts_mov_b32 s{56 + offset}, s{32 + offset}"
                for offset in range(12)
            ],
        ]
        save_extra_tail = [
            "\t; s54:s55 carry the next-task coordinates between payload and",
            "\t; Scale setup. Preserve the extra descriptor tail in dead VGPRs.",
            "\ts_set_vgpr_msb 0",
            "\tv_mov_b32_e32 v250, s54",
            "\tv_mov_b32_e32 v251, s55",
        ]
        if role < 2:
            # Waves 0/1 build primary A first, then opposite B.
            replacements[first_payload] = save_primary
            replacements[second_payload] = save_extra_tail
        else:
            # Waves 2/3 build opposite A first, restore coordinates, then
            # build primary B. Save the extra tail before that restoration.
            replacements[first_payload] = save_extra_tail
            replacements[second_payload] = save_primary
        replacements[scale] = [
            lines[scale],
            "\t; Scale is disjoint from output LDS; preserve the opposite payload",
            "\t; descriptor after restoring the",
            "\t; two words temporarily reused as next-task coordinates.",
            "\ts_wait_alu depctr_va_vdst(0)",
            "\ts_set_vgpr_msb 0",
            "\tv_readfirstlane_b32 s54, v250",
            "\tv_readfirstlane_b32 s55, v251",
            *[
                f"\ts_mov_b32 s{68 + offset}, s{44 + offset}"
                for offset in range(12)
            ],
        ]

    output: list[str] = []
    for index, line in enumerate(lines):
        output.extend(replacements.get(index, [line]))
    lines = output

    stores = [
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith("tensor_store_from_lds")
    ]
    if len(stores) != 2:
        raise RuntimeError(f"expected two output stores, got {len(stores)}")
    insert_at = stores[-1] + 1
    lines[insert_at:insert_at] = [
        "\t; The stage-0 A/B rings overlap the current output allocation for",
        "\t; this K-ring phase. Retire every local output TDM before reuse.",
        "\ts_wait_tensorcnt 0x0",
        "\ts_barrier_signal -1",
        "\ts_barrier_wait 0xffff",
        "\ts_cmp_lt_u32 s29, 0x240",
        "\ts_cbranch_scc0 .Lmoe_late_payload_tail_done",
        "\t; Preserve A -> B ordering in every wave. Waves 0/1 have primary A",
        "\t; and opposite B; waves 2/3 have opposite A and primary B.",
        "\ts_cmp_lt_u32 s22, 2",
        "\ts_cbranch_scc0 .Lmoe_late_payload_tail_extra_first",
        "\ttensor_load_to_lds s[56:59], s[60:67] th:TH_LOAD_NT_RT",
        "\ttensor_load_to_lds s[68:71], s[72:79] th:TH_LOAD_NT_RT",
        "\ts_branch .Lmoe_late_payload_tail_done",
        ".Lmoe_late_payload_tail_extra_first:",
        "\ttensor_load_to_lds s[68:71], s[72:79] th:TH_LOAD_NT_RT",
        "\ttensor_load_to_lds s[56:59], s[60:67] th:TH_LOAD_NT_RT",
        ".Lmoe_late_payload_tail_done:",
    ]
    return "\n".join(lines) + "\n"


def build_full_tail_wg_sync(text: str) -> str:
    """Build a correct complete 2+3 tail around the shared output LDS."""

    lines = retime_three_tdm_pipeline(
        split_tail_payload_after_output(text)
    ).splitlines()

    # The early Scale tail and both current output stores are fully retired
    # before the late A/B pair is issued.  A prefetched task therefore waits
    # for exactly those two payload TDMs before consuming stage 0.
    role_names = ("a", "b", "scale_a", "scale_b")
    for role in role_names:
        prefetched_wait = lines.index(
            f".Lmoe_{role}_prefetched_input_wait:"
        ) + 1
        if lines[prefetched_wait].strip() != "s_wait_tensorcnt 0x6":
            raise RuntimeError(
                f"{role}: prefetched stage-0 wait marker changed"
            )
        lines[prefetched_wait] = "\ts_wait_tensorcnt 0x0"

    # Full-setup tasks retain the original cluster barrier.  Prefetched tasks
    # have already crossed the persistent task-boundary cluster barrier after
    # O1 was issued; after the per-wave wait and workgroup barrier, their local
    # stage-0 LDS is ready without a second cluster synchronization.
    for role in reversed(role_names):
        input_ready = lines.index(f".Lmoe_{role}_input_wait_done:")
        signal = lines[input_ready + 1]
        wait_wg = lines[input_ready + 2]
        if signal.strip().split(";", 1)[0].strip() != "s_barrier_signal -1":
            raise RuntimeError(f"{role}: workgroup signal marker changed")
        if wait_wg.strip().split(";", 1)[0].strip() != "s_barrier_wait 0xffff":
            raise RuntimeError(f"{role}: workgroup wait marker changed")

        no_cluster = f".Lmoe_{role}_input_ready_no_cluster"
        if role == "a":
            signal_cluster = lines[input_ready + 3]
            wait_cluster = lines[input_ready + 4]
            if signal_cluster.strip().split(";", 1)[0].strip() != "s_barrier_signal -3":
                raise RuntimeError("a: cluster signal marker changed")
            if wait_cluster.strip().split(";", 1)[0].strip() != "s_barrier_wait 0xfffd":
                raise RuntimeError("a: cluster wait marker changed")
            replace_end = input_ready + 5
            cluster_ops = [signal_cluster, wait_cluster]
        else:
            wait_cluster = lines[input_ready + 3]
            if wait_cluster.strip().split(";", 1)[0].strip() != "s_barrier_wait 0xfffd":
                raise RuntimeError(f"{role}: cluster wait marker changed")
            replace_end = input_ready + 4
            cluster_ops = [wait_cluster]

        lines[input_ready + 1 : replace_end] = [
            signal,
            wait_wg,
            "\t; Full-setup tasks retain the initial cluster rendezvous; the",
            "\t; prefetched path already crossed the persistent task boundary.",
            "\ts_cmp_eq_u32 s101, 0",
            f"\ts_cbranch_scc0 {no_cluster}",
            *cluster_ops,
            no_cluster + ":",
        ]

    lines[3] = (
        "\t; Resident 2+3 TDM with early Scale and late A/B next-task tail."
    )
    result = "\n".join(lines) + "\n"

    for role in role_names:
        marker = (
            "s_wait_tensorcnt 0x0\n"
            f".Lmoe_{role}_input_wait_done:"
        )
        if result.count(marker) != 1:
            raise RuntimeError(
                f"{role}: prefetched stage-0 wait was not preserved"
            )
    if result.count("tensor_load_to_lds s[56:59], s[60:67]") != 2:
        raise RuntimeError("late primary payload issue sequence changed")
    if result.count("tensor_load_to_lds s[68:71], s[72:79]") != 2:
        raise RuntimeError("late opposite payload issue sequence changed")
    if result.count("Retire every local output TDM before reuse") != 1:
        raise RuntimeError("late payload output-LDS retirement marker missing")
    return result


def main() -> None:
    actual = sha256(SOURCE)
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"winner checksum changed: expected {EXPECTED_SOURCE_SHA256}, got {actual}"
        )
    source = SOURCE.read_text(encoding="utf-8")
    # The retained full-setup path deletes the next-task tail, so construct the
    # native tail directly and avoid depending on discarded WPT23 probe files.
    resident = build(source, ROLE_SPECS, use_wpt23_tail=False)
    full_setup_wait6 = retime_three_tdm_pipeline(
        force_full_setup_each_task(resident)
    )
    OUTPUT_FULL_SETUP_LOOP_WAIT6.write_text(
        full_setup_wait6,
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {OUTPUT_FULL_SETUP_LOOP_WAIT6}")
    print(f"sha256={sha256(OUTPUT_FULL_SETUP_LOOP_WAIT6)}")


if __name__ == "__main__":
    main()
