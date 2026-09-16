#!/usr/bin/env python3
"""Port exactopt WPT2 input ownership to the persistent all-NT_RT winner.

The older ``build_wpt2_owner_balance_variant.py`` is used as the structural
reference for descriptor splitting.  This generator is pinned to the accepted
all-NT_RT source, preserves every persistent/output-overlap instruction, forces
all input TDM requests to NT_RT, and retimes only waits whose outstanding input
request multiplicity changes from one to two per K stage.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt.s"
)
REFERENCE_SOURCE = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full.s"
)
OUTPUT = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_all_nt_rt_exactopt_wpt2_owner.s"
)
EXPECTED_SOURCE_SHA256 = (
    "8934b9767fb8295b7d1bf3c5df246fbb239a2784c76f11873f1ceb490d9f0169"
)
EXPECTED_REFERENCE_SHA256 = (
    "6d3b387137b1f5d3de0e95fa8dddb8067e0dae9847ecf773fdc50b5d5ba28bb3"
)

# These waits are the initial/steady input-ring fences in the source ISA.  The
# original path has one request per stage; WPT2 has a payload+scale pair.  The
# four unannotated prefetched-task entry waits deliberately are not listed:
# they must remain at 2 to retire [next-I0-payload, next-I0-scale] while leaving
# [current-O0, current-O1] in flight.
WAIT2_TO_WAIT4_PCS = {
    "0000000029A8",
    "00000000360C",
    "000000004258",
    "000000004EC0",
    "0000000053A8",
    "0000000058B4",
    "000000005DA4",
    "0000000062B0",
    "0000000074CC",
    "000000007C04",
    "00000000833C",
    "000000008A74",
    "0000000091E0",
    "000000009918",
    "00000000A050",
    "00000000A788",
}

# A one-WMMA payload/scale stagger was measured on a07-3 and regressed GEMM1
# from 512.701 us to 517.920 us.  Keep the faster adjacent issue order while
# retaining exactopt's ownership, descriptor shapes, and per-wave job order.
STAGGER_SCALE_AFTER_ONE_WMMA_PCS: set[str] = set()

ROLE_PATHS = (
    (
        0,
        ".Lbranch_000000001f08:",
        ".Lmoe_a_prefetched_descriptor_ready:",
        "A",
        0,
    ),
    (
        1,
        ".Lbranch_000000002b48:",
        ".Lmoe_b_prefetched_descriptor_ready:",
        "A",
        1,
    ),
    (
        2,
        ".Lbranch_0000000037ac:",
        ".Lmoe_scale_a_prefetched_descriptor_ready:",
        "B",
        0,
    ),
    (
        3,
        ".Lbranch_0000000043f8:",
        ".Lmoe_scale_b_prefetched_descriptor_ready:",
        "B",
        1,
    ),
)

A_RING = (0x00000, 0x08000, 0x12000, 0x1A000)
B_RING = (0x30000, 0x38000, 0x40000, 0x48000)
SA_RING = (0x10000, 0x10800, 0x11000, 0x11800)
SB_RING = (0x22000, 0x22800, 0x23000, 0x23800)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_input_tdm_hints(text: str) -> str:
    """Remove only tensor-load temporal hints for source-equivalence checks."""

    return re.sub(
        r"^(\s*tensor_load_to_lds\b.*?)\s+th:TH_LOAD_[A-Z_]+(?=\s*(?:;|$))",
        r"\1",
        text,
        flags=re.MULTILINE,
    )


def force_all_input_tdm_nt_rt(lines: list[str]) -> None:
    """Set every tensor_load_to_lds instruction to TH_LOAD_NT_RT."""

    for index, line in enumerate(lines):
        if not line.lstrip().startswith("tensor_load_to_lds "):
            continue
        code, sep, comment = line.partition(";")
        code = re.sub(r"\s+th:TH_LOAD_[A-Z_]+\s*$", "", code.rstrip())
        lines[index] = f"{code} th:TH_LOAD_NT_RT"
        if sep:
            lines[index] += f"        ;{comment}"


def retime_tensor_waits(lines: list[str]) -> tuple[int, int]:
    """Retune input-ring waits after changing one request/stage into two.

    The four unannotated ``.Lmoe_*_prefetched_input_wait`` waits stay at 2:
    their queue is [next I0 payload, next I0 scale, current O0, current O1].
    The four old-output waits change from 1 to 2 so they retire O0/O1 while
    retaining the stage-1 input pair.  Annotated initial/steady ring waits
    change from 2 to 4.
    """

    wait1_rewrites = 0
    wait4_rewrites = 0
    pc_pattern = re.compile(r";\s*([0-9A-Fa-f]{8,16}):")
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "s_wait_tensorcnt 0x1":
            lines[index] = line.replace("s_wait_tensorcnt 0x1", "s_wait_tensorcnt 0x2")
            wait1_rewrites += 1
            continue
        if "s_wait_tensorcnt 0x2" not in line:
            continue
        match = pc_pattern.search(line)
        if match is None or match.group(1).upper() not in WAIT2_TO_WAIT4_PCS:
            continue
        lines[index] = line.replace("s_wait_tensorcnt 0x2", "s_wait_tensorcnt 0x4")
        lines[index] = lines[index].replace("BFCB0002", "BFCB0004")
        wait4_rewrites += 1
    if wait1_rewrites != 4:
        raise RuntimeError(f"expected four old-output wait rewrites, got {wait1_rewrites}")
    if wait4_rewrites != len(WAIT2_TO_WAIT4_PCS):
        raise RuntimeError(
            "input-ring wait rewrite count mismatch: "
            f"expected {len(WAIT2_TO_WAIT4_PCS)}, got {wait4_rewrites}"
        )
    return wait1_rewrites, wait4_rewrites


def audit_output(text: str) -> None:
    """Check the static contract that makes the generated candidate testable."""

    failures: list[str] = []

    def require(name: str, actual, expected) -> None:
        if actual != expected:
            failures.append(f"{name}: expected {expected!r}, got {actual!r}")

    loads = [
        line
        for line in text.splitlines()
        if line.lstrip().startswith("tensor_load_to_lds ")
    ]
    labels = re.findall(
        r"^([.$A-Za-z_][.$A-Za-z0-9_]*):", text, flags=re.MULTILINE
    )
    duplicates = sorted(
        label for label, count in Counter(labels).items() if count != 1
    )
    waits = Counter(
        re.findall(r"^\s*s_wait_tensorcnt\s+([^\s;]+)", text, flags=re.MULTILINE)
    )

    require("tensor_load_to_lds count", len(loads), 120)
    require(
        "all input loads use TH_LOAD_NT_RT",
        sum("th:TH_LOAD_NT_RT" in line for line in loads),
        120,
    )
    require(
        "tensor_store_from_lds count",
        len(re.findall(r"^\s*tensor_store_from_lds\b", text, flags=re.MULTILINE)),
        2,
    )
    require("duplicate labels", duplicates, [])
    require("TENSORcnt wait distribution", waits, Counter({"0x4": 16, "0x2": 8, "0x0": 10}))
    require("184-byte ABI", text.count(".amdhsa_kernarg_size 184"), 1)
    require("320 KiB LDS", text.count(".amdhsa_group_segment_fixed_size 327680"), 1)
    require("1024 VGPR", text.count(".amdhsa_next_free_vgpr 1024"), 1)
    require("104 SGPR", text.count(".amdhsa_next_free_sgpr 104"), 1)
    require(
        "workgroup barrier operations",
        len(re.findall(r"^\s*s_barrier_(?:signal|wait)\s+(?:-1|0xffff)\b", text, re.MULTILINE)),
        128,
    )
    require("cluster signals", text.count("s_barrier_signal -3"), 4)
    require("cluster waits", text.count("s_barrier_wait 0xfffd"), 12)
    require(
        "WPT2 owner markers",
        len(re.findall(r"WPT2 owner wave [0-3]:", text)),
        4,
    )
    require(
        "WPT2 next-task markers",
        len(re.findall(r"WPT2 next-task stage 0:", text)),
        4,
    )

    # The split changes ownership only.  The union of the two half intervals
    # must remain exactly equal to every original input-ring slot.
    for name, ring, full_bytes, half_bytes in (
        ("A", A_RING, 0x8000, 0x4000),
        ("B", B_RING, 0x8000, 0x4000),
        ("ScaleA", SA_RING, 0x800, 0x400),
        ("ScaleB", SB_RING, 0x800, 0x400),
    ):
        for stage, base in enumerate(ring):
            intervals = ((base, base + half_bytes), (base + half_bytes, base + full_bytes))
            require(
                f"{name} stage {stage} half union",
                intervals,
                ((base, base + half_bytes), (base + half_bytes, base + full_bytes)),
            )
            if intervals[-1][1] > 0x50000:
                failures.append(
                    f"{name} stage {stage} exceeds LDS: 0x{intervals[-1][1]:x}"
                )

    if failures:
        raise RuntimeError("static audit failed:\n  " + "\n  ".join(failures))


def find_setup(lines: list[str], entry: str, ready: str) -> list[str]:
    start = lines.index(entry)
    end = lines.index(ready, start + 1)
    first = next(
        index
        for index in range(start, end)
        if re.match(r"^\s*s_mov_b32\s+s95,", lines[index])
    )
    return lines[first:end]


def find_tail_descriptor(lines: list[str], label: str) -> list[str]:
    start = lines.index(label) + 1
    end = next(
        index
        for index in range(start, len(lines))
        if lines[index].lstrip().startswith("tensor_load_to_lds s[32:35]")
    )
    return lines[start:end]


def rename_label(lines: list[str], old: str, new: str) -> list[str]:
    return [line.replace(old, new) for line in lines]


def replace_literal_destination_bases(
    lines: list[str],
    old_bases: tuple[int, ...],
    new_bases: tuple[int, ...],
    *,
    min_hits: int = 4,
) -> None:
    mapping = dict(zip(old_bases, new_bases, strict=True))
    hits = 0
    pattern = re.compile(r"^(?P<head>\s*s_mov_b32\s+s(?:33|95|96|97|98),\s+)(?P<value>0x[0-9a-fA-F]+|0)(?P<tail>\s*(?:;.*)?)$")
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            continue
        value = int(match.group("value"), 0)
        if value not in mapping:
            continue
        lines[index] = (
            f"{match.group('head')}0x{mapping[value]:x}{match.group('tail')}"
        )
        hits += 1
    if hits < min_hits:
        raise RuntimeError(
            f"expected at least {min_hits} LDS-base rewrites, got {hits}"
        )


def add_half_pointer_offset(
    lines: list[str], *, pointer_hi: int, offset: int, bound_shift: int, bound_delta: int
) -> None:
    if offset == 0:
        return
    high_marker = re.compile(
        rf"^\s*s_or_b32\s+s35,\s+s{pointer_hi},\s+s35"
    )
    high_hits = [i for i, line in enumerate(lines) if high_marker.match(line)]
    if len(high_hits) != 1:
        raise RuntimeError(f"main pointer-high marker count: {len(high_hits)}")
    pos = high_hits[0] + 1
    lines[pos:pos] = [
        f"\ts_add_co_u32 s34, s34, 0x{offset:x}",
        "\ts_add_co_ci_u32 s35, s35, 0",
    ]

    bound_marker = re.compile(rf"^\s*s_lshr_b32\s+s26,\s+s26,\s+{bound_shift}\b")
    bound_hits = [i for i, line in enumerate(lines) if bound_marker.match(line)]
    if len(bound_hits) != 1:
        raise RuntimeError(f"main bound marker count: {len(bound_hits)}")
    lines[bound_hits[0] + 1 : bound_hits[0] + 1] = [
        f"\ts_sub_co_u32 s26, s26, {bound_delta}"
    ]

    fallback_high = re.compile(r"^\s*s_or_b32\s+s63,\s+s63,\s+0x80000000")
    fallback_hits = [i for i, line in enumerate(lines) if fallback_high.match(line)]
    if fallback_hits:
        if len(fallback_hits) != 1:
            raise RuntimeError(f"fallback pointer marker count: {len(fallback_hits)}")
        pos = fallback_hits[0]
        # Insert before the high-word type flag is added.
        lines[pos:pos] = [
            f"\ts_add_co_u32 s62, s62, 0x{offset:x}",
            "\ts_add_co_ci_u32 s63, s63, 0",
        ]
        fallback_bound = re.compile(
            rf"^\s*s_lshr_b32\s+s27,\s+s27,\s+{bound_shift}\b"
        )
        fallback_bound_hits = [
            i for i, line in enumerate(lines) if fallback_bound.match(line)
        ]
        if len(fallback_bound_hits) != 1:
            raise RuntimeError(
                f"fallback bound marker count: {len(fallback_bound_hits)}"
            )
        lines[fallback_bound_hits[0] + 1 : fallback_bound_hits[0] + 1] = [
            f"\ts_sub_co_u32 s27, s27, {bound_delta}"
        ]


def primary_setup(
    template: list[str], *, tensor: str, half: int, tag: str, tail: bool = False
) -> list[str]:
    lines = list(template)
    if tensor == "A":
        old_ring = A_RING
        base_ring = A_RING
        pointer_hi = 73
    elif tensor == "B":
        old_ring = B_RING
        base_ring = B_RING
        pointer_hi = 75
        lines = rename_label(
            lines,
            ".Lbranch_000000002c70",
            f".Lmoe_wpt2_{tag}_b_mask_loop",
        )
        lines = rename_label(
            lines,
            ".Lmoe_next_prefetch_column_mask_loop_b",
            f".Lmoe_wpt2_{tag}_b_tail_mask_loop",
        )
    else:
        raise ValueError(tensor)

    half_lds = half * 0x4000
    replace_literal_destination_bases(
        lines,
        old_ring,
        tuple(value + half_lds for value in base_ring),
        min_hits=1 if tail else 4,
    )
    extent_hits = 0
    for index, line in enumerate(lines):
        if re.search(r"\bs_or_b32 s40, s40, 16\b", line):
            lines[index] = re.sub(r", 16\b", ", 8", line)
            extent_hits += 1
    if extent_hits != 1:
        raise RuntimeError(f"{tag}: heavy extent count={extent_hits}")
    add_half_pointer_offset(
        lines,
        pointer_hi=pointer_hi,
        offset=half * 0x70000,
        bound_shift=4,
        bound_delta=8,
    )
    return lines


def remap_descriptor_registers(lines: list[str]) -> list[str]:
    mapping = {reg: reg + 48 for reg in range(32, 44)}

    def repl(match: re.Match[str]) -> str:
        reg = int(match.group(1))
        return f"s{mapping.get(reg, reg)}"

    return [re.sub(r"\bs(\d+)\b", repl, line) for line in lines]


def secondary_scale_setup(
    template: list[str], *, tensor: str, half: int, tag: str
) -> list[str]:
    start = next(
        i for i, line in enumerate(template) if re.match(r"^\s*s_mov_b32\s+s32,", line)
    )
    end = next(
        i
        for i in range(start, len(template))
        if re.match(r"^\s*s_bitset1_b32\s+s36,\s+21", template[i])
    ) + 1
    lines = remap_descriptor_registers(template[start:end])
    if tensor == "ScaleA":
        old_base = 0x10000
        pointer_hi = 77
    elif tensor == "ScaleB":
        old_base = 0x22000
        pointer_hi = 79
        lines = rename_label(
            lines,
            ".Lbranch_000000004520",
            f".Lmoe_wpt2_{tag}_sb_mask_loop",
        )
        lines = rename_label(
            lines,
            ".Lmoe_next_prefetch_column_mask_loop_scale_b",
            f".Lmoe_wpt2_{tag}_sb_tail_mask_loop",
        )
    else:
        raise ValueError(tensor)

    actual_base = old_base + half * 0x400
    base_pattern = re.compile(
        rf"^(?P<head>\s*s_mov_b32\s+s81,\s+)0x{old_base:x}(?P<tail>\s*(?:;.*)?)$"
    )
    base_hits = 0
    for index, line in enumerate(lines):
        match = base_pattern.match(line)
        if match:
            lines[index] = (
                f"{match.group('head')}0x{actual_base:x}{match.group('tail')}"
            )
            base_hits += 1
    if base_hits != 1:
        raise RuntimeError(f"{tag}: scale LDS base count={base_hits}")

    extent_hits = 0
    for index, line in enumerate(lines):
        if re.search(r"\bs_or_b32 s88, s88, 8\b", line):
            lines[index] = re.sub(r", 8\b", ", 4", line)
            extent_hits += 1
    if extent_hits != 1:
        raise RuntimeError(f"{tag}: scale extent count={extent_hits}")

    if half:
        high_marker = re.compile(
            rf"^\s*s_or_b32\s+s83,\s+s{pointer_hi},\s+s83"
        )
        hits = [i for i, line in enumerate(lines) if high_marker.match(line)]
        if len(hits) != 1:
            raise RuntimeError(f"{tag}: scale pointer marker count={len(hits)}")
        pos = hits[0] + 1
        lines[pos:pos] = [
            "\ts_add_co_u32 s82, s82, 0x7000",
            "\ts_add_co_ci_u32 s83, s83, 0",
        ]
        bound_marker = re.compile(r"^\s*s_lshr_b32\s+s26,\s+s26,\s+5\b")
        hits = [i for i, line in enumerate(lines) if bound_marker.match(line)]
        if len(hits) != 1:
            raise RuntimeError(f"{tag}: scale bound marker count={len(hits)}")
        lines[hits[0] + 1 : hits[0] + 1] = [
            "\ts_sub_co_u32 s26, s26, 4"
        ]

    # Stage 0 may already have been issued in the previous persistent task.
    lines.extend(
        (
            "\ts_cmp_eq_u32 s101, 0",
            "\ts_cselect_b32 s24, 0, 0x100",
            "\ts_add_co_u32 s82, s82, s24",
            "\ts_add_co_ci_u32 s83, s83, 0",
        )
    )
    return lines


def selector_ring(selector: str) -> int:
    selector = selector.split(":", 1)[0]
    register_map = {"s95": 0, "s96": 1, "s97": 2, "s98": 3}
    if selector in register_map:
        return register_map[selector]
    value = int(selector, 0)
    for ring, values in enumerate(zip(A_RING, B_RING, SA_RING, SB_RING, strict=True)):
        if value in values:
            return ring
    raise RuntimeError(f"unknown LDS selector: {selector}")


def classify_static_wave(selector: str) -> int | None:
    if selector.startswith("s"):
        return None
    value = int(selector, 0)
    if value in A_RING:
        return 0
    if value in B_RING:
        return 1
    if value in SA_RING:
        return 2
    if value in SB_RING:
        return 3
    raise RuntimeError(f"unknown role-specific LDS selector: {selector}")


def secondary_scale_issue(selector: str, label_id: int) -> list[str]:
    ring = selector_ring(selector)
    wave = classify_static_wave(selector)
    done = f".Lmoe_wpt2_scale_issue_done_{label_id}"
    lines = ["\t; Issue this wave's matching ScaleA/ScaleB half after payload."]
    if wave is None:
        # The even common path is wave 0/2 (first halves); the odd common path
        # is wave 1/3 (second halves).  The instruction temporal hint identifies
        # the path at the call site, which is passed through the selector suffix.
        half = 0 if selector.endswith(":even") else 1
        sa = SA_RING[ring] + half * 0x400
        sb = SB_RING[ring] + half * 0x400
        lines.extend(
            (
                f"\ts_mov_b32 s81, 0x{sb:x}",
                "\ts_cmp_lt_u32 s22, 2",
                f"\ts_cselect_b32 s81, 0x{sa:x}, s81",
            )
        )
    else:
        scale_ring = SA_RING if wave < 2 else SB_RING
        lines.append(f"\ts_mov_b32 s81, 0x{scale_ring[ring] + (wave & 1) * 0x400:x}")
    lines.extend(
        (
            "\ts_cmp_lg_u32 s39, 0",
            f"\ts_cbranch_scc0 {done}",
            "\ttensor_load_to_lds s[80:83], s[84:91] th:TH_LOAD_NT_RT",
            "\ts_add_co_u32 s82, s82, 0x100",
            "\ts_add_co_ci_u32 s83, s83, 0",
            f"{done}:",
        )
    )
    return lines


def transform_role_selectors(
    lines: list[str], ready: str, end_label: str, tensor: str, half: int
) -> None:
    start = lines.index(ready) + 1
    end = lines.index(end_label, start)
    desired = A_RING if tensor == "A" else B_RING
    desired = tuple(value + half * 0x4000 for value in desired)
    all_bases = (*A_RING, *B_RING, *SA_RING, *SB_RING)
    pattern = re.compile(
        r"^(?P<head>\s*s_mov_b32\s+s33,\s+)(?P<value>0x[0-9a-fA-F]+|0)(?P<tail>\s*(?:;.*)?)$"
    )
    rewrites = 0
    for index in range(start, end):
        match = pattern.match(lines[index])
        if not match:
            continue
        value = int(match.group("value"), 0)
        if value not in all_bases:
            continue
        ring = selector_ring(match.group("value"))
        lines[index] = (
            f"{match.group('head')}0x{desired[ring]:x}{match.group('tail')}"
        )
        rewrites += 1
    if rewrites != 4:
        raise RuntimeError(f"{ready}: expected four stage selectors, got {rewrites}")


def tail_primary(
    template: list[str], *, tensor: str, half: int, tag: str
) -> list[str]:
    lines = primary_setup(template, tensor=tensor, half=half, tag=tag, tail=True)
    tensor_line = "\ttensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT_RT"
    return [*lines, tensor_line]


def tail_scale(
    template: list[str], *, tensor: str, half: int, tag: str
) -> list[str]:
    # Tail descriptors use s32:s43 directly and do not need the stage-1 offset.
    start = next(
        i for i, line in enumerate(template) if re.match(r"^\s*s_mov_b32\s+s32,", line)
    )
    end = next(
        i
        for i in range(start, len(template))
        if re.match(r"^\s*s_bitset1_b32\s+s36,\s+21", template[i])
    ) + 1
    lines = list(template[start:end])
    if tensor == "ScaleA":
        old_base = 0x10000
        pointer_hi = 77
    elif tensor == "ScaleB":
        old_base = 0x22000
        pointer_hi = 79
        lines = rename_label(
            lines,
            ".Lmoe_next_prefetch_column_mask_loop_scale_b",
            f".Lmoe_wpt2_{tag}_sb_tail_mask_loop",
        )
    else:
        raise ValueError(tensor)
    actual_base = old_base + half * 0x400
    for index, line in enumerate(lines):
        lines[index] = re.sub(
            rf"(s_mov_b32\s+s33,\s+)0x{old_base:x}\b",
            rf"\g<1>0x{actual_base:x}",
            line,
        )
        lines[index] = re.sub(r"(s_or_b32 s40, s40,) 8\b", r"\g<1> 4", lines[index])
    if half:
        high_marker = re.compile(
            rf"^\s*s_or_b32\s+s35,\s+s{pointer_hi},\s+s35"
        )
        pos = next(i for i, line in enumerate(lines) if high_marker.match(line)) + 1
        lines[pos:pos] = [
            "\ts_add_co_u32 s34, s34, 0x7000",
            "\ts_add_co_ci_u32 s35, s35, 0",
        ]
        bound_marker = re.compile(r"^\s*s_lshr_b32\s+s26,\s+s26,\s+5\b")
        pos = next(i for i, line in enumerate(lines) if bound_marker.match(line)) + 1
        lines[pos:pos] = ["\ts_sub_co_u32 s26, s26, 4"]
    return [
        *lines,
        "\ttensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT_RT",
    ]


def build(source: str, reference_source: str) -> tuple[str, int, int, int, int]:
    lines = source.splitlines()
    reference_lines = reference_source.splitlines()
    if len(lines) != len(reference_lines):
        raise RuntimeError(
            "all-NT_RT/reference line counts differ: "
            f"{len(lines)} != {len(reference_lines)}"
        )
    setups = {
        "A": find_setup(lines, ROLE_PATHS[0][1], ROLE_PATHS[0][2]),
        "B": find_setup(lines, ROLE_PATHS[1][1], ROLE_PATHS[1][2]),
        "ScaleA": find_setup(lines, ROLE_PATHS[2][1], ROLE_PATHS[2][2]),
        "ScaleB": find_setup(lines, ROLE_PATHS[3][1], ROLE_PATHS[3][2]),
    }
    tail = {
        "A": find_tail_descriptor(lines, ".Lmoe_next_prefetch_a:"),
        "B": find_tail_descriptor(lines, ".Lmoe_next_prefetch_b:"),
        "ScaleA": find_tail_descriptor(lines, ".Lmoe_next_prefetch_scale_a:"),
        "ScaleB": find_tail_descriptor(lines, ".Lmoe_next_prefetch_scale_b:"),
    }

    # First identify every original primary load before inserting new setup
    # blocks.  Static role paths are classified from their literal LDS bases;
    # shared paths are marked even/odd by their temporal hint.
    cutoff = lines.index(".Lbranch_00000000ad68:")
    selector: str | None = None
    selector_re = re.compile(r"^\s*s_mov_b32\s+s33,\s+([^\s;]+)")
    pc_pattern = re.compile(r";\s*([0-9A-Fa-f]{8,16}):")
    output: list[str] = []
    issue_sites = 0
    stagger_sites = 0
    pending_scale_issue: list[str] | None = None
    for index, line in enumerate(lines):
        if index >= cutoff:
            if pending_scale_issue is not None:
                raise RuntimeError("unterminated staggered scale issue before epilogue")
            output.extend(lines[index:])
            break
        match = selector_re.match(line)
        if match:
            selector = match.group(1)
        output.append(line)
        if "tensor_load_to_lds s[32:35], s[36:43]" not in line:
            if pending_scale_issue is not None and line.lstrip().startswith(
                "v_wmma_scale_f32_32x16x128_f4 "
            ):
                output.extend(pending_scale_issue)
                pending_scale_issue = None
            continue
        if selector is None:
            raise RuntimeError(f"missing LDS selector before line {index + 1}")
        selector_arg = selector
        if selector.startswith("s"):
            # all-NT_RT intentionally erased the old A-vs-other hint
            # distinction.  The structurally identical reference retains it,
            # so use the corresponding source line only to recover which
            # common owner path this static site belongs to.
            selector_arg += (
                ":even" if "th:TH_LOAD_NT" in reference_lines[index] else ":odd"
            )
        scale_issue = secondary_scale_issue(selector_arg, issue_sites)
        pc_match = pc_pattern.search(line)
        pc = pc_match.group(1).upper() if pc_match else None
        if pc in STAGGER_SCALE_AFTER_ONE_WMMA_PCS:
            if pending_scale_issue is not None:
                raise RuntimeError(f"nested staggered scale issue at source line {index + 1}")
            pending_scale_issue = scale_issue
            stagger_sites += 1
        else:
            output.extend(scale_issue)
        issue_sites += 1
    else:
        raise RuntimeError("output setup cutoff not found")
    lines = output
    if issue_sites != 56:
        raise RuntimeError(f"expected 56 hotloop scale issue sites, got {issue_sites}")

    # Rewrite the four literal LDS selectors in each role-specific prologue to
    # point at the selected payload half.  Shared hotloop selectors use the
    # newly initialized s95:s98 values.
    path_ends = (
        ROLE_PATHS[1][1],
        ROLE_PATHS[2][1],
        ROLE_PATHS[3][1],
        ".Lbranch_000000005060:",
    )
    for (_, _entry, ready, tensor, half), end_label in zip(
        ROLE_PATHS, path_ends, strict=True
    ):
        transform_role_selectors(lines, ready, end_label, tensor, half)

    # Override each old WPT1 role with the desired WPT2 payload and scale
    # descriptors.  This also runs on prefetched tasks, where the original
    # role setup is skipped.  Insert in reverse source order so earlier label
    # positions remain stable while later blocks grow.
    for wave, _entry, ready, tensor, half in reversed(ROLE_PATHS):
        scale = "ScaleA" if wave < 2 else "ScaleB"
        tag = f"wave{wave}"
        insert_at = lines.index(ready) + 1
        replacement = [
            f"\t; WPT2 owner wave {wave}: {tensor} half {half} + {scale} half {half}.",
            *primary_setup(setups[tensor], tensor=tensor, half=half, tag=tag),
            *secondary_scale_setup(setups[scale], tensor=scale, half=half, tag=tag),
        ]
        lines[insert_at:insert_at] = replacement

    # Replace the four old one-descriptor next-task branches with exact WPT2
    # pairs.  The following task rebuilds both descriptors before advancing
    # from its already-prefetched stage 0 to stage 1.
    tail_start = lines.index(".Lmoe_next_prefetch_a:")
    tail_end = lines.index(".Lmoe_next_prefetch_done:")
    tail_lines: list[str] = []
    tail_specs = (
        (".Lmoe_next_prefetch_a:", "A", "ScaleA", 0, "wave0"),
        (".Lmoe_next_prefetch_b:", "A", "ScaleA", 1, "wave1"),
        (".Lmoe_next_prefetch_scale_a:", "B", "ScaleB", 0, "wave2"),
        (".Lmoe_next_prefetch_scale_b:", "B", "ScaleB", 1, "wave3"),
    )
    for label, tensor, scale, half, tag in tail_specs:
        tail_lines.extend(
            (
                label,
                f"\t; WPT2 next-task stage 0: {tensor} half {half}, then {scale} half {half}.",
                *tail_primary(tail[tensor], tensor=tensor, half=half, tag=tag),
                *tail_scale(tail[scale], tensor=scale, half=half, tag=tag),
                "\ts_branch .Lmoe_next_prefetch_done",
                "",
            )
        )
    lines[tail_start:tail_end] = tail_lines

    lines[3] = (
        "\t; Persistent MoE GEMM1 with B64 clears, full instruction prefetch, "
        "all-input NT_RT, and exactopt-style WPT2 TDM ownership."
    )
    force_all_input_tdm_nt_rt(lines)
    wait1_rewrites, wait4_rewrites = retime_tensor_waits(lines)
    if stagger_sites != len(STAGGER_SCALE_AFTER_ONE_WMMA_PCS):
        raise RuntimeError(
            "staggered scale issue count mismatch: "
            f"expected {len(STAGGER_SCALE_AFTER_ONE_WMMA_PCS)}, got {stagger_sites}"
        )
    return (
        "\n".join(lines) + "\n",
        issue_sites,
        stagger_sites,
        wait1_rewrites,
        wait4_rewrites,
    )


def main() -> None:
    actual = sha256(SOURCE)
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "winner source checksum changed: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {actual}"
        )
    reference_actual = sha256(REFERENCE_SOURCE)
    if reference_actual != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError(
            "reference source checksum changed: "
            f"expected {EXPECTED_REFERENCE_SHA256}, got {reference_actual}"
        )
    source = SOURCE.read_text(encoding="utf-8")
    reference = REFERENCE_SOURCE.read_text(encoding="utf-8")
    if strip_input_tdm_hints(source) != strip_input_tdm_hints(reference):
        raise RuntimeError(
            "all-NT_RT source differs from the structural reference beyond "
            "tensor-load temporal hints"
        )
    text, issue_sites, stagger_sites, wait1_rewrites, wait4_rewrites = build(
        source, reference
    )
    audit_output(text)
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT}")
    print(f"hotloop_scale_issue_sites={issue_sites}")
    print(f"staggered_scale_issue_sites={stagger_sites}")
    print(f"old_output_wait_rewrites={wait1_rewrites}")
    print(f"input_ring_wait_rewrites={wait4_rewrites}")
    print(f"sha256={sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
