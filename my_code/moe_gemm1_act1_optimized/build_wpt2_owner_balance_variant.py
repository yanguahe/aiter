#!/usr/bin/env python3
"""Port only the FlyDSL WPT2 TDM ownership pattern to the assembly winner."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full.s"
OUTPUT = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_"
    "iprefetch_full_wpt2_owner.s"
)
EXPECTED_SOURCE_SHA256 = (
    "6d3b387137b1f5d3de0e95fa8dddb8067e0dae9847ecf773fdc50b5d5ba28bb3"
)

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
            "\ttensor_load_to_lds s[80:83], s[84:91]",
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
    tensor_line = (
        "\ttensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT"
        if tensor == "A"
        else "\ttensor_load_to_lds s[32:35], s[36:43]"
    )
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
    return [*lines, "\ttensor_load_to_lds s[32:35], s[36:43]"]


def build(source: str) -> tuple[str, int]:
    lines = source.splitlines()
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
    output: list[str] = []
    issue_sites = 0
    for index, line in enumerate(lines):
        if index >= cutoff:
            output.extend(lines[index:])
            break
        match = selector_re.match(line)
        if match:
            selector = match.group(1)
        output.append(line)
        if "tensor_load_to_lds s[32:35], s[36:43]" not in line:
            continue
        if selector is None:
            raise RuntimeError(f"missing LDS selector before line {index + 1}")
        selector_arg = selector
        if selector.startswith("s"):
            selector_arg += ":even" if "th:TH_LOAD_NT" in line else ":odd"
        output.extend(secondary_scale_issue(selector_arg, issue_sites))
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
        "and production-style WPT2 TDM ownership."
    )
    return "\n".join(lines) + "\n", issue_sites


def main() -> None:
    actual = sha256(SOURCE)
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "winner source checksum changed: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {actual}"
        )
    text, issue_sites = build(SOURCE.read_text(encoding="utf-8"))
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT}")
    print(f"hotloop_scale_issue_sites={issue_sites}")
    print(f"sha256={sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
