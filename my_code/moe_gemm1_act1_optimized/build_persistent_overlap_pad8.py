#!/usr/bin/env python3
"""Build the retained output-pad8 kernel from persistent_overlap."""

from __future__ import annotations

import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE
    / "moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent_overlap.s"
)
OUTPUT = HERE / "persistent_overlap_output_pad8.s"
EXPECTED_SOURCE_SHA256 = (
    "ec3af906acebfdef77db5f734d01fbef45035dfcfdd799f557693aa73dd0d601"
)
EXPECTED_OUTPUT_SHA256 = (
    "039ed787b1f136ee402b76e3bd0b7c9bf439148a0156d0fcca856ed6ab25ccad"
)

LDS_BYTES = 320 * 1024
OUTPUT_REGION_BASES = (
    0x00000,
    0x08000,
    0x12000,
    0x1A000,
    0x30000,
    0x38000,
    0x40000,
    0x48000,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_exact(text: str, old: str, new: str, *, count: int = 1) -> str:
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(f"expected {count} matches for {old!r}, got {actual}")
    return text.replace(old, new, count)


def validate_output_layout(text: str, *, row_pitch: int) -> None:
    """Prove that every double-buffered output region is disjoint and in LDS."""

    base_markers = (
        "\ts_mov_b32 s93, 0",
        "\ts_cselect_b32 s93, 0x8000, s93",
        "\ts_cselect_b32 s93, 0x12000, s93",
        "\ts_cselect_b32 s93, 0x1a000, s93",
        "\ts_mov_b32 s25, 0x30000",
        "\ts_cselect_b32 s25, 0x38000, s25",
        "\ts_cselect_b32 s25, 0x40000, s25",
        "\ts_cselect_b32 s25, 0x48000, s25",
        "\ts_cselect_b32 s93, s93, s25",
    )
    for marker in base_markers:
        if text.count(marker) != 1:
            raise RuntimeError(f"output LDS base marker changed: {marker!r}")

    half_bytes = 64 * row_pitch
    region_bytes = 2 * half_bytes
    intervals = [
        (base, base + region_bytes) for base in OUTPUT_REGION_BASES
    ]
    for (_, previous_end), (current_start, _) in zip(intervals, intervals[1:]):
        if previous_end > current_start:
            raise RuntimeError(
                "double-buffered output LDS regions overlap: "
                f"previous_end=0x{previous_end:x}, current_start=0x{current_start:x}"
            )
    if intervals[-1][1] > LDS_BYTES:
        raise RuntimeError(
            "double-buffered output LDS exceeds 320 KiB: "
            f"end=0x{intervals[-1][1]:x}, limit=0x{LDS_BYTES:x}"
        )


def build_output_pad8(source: str) -> str:
    """Use a 144-byte row pitch for each wave's 64-column BF16 output."""

    text = replace_exact(
        source,
        "\t; Persistent MoE GEMM1 with cross-task output-drain overlap.",
        "\t; Persistent MoE GEMM1 output-pad8 candidate.",
    )
    text = replace_exact(
        text,
        "\t; Compact activated output uses a 128-byte LDS row pitch.",
        "\t; Activated output uses a 144-byte LDS row pitch: 128-byte payload "
        "+ 16-byte skew.",
    )
    text = replace_exact(
        text,
        "\tv_mul_u32_u24_e64 v91, v4, 0x80",
        "\tv_mul_u32_u24_e64 v91, v4, 0x90",
    )
    text = replace_exact(
        text,
        "\ts_or_b32 s87, s87, 0x800000",
        "\ts_or_b32 s87, s87, 0x900000",
    )

    offset_map = {
        2048: 2304,
        2064: 2320,
        2080: 2336,
        2096: 2352,
        4096: 4608,
        4112: 4624,
        4128: 4640,
        4144: 4656,
        6144: 6912,
        6160: 6928,
        6176: 6944,
        6192: 6960,
    }
    epilogue = text.index("\t; Activated output uses a 144-byte LDS row pitch")
    prefix, tail = text[:epilogue], text[epilogue:]
    for old, new in offset_map.items():
        marker = f"offset:{old}"
        actual = tail.count(marker)
        if actual != 4:
            raise RuntimeError(f"expected four output {marker} sites, got {actual}")
        tail = tail.replace(marker, f"offset:{new}")
    text = prefix + tail

    text = replace_exact(
        text,
        "\tv_add_nc_u32_e32 v91, 0x2000, v91",
        "\tv_add_nc_u32_e32 v91, 0x2400, v91",
        count=2,
    )
    text = replace_exact(
        text,
        "\ts_add_co_u32 s81, s81, 0x2000",
        "\ts_add_co_u32 s81, s81, 0x2400",
    )
    validate_output_layout(text, row_pitch=144)
    return text


def main() -> None:
    actual = sha256(SOURCE)
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "persistent_overlap source checksum changed: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {actual}"
        )
    source = SOURCE.read_text(encoding="utf-8")
    validate_output_layout(source, row_pitch=128)
    OUTPUT.write_text(build_output_pad8(source), encoding="utf-8", newline="\n")
    generated = sha256(OUTPUT)
    if generated != EXPECTED_OUTPUT_SHA256:
        raise RuntimeError(
            "generated output checksum changed: "
            f"expected {EXPECTED_OUTPUT_SHA256}, got {generated}"
        )
    print(f"wrote {OUTPUT}")
    print(f"sha256={generated}")


if __name__ == "__main__":
    main()
