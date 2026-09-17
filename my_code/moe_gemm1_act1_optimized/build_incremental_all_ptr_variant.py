#!/usr/bin/env python3
"""Build the retained all-NT-RT variant with incremental next-task pointers."""

from __future__ import annotations

import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE
    / "persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt.s"
)
OUTPUT = (
    HERE
    / "persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_incremental_all_ptr.s"
)
EXPECTED_SOURCE_SHA256 = (
    "8934b9767fb8295b7d1bf3c5df246fbb239a2784c76f11873f1ceb490d9f0169"
)
EXPECTED_OUTPUT_SHA256 = (
    "5cb009b909427b231f281c014cfafc578e8e9d21c90049413437970234dea72a"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_exact(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match, got {count}: {old!r}")
    return text.replace(old, new, 1)


def build(source: str) -> str:
    text = replace_exact(
        source,
        "\ts_mul_i32 s24, s29, 0x380000\n"
        "\ts_add_nc_u64 s[4:5], s[4:5], s[24:25]",
        "\ts_mul_i32 s24, s29, 0x380000\n"
        "\ts_add_nc_u64 s[4:5], s[4:5], s[24:25]\n"
        "\ts_add_nc_u64 s[72:73], s[72:73], s[24:25]",
    )
    text = replace_exact(
        text,
        "\ts_mul_i32 s24, s29, 0x38000\n"
        "\ts_add_nc_u64 s[8:9], s[8:9], s[24:25]",
        "\ts_mul_i32 s24, s29, 0x38000\n"
        "\ts_add_nc_u64 s[8:9], s[8:9], s[24:25]\n"
        "\ts_add_nc_u64 s[76:77], s[76:77], s[24:25]",
    )
    text = replace_exact(
        text,
        "\t; Materialize all four tensor bases for the next task.\n"
        "\ts_mul_i32 s24, s55, 0x100\n"
        "\ts_mul_hi_u32 s73, s24, s13\n"
        "\ts_mul_i32 s24, s24, s13\n"
        "\ts_add_co_u32 s72, s4, s24\n"
        "\ts_add_co_ci_u32 s73, s73, s5\n"
        "\ts_mul_i32 s24, s55, 0x100\n"
        "\ts_mul_hi_u32 s77, s24, s15\n"
        "\ts_mul_i32 s24, s24, s15\n"
        "\ts_add_co_u32 s76, s8, s24\n"
        "\ts_add_co_ci_u32 s77, s77, s9\n"
        "\ts_mul_i32 s24, s54, 0x100\n"
        "\ts_mul_hi_u32 s75, s24, s14\n"
        "\ts_mul_i32 s24, s24, s14\n"
        "\ts_add_co_u32 s74, s6, s24\n"
        "\ts_add_co_ci_u32 s75, s75, s7\n"
        "\ts_mul_i32 s24, s54, 0x100\n"
        "\ts_mul_hi_u32 s79, s24, s16\n"
        "\ts_mul_i32 s24, s24, s16\n"
        "\ts_add_co_u32 s78, s10, s24\n"
        "\ts_add_co_ci_u32 s79, s79, s11\n",
        "\t; All tensor bases advance incrementally. A/ScaleA use their expert deltas.\n"
        "\t; With fixed s14=0xe00 and s16=0xe0, B/ScaleB deltas are exact:\n"
        "\t; same expert (+16 N): 0xe00000 / 0xe0000; wrap (+4 experts,-8 N): 0x4d00000 / 0x4d0000.\n"
        "\ts_mul_i32 s24, s29, 0xfc0000\n"
        "\ts_add_co_u32 s24, s24, 0xe00000\n"
        "\ts_add_nc_u64 s[74:75], s[74:75], s[24:25]\n"
        "\ts_mul_i32 s24, s29, 0xfc000\n"
        "\ts_add_co_u32 s24, s24, 0xe0000\n"
        "\ts_add_nc_u64 s[78:79], s[78:79], s[24:25]\n",
    )
    return text


def main() -> None:
    actual = sha256(SOURCE)
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"source checksum changed: expected {EXPECTED_SOURCE_SHA256}, got {actual}"
        )
    OUTPUT.write_text(build(SOURCE.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    actual_output = sha256(OUTPUT)
    if actual_output != EXPECTED_OUTPUT_SHA256:
        raise RuntimeError(
            f"output checksum changed: expected {EXPECTED_OUTPUT_SHA256}, got {actual_output}"
        )
    print(f"wrote {OUTPUT}")
    print(f"sha256={actual_output}")


if __name__ == "__main__":
    main()
