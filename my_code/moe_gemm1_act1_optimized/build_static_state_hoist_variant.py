#!/usr/bin/env python3
"""Build the retained persistent variant with one-time static wave setup."""

from __future__ import annotations

import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE
    / "persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_incremental_all_ptr.s"
)
OUTPUT = (
    HERE
    / "persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist.s"
)
EXPECTED_SOURCE_SHA256 = (
    "5cb009b909427b231f281c014cfafc578e8e9d21c90049413437970234dea72a"
)
EXPECTED_OUTPUT_SHA256 = (
    "283f2b5a1a0d4e82134211b3724816549450ba5abe6806554bfe67f1a9e6f332"
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
        ".Lmoe_persistent_state_ready:\n"
        "\t; Legacy core scalar contract for one expert-local 1024x6144x7168 GEMM.",
        "\t; Static per-wave state is initialized once on the full-setup path.\n"
        "\t; Legacy core scalar contract for one expert-local 1024x6144x7168 GEMM.",
    )
    text = replace_exact(
        text,
        "\ts_mov_b32 s44, s2\n"
        "\ts_mov_b32 s45, s3\n\n"
        "\t; Preserve the source kernel",
        "\t; Preserve the source kernel",
    )
    text = replace_exact(
        text,
        "\ts_mov_b32 s61, 24\n"
        "\ts_mov_b32 s68, s54",
        "\ts_mov_b32 s61, 24\n\n"
        ".Lmoe_persistent_state_ready:\n"
        "\ts_mov_b32 s44, s2\n"
        "\ts_mov_b32 s45, s3\n"
        "\ts_mov_b32 s68, s54",
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
