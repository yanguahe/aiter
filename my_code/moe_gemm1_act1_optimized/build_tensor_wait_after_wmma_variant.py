#!/usr/bin/env python3
"""Build the retained K-ring schedule that overlaps TDM wait with four WMMAs."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_"
    "static_state_hoist_descriptor_b64.s"
)
OUTPUT = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_"
    "static_state_hoist_descriptor_b64_tensor_wait_after_wmma.s"
)
EXPECTED_SOURCE_SHA256 = (
    "44e6768485b99328561e22c754873bf13ea92cd561ed1c5817fdfbf64e228cae"
)
EXPECTED_OUTPUT_SHA256 = (
    "02320b67ae67a8349f7942a31f89e3f2f3de6efe130e937d04d1d4bdb9273b2d"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: str) -> str:
    pattern = re.compile(
        r"(?P<ds>\ts_wait_dscnt 0x8[^\n]*\n)"
        r"(?P<tw>\ts_wait_tensorcnt 0x2[^\n]*\n)"
        r"(?P<sig>\ts_barrier_signal -1[^\n]*\n)"
        r"(?P<body>\ts_set_vgpr_msb[^\n]*\n"
        r"(?:\tv_wmma_scale_f32_32x16x128_f4[^\n]*\n){4})"
        r"(?P<tail>\ts_mov_b32 s33,[^\n]*\n"
        r"\ts_barrier_wait 0xffff[^\n]*\n)"
    )

    def move_wait(match: re.Match[str]) -> str:
        return (
            match.group("ds")
            + match.group("body")
            + "\ts_wait_tensorcnt 0x2\n"
            + "\ts_barrier_signal -1\n"
            + match.group("tail")
        )

    text, replaced = pattern.subn(move_wait, source)
    if replaced != 8:
        raise RuntimeError(f"expected eight K-ring boundary sequences, got {replaced}")
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
