#!/usr/bin/env python3
"""Build the retained full-SQC-prefetch variant from the B64-clear kernel."""

from __future__ import annotations

import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "persistent_overlap_pad8_prefetch_stage0_b64_clear.s"
OUTPUT = HERE / (
    "persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full.s"
)
EXPECTED_SOURCE_SHA256 = (
    "c863c8a6f2f51cd2ce85381addf782cefa6ab0dbe9a2ffee2f9e44e875786788"
)
PREFETCH_OFFSETS = tuple(range(0x0000, 0xC000, 0x1000))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: str) -> str:
    marker = "\ts_bfe_u32 s22, ttmp8, 0x50019"
    if source.count(marker) != 3:
        raise RuntimeError(f"unexpected wave-id marker count: {source.count(marker)}")
    prefetch = [
        marker,
        "\t; Only wave 0 in each WGP warms the shared SQC instruction cache.",
        "\ts_cmp_eq_u32 s22, 0",
        "\ts_cbranch_scc0 .Lmoe_instruction_prefetch_done",
        "\ts_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 24, 1), 1",
        "\ts_get_pc_i64 s[24:25]",
        "\ts_mov_b32 s26, 31",
    ]
    prefetch.extend(
        f"\ts_prefetch_inst s[24:25], 0x{offset:x}, s26, 0"
        for offset in PREFETCH_OFFSETS
    )
    prefetch.append(".Lmoe_instruction_prefetch_done:")
    text = source.replace(marker, "\n".join(prefetch), 1)
    return text.replace(
        "\t; Persistent MoE GEMM1 output-pad8 with stage-0 prefetch and B64 accumulator clears.",
        "\t; Persistent MoE GEMM1 output-pad8 with B64 clears and 12 instruction prefetches.",
        1,
    )


def main() -> None:
    actual = sha256(SOURCE)
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "B64-clear source checksum changed: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {actual}"
        )
    OUTPUT.write_text(
        build(SOURCE.read_text(encoding="utf-8")),
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {OUTPUT}")
    print(f"prefetches={len(PREFETCH_OFFSETS)} sha256={sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
