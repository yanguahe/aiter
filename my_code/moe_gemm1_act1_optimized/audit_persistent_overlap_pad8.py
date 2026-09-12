#!/usr/bin/env python3
"""Static contract and LDS-layout audit for persistent_overlap_pad8."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_ISA = HERE / "persistent_overlap_output_pad8.s"
EXPECTED_SYMBOL = (
    "moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1"
)
EXPECTED_SHA256 = (
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


def count(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("isa", type=Path, nargs="?", default=DEFAULT_ISA)
    args = parser.parse_args()

    text = args.isa.read_text(encoding="utf-8")
    digest = hashlib.sha256(args.isa.read_bytes()).hexdigest()
    checks = {
        "symbol": count(
            text, rf"^\s*\.globl\s+{re.escape(EXPECTED_SYMBOL)}\s*$"
        ),
        "kernarg_184": count(
            text,
            r"^\s*\.(?:kernarg_segment_size|amdhsa_kernarg_size)[: ]+\s*184$",
        ),
        "cluster_4x4x1": count(
            text,
            r"^\s*\.cluster_dims:\s*$\n\s*- 4\s*$\n\s*- 4\s*$\n\s*- 1\s*$",
        ),
        "lds_320k": count(
            text,
            r"^\s*\.(?:group_segment_fixed_size|amdhsa_group_segment_fixed_size)"
            r"[: ]+\s*327680$",
        ),
        "vgpr_1024": count(text, r"^\s*\.amdhsa_next_free_vgpr\s+1024$"),
        "sgpr_104": count(text, r"^\s*\.amdhsa_next_free_sgpr\s+104$"),
        "sched_mode_2": count(
            text,
            r"s_setreg_imm32_b32 hwreg\(HW_REG_WAVE_SCHED_MODE, 0, 2\), 2",
        ),
        "xdl_bit_1": count(
            text,
            r"s_setreg_imm32_b32 hwreg\(HW_REG_WAVE_SCHED_MODE, 2, 1\), 1",
        ),
        "tensor_load": count(text, r"^\s*tensor_load_to_lds\b"),
        "tensor_store": count(text, r"^\s*tensor_store_from_lds\b"),
        "cluster_signal": count(text, r"^\s*s_barrier_signal\s+-3\b"),
        "cluster_wait": count(text, r"^\s*s_barrier_wait\s+(?:0xfffd|-3)\b"),
        "wmma": count(text, r"^\s*v_wmma_scale_f32_32x16x128_f4\b"),
        "reuse_a": text.count("matrix_a_reuse"),
        "reuse_b": text.count("matrix_b_reuse"),
        "exp": count(text, r"^\s*v_exp_f32"),
        "rcp": count(text, r"^\s*v_rcp_f32"),
        "pk_mul": count(text, r"^\s*v_pk_mul_f32"),
        "ds_store_b64": count(text, r"^\s*ds_store_b64\b"),
        "output_pitch_0x90": text.count("v_mul_u32_u24_e64 v91, v4, 0x90"),
        "output_tile_144B": text.count("s_or_b32 s87, s87, 0x900000"),
        "second_half_vgpr_0x2400": text.count(
            "v_add_nc_u32_e32 v91, 0x2400, v91"
        ),
        "second_half_tdm_0x2400": text.count(
            "s_add_co_u32 s81, s81, 0x2400"
        ),
    }
    expected = {
        "symbol": 1,
        "kernarg_184": 2,
        "cluster_4x4x1": 1,
        "lds_320k": 2,
        "vgpr_1024": 1,
        "sgpr_104": 1,
        "sched_mode_2": 1,
        "xdl_bit_1": 1,
        "tensor_load": 56,
        "tensor_store": 2,
        "cluster_signal": 4,
        "cluster_wait": 12,
        "wmma": 512,
        "reuse_a": 128,
        "reuse_b": 128,
        "exp": 256,
        "rcp": 256,
        "pk_mul": 384,
        "ds_store_b64": 64,
        "output_pitch_0x90": 1,
        "output_tile_144B": 1,
        "second_half_vgpr_0x2400": 2,
        "second_half_tdm_0x2400": 1,
    }

    failures = []
    for name, wanted in expected.items():
        actual = checks[name]
        state = "ok" if actual == wanted else "FAIL"
        print(f"{state:4} {name:28} actual={actual} expected={wanted}")
        if actual != wanted:
            failures.append((name, actual, wanted))

    half_bytes = 64 * 144
    region_bytes = 2 * half_bytes
    intervals = [
        (base, base + region_bytes) for base in OUTPUT_REGION_BASES
    ]
    disjoint = all(
        previous_end <= current_start
        for (_, previous_end), (current_start, _) in zip(intervals, intervals[1:])
    )
    in_bounds = intervals[-1][1] <= LDS_BYTES
    for name, actual in (
        ("output_regions_disjoint", disjoint),
        ("output_regions_in_320k", in_bounds),
    ):
        state = "ok" if actual else "FAIL"
        print(f"{state:4} {name:28} actual={actual} expected=True")
        if not actual:
            failures.append((name, actual, True))

    print(
        "info output_layout "
        f"row_pitch=144 half_bytes=0x{half_bytes:x} "
        f"max_end=0x{intervals[-1][1]:x} lds_limit=0x{LDS_BYTES:x}"
    )
    print(f"sha256={digest}")
    if digest != EXPECTED_SHA256:
        failures.append(("sha256", digest, EXPECTED_SHA256))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
