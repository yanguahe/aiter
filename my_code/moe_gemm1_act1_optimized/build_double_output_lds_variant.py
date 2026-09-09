#!/usr/bin/env python3
"""Build a variant that overlaps the first output TDM store with epilogue 2."""

from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE
    / "moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_opt.s"
)
OUTPUT = (
    HERE
    / "moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_double_lds.s"
)

# Each wave's tensor_store_from_lds consumes an 8 KiB slice.  The existing
# wave-specific bases are separated by at least 32 KiB, so the immediately
# adjacent 8 KiB slice is free for the second half while the first store is in
# flight.  A larger global offset would cross into another wave's slice.
SECOND_LDS = "0x2000"


def replace_exact(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one match for {old!r}, got {count}")
    return text.replace(old, new, 1)


text = SOURCE.read_text(encoding="utf-8")
text = replace_exact(
    text,
    "\ttensor_store_from_lds s[80:83], s[84:91]                   ; 00000000B278: D0314000 00000000 7C7C5450\n"
    "\ts_wait_tensorcnt 0x0\n"
    "\t; Optimized accumulator bank 2: packed gate/up pairs,",
    "\ttensor_store_from_lds s[80:83], s[84:91]                   ; first output half\n"
    "\t; The second output half uses a disjoint LDS region, so this store can\n"
    "\t; remain in flight while banks 2 and 3 execute their SiLU epilogue.\n"
    "\t; Optimized accumulator bank 2: packed gate/up pairs,",
)
for bank, msb in ((2, "0xaa"), (3, "0xff")):
    marker = (
        f"\t; Optimized accumulator bank {bank}: packed gate/up pairs,\n"
        "\t; dual-issue clamps/adds, and cross-batch EXP/RCP software pipeline.\n"
        f"\ts_set_vgpr_msb {msb}"
    )
    text = replace_exact(
        text,
        marker,
        marker + f"\n\tv_add_nc_u32_e32 v91, {SECOND_LDS}, v91",
    )

text = replace_exact(
    text,
    "\ttensor_store_from_lds s[80:83], s[84:91]                   ; 00000000B7FC: D0314000 00000000 7C7C5450",
    f"\ts_add_co_u32 s81, s81, {SECOND_LDS}\n"
    "\ttensor_store_from_lds s[80:83], s[84:91]                   ; second output half",
)
text = replace_exact(
    text,
    "\t; Optimized MoE GEMM1: production multicast plus packed/pipelined SiLU.",
    "\t; Optimized MoE GEMM1 with double-buffered output LDS/TDM overlap.",
)

OUTPUT.write_text(text, encoding="utf-8", newline="\n")
print(f"wrote {OUTPUT}")
print(f"tensor stores: {text.count('tensor_store_from_lds')}")
print(f"tensorcnt-zero waits: {text.count('s_wait_tensorcnt 0x0')}")
