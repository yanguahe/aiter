#!/usr/bin/env python3
"""Build the retained variant with compact 64-bit descriptor zeroing."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE
    / "persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist.s"
)
OUTPUT = (
    HERE
    / "persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64.s"
)
EXPECTED_SOURCE_SHA256 = (
    "283f2b5a1a0d4e82134211b3724816549450ba5abe6806554bfe67f1a9e6f332"
)
EXPECTED_OUTPUT_SHA256 = (
    "44e6768485b99328561e22c754873bf13ea92cd561ed1c5817fdfbf64e228cae"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: str) -> str:
    text, removed = re.subn(
        r"(?P<head>\.Lmoe_next_prefetch_(?:a|b|scale_a|scale_b):\n"
        r"\ts_mov_b32 s32, 1[^\n]*\n)"
        r"\ts_mov_b32 s33, 0[^\n]*\n"
        r"\ts_mov_b32 s34, 0[^\n]*\n",
        r"\g<head>",
        source,
    )
    if removed != 4:
        raise RuntimeError(
            f"expected four overwritten descriptor pointer initializers, got {removed}"
        )
    # Disassembly comments carry different PCs in each role, so replace the
    # instruction sequence line-by-line while preserving the following code.
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    replaced = 0
    i = 0
    in_next_prefetch_role = False
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped in {
            ".Lmoe_next_prefetch_a:",
            ".Lmoe_next_prefetch_b:",
            ".Lmoe_next_prefetch_scale_a:",
            ".Lmoe_next_prefetch_scale_b:",
        }:
            in_next_prefetch_role = True
        elif stripped == ".Lmoe_next_prefetch_done:":
            in_next_prefetch_role = False
        if in_next_prefetch_role and i + 8 <= len(lines) and all(
            lines[i + j].lstrip().startswith(f"s_mov_b32 s{36 + j}, 0 ")
            for j in range(8)
        ):
            out.extend(
                [
                    "\ts_mov_b64 s[36:37], 0\n",
                    "\ts_mov_b64 s[38:39], 0\n",
                    "\ts_mov_b64 s[40:41], 0\n",
                    "\ts_mov_b64 s[42:43], 0\n",
                ]
            )
            replaced += 1
            i += 8
            continue
        out.append(lines[i])
        i += 1
    if replaced != 4:
        raise RuntimeError(f"expected four descriptor zero blocks, got {replaced}")
    return "".join(out)


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
