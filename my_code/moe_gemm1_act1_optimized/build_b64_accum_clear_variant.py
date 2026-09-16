#!/usr/bin/env python3
"""Pack adjacent accumulator clears into V_MOV_B64_E32 instructions."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "persistent_overlap_pad8_prefetch_stage0.s"
OUTPUT = HERE / "persistent_overlap_pad8_prefetch_stage0_b64_clear.s"
EXPECTED_SOURCE_SHA256 = (
    "c96f89fccf4ee385d4b3ce5f54a8d0c5efee4f6c05ff90f3ce4e271ff7a51796"
)
ZERO_RE = re.compile(
    r"^\s*v_mov_b32_e32\s+v(?P<reg>\d+)(?:\s*/\*v\d+\*/)?\s*,\s*0\s*(?:;.*)?$"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: str) -> tuple[str, int]:
    lines = source.splitlines()
    header = "\t; Persistent MoE GEMM1 output-pad8 with next-task stage-0 prefetch."
    if lines.count(header) != 1:
        raise RuntimeError("source header changed")
    lines[lines.index(header)] = (
        "\t; Persistent MoE GEMM1 output-pad8 with stage-0 prefetch and B64 accumulator clears."
    )

    output: list[str] = []
    pairs = 0
    index = 0
    while index < len(lines):
        first = ZERO_RE.match(lines[index])
        second = ZERO_RE.match(lines[index + 1]) if index + 1 < len(lines) else None
        if first and second:
            first_reg = int(first.group("reg"))
            second_reg = int(second.group("reg"))
            if (
                100 <= first_reg <= 226
                and first_reg % 2 == 0
                and second_reg == first_reg + 1
            ):
                output.append(f"\tv_mov_b64_e32 v[{first_reg}:{second_reg}], 0")
                pairs += 1
                index += 2
                continue
        output.append(lines[index])
        index += 1

    if pairs != 1280:
        raise RuntimeError(f"expected 1280 accumulator-clear pairs, got {pairs}")
    return "\n".join(output) + "\n", pairs


def main() -> None:
    actual = sha256(SOURCE)
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "stage-0 source checksum changed: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {actual}"
        )
    text, pairs = build(SOURCE.read_text(encoding="utf-8"))
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT}")
    print(f"packed_pairs={pairs}")
    print(f"sha256={sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
