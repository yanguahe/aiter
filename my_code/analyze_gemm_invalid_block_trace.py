#!/usr/bin/env python3
"""Measure early-exit blocks in decoded gfx1250 ATT wave traces."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


WAVE_FILE_RE = re.compile(
    r"se(?P<se>\d+)_sm(?P<simd>\d+)_sl(?P<slot>\d+)_wv(?P<wave>\d+)\.json$"
)


@dataclass(frozen=True)
class WaveBlock:
    path: Path
    se: int
    simd: int
    slot: int
    sequence: int
    begin: int
    end: int
    duration: int
    instruction_count: int
    wmma_count: int

    @property
    def valid(self) -> bool:
        return self.wmma_count > 0


def load_blocks(ui_dir: Path) -> list[WaveBlock]:
    code_path = ui_dir / "code.json"
    if not code_path.is_file():
        raise ValueError(f"Missing code.json: {code_path}")
    code = json.loads(code_path.read_text(encoding="utf-8"))["code"]

    blocks = []
    for path in sorted(ui_dir.glob("se*_sm*_sl*_wv*.json")):
        match = WAVE_FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        wave = payload["wave"]
        wmma_count = 0
        for record in wave["instructions"]:
            # ATT instruction tuples are
            # [timestamp, latency, stall, idle, code_index].
            code_index = record[4]
            if "wmma" in str(code[code_index][0]).lower():
                wmma_count += 1
        blocks.append(
            WaveBlock(
                path=path,
                se=int(match.group("se")),
                simd=int(match.group("simd")),
                slot=int(match.group("slot")),
                sequence=int(match.group("wave")),
                begin=int(wave["begin"]),
                end=int(wave["end"]),
                duration=int(payload["duration"]),
                instruction_count=int(payload["num_insts"]),
                wmma_count=wmma_count,
            )
        )
    if not blocks:
        raise ValueError(f"No decoded wave JSON files found under {ui_dir}")
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ui_dir", type=Path)
    args = parser.parse_args()

    blocks = load_blocks(args.ui_dir)
    by_slot: dict[tuple[int, int, int], list[WaveBlock]] = defaultdict(list)
    for block in blocks:
        by_slot[(block.se, block.simd, block.slot)].append(block)

    rows = []
    for key, slot_blocks in by_slot.items():
        invalid = [block for block in slot_blocks if not block.valid]
        total_cycles = sum(block.duration for block in slot_blocks)
        invalid_cycles = sum(block.duration for block in invalid)
        rows.append(
            (
                total_cycles,
                key,
                len(slot_blocks),
                len(invalid),
                invalid_cycles,
                100.0 * invalid_cycles / total_cycles,
            )
        )
    rows.sort(reverse=True)

    valid_blocks = [block for block in blocks if block.valid]
    invalid_blocks = [block for block in blocks if not block.valid]
    print(f"ui_dir={args.ui_dir.resolve()}")
    print(
        f"decoded_blocks={len(blocks)} valid={len(valid_blocks)} "
        f"invalid_early_exit={len(invalid_blocks)}"
    )
    print(
        "valid_duration_range="
        f"{min(block.duration for block in valid_blocks)}.."
        f"{max(block.duration for block in valid_blocks)} cycles"
    )
    print(
        "invalid_duration_range="
        f"{min(block.duration for block in invalid_blocks)}.."
        f"{max(block.duration for block in invalid_blocks)} cycles"
    )
    print(
        "valid_instruction_count_range="
        f"{min(block.instruction_count for block in valid_blocks)}.."
        f"{max(block.instruction_count for block in valid_blocks)}"
    )
    print(
        "invalid_instruction_count_range="
        f"{min(block.instruction_count for block in invalid_blocks)}.."
        f"{max(block.instruction_count for block in invalid_blocks)}"
    )
    print()
    print(
        "| rank | SIMD slot | blocks | valid | invalid | total cycles | "
        "invalid cycles | invalid share |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(rows, 1):
        total, (se, simd, slot), count, invalid_count, invalid_cycles, share = row
        print(
            f"| {rank} | SE{se}/SIMD{simd}/slot{slot} | {count} | "
            f"{count-invalid_count} | {invalid_count} | {total} | "
            f"{invalid_cycles} | {share:.6f}% |"
        )

    total, key, count, invalid_count, invalid_cycles, share = rows[0]
    se, simd, slot = key
    print()
    print(
        f"slowest_slot=SE{se}/SIMD{simd}/slot{slot} total_cycles={total} "
        f"blocks={count} valid={count-invalid_count} invalid={invalid_count} "
        f"invalid_cycles={invalid_cycles} invalid_share={share:.6f}%"
    )
    for block in sorted(by_slot[key], key=lambda item: item.sequence):
        kind = "valid/compute" if block.valid else "invalid/early-exit"
        print(
            f"  wv{block.sequence}: cycles={block.duration} "
            f"instructions={block.instruction_count} wmma={block.wmma_count} {kind}"
        )


if __name__ == "__main__":
    main()
