#!/usr/bin/env python3
"""Measure early-exit GEMM blocks from rocprof ATT occupancy data.

The gfx1250 ATT decoder records one allocation and one deallocation event for
every wave lifetime.  An 8-wave workgroup contributes one lifetime to each
physical (SIMD, slot) pair.  For grouped GEMM dispatches, blocks whose binary
search returns ``expert >= n_experts`` take the early-exit path and form a
clearly separated short-lifetime population.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Lifetime:
    se: int
    packed_sa_wgp: int
    simd: int
    slot: int
    kernel_index: int
    start: int
    end: int

    @property
    def cycles(self) -> int:
        return self.end - self.start


def _pair_lifetimes(path: Path, kernel_pattern: re.Pattern[str]) -> list[Lifetime]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = {int(key): value for key, value in payload.get("dispatches", {}).items()}
    selected = {
        index for index, label in labels.items() if kernel_pattern.search(label)
    }
    if not selected:
        raise ValueError(f"No occupancy kernel label matches {kernel_pattern.pattern!r}")

    lifetimes: list[Lifetime] = []
    for raw_se, raw_events in payload.items():
        if not raw_se.isdigit():
            continue
        se = int(raw_se)
        active: dict[tuple[int, int, int, int], int] = {}
        for event in sorted(raw_events, key=lambda item: (item[0], item[4])):
            timestamp, packed, simd, slot, start, kernel_index = event
            if kernel_index not in selected:
                continue
            key = (packed, simd, slot, kernel_index)
            if start:
                if key in active:
                    raise ValueError(f"Double allocation for {key} at {timestamp}")
                active[key] = timestamp
            else:
                if key not in active:
                    raise ValueError(f"Unmatched deallocation for {key} at {timestamp}")
                begin = active.pop(key)
                lifetimes.append(
                    Lifetime(
                        se=se,
                        packed_sa_wgp=packed,
                        simd=simd,
                        slot=slot,
                        kernel_index=kernel_index,
                        start=begin,
                        end=timestamp,
                    )
                )
        if active:
            raise ValueError(f"Unfinished wave allocations for SE{se}: {sorted(active)}")
    if not lifetimes:
        raise ValueError("No complete wave lifetimes were found")
    return lifetimes


def _automatic_threshold(cycles: list[int]) -> tuple[float, float]:
    unique = sorted(set(cycles))
    if len(unique) < 2:
        raise ValueError("Need at least two distinct durations for automatic classification")
    low, high = max(zip(unique, unique[1:]), key=lambda pair: pair[1] / pair[0])
    ratio = high / low
    if ratio < 4.0:
        raise ValueError(
            f"No clear early-exit duration gap: largest ratio is only {ratio:.3f}x"
        )
    return math.sqrt(low * high), ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("occupancy", type=Path)
    parser.add_argument(
        "--kernel-regex",
        default=r"a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1",
    )
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    lifetimes = _pair_lifetimes(args.occupancy, re.compile(args.kernel_regex))
    threshold, gap_ratio = (
        (args.threshold, float("nan"))
        if args.threshold is not None
        else _automatic_threshold([item.cycles for item in lifetimes])
    )

    physical_slots: dict[tuple[int, int, int, int], list[Lifetime]] = defaultdict(list)
    logical_slots: dict[tuple[int, int, int], list[Lifetime]] = defaultdict(list)
    for item in lifetimes:
        physical_slots[(item.se, item.packed_sa_wgp, item.simd, item.slot)].append(item)
        logical_slots[(item.se, item.simd, item.slot)].append(item)

    def summarize(groups):
        rows = []
        for key, items in groups.items():
            invalid = [item for item in items if item.cycles < threshold]
            total_cycles = sum(item.cycles for item in items)
            invalid_cycles = sum(item.cycles for item in invalid)
            rows.append(
                (
                    total_cycles,
                    key,
                    len(items),
                    len(invalid),
                    invalid_cycles,
                    100.0 * invalid_cycles / total_cycles,
                    min(item.cycles for item in items),
                    max(item.cycles for item in items),
                )
            )
        rows.sort(reverse=True)
        return rows

    physical_rows = summarize(physical_slots)
    logical_rows = summarize(logical_slots)

    def row_text(rank, label, row):
        total, _, count, invalid_count, invalid_cycles, share, minimum, maximum = row
        return (
            f"| {rank} | {label} | {count} | {count - invalid_count} | "
            f"{invalid_count} | {total} | {invalid_cycles} | {share:.6f}% | "
            f"{minimum} | {maximum} |"
        )

    all_invalid = [item for item in lifetimes if item.cycles < threshold]
    all_valid = [item for item in lifetimes if item.cycles >= threshold]
    print(f"occupancy={args.occupancy.resolve()}")
    print(f"kernel_regex={args.kernel_regex}")
    print(f"classification_threshold_cycles={threshold:.3f}")
    if not math.isnan(gap_ratio):
        print(f"largest_duration_gap_ratio={gap_ratio:.3f}x")
    print(
        "all_lifetimes="
        f"{len(lifetimes)} valid={len(all_valid)} invalid={len(all_invalid)} "
        f"valid_range={min(x.cycles for x in all_valid)}..{max(x.cycles for x in all_valid)} "
        f"invalid_range={min(x.cycles for x in all_invalid)}..{max(x.cycles for x in all_invalid)}"
    )
    print()
    print("Physical SIMD slots (top 20 by summed resident cycles):")
    print("| rank | physical SIMD slot | blocks | valid | invalid | total cycles | invalid cycles | invalid share | min cycles | max cycles |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(physical_rows[:20], 1):
        _, (se, packed, simd, slot), *_ = row
        sa, wgp = (packed >> 7) & 1, packed & 0x7F
        print(row_text(rank, f"SE{se}/SA{sa}/WGP{wgp}/SIMD{simd}/slot{slot}", row))

    total, key, count, invalid_count, invalid_cycles, share, _, _ = physical_rows[0]
    se, packed, simd, slot = key
    sa, wgp = (packed >> 7) & 1, packed & 0x7F
    physical_label = f"SE{se}/SA{sa}/WGP{wgp}/SIMD{simd}/slot{slot}"
    selected = sorted(physical_slots[key], key=lambda item: item.start)
    print()
    print(
        f"slowest_physical_slot={physical_label} total_cycles={total} "
        f"blocks={count} valid={count-invalid_count} invalid={invalid_count} "
        f"invalid_cycles={invalid_cycles} invalid_share={share:.6f}%"
    )
    print("slowest_physical_slot_lifetimes:")
    for index, item in enumerate(selected):
        kind = "invalid/early-exit" if item.cycles < threshold else "valid/compute"
        print(
            f"  {index:02d}: start={item.start} end={item.end} "
            f"cycles={item.cycles} {kind}"
        )

    reused_rows = [row for row in physical_rows if row[2] > 1]
    if reused_rows:
        total, key, count, invalid_count, invalid_cycles, share, _, _ = reused_rows[0]
        se, packed, simd, slot = key
        sa, wgp = (packed >> 7) & 1, packed & 0x7F
        print()
        print(
            "slowest_reused_physical_slot="
            f"SE{se}/SA{sa}/WGP{wgp}/SIMD{simd}/slot{slot} "
            f"total_cycles={total} blocks={count} valid={count-invalid_count} "
            f"invalid={invalid_count} invalid_cycles={invalid_cycles} "
            f"invalid_share={share:.6f}%"
        )

    print()
    print("Logical SIMD/slot aggregation across physical WGPs (diagnostic only):")
    print("| rank | logical SIMD slot | blocks | valid | invalid | summed cycles | invalid cycles | invalid share | min cycles | max cycles |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(logical_rows, 1):
        _, (se, simd, slot), *_ = row
        print(row_text(rank, f"SE{se}/SIMD{simd}/slot{slot}", row))


if __name__ == "__main__":
    main()
