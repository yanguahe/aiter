#!/usr/bin/env python3
"""Aggregate wait and opcode latency from decoded gfx1250 ATT wave JSON."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


WAIT_PREFIXES = {
    "s_wait_tensorcnt": "s_wait_tensorcnt",
    "s_wait_dscnt": "s_wait_dscnt",
    "s_barrier_wait": "s_barrier_wait",
    "s_wait_kmcnt": "s_wait_kmcnt",
    "s_wait_loadcnt": "s_wait_loadcnt",
    "s_wait_storecnt": "s_wait_storecnt",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ui_dir", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    code_path = args.ui_dir / "code.json"
    code_rows = json.loads(code_path.read_text(encoding="utf-8"))["code"]
    waves = []
    opcode_stats = defaultdict(lambda: [0, 0, 0, 0])
    immediate_stats = defaultdict(lambda: [0, 0, 0, 0])

    for path in sorted(args.ui_dir.glob("se*_sm*_sl*_wv*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        wave = payload["wave"]
        instructions = wave["instructions"]
        opcodes = [str(code_rows[event[4]][0]).strip() for event in instructions]
        if not any("v_wmma" in opcode.lower() for opcode in opcodes):
            continue

        span = int(wave["end"]) - int(wave["begin"])
        per_wait = {}
        for label, prefix in WAIT_PREFIXES.items():
            selected = [
                event
                for event, opcode in zip(instructions, opcodes)
                if opcode.startswith(prefix)
            ]
            per_wait[label] = {
                "count": len(selected),
                "latency": sum(int(event[3]) for event in selected),
                "stall": sum(int(event[2]) for event in selected),
            }

        waves.append(
            {
                "file": path.name,
                "span": span,
                "instruction_latency": sum(int(event[3]) for event in instructions),
                "instruction_stall": sum(int(event[2]) for event in instructions),
                "wait": per_wait,
            }
        )

        for event, instruction in zip(instructions, opcodes):
            opcode = instruction.split()[0] if instruction else "<blank>"
            for table, key in ((opcode_stats, opcode), (immediate_stats, instruction)):
                stats = table[key]
                stats[0] += 1
                stats[1] += int(event[3])
                stats[2] += int(event[2])
                stats[3] = max(stats[3], int(event[3]))

    if not waves:
        raise ValueError(f"No valid compute waves found under {args.ui_dir}")

    total_span = sum(wave["span"] for wave in waves)
    total_latency = sum(wave["instruction_latency"] for wave in waves)
    total_stall = sum(wave["instruction_stall"] for wave in waves)
    print(f"ui_dir={args.ui_dir.resolve()}")
    print(f"valid_wave_count={len(waves)}")
    print(f"total_wave_span_cycles={total_span}")
    print(f"mean_wave_span_cycles={total_span / len(waves):.3f}")
    print(f"instruction_latency_over_span={100.0 * total_latency / total_span:.6f}%")
    print(f"instruction_stall_over_span={100.0 * total_stall / total_span:.6f}%")

    print("\nAggregate wait classes:")
    print("| wait class | count | latency cycles | latency/span | stall cycles | stall/span | per-wave latency/span range |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for label in WAIT_PREFIXES:
        count = sum(wave["wait"][label]["count"] for wave in waves)
        latency = sum(wave["wait"][label]["latency"] for wave in waves)
        stall = sum(wave["wait"][label]["stall"] for wave in waves)
        shares = [100.0 * wave["wait"][label]["latency"] / wave["span"] for wave in waves]
        print(
            f"| `{label}` | {count} | {latency} | {100.0 * latency / total_span:.6f}% | "
            f"{stall} | {100.0 * stall / total_span:.6f}% | "
            f"{min(shares):.6f}% / {statistics.median(shares):.6f}% / {max(shares):.6f}% |"
        )

    print("\nPer-wave wait shares:")
    print("| wave | span | tensor cycles | tensor % | ds cycles | ds % | barrier cycles | barrier % | km cycles | km % |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for wave in sorted(waves, key=lambda item: item["file"]):
        span = wave["span"]
        values = wave["wait"]
        print(
            f"| {wave['file']} | {span} | "
            f"{values['s_wait_tensorcnt']['latency']} | "
            f"{100.0 * values['s_wait_tensorcnt']['latency'] / span:.6f}% | "
            f"{values['s_wait_dscnt']['latency']} | "
            f"{100.0 * values['s_wait_dscnt']['latency'] / span:.6f}% | "
            f"{values['s_barrier_wait']['latency']} | "
            f"{100.0 * values['s_barrier_wait']['latency'] / span:.6f}% | "
            f"{values['s_wait_kmcnt']['latency']} | "
            f"{100.0 * values['s_wait_kmcnt']['latency'] / span:.6f}% |"
        )

    print("\nWait immediates:")
    print("| instruction | count | latency cycles | latency/span | stall cycles | stall/span | max latency |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    waits = [
        (instruction, stats)
        for instruction, stats in immediate_stats.items()
        if instruction.startswith(tuple(WAIT_PREFIXES.values()))
    ]
    for instruction, (count, latency, stall, maximum) in sorted(
        waits, key=lambda item: item[1][1], reverse=True
    ):
        print(
            f"| `{instruction}` | {count} | {latency} | "
            f"{100.0 * latency / total_span:.6f}% | {stall} | "
            f"{100.0 * stall / total_span:.6f}% | {maximum} |"
        )

    print(f"\nTop {args.top} opcode classes by decoder latency:")
    print("| rank | opcode | count | latency cycles | latency/span | stall cycles | stall/span | max latency |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|")
    for rank, (opcode, (count, latency, stall, maximum)) in enumerate(
        sorted(opcode_stats.items(), key=lambda item: item[1][1], reverse=True)[: args.top],
        1,
    ):
        print(
            f"| {rank} | `{opcode}` | {count} | {latency} | "
            f"{100.0 * latency / total_span:.6f}% | {stall} | "
            f"{100.0 * stall / total_span:.6f}% | {maximum} |"
        )


if __name__ == "__main__":
    main()
