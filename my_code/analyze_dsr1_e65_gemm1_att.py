#!/usr/bin/env python3
"""Summarize exact decoded G1 wave instruction counts and wait cycles."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    code = json.loads((args.directory / "code.json").read_text())["code"]
    ops = {row[2]: row[0] for row in code}
    waves = []
    missing = Counter()
    totals = defaultdict(list)
    per_instruction = defaultdict(list)
    phases_by_wmma = defaultdict(list)
    for path in sorted(args.directory.glob("se*_sm*_sl*_wv*.json")):
        data = json.loads(path.read_text())
        instructions = data["wave"]["instructions"]
        counts = Counter()
        waits = Counter()
        for row in instructions:
            index = row[-1]
            if index not in ops:
                missing[index] += 1
                continue
            op = ops[index].split(" ")[0]
            counts[op] += 1
            if op.startswith("s_wait") or op == "s_barrier_wait":
                waits[op] += row[3]
                per_instruction[index].append(row[3])
        wmma = sum(value for op, value in counts.items() if op.startswith("v_wmma"))
        tensors = {op: value for op, value in counts.items() if op.startswith("tensor_")}
        wmma_rows = [row for row in instructions
                     if ops.get(row[-1], "").startswith("v_wmma")]
        phases = None
        if wmma_rows:
            first_wmma, last_wmma = wmma_rows[0][0], wmma_rows[-1][0]
            bounds = (data["wave"]["begin"], first_wmma, last_wmma, data["wave"]["end"])
            phases = {}
            for phase, begin, end in zip(("entry", "compute", "epilogue"), bounds, bounds[1:]):
                phase_rows = [row for row in instructions if begin <= row[0] < end]
                phase_ops = Counter(ops.get(row[-1], "unmapped").split(" ")[0]
                                    for row in phase_rows)
                phase_waits = Counter()
                for row in phase_rows:
                    op = ops.get(row[-1], "unmapped").split(" ")[0]
                    if op.startswith("s_wait") or op == "s_barrier_wait":
                        phase_waits[op] += row[3]
                phases[phase] = {"span_cycles": end - begin,
                                 "instruction_count": len(phase_rows),
                                 "wait_cycles": dict(phase_waits),
                                 "opcode_counts": dict(phase_ops)}
            phases_by_wmma[wmma].append(phases)
        wave = {"file": path.name, "duration": data["duration"],
                "begin": data["wave"]["begin"], "end": data["wave"]["end"],
                "wmma": wmma, "tensors": tensors,
                "wait_cycles": dict(waits), "instruction_count": len(instructions),
                "phases": phases}
        waves.append(wave)
        totals[wmma].append(data["duration"])
    summary = {
        "wave_count": len(waves), "missing_pcs": dict(missing),
        "phase_boundaries": "first and last WMMA issue timestamps; last WMMA belongs to epilogue",
        "phase_groups": {
            count: {phase: {
                "span_cycles_median": median(value[phase]["span_cycles"] for value in values),
                "instruction_count_median": median(value[phase]["instruction_count"] for value in values),
                "wait_cycles_median": {
                    op: median(value[phase]["wait_cycles"].get(op, 0) for value in values)
                    for op in sorted({op for value in values for op in value[phase]["wait_cycles"]})
                },
            } for phase in ("entry", "compute", "epilogue")}
            for count, values in sorted(phases_by_wmma.items())
        },
        "wmma_groups": {count: {"waves": len(values), "duration_median": median(values),
                                 "duration_min": min(values), "duration_max": max(values)}
                        for count, values in sorted(totals.items())},
        "longest_waves": sorted(waves, key=lambda row: row["duration"], reverse=True)[:16],
        "wait_instructions": [{"index": index, "isa": ops[index],
                               "count": len(values), "median_cycles": median(values),
                               "sum_cycles": sum(values)}
                              for index, values in sorted(per_instruction.items(),
                                                          key=lambda pair: sum(pair[1]), reverse=True)],
        "waves": waves,
    }
    (args.directory / "g1_wave_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"wave_count": len(waves), "missing_pcs": dict(missing),
                      "wmma_groups": summary["wmma_groups"],
                      "phase_groups": summary["phase_groups"],
                      "top_waits": summary["wait_instructions"][:12]}, indent=2))
    if missing:
        raise RuntimeError(f"unmapped ISA indices: {dict(missing)}")


if __name__ == "__main__":
    main()
