#!/usr/bin/env python3
"""Map device ATT PCs to ISA and inspect one exact Mega MoE kernel.

Instruction duration is not additive wall time. This report uses timestamp
differences for sequential regions and lists duration/stall only as diagnostics.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def stats(values):
    values = sorted(values)
    if not values:
        return None
    return {
        "count": len(values),
        "min": values[0],
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p90": values[min(len(values) - 1, int(len(values) * 0.9))],
        "max": values[-1],
    }


def disassemble(objdump, path):
    result = subprocess.run(
        [objdump, "-d", "--mcpu=gfx1250", str(path)],
        capture_output=True, text=True, check=True,
    )
    rows = {}
    symbols = {}
    symbol = ""
    for number, line in enumerate(result.stdout.splitlines(), 1):
        label = re.match(r"^([0-9a-fA-F]+) <(.+)>:$", line)
        if label:
            symbol = label[2]
            symbols[symbol] = int(label[1], 16)
            continue
        instruction = re.match(r"^\s*(.+?)\s*//\s*([0-9a-fA-F]+):", line)
        if instruction:
            address = int(instruction[2], 16)
            rows[address] = {
                "isa": instruction[1].strip(), "line": number, "symbol": symbol,
            }
            branch = re.search(r"<.*\+0x([0-9a-f]+)>", line)
            if branch:
                rows[address]["branch_target"] = symbols[symbol] + int(branch[1], 16)
    return result.stdout, rows, symbols


def read_jsonl(path):
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="PID directory with code objects and decoded/")
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--objdump", required=True)
    parser.add_argument("--decoded", default="decoded")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    decoded = args.input / args.decoded
    objects = list(read_jsonl(args.input / "code_objects.jsonl"))
    targets = [row for row in objects if args.kernel.encode() + b"\0" in (
        args.input / row["file"]).read_bytes()]
    if not targets:
        raise RuntimeError("Exact target symbol is absent from captured code objects")
    mappings = {}
    outer_entries = {}
    for row in targets:
        assembly, instructions, symbols = disassemble(args.objdump, args.input / row["file"])
        if args.kernel not in symbols:
            raise RuntimeError("Exact target symbol is absent from disassembly")
        if re.search(r"\bv_illegal\b|^\s*\.long\b", assembly, re.MULTILINE):
            raise RuntimeError("Unknown/illegal target ISA")
        (args.output / f"code_object_{row['id']}.disasm").write_text(assembly)
        mappings[row["id"]] = {
            pc: info for pc, info in instructions.items() if info["symbol"] == args.kernel
        }
        # Kernels without WMMA (for example communication kernels) or without a
        # persistent outer loop are analyzed as one task per wave.
        outer_entries[row["id"]] = None
        mma_pcs = [pc for pc, info in mappings[row["id"]].items() if "wmma" in info["isa"]]
        if mma_pcs:
            first_mma = min(mma_pcs)
            backedges = []
            for pc, info in mappings[row["id"]].items():
                if info["isa"].startswith("s_branch ") and "branch_target" in info:
                    destination = info["branch_target"]
                    if destination < first_mma < pc:
                        backedges.append((pc, destination))
            if backedges:
                outer_entries[row["id"]] = max(backedges)[1]
    target_ids = set(mappings)
    code = []
    code_index = {}
    for obj_id, mapping in mappings.items():
        for pc, info in sorted(mapping.items()):
            code_index[(obj_id, pc)] = len(code)
            code.append([info["isa"], 0, info["line"], info["symbol"], obj_id, pc, 0, 0, 0, 0])

    waves = []
    missing = Counter()
    op_counts = Counter()
    by_pc = defaultdict(lambda: {"hits": 0, "duration": [], "stall": [], "next_issue_gap": []})
    per_se = defaultdict(list)
    phase_times = defaultdict(list)
    representatives = []
    for path in decoded.glob("se*_wave*.json"):
        wave = json.loads(path.read_text())
        original = wave["instructions"]
        selected = [event for event in original if event[5] in target_ids]
        if not selected:
            continue
        mixed = len(selected) != len(original)
        mapped = []
        for event in selected:
            time, _category, stall, duration, pc, obj_id = event
            info = mappings[obj_id].get(pc)
            if info is None:
                missing[(obj_id, pc)] += 1
                continue
            idx = code_index[(obj_id, pc)]
            op = info["isa"].split()[0]
            mapped.append((event, info["isa"], idx, op))
            op_counts[op] += 1
            row = by_pc[(obj_id, pc)]
            row["hits"] += 1
            row["duration"].append(duration)
            row["stall"].append(stall)
            code[idx][6] += 1
            code[idx][7] += duration
            code[idx][8] += stall
        if not mapped:
            continue
        for i, (event, _, _, _) in enumerate(mapped[:-1]):
            gap = mapped[i + 1][0][0] - event[0]
            by_pc[(event[5], event[4])]["next_issue_gap"].append(gap)
        span = wave["end"] - wave["begin"]
        counts = Counter(op for _, _, _, op in mapped)
        timeline = defaultdict(int)
        for state, duration in wave["timeline"]:
            timeline[str(state)] += duration
        summary = {
            "file": path.name, **{k: v for k, v in wave.items() if k not in ("instructions", "timeline")},
            "span_cycles": span, "mixed_code_objects": mixed,
            "instruction_count": len(mapped), "op_counts": dict(counts),
            "timeline_cycles": dict(timeline),
        }
        # Split only the persistent outer N-task loop. The inner K loop also
        # jumps backward, so an arbitrary PC decrease is not a task boundary.
        task_begin = 0
        task_regions = []
        for i, (event, _, _, _) in enumerate(mapped):
            entry = outer_entries[event[5]]
            if i and entry is not None and event[4] == entry:
                segment = mapped[task_begin:i]
                if any("wmma" in item[3] for item in segment):
                    task_regions.append((task_begin, i))
                task_begin = i
        segment = mapped[task_begin:]
        if any("wmma" in item[3] for item in segment):
            task_regions.append((task_begin, len(mapped)))
        tasks = []
        for start, end in task_regions:
            segment = mapped[start:end]
            mma = [j for j, item in enumerate(segment) if "wmma" in item[3]]
            stores = [j for j, item in enumerate(segment) if item[3] == "tensor_store_from_lds"]
            first, last = mma[0], mma[-1]
            t0 = segment[0][0][0]
            next_time = mapped[end][0][0] if end < len(mapped) else wave["end"]
            times = {
                "task_timestamp_span": next_time - t0,
                "before_first_wmma": segment[first][0][0] - t0,
                "first_to_last_wmma": segment[last][0][0] - segment[first][0][0],
                "after_last_wmma": next_time - segment[last][0][0],
            }
            if stores:
                times["last_wmma_to_first_output_store"] = segment[stores[0]][0][0] - segment[last][0][0]
                times["output_store_issue_span"] = segment[stores[-1]][0][0] - segment[stores[0]][0][0]
            task = {"times": times, "wmma_count": len(mma), "output_stores": len(stores)}
            tasks.append(task)
            for name, value in times.items():
                phase_times[name].append(value)
        summary["tasks"] = tasks
        waves.append(summary)
        per_se[str(wave["se"])].append(span)
        converted = {
            "wave": {**{k: v for k, v in wave.items() if k not in ("instructions", "timeline")},
                     "instructions": [[e[0], e[1], e[2], e[3], idx] for e, _, idx, _ in mapped],
                     "timeline": wave["timeline"]},
        }
        ui_name = f"se{wave['se']}_sm{wave['cu']}_sl{wave['simd']}_wv{len(waves)}.json"
        (args.output / ui_name).write_text(json.dumps(converted, separators=(",", ":")) + "\n")
        representatives.append((span, path.name, mapped))
    if not waves:
        raise RuntimeError("No decoded instructions belong to target kernel")
    active = [wave for wave in waves if wave["tasks"]]
    (args.output / "code.json").write_text(json.dumps({"code": code}, separators=(",", ":")) + "\n")
    (args.output / "target_waves.json").write_text(json.dumps(waves, indent=2) + "\n")
    wait_rows = []
    for (obj_id, pc), data in by_pc.items():
        isa = mappings[obj_id][pc]["isa"]
        if isa.startswith(("s_wait", "s_barrier", "tensor_", "v_readlane", "v_readfirstlane")):
            wait_rows.append({"code_object_id": obj_id, "pc": hex(pc), "isa": isa,
                              "hits": data["hits"], "duration_cycles": stats(data["duration"]),
                              "stall_cycles": stats(data["stall"]),
                              "next_first_attempt_gap_cycles": stats(data["next_issue_gap"])})
    wait_rows.sort(key=lambda row: row["duration_cycles"]["mean"] * row["hits"], reverse=True)
    clock_rows = defaultdict(list)
    for se, shader, realtime in read_jsonl(decoded / "realtime.jsonl"):
        clock_rows[str(se)].append((shader, realtime))
    frequency = []
    if (decoded / "frequency.txt").exists():
        frequency = [int(x) for x in (decoded / "frequency.txt").read_text().split()]
    clocks = {}
    if frequency:
        for se, rows in clock_rows.items():
            if len(rows) > 1:
                shader = rows[-1][0] - rows[0][0]
                real = rows[-1][1] - rows[0][1]
                if real > 0:
                    clocks[se] = shader / real * frequency[0] / 1e6
    chunks = list(read_jsonl(args.input / "chunks.jsonl"))
    warnings = []
    if any(row["flags"] & 6 for row in chunks):
        warnings.append("ATT buffer-full flag")
    if (decoded / "info.jsonl").exists() and (decoded / "info.jsonl").stat().st_size:
        warnings.append("Decoder INFO records need review")
    if missing:
        warnings.append("Instruction PCs absent from captured ISA")
    notes = []
    if any(entry is None for entry in outer_entries.values()):
        notes.append("No persistent outer-loop backedge found; each wave is analyzed as one task")
    report = {
        "kernel": args.kernel, "target_code_objects": targets,
        "target_wave_count": len(waves), "target_waves_by_se": {se: stats(v) for se, v in per_se.items()},
        "op_counts": dict(op_counts), "wave_span_cycles": stats([w["span_cycles"] for w in waves]),
        "active_wave_count": len(active),
        "active_wave_span_cycles": stats([w["span_cycles"] for w in active]),
        "active_waves_by_se": {se: stats([w["span_cycles"] for w in active if str(w["se"]) == se])
                               for se in per_se},
        "persistent_tasks_per_active_wave": dict(Counter(len(w["tasks"]) for w in active)),
        "outer_entry_pcs": {
            str(obj): (hex(pc) if pc is not None else None) for obj, pc in outer_entries.items()
        },
        "notes": notes,
        "phase_timestamp_cycles": {name: stats(v) for name, v in phase_times.items()},
        "wait_and_tensor_sites": wait_rows, "clock_mhz_by_se": clocks,
        "raw_chunks": chunks, "warnings": warnings,
        "missing_pcs": [{"code_object": o, "pc": hex(pc), "count": n} for (o, pc), n in missing.items()],
        "timing_note": "Timestamp regions are sequential elapsed cycles; duration sums are not kernel wall time.",
    }
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    compute = [row for row in representatives if any("wmma" in item[3] for item in row[2])]
    representatives = compute or representatives
    representatives.sort(key=lambda row: row[0])
    _, name, mapped = representatives[len(representatives) // 2]
    lines = [f"wave={name}", "first_attempt successful_issue completion next_gap PC ISA"]
    for i, (event, isa, _, _) in enumerate(mapped):
        time, _, stall, duration, pc, _ = event
        gap = mapped[i + 1][0][0] - time if i + 1 < len(mapped) else 0
        lines.append(f"{time:12} {time + stall:12} {time + duration:12} {gap:8} {pc:08x} {isa}")
    (args.output / "representative_trace.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("wait_and_tensor_sites", "missing_pcs")}, indent=2))
    return int(bool(warnings))


if __name__ == "__main__":
    raise SystemExit(main())
