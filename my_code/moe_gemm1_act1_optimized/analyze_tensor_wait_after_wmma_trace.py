#!/usr/bin/env python3
"""Analyze a fresh tensor_wait_after_wmma capture without changing its ISA.

Use the canonical trace_segment_cycles parser and its representative-trace CLI.
Keep whole-wave, consecutive-task, issue-gap, and completion-latency statistics
separate. Export physical-WGP summaries through analyze_att_capture.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path


TRACE_SHA = "6684004f30ac4336160f41e89a3eb4313f77943b47f0eab553c6c7d69fa4420a"
ROLES = {
    "s_mov_b32 s95, 0": "A",
    "s_mov_b32 s95, 0x30000": "B",
    "s_mov_b32 s95, 0x10000": "ScaleA",
    "s_mov_b32 s95, 0x22000": "ScaleB",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    values = list(values)
    mean = statistics.mean(values)
    ordered = sorted(values)
    position = (len(values) - 1) * 0.9
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return {
        "count": len(values), "min": min(values), "median": statistics.median(values),
        "mean": mean, "max": max(values),
        "p90": ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower),
        "cv": statistics.pstdev(values) / mean if mean else 0.0,
    }


def group(isa: str) -> str:
    op = isa.split()[0]
    if op.startswith("s_wait") or op == "s_barrier_wait":
        return op
    for prefix, label in (
        ("v_wmma", "WMMA"), ("tensor_", "TDM issue"),
        ("ds_load", "LDS read issue"), ("ds_store", "LDS write issue"),
        ("v_exp", "EXP/RCP"), ("v_rcp", "EXP/RCP"),
        ("v_", "other VALU"), ("s_nop", "NOP"), ("s_", "SALU/control"),
    ):
        if op.startswith(prefix):
            return label
    return "other"


def empty_aggregate():
    return {"cycles": 0, "wave_count": 0, "opcodes": Counter(),
            "wait_stall": Counter(), "issue_timeline": Counter(), "pc": {}}


def add_events(aggregate, rows, events, start_pos, end_pos, end_timestamp):
    begin = int(events[start_pos][0])
    duration = int(end_timestamp) - begin
    aggregate["cycles"] += duration
    aggregate["wave_count"] += 1
    gaps = 0
    local_waits = Counter()
    for pos in range(start_pos, end_pos):
        event = events[pos]
        row = rows[int(event[4])]
        next_ts = int(events[pos + 1][0]) if pos + 1 < end_pos else int(end_timestamp)
        gap = next_ts - int(event[0])
        if gap < 0:
            raise RuntimeError("Non-monotonic instruction timestamps")
        gaps += gap
        op = row.isa.split()[0]
        category = group(row.isa)
        aggregate["opcodes"][op] += 1
        aggregate["issue_timeline"][category] += gap
        key = (row.vaddr, row.isa)
        pc = aggregate["pc"].setdefault(key, {
            "pc": f"0x{row.vaddr:x}", "isa": row.isa, "hits": 0,
            "stall": 0, "issue_gap": 0, "latency_sum": 0, "max_latency": 0,
        })
        pc["hits"] += 1
        pc["stall"] += int(event[2])
        pc["issue_gap"] += gap
        pc["latency_sum"] += int(event[3])
        pc["max_latency"] = max(pc["max_latency"], int(event[3]))
        if op.startswith("s_wait") or op == "s_barrier_wait":
            aggregate["wait_stall"][op] += int(event[2])
            local_waits[op] += int(event[2])
    if gaps != duration:
        raise RuntimeError(f"Timeline does not close: {gaps} != {duration}")
    return duration, dict(local_waits)


def finish_aggregate(aggregate):
    total = aggregate["cycles"]
    waves = aggregate["wave_count"]
    pcs = []
    for pc in aggregate["pc"].values():
        pcs.append({
            **pc, "hits_per_wave": pc["hits"] / waves,
            "stall_cycles_per_wave": pc["stall"] / waves,
            "stall_percent": pc["stall"] / total * 100,
            "issue_gap_cycles_per_wave": pc["issue_gap"] / waves,
            "issue_percent": pc["issue_gap"] / total * 100,
            "mean_latency": pc["latency_sum"] / pc["hits"],
        })
    is_wait = lambda p: p["isa"].startswith("s_wait") or p["isa"].startswith("s_barrier_wait")
    return {
        "total_observed_wave_cycles": total, "wave_count": waves,
        "mean_cycles_per_wave": total / waves,
        "opcode_counts": dict(aggregate["opcodes"]),
        "wait_stall": {k: {"cycles_per_wave": v / waves, "percent": v / total * 100}
                       for k, v in aggregate["wait_stall"].most_common()},
        "issue_timeline": {k: {"cycles_per_wave": v / waves, "percent": v / total * 100}
                           for k, v in aggregate["issue_timeline"].most_common()},
        "top_wait_sites": sorted((p for p in pcs if is_wait(p)),
                                 key=lambda p: p["stall"], reverse=True)[:40],
        "top_nonwait_issue_sites": sorted((p for p in pcs if not is_wait(p)),
                                         key=lambda p: p["issue_gap"], reverse=True)[:30],
        "top_nonwait_latency_sites": sorted((p for p in pcs if not is_wait(p)),
                                           key=lambda p: p["mean_latency"], reverse=True)[:30],
        "top_nonwait_max_latency_sites": sorted((p for p in pcs if not is_wait(p)),
                                               key=lambda p: p["max_latency"], reverse=True)[:30],
    }


def marker(rows, prefix: str):
    matches = [i for i, row in enumerate(rows) if row.isa == prefix]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one boundary {prefix!r}, got {matches}")
    index = matches[0]
    return [row.idx for row in rows[index:index + 3]], [row.isa for row in rows[index:index + 3]]


def analyze_instructions(tool, root: Path, output: Path):
    whole, body = empty_aggregate(), empty_aggregate()
    wave_summaries, task_cycles, anchors, configurations = [], [], [], []
    for simd_dir in sorted((root / "thread_trace").glob("simd[0-3]")):
        trace_dirs = list((simd_dir / "kernel/rpf_v3").glob("ui_output_agent_*"))
        if len(trace_dirs) != 1:
            raise RuntimeError(f"Ambiguous UI path under {simd_dir}")
        trace_dir = trace_dirs[0]
        rows = tool.load_code(trace_dir)
        indices, instructions = marker(rows, "s_add_co_u32 s28, s28, 16")
        csv_files = list(trace_dir.parent.glob("stats_ui_*.csv"))
        if len(csv_files) != 1:
            raise RuntimeError(f"Missing/ambiguous instruction statistics in {trace_dir.parent}")
        with csv_files[0].open(newline="", encoding="utf-8") as handle:
            csv_instructions = [r["Instruction"] for r in csv.DictReader(handle)]
        if not any(csv_instructions[i:i + 3] == instructions for i in range(len(csv_instructions) - 2)):
            raise RuntimeError("Task anchor is not three consecutive instructions in trace CSV")
        anchors.append({"capture": simd_dir.name, "stats_csv": str(csv_files[0]),
                        "code_sha256": sha256(trace_dir / "code.json"), "instructions": instructions})
        config = {
            "trace_dir": str(trace_dir), "se": "*", "sm": "*", "sl": "*", "wv": "*",
            "sample points": [{"task_start": {"instruction": instructions}},
                              {"task_end": {"instruction": instructions}}],
            "interval information": [{"key": "desc", "value": "persistent task"}],
            "interval-print-limit": "1",
        }
        config_path = output / f"seg_asm_{simd_dir.name}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        representative = {"kernel versions": [{
            "key": "version", "value": simd_dir.name, "baseline": True, "config": config,
            "specific_part": {"interval_name": "persistent task", "long_lat_thr": 300},
        }]}
        representative_path = output / f"seg_asm_{simd_dir.name}_representative.json"
        representative_path.write_text(json.dumps(representative, indent=2) + "\n", encoding="utf-8")
        configurations.append((config_path, representative_path))
        parsed = tool.analyze_kernel_config(simd_dir.name, True, config, config_path)
        parser_intervals = parsed.sample_intervals[0].primary_intervals
        by_wave = Counter(interval.wave_file for interval in parser_intervals)
        for wave_path in sorted(trace_dir.glob("se*_sm*_sl*_wv*.json")):
            payload = json.loads(wave_path.read_text(encoding="utf-8"))
            events = payload["wave"]["instructions"]
            codes = [int(event[4]) for event in events]
            boundaries = [i for i, code in enumerate(codes)
                          if code == indices[0] and codes[i:i + 3] == indices]
            if len(boundaries) != 36:
                raise RuntimeError(f"Expected 36 tasks in {wave_path}, found {len(boundaries)}")
            # Two identical sample points alternate start/end in the canonical
            # parser: 0->1, 2->3, ... . Validate those 18 sampled pairs, but use
            # every adjacent boundary below to account for all 35 body tasks.
            canonical_pairs = [(x.start_pos, x.end_pos) for x in parser_intervals
                               if x.wave_file == wave_path.name]
            expected_pairs = list(zip(boundaries[::2], boundaries[1::2]))
            if canonical_pairs != expected_pairs:
                raise RuntimeError(f"Canonical sampled pairs disagree with the complete boundary ledger in {wave_path}")
            roles = {ROLES[rows[code].isa] for code in codes if rows[code].isa in ROLES}
            if len(roles) != 1:
                raise RuntimeError(f"Ambiguous owner in {wave_path}: {roles}")
            end = int(events[-1][0]) + int(events[-1][3])
            span, waits = add_events(whole, rows, events, 0, len(events), end)
            body_span, body_waits = add_events(body, rows, events, boundaries[0], boundaries[-1],
                                             int(events[boundaries[-1]][0]))
            first = int(events[boundaries[0]][0]) - int(events[0][0])
            drain = end - int(events[boundaries[-1]][0])
            if first + body_span + drain != span:
                raise RuntimeError("First/body/drain partition is not additive")
            durations = [int(events[b][0]) - int(events[a][0])
                         for a, b in zip(boundaries, boundaries[1:])]
            task_cycles.extend(durations)
            opcodes = Counter(rows[code].isa.split()[0] for code in codes)
            if opcodes["v_wmma_scale_f32_32x16x128_f4"] != 36 * 1792:
                raise RuntimeError(f"Unexpected WMMA count in {wave_path}")
            transcendental_counts = {
                prefix: sum(count for op, count in opcodes.items() if op.startswith(prefix))
                for prefix in ("v_exp_f32", "v_rcp_f32")
            }
            for op in ("v_exp_f32", "v_rcp_f32"):
                if transcendental_counts[op] != 36 * 256:
                    raise RuntimeError(f"Unexpected {op} count in {wave_path}: {transcendental_counts[op]}")
            wave_summaries.append({
                "capture": simd_dir.name, "wave_file": wave_path.name, "owner": next(iter(roles)),
                "task_count": 36, "body_tasks": 35, "whole_wave_cycles": span,
                "canonical_sampled_task_pairs": len(canonical_pairs),
                "first_setup_and_task_cycles": first, "body_cycles": body_span,
                "final_drain_cycles": drain, "body_task_distribution": distribution(durations),
                "wait_stall": waits, "body_wait_stall": body_waits,
                "wmma_count": opcodes["v_wmma_scale_f32_32x16x128_f4"],
                "exp_count": transcendental_counts["v_exp_f32"],
                "rcp_count": transcendental_counts["v_rcp_f32"],
                "instruction_count": len(events),
                "metadata_wave_begin": payload["wave"].get("begin"),
                "metadata_wave_end": payload["wave"].get("end"),
            })
        print(f"Parsed {simd_dir.name}: {len(by_wave)} waves, {len(parser_intervals)} canonical sampled task pairs", flush=True)
    owners = []
    for role in ROLES.values():
        waves = [w for w in wave_summaries if w["owner"] == role]
        if not waves:
            raise RuntimeError(f"No {role} wave")
        owner_cycles = sum(w["whole_wave_cycles"] for w in waves)
        waits = Counter()
        for wave in waves:
            waits.update(wave["wait_stall"])
        owners.append({"role": role, "waves": len(waves),
                       "full_wave_distribution": distribution(w["whole_wave_cycles"] for w in waves),
                       "body_mean_cycles_per_task": sum(w["body_cycles"] for w in waves) / (35 * len(waves)),
                       "waits": {k: {"cycles_per_wave": v / len(waves), "percent": 100 * v / owner_cycles}
                                 for k, v in waits.items()}})
    return {"whole_wave": finish_aggregate(whole), "body_35_tasks": finish_aggregate(body),
            "task_distribution": distribution(task_cycles), "waves": wave_summaries,
            "owners": owners, "anchors": anchors,
            "partition_mean_cycles": {k: statistics.mean(w[k] for w in wave_summaries)
                                      for k in ("first_setup_and_task_cycles", "body_cycles", "final_drain_cycles")}}, configurations


def analyze_occupancy(capture_tool, root: Path):
    results = []
    for simd, summary in capture_tool.analyze_directory_capture(root).items():
        occ = summary.occupancy
        results.append({
            "simd_select": simd, "realtime_path": str(summary.files.realtime_path),
            "occupancy_path": str(summary.files.occupancy_path),
            "wave_lifetimes": len(occ.wave_lifetimes),
            "wave_counts_per_physical_simd": distribution(x.wave_count for x in occ.wave_reuse),
            "max_concurrent_slots": max(x.max_concurrent_active_slots for x in occ.slot_usage),
            "wgp_usage": [asdict(x) for x in occ.wgp_usage],
            "completion": [asdict(x) for x in occ.completion_by_group],
            "physical_wgps": [asdict(x) for x in occ.physical_wgp_summaries],
            "occupancy_cycle_spans": [asdict(x) for x in occ.cycle_spans],
            "weighted_mean_gfx_mhz": summary.realtime.weighted_mean_mhz,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_root", type=Path)
    parser.add_argument("--trace-tool", required=True, type=Path)
    parser.add_argument("--capture-tool", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    trace_path = args.trace_tool.resolve()
    if sha256(trace_path) != TRACE_SHA:
        raise SystemExit("The requested canonical trace_segment_cycles.py SHA256 does not match")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tool = load_module("tensor_wait_trace_segment_cycles", trace_path)
    capture_tool = load_module("tensor_wait_analyze_att_capture", args.capture_tool.resolve())
    instruction_metrics, configurations = analyze_instructions(tool, args.case_root.resolve(), output)
    payload = {"case_root": str(args.case_root.resolve()),
               "trace_tool_sha256": TRACE_SHA,
               "capture_tool_sha256": sha256(args.capture_tool),
               "method": "Whole-wave and consecutive-task issue timelines; latency is not additive.",
               **instruction_metrics,
               "occupancy": analyze_occupancy(capture_tool, args.case_root.resolve())}
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for config, representative in configurations:
        for path, flags in ((config, []), (representative, ["--specific-part-representative-trace"])):
            with path.with_suffix(".log").open("w", encoding="utf-8") as log:
                subprocess.run([sys.executable, str(trace_path), str(path), *flags],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({"wave_count": payload["whole_wave"]["wave_count"],
                      "whole_wave_mean": payload["whole_wave"]["mean_cycles_per_wave"],
                      "task_distribution": payload["task_distribution"],
                      "waits": payload["whole_wave"]["wait_stall"],
                      "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
