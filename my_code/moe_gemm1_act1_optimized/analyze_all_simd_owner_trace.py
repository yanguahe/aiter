#!/usr/bin/env python3
"""Compare persistent-task stalls for all four logical owner waves in one ATT run."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import analyze_stage0_thread_trace as stage0


ROLE_MARKERS = {
    "A": "s_mov_b32 s95, 0",
    "B": "s_mov_b32 s95, 0x30000",
    "ScaleA": "s_mov_b32 s95, 0x10000",
    "ScaleB": "s_mov_b32 s95, 0x22000",
}


def inclusive_p90(values: list[int]) -> float:
    if len(values) == 1:
        return float(values[0])
    return statistics.quantiles(values, n=10, method="inclusive")[8]


def percentile(values: list[int], fraction: float) -> float:
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def classify_capture(trace_dir: Path, rows) -> tuple[str, dict[str, int]]:
    counts = Counter()
    for path in sorted(trace_dir.glob("se*_sm*_sl*_wv*.json")):
        events = json.loads(path.read_text(encoding="utf-8"))["wave"]["instructions"]
        for event in events:
            isa = rows[int(event[4])].isa
            for role, marker in ROLE_MARKERS.items():
                if isa == marker:
                    counts[role] += 1
    present = [role for role, hits in counts.items() if hits]
    if len(present) != 1:
        raise RuntimeError(
            f"cannot identify one logical owner for {trace_dir}: {dict(counts)}"
        )
    return present[0], dict(counts)


def analyze_capture(tool, simd_dir: Path) -> dict:
    matches = [
        path
        for path in (simd_dir / "kernel" / "rpf_v3").glob("ui_output_agent_*")
        if path.is_dir()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one UI directory under {simd_dir}, got {matches}")
    trace_dir = matches[0]
    result = stage0.analyze_interval(
        tool,
        f"owner_{simd_dir.name}",
        trace_dir,
        stage0.TASK_BOUNDARY,
        stage0.TASK_BOUNDARY,
        "persistent task",
    )
    role, role_marker_hits = classify_capture(trace_dir, result.rows)
    intervals = stage0.drop_first_per_wave(
        result.sample_intervals[0].primary_intervals
    )
    if not intervals:
        raise RuntimeError(f"no steady task intervals in {trace_dir}")

    wait_stall = Counter()
    issue_gap = Counter()
    exact_wait_stall = Counter()
    exact_wait_hits = Counter()
    pc = defaultdict(
        lambda: {"issue_gap": 0, "stall": 0, "hits": 0, "latencies": []}
    )
    for interval in intervals:
        events = [
            event for event in interval.events if event.start_ts < interval.end_ts
        ]
        for event_index, event in enumerate(events):
            row = result.rows[event.code_idx]
            next_timestamp = (
                events[event_index + 1].start_ts
                if event_index + 1 < len(events)
                else interval.end_ts
            )
            gap = max(0, next_timestamp - event.start_ts)
            issue_gap[stage0.issue_group(row.isa)] += gap
            item = pc[(row.vaddr, row.isa)]
            item["issue_gap"] += gap
            item["stall"] += event.stall
            item["hits"] += 1
            item["latencies"].append(event.latency)
            group = stage0.wait_group(row.isa)
            if group is None:
                continue
            wait_stall[group] += event.stall
            exact_wait_stall[row.isa] += event.stall
            exact_wait_hits[row.isa] += 1

    tasks = len(intervals)
    task_mean = statistics.mean(item.cycles for item in intervals)
    wave_spans = []
    for path in sorted(trace_dir.glob("se*_sm*_sl*_wv*.json")):
        events = json.loads(path.read_text(encoding="utf-8"))["wave"]["instructions"]
        wave_spans.append(
            int(events[-1][0]) + int(events[-1][3]) - int(events[0][0])
        )
    top_wait_pcs = []
    nonwait_pcs = []
    for (vaddr, isa), values in pc.items():
        stall = values["stall"] / tasks
        row = {
            "vaddr": vaddr,
            "isa": isa,
            "issue_gap_cycles_per_task": values["issue_gap"] / tasks,
            "stall_cycles_per_task": stall,
            "task_percent": stall / task_mean * 100.0,
            "hits_per_task": values["hits"] / tasks,
            "mean_latency": statistics.mean(values["latencies"]),
            "p90_latency": percentile(values["latencies"], 0.9),
            "max_latency": max(values["latencies"]),
        }
        if stage0.wait_group(isa) is None:
            nonwait_pcs.append(row)
        else:
            top_wait_pcs.append(row)
    top_wait_pcs.sort(key=lambda item: item["stall_cycles_per_task"], reverse=True)
    top_nonwait_issue_gap_pcs = sorted(
        nonwait_pcs,
        key=lambda item: (item["issue_gap_cycles_per_task"], item["max_latency"]),
        reverse=True,
    )
    top_nonwait_latency_pcs = sorted(
        nonwait_pcs,
        key=lambda item: (item["mean_latency"], item["max_latency"]),
        reverse=True,
    )
    return {
        "simd_select": simd_dir.name,
        "role": role,
        "role_marker_hits": role_marker_hits,
        "trace_dir": str(trace_dir),
        "steady_tasks": tasks,
        "task_mean_cycles": task_mean,
        "task_median_cycles": statistics.median(item.cycles for item in intervals),
        "task_p90_cycles": inclusive_p90([item.cycles for item in intervals]),
        "full_wave_mean_cycles": statistics.mean(wave_spans),
        "full_wave_min_cycles": min(wave_spans),
        "full_wave_max_cycles": max(wave_spans),
        "issue_gap_cycles_per_task": {
            key: value / tasks for key, value in sorted(issue_gap.items())
        },
        "wait_stall_cycles_per_task": {
            key: value / tasks for key, value in sorted(wait_stall.items())
        },
        "exact_waits": [
            {
                "isa": isa,
                "stall_cycles_per_task": stall / tasks,
                "hits_per_task": exact_wait_hits[isa] / tasks,
            }
            for isa, stall in exact_wait_stall.most_common()
        ],
        "top_wait_pcs": top_wait_pcs[:32],
        "top_nonwait_issue_gap_pcs": top_nonwait_issue_gap_pcs[:32],
        "top_nonwait_latency_pcs": top_nonwait_latency_pcs[:32],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.trace_root.resolve()
    simd_dirs = sorted(path for path in (root / "thread_trace").glob("simd[0-3]") if path.is_dir())
    if len(simd_dirs) != 4:
        raise SystemExit(f"expected four SIMD trace directories under {root}, got {simd_dirs}")

    tool = stage0.load_trace_tool()
    captures = [analyze_capture(tool, path) for path in simd_dirs]
    payload = {"trace_root": str(root), "captures": captures}
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")

    print("\nowner summary")
    print("role\tsimd\ttask_mean\tbarrier\tdscnt\ttensorcnt\ttop_wait")
    for capture in sorted(captures, key=lambda item: item["role"]):
        waits = capture["wait_stall_cycles_per_task"]
        top = capture["top_wait_pcs"][0]
        print(
            f"{capture['role']}\t{capture['simd_select']}\t"
            f"{capture['task_mean_cycles']:.1f}\t"
            f"{waits.get('s_barrier_wait', 0.0):.1f}\t"
            f"{waits.get('s_wait_dscnt', 0.0):.1f}\t"
            f"{waits.get('s_wait_tensorcnt', 0.0):.1f}\t"
            f"0x{top['vaddr']:x} {top['isa']} {top['stall_cycles_per_task']:.1f}"
        )


if __name__ == "__main__":
    main()
