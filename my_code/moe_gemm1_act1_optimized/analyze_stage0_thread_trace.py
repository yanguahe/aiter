#!/usr/bin/env python3
"""Analyze persistent_overlap_pad8_prefetch_stage0 ATT captures.

The parser is the checked-in trace_segment_cycles.py required by the local
FlyDSL alignment workflow.  The report is emitted as JSON so the measured
cycles can be quoted without reimplementing rocprof's trace format.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
TRACE_TOOL = HERE / "trace_segment_cycles.py"
EXPECTED_TRACE_TOOL_SHA256 = (
    "6684004f30ac4336160f41e89a3eb4313f77943b47f0eab553c6c7d69fa4420a"
)
CASE = "persistent_overlap_pad8_prefetch_stage0"
HOST = "heliosr-1b114-d01-3"

# The task-entry instruction is different for prefetched and full-setup tasks.
# The boundary sequence below executes exactly once after every completed task,
# so consecutive dynamic occurrences delimit complete steady tasks regardless
# of which entry path the following task takes.
TASK_BOUNDARY = (
    "s_add_co_u32 s28, s28, 16",
    "s_cmp_lt_u32 s28, 0x240",
    "s_cbranch_scc0 5",
)

PHASES = (
    (
        "boundary + next-task setup + K hotloop",
        TASK_BOUNDARY,
        (
            "s_wait_idle",
            "s_add_co_u32 s26, s92, 3",
            "s_and_b32 s27, s22, 2",
        ),
    ),
    (
        "output descriptor/address setup + next-stage0 prefetch",
        (
            "s_wait_idle",
            "s_add_co_u32 s26, s92, 3",
            "s_and_b32 s27, s22, 2",
        ),
        (
            "v_and_b32_e32 v4, 15, v0",
            "v_mul_u32_u24_e64 v91, v4, 0x90",
            "v_lshrrev_b32_e32 v4, 4, v0",
        ),
    ),
    (
        "SiLU banks 0-1 + first output launch",
        (
            "v_and_b32_e32 v4, 15, v0",
            "v_mul_u32_u24_e64 v91, v4, 0x90",
            "v_lshrrev_b32_e32 v4, 4, v0",
        ),
        (
            "s_wait_dscnt 0x0",
            "s_barrier_signal -1",
            "s_barrier_wait 0xffff",
            "tensor_store_from_lds s[80:83], s[84:91]",
        ),
    ),
    (
        "SiLU banks 2-3",
        (
            "s_wait_dscnt 0x0",
            "s_barrier_signal -1",
            "s_barrier_wait 0xffff",
            "tensor_store_from_lds s[80:83], s[84:91]",
        ),
        (
            "s_wait_dscnt 0x0",
            "s_barrier_signal -1",
            "s_barrier_wait 0xffff",
            "s_mov_b32 s24, 64",
        ),
    ),
    (
        "second output setup/launch + task boundary",
        (
            "s_wait_dscnt 0x0",
            "s_barrier_signal -1",
            "s_barrier_wait 0xffff",
            "s_mov_b32 s24, 64",
        ),
        TASK_BOUNDARY,
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trace_tool():
    actual = sha256(TRACE_TOOL)
    if actual != EXPECTED_TRACE_TOOL_SHA256:
        raise RuntimeError(
            "trace_segment_cycles.py checksum changed: "
            f"expected {EXPECTED_TRACE_TOOL_SHA256}, got {actual}"
        )
    spec = importlib.util.spec_from_file_location("stage0_trace_segment_cycles", TRACE_TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TRACE_TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_ui_dir(run_id: str) -> Path:
    pattern = str(
        HERE
        / "history_runs"
        / f"{HOST}_{run_id}_att"
        / "att"
        / CASE
        / "thread_trace"
        / "kernel"
        / "rpf_v3"
        / "ui_output_agent_*"
    )
    matches = [Path(item) for item in glob.glob(pattern) if Path(item).is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one trace directory for {pattern}, got {matches}")
    return matches[0]


def analyze_interval(tool, name: str, trace_dir: Path, start, end, description: str):
    config = {
        "sample points": [
            {"start": {"instruction": list(start)}},
            {"end": {"instruction": list(end)}},
        ],
        "interval information": [{"key": "desc", "value": description}],
        "trace_dir": str(trace_dir),
        "se": "*",
        "sm": "*",
        "sl": "*",
        "wv": "*",
    }
    return tool.analyze_kernel_config(name, False, config, Path("<generated>"))


def opcode(isa: str) -> str:
    return isa.split(maxsplit=1)[0] if isa else ""


def wait_group(isa: str) -> str | None:
    op = opcode(isa)
    if op.startswith("s_wait") or op == "s_barrier_wait":
        return op
    return None


def issue_group(isa: str) -> str:
    op = opcode(isa)
    if op == "s_wait_tensorcnt":
        return "TENSORcnt wait"
    if op == "s_barrier_wait":
        return "barrier wait"
    if op == "s_wait_dscnt":
        return "DScnt wait"
    if op.startswith("s_wait"):
        return "other wait"
    if op.startswith("v_wmma"):
        return "WMMA issue"
    if op.startswith("tensor_"):
        return "TDM issue"
    if op.startswith("ds_load"):
        return "LDS read issue"
    if op.startswith("ds_store"):
        return "LDS write issue"
    if op.startswith("ds_"):
        return "other LDS issue"
    if op.startswith("v_exp_f32") or op.startswith("v_rcp_f32"):
        return "EXP/RCP issue"
    if (
        op.startswith("v_dual_")
        or op.startswith("v_pk_mul_f32")
        or op.startswith("v_cvt_pk_bf16_f32")
        or op.startswith("v_swap_b32")
    ):
        return "packed SiLU VALU issue"
    if op.startswith("v_"):
        return "other VALU issue"
    if op == "s_nop":
        return "explicit NOP"
    if op.startswith("s_"):
        return "SALU/control issue"
    return "other issue"


def drop_first_per_wave(intervals):
    kept = []
    seen = set()
    for interval in sorted(intervals, key=lambda item: (item.wave_file, item.start_ts)):
        if interval.wave_file in seen:
            kept.append(interval)
        else:
            seen.add(interval.wave_file)
    return kept


def dispatch_cycles(run_id: str) -> int:
    log = (
        HERE
        / "history_runs"
        / f"{HOST}_{run_id}_att"
        / "att"
        / CASE
        / "logs"
        / "analyze_att_capture.log"
    )
    values = re.findall(
        r"Max observed.*?per_se_cycle=(\d+).*?occupancy\.json",
        log.read_text(encoding="utf-8", errors="replace"),
    )
    if not values:
        raise RuntimeError(f"cannot extract occupancy cycle span from {log}")
    return int(values[-1])


def full_wave_profile(trace_dir: Path, rows):
    spans = []
    waits = Counter()
    for path in sorted(trace_dir.glob("se*_sm*_sl*_wv*.json")):
        events = json.loads(path.read_text(encoding="utf-8"))["wave"]["instructions"]
        spans.append(int(events[-1][0]) + int(events[-1][3]) - int(events[0][0]))
        for event in events:
            group = wait_group(rows[int(event[4])].isa)
            if group:
                waits[group] += int(event[2])
    count = len(spans)
    return statistics.mean(spans), {key: value / count for key, value in waits.items()}


def capture_summary(tool, run_id: str) -> dict:
    trace_dir = resolve_ui_dir(run_id)
    analysis = analyze_interval(
        tool, f"stage0_{run_id}", trace_dir, TASK_BOUNDARY, TASK_BOUNDARY, "task"
    )
    intervals = analysis.sample_intervals[0].primary_intervals
    steady = drop_first_per_wave(intervals)
    if not steady:
        raise RuntimeError(f"{run_id}: no steady task intervals")

    issue = Counter()
    wait_stall = Counter()
    wait_hits = Counter()
    exact_wait_stall = Counter()
    exact_wait_hits = Counter()
    pc = defaultdict(lambda: {"issue": 0, "stall": 0, "hits": 0, "latencies": []})
    for interval in steady:
        events = [event for event in interval.events if event.start_ts < interval.end_ts]
        for index, event in enumerate(events):
            row = analysis.rows[event.code_idx]
            next_ts = events[index + 1].start_ts if index + 1 < len(events) else interval.end_ts
            issue_delta = max(0, next_ts - event.start_ts)
            issue[issue_group(row.isa)] += issue_delta
            group = wait_group(row.isa)
            if group:
                wait_stall[group] += event.stall
                wait_hits[group] += 1
                exact_wait_stall[row.isa] += event.stall
                exact_wait_hits[row.isa] += 1
            key = (row.vaddr, row.isa)
            pc[key]["issue"] += issue_delta
            pc[key]["stall"] += event.stall
            pc[key]["hits"] += 1
            pc[key]["latencies"].append(event.latency)

    phases = {}
    for index, (description, start, end) in enumerate(PHASES):
        phase = analyze_interval(
            tool, f"stage0_{run_id}_phase_{index}", trace_dir, start, end, description
        )
        phase_intervals = drop_first_per_wave(
            phase.sample_intervals[0].primary_intervals
        )
        phases[description] = statistics.mean(item.cycles for item in phase_intervals)

    tasks = len(steady)
    wave_cycles, wave_waits = full_wave_profile(trace_dir, analysis.rows)
    return {
        "run_id": run_id,
        "trace_dir": str(trace_dir),
        "matched_tasks": len(intervals),
        "steady_tasks": tasks,
        "task_mean": statistics.mean(item.cycles for item in steady),
        "task_median": statistics.median(item.cycles for item in steady),
        "task_p90": statistics.quantiles(
            [item.cycles for item in steady], n=10, method="inclusive"
        )[8],
        "dispatch_cycles": dispatch_cycles(run_id),
        "full_wave_mean": wave_cycles,
        "full_wave_waits": wave_waits,
        "phases": phases,
        "issue": {key: value / tasks for key, value in issue.items()},
        "wait_stall": {key: value / tasks for key, value in wait_stall.items()},
        "wait_hits": {key: value / tasks for key, value in wait_hits.items()},
        "exact_wait_stall": {
            key: value / tasks for key, value in exact_wait_stall.items()
        },
        "exact_wait_hits": {key: value / tasks for key, value in exact_wait_hits.items()},
        "pc": {
            f"0x{vaddr:x}|{isa}": {
                "vaddr": vaddr,
                "isa": isa,
                "issue_per_task": values["issue"] / tasks,
                "stall_per_task": values["stall"] / tasks,
                "hits_per_task": values["hits"] / tasks,
                "max_latency": max(values["latencies"]),
                "p90_latency": statistics.quantiles(
                    values["latencies"], n=10, method="inclusive"
                )[8],
            }
            for (vaddr, isa), values in pc.items()
        },
    }


def median_map(captures: list[dict], key: str) -> dict[str, float]:
    names = sorted({name for capture in captures for name in capture[key]})
    return {
        name: statistics.median(capture[key].get(name, 0.0) for capture in captures)
        for name in names
    }


def aggregate(captures: list[dict]) -> dict:
    task_cycles = statistics.median(item["task_mean"] for item in captures)
    pc_keys = sorted({key for capture in captures for key in capture["pc"]})
    pc_rows = []
    for key in pc_keys:
        rows = [capture["pc"].get(key) for capture in captures]
        present = [row for row in rows if row is not None]
        template = present[0]
        pc_rows.append(
            {
                "vaddr": template["vaddr"],
                "isa": template["isa"],
                "issue_per_task": statistics.median(
                    row["issue_per_task"] if row else 0.0 for row in rows
                ),
                "stall_per_task": statistics.median(
                    row["stall_per_task"] if row else 0.0 for row in rows
                ),
                "hits_per_task": statistics.median(
                    row["hits_per_task"] if row else 0.0 for row in rows
                ),
                "max_latency_median": statistics.median(
                    row["max_latency"] if row else 0.0 for row in rows
                ),
                "max_latency_observed": max(
                    row["max_latency"] if row else 0.0 for row in rows
                ),
            }
        )
    waits = median_map(captures, "wait_stall")
    exact_waits = median_map(captures, "exact_wait_stall")
    exact_hits = median_map(captures, "exact_wait_hits")
    issue = median_map(captures, "issue")
    phases = median_map(captures, "phases")
    for mapping in (waits, exact_waits, issue, phases):
        for name, value in list(mapping.items()):
            mapping[name] = {
                "cycles_per_task": value,
                "task_percent": value / task_cycles * 100.0,
            }
    exact_wait_table = []
    for name, values in exact_waits.items():
        exact_wait_table.append(
            {
                "isa": name,
                **values,
                "hits_per_task": exact_hits.get(name, 0.0),
            }
        )
    exact_wait_table.sort(key=lambda row: row["cycles_per_task"], reverse=True)
    for row in pc_rows:
        row["task_percent"] = row["issue_per_task"] / task_cycles * 100.0
    wait_pc = sorted(
        (row for row in pc_rows if wait_group(row["isa"])),
        key=lambda row: row["stall_per_task"],
        reverse=True,
    )
    long_nonwait = sorted(
        (
            row
            for row in pc_rows
            if not wait_group(row["isa"]) and row["max_latency_observed"] >= 100
        ),
        key=lambda row: (row["issue_per_task"], row["max_latency_observed"]),
        reverse=True,
    )
    return {
        "trace_tool_sha256": sha256(TRACE_TOOL),
        "task_cycles": task_cycles,
        "dispatch_cycles": statistics.median(
            item["dispatch_cycles"] for item in captures
        ),
        "full_wave_mean_cycles": statistics.median(
            item["full_wave_mean"] for item in captures
        ),
        "wait_groups": waits,
        "exact_waits": exact_wait_table,
        "issue_groups": issue,
        "phases": phases,
        "top_wait_pcs": wait_pc[:20],
        "long_nonwait_pcs": long_nonwait[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tool = load_trace_tool()
    captures = [capture_summary(tool, run_id) for run_id in args.run_ids]
    capture_overview_keys = (
        "run_id",
        "trace_dir",
        "matched_tasks",
        "steady_tasks",
        "task_mean",
        "task_median",
        "task_p90",
        "dispatch_cycles",
        "full_wave_mean",
    )
    payload = {
        "captures": [
            {key: capture[key] for key in capture_overview_keys}
            for capture in captures
        ],
        "aggregate": aggregate(captures),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
