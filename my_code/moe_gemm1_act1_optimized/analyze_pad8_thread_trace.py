#!/usr/bin/env python3
"""Analyze persistent_overlap_pad8 ATT traces with trace_segment_cycles.py.

The script intentionally imports the rule-provided trace parser so interval
matching follows the same code.json and per-wave timestamp semantics as the
documented FlyDSL alignment workflow.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
TRACE_TOOL = HERE / "trace_segment_cycles.py"
EXPECTED_TRACE_TOOL_SHA256 = (
    "6684004f30ac4336160f41e89a3eb4313f77943b47f0eab553c6c7d69fa4420a"
)

CANDIDATE_RUNS = (
    "20260912T154121Z",
    "20260912T154441Z",
    "20260912T154643Z",
)

TASK_START = (
    "s_cmp_lt_u32 s28, 0x240",
    "s_cbranch_scc0 12477",
    "s_load_b64 s[2:3], s[0:1], 0x0 nv",
)

PHASES = (
    (
        "Task prologue + K hotloop",
        TASK_START,
        (
            "s_wait_idle",
            "s_add_co_u32 s26, s92, 3",
            "s_and_b32 s27, s22, 2",
        ),
    ),
    (
        "Output descriptor/address setup",
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
        "SiLU banks 0-1 + first LDS drain",
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
        "Second output descriptor finalization",
        (
            "s_wait_dscnt 0x0",
            "s_barrier_signal -1",
            "s_barrier_wait 0xffff",
            "s_mov_b32 s24, 64",
        ),
        (
            "s_wait_dscnt 0x0",
            "s_barrier_signal -1",
            "s_barrier_wait 0xffff",
            "s_add_co_u32 s81, s81, 0x2400",
            "tensor_store_from_lds s[80:83], s[84:91]",
        ),
    ),
    (
        "Second output launch + persistent boundary",
        (
            "s_wait_dscnt 0x0",
            "s_barrier_signal -1",
            "s_barrier_wait 0xffff",
            "s_add_co_u32 s81, s81, 0x2400",
            "tensor_store_from_lds s[80:83], s[84:91]",
        ),
        TASK_START,
    ),
)


@dataclass
class CaptureResult:
    name: str
    trace_dir: Path
    full_intervals: list
    steady_intervals: list
    phase_means: dict[str, float]
    phase_medians: dict[str, float]
    issue_cycles: Counter[str]
    wait_stalls: Counter[str]
    wait_hits: Counter[str]
    pc_stats: dict[int, tuple[int, int, list[int]]]
    full_wave_mean_cycles: float
    full_wave_wait_stalls: Counter[str]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trace_tool():
    actual = sha256(TRACE_TOOL)
    if actual != EXPECTED_TRACE_TOOL_SHA256:
        raise RuntimeError(
            "trace_segment_cycles.py checksum changed: "
            f"expected {EXPECTED_TRACE_TOOL_SHA256}, got {actual}"
        )
    spec = importlib.util.spec_from_file_location("pad8_trace_segment_cycles", TRACE_TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TRACE_TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_ui_dir(run_id: str, case: str) -> Path:
    pattern = str(
        HERE
        / "history_runs"
        / f"heliosr-1b114-d01-3_{run_id}_att"
        / "att"
        / case
        / "thread_trace"
        / "kernel"
        / "rpf_v3"
        / "ui_output_agent_*"
    )
    matches = [Path(path) for path in glob.glob(pattern) if Path(path).is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one trace directory for {pattern}, got {matches}")
    return matches[0]


def analyze_config(tool, name: str, trace_dir: Path, points, descriptions):
    config = {
        "sample points": [
            {point_name: {"instruction": list(instructions)}}
            for point_name, instructions in points
        ],
        "interval information": [
            {"key": "desc", "value": description}
            for description in descriptions
        ],
        "trace_dir": str(trace_dir),
        "se": "*",
        "sm": "*",
        "sl": "*",
        "wv": "*",
    }
    return tool.analyze_kernel_config(name, False, config, Path("<generated>"))


def all_task_analysis(tool, name: str, trace_dir: Path):
    # The trace contains 36 task-entry occurrences per wave.  Consecutive
    # task-entry anchors therefore describe the first 35 complete tasks; the
    # final task terminates at s_endpgm and has no following task-entry anchor.
    points = [(f"task_{index}", TASK_START) for index in range(36)]
    descriptions = [f"Persistent task {index}" for index in range(35)]
    analysis = analyze_config(tool, name, trace_dir, points, descriptions)
    counts = [len(sample.primary_intervals) for sample in analysis.sample_intervals]
    if counts != [4] * 35:
        raise RuntimeError(f"{name}: expected 4 waves x 35 complete tasks, got {counts}")
    full = [
        interval
        for sample in analysis.sample_intervals
        for interval in sample.primary_intervals
    ]
    steady = [
        interval
        for sample in analysis.sample_intervals[1:]
        for interval in sample.primary_intervals
    ]
    return analysis, full, steady


def drop_first_per_wave(intervals: Iterable) -> list:
    result = []
    seen = set()
    for interval in sorted(intervals, key=lambda item: (item.wave_file, item.start_ts)):
        if interval.wave_file in seen:
            result.append(interval)
        else:
            seen.add(interval.wave_file)
    return result


def opcode(isa: str) -> str:
    fields = isa.split()
    return fields[0] if fields else ""


def issue_category(isa: str) -> str:
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


def wait_category(isa: str) -> str | None:
    op = opcode(isa)
    return {
        "s_wait_tensorcnt": "s_wait_tensorcnt",
        "s_barrier_wait": "s_barrier_wait",
        "s_wait_dscnt": "s_wait_dscnt",
        "s_wait_kmcnt": "s_wait_kmcnt",
        "s_wait_idle": "s_wait_idle",
        "s_wait_loadcnt": "s_wait_loadcnt",
        "s_wait_storecnt": "s_wait_storecnt",
        "s_wait_asynccnt": "s_wait_asynccnt",
        "s_wait_xcnt": "s_wait_xcnt",
        "s_wait_alu": "s_wait_alu",
    }.get(op)


def active_events(interval):
    return [event for event in interval.events if event.start_ts < interval.end_ts]


def full_wave_wait_profile(trace_dir: Path, rows) -> tuple[float, Counter[str]]:
    wave_spans = []
    wait_stalls: Counter[str] = Counter()
    wave_files = sorted(trace_dir.glob("se*_sm*_sl*_wv*.json"))
    if not wave_files:
        raise RuntimeError(f"no wave traces under {trace_dir}")
    for wave_path in wave_files:
        instructions = json.loads(wave_path.read_text(encoding="utf-8"))["wave"][
            "instructions"
        ]
        wave_spans.append(
            int(instructions[-1][0])
            + int(instructions[-1][3])
            - int(instructions[0][0])
        )
        for event in instructions:
            category = wait_category(rows[int(event[4])].isa)
            if category is not None:
                wait_stalls[category] += int(event[2])
    wave_count = len(wave_files)
    return (
        statistics.mean(wave_spans),
        Counter({name: value / wave_count for name, value in wait_stalls.items()}),
    )


def summarize_capture(tool, name: str, trace_dir: Path) -> tuple[CaptureResult, object]:
    analysis, full, steady = all_task_analysis(tool, name, trace_dir)

    phase_means = {}
    phase_medians = {}
    for phase_idx, (description, start, end) in enumerate(PHASES):
        phase = analyze_config(
            tool,
            f"{name}_phase_{phase_idx}",
            trace_dir,
            (("start", start), ("end", end)),
            (description,),
        )
        intervals = drop_first_per_wave(phase.sample_intervals[0].primary_intervals)
        values = [interval.cycles for interval in intervals]
        phase_means[description] = statistics.mean(values)
        phase_medians[description] = statistics.median(values)

    issue_cycles: Counter[str] = Counter()
    wait_stalls: Counter[str] = Counter()
    wait_hits: Counter[str] = Counter()
    pc_accumulator: dict[int, list] = defaultdict(lambda: [0, 0, []])

    for interval in steady:
        events = active_events(interval)
        for index, event in enumerate(events):
            next_timestamp = (
                events[index + 1].start_ts
                if index + 1 < len(events)
                else interval.end_ts
            )
            issue_cycles[issue_category(analysis.rows[event.code_idx].isa)] += max(
                0, next_timestamp - event.start_ts
            )

            category = wait_category(analysis.rows[event.code_idx].isa)
            if category is not None:
                wait_stalls[category] += event.stall
                wait_hits[category] += 1

            entry = pc_accumulator[event.code_idx]
            entry[0] += event.stall
            entry[1] += 1
            entry[2].append(event.latency)

    pc_stats = {
        code_idx: (values[0], values[1], values[2])
        for code_idx, values in pc_accumulator.items()
    }
    full_wave_mean_cycles, full_wave_wait_stalls = full_wave_wait_profile(
        trace_dir, analysis.rows
    )
    return (
        CaptureResult(
            name=name,
            trace_dir=trace_dir,
            full_intervals=full,
            steady_intervals=steady,
            phase_means=phase_means,
            phase_medians=phase_medians,
            issue_cycles=issue_cycles,
            wait_stalls=wait_stalls,
            wait_hits=wait_hits,
            pc_stats=pc_stats,
            full_wave_mean_cycles=full_wave_mean_cycles,
            full_wave_wait_stalls=full_wave_wait_stalls,
        ),
        analysis,
    )


def median(values: Iterable[float]) -> float:
    return float(statistics.median(list(values)))


def fmt(value: float) -> str:
    return f"{value:,.1f}"


def full_kernel_cycles(run_id: str, case: str) -> int:
    path = (
        HERE
        / "history_runs"
        / f"heliosr-1b114-d01-3_{run_id}_att"
        / "att"
        / case
        / "logs"
        / "analyze_att_capture.log"
    )
    text = path.read_text(encoding="utf-8", errors="replace")
    values = re.findall(
        r"Max observed.*?per_se_cycle=(\d+).*?occupancy\.json", text
    )
    if not values:
        raise RuntimeError(f"cannot find occupancy cycle span in {path}")
    return int(values[-1])


def render_report(results: list[CaptureResult], analyses: list, baseline_waits) -> str:
    task_means = [statistics.mean(i.cycles for i in result.steady_intervals) for result in results]
    task_p50 = [statistics.median(i.cycles for i in result.steady_intervals) for result in results]
    task_p90 = [
        statistics.quantiles(
            [i.cycles for i in result.steady_intervals], n=10, method="inclusive"
        )[8]
        for result in results
    ]
    median_task_mean = median(task_means)
    dispatch_cycles = [full_kernel_cycles(run_id, "candidate") for run_id in CANDIDATE_RUNS]

    lines = [
        "# `persistent_overlap_pad8` thread-trace 分析",
        "",
        "## 结论",
        "",
        "当前 kernel 的主要剩余瓶颈在 K hotloop 的 input TDM 同步和 cluster 同步。",
        "三次 ATT 的 steady persistent task 中，所有显式 wait 的 stall 合计约占",
        "`30%`；其中 `s_wait_tensorcnt` 约占 `12.5%`，`s_barrier_wait` 约占",
        "`11.6%`，`s_wait_dscnt` 约占 `5.8%`。output-pad8 已把 output LDS",
        "drain 的两个主要 `s_wait_dscnt 0` 压到约 `150 cycles/task`，它们已经不是",
        "首要瓶颈；但下一 task prologue 中等待前一 task output TDM 完成的",
        "`s_wait_tensorcnt 0` 仍约占 `3.5%`。",
        "",
        "按 issue timeline 划分，WMMA 占约三分之一，LDS read/write 约占八分之一，",
        "显式同步等待约占三分之一。下一轮优化应优先减少 TDM ready 时间和 cluster",
        "peer 到达偏差，而不是继续压缩 SiLU 指令条数。",
        "",
        "## 方法",
        "",
        "分析使用规则文件指定的 `trace_segment_cycles.py`：",
        "",
        f"```text\nSHA256={EXPECTED_TRACE_TOOL_SHA256}\n```",
        "",
        "数据来自 d01-3 的三次 `persistent_overlap_pad8` ATT capture。每个 capture",
        "包含四个 SE 的 `SIMD3-select` wave trace。脚本使用 task-entry 到下一次",
        "task-entry 的动态时间戳作为一个 persistent task 区间，并展开 36 个相同",
        "sample point。每个 wave 的最后一个 task 没有后继 task-entry，因此得到每个",
        "capture `4 waves × 35 tasks = 140` 个完整区间。占比统计再去掉每个 wave",
        "的第一个 cold task，保留 `136` 个 steady task。",
        "",
        "`trace_segment_cycles.py` 的 interval cycle 使用 `end_ts - start_ts`，包含区间",
        "内的 wait stall。指令级 latency 可以与后续独立指令重叠，因此非 wait 指令的",
        "latency 只用于寻找异常长事件；全周期占比采用相邻 instruction issue timestamp",
        "之差，保证各类别加总接近完整 task cycle。",
        "",
        "## 完整 task cycle",
        "",
        "| capture | steady tasks | mean cycles/task | p50 | p90 |",
        "|---|---:|---:|---:|---:|",
    ]
    for result, mean_value, p50_value, p90_value in zip(results, task_means, task_p50, task_p90):
        lines.append(
            f"| `{result.name}` | {len(result.steady_intervals)} | {mean_value:,.1f} | "
            f"{p50_value:,.1f} | {p90_value:,.1f} |"
        )
    lines.extend(
        [
            "",
            f"三次 capture 的 steady-task mean 中位数为 `{median_task_mean:,.1f} cycles/task`。",
            "首个 task 明显受 cold-start/TDM 建链影响，因此不用于 steady-state 占比。",
            "",
            "## 整个 kernel wave-lifetime 的 wait 占比",
            "",
            "这一统计直接覆盖每个 trace wave 从第一条到最后一条指令，包括 cold",
            "task、后续 steady task 和 final drain。每个 capture 先在四个 SE wave",
            "之间求平均，再对三次 capture 取中位数。",
            "",
            "| capture | occupancy max kernel cycles | mean traced-wave cycles |",
            "|---|---:|---:|",
        ]
    )
    for result, dispatch_cycle in zip(results, dispatch_cycles):
        lines.append(
            f"| `{result.name}` | {dispatch_cycle:,} | {result.full_wave_mean_cycles:,.1f} |"
        )
    lines.extend(
        [
            "",
            f"occupancy max 的中位数为 `{median(dispatch_cycles):,.0f} cycles`；四个 traced",
            "wave lifetime 的中位数为",
            f"`{median(result.full_wave_mean_cycles for result in results):,.1f} cycles`。",
            "两种口径相差不到 1%，下面用 traced-wave lifetime 作为 wait 占比分母。",
            "",
            "| 指令组 | cycles/wave | 整个 wave lifetime 占比 |",
            "|---|---:|---:|",
        ]
    )
    full_wave_cycles = median(result.full_wave_mean_cycles for result in results)
    full_wait_total = 0.0
    for name in (
        "s_wait_tensorcnt",
        "s_barrier_wait",
        "s_wait_dscnt",
        "s_wait_kmcnt",
        "s_wait_idle",
        "s_wait_alu",
    ):
        value = median(result.full_wave_wait_stalls[name] for result in results)
        if value == 0:
            continue
        full_wait_total += value
        lines.append(
            f"| `{name}` | {value:,.1f} | {value / full_wave_cycles * 100:.2f}% |"
        )
    lines.extend(
        [
            f"| 合计 | {full_wait_total:,.1f} | {full_wait_total / full_wave_cycles * 100:.2f}% |",
            "",
            f"三次 capture 的 mean wave lifetime 中位数为 `{full_wave_cycles:,.1f} cycles`。",
            "因此从整个 kernel 生命周期看，显式 wait stall 约占 `28%`；去掉 cold",
            "task 后，steady task 中该比例约为 `29%`。两种口径结论一致。",
            "",
            "## 阶段占比",
            "",
            "下表先对每个 capture 求 steady-task phase mean，再取三次 capture 的中位数。",
            "",
            "| 阶段 | cycles/task | 占完整 task |",
            "|---|---:|---:|",
        ]
    )
    phase_values = []
    for description, _start, _end in PHASES:
        value = median(result.phase_means[description] for result in results)
        phase_values.append(value)
        lines.append(
            f"| {description} | {value:,.1f} | {value / median_task_mean * 100:.2f}% |"
        )
    lines.extend(
        [
            f"| 分段合计 | {sum(phase_values):,.1f} | {sum(phase_values) / median_task_mean * 100:.2f}% |",
            "",
            "`Task prologue + K hotloop` 占约 `85%`。两个 SiLU/output staging 段合计约",
            "`12.5%`，descriptor setup 和 persistent boundary 合计约 `2%`。因此即使",
            "把整个 SiLU/output 部分理想化为零成本，上限也只是约 `14%`；实际可优化",
            "空间显著更小。",
            "",
            "## `s_wait_*` 与 barrier stall",
            "",
            "这些值直接累加 trace event 的 `stall` 字段。对 wait 指令而言，stall",
            "期间 wave 不能 issue，因此可以与完整 task cycle 相除。",
            "",
            "| 指令组 | stall cycles/task | task 占比 | 动态次数/task | 平均 stall/次 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    wait_names = (
        "s_wait_tensorcnt",
        "s_barrier_wait",
        "s_wait_dscnt",
        "s_wait_kmcnt",
        "s_wait_idle",
        "s_wait_alu",
    )
    wait_total = 0.0
    for name in wait_names:
        stalls = [result.wait_stalls[name] / len(result.steady_intervals) for result in results]
        hits = [result.wait_hits[name] / len(result.steady_intervals) for result in results]
        stall_value = median(stalls)
        hit_value = median(hits)
        if stall_value == 0 and hit_value == 0:
            continue
        wait_total += stall_value
        lines.append(
            f"| `{name}` | {stall_value:,.1f} | {stall_value / median_task_mean * 100:.2f}% | "
            f"{hit_value:,.1f} | {stall_value / hit_value if hit_value else 0:,.1f} |"
        )
    lines.extend(
        [
            f"| 合计 | {wait_total:,.1f} | {wait_total / median_task_mean * 100:.2f}% | — | — |",
            "",
            "按具体 wait immediate 合并后：",
            "",
            "| 指令 | stall cycles/task | task 占比 | 动态次数/task |",
            "|---|---:|---:|---:|",
        ]
    )
    exact_wait_names = set()
    for analysis, result in zip(analyses, results):
        for code_idx in result.pc_stats:
            isa = analysis.rows[code_idx].isa
            if wait_category(isa) is not None:
                exact_wait_names.add(isa)
    exact_wait_rows = []
    for isa in exact_wait_names:
        stalls = []
        hits = []
        for analysis, result in zip(analyses, results):
            stall_sum = 0
            hit_sum = 0
            for code_idx, (stall, hit, _latencies) in result.pc_stats.items():
                if analysis.rows[code_idx].isa == isa:
                    stall_sum += stall
                    hit_sum += hit
            stalls.append(stall_sum / len(result.steady_intervals))
            hits.append(hit_sum / len(result.steady_intervals))
        exact_wait_rows.append((median(stalls), isa, median(hits)))
    for stall_value, isa, hit_value in sorted(exact_wait_rows, reverse=True):
        if stall_value == 0 and hit_value == 0:
            continue
        lines.append(
            f"| `{isa}` | {stall_value:,.1f} | "
            f"{stall_value / median_task_mean * 100:.2f}% | {hit_value:,.1f} |"
        )
    lines.extend(
        [
            "",
            "最重要的具体 wait PC 是：",
            "",
            "| PC | 指令 | stall cycles/task | task 占比 | 解释 |",
            "|---|---|---:|---:|---|",
        ]
    )

    all_indices = set.intersection(*(set(result.pc_stats) for result in results))
    ranked = []
    for code_idx in all_indices:
        isa = analyses[0].rows[code_idx].isa
        if wait_category(isa) is None and opcode(isa) != "s_nop":
            continue
        stalls = [
            result.pc_stats[code_idx][0] / len(result.steady_intervals)
            for result in results
        ]
        hits = [
            result.pc_stats[code_idx][1] / len(result.steady_intervals)
            for result in results
        ]
        max_latency = median(max(result.pc_stats[code_idx][2]) for result in results)
        ranked.append((median(stalls), code_idx, median(hits), max_latency, isa))
    explanations = {
        6389: "每个 K-ring generation 末尾的 cluster barrier；peer 到达偏差最大的单一热点。",
        1714: "初始 input TDM 发出后等待 `TENSORcnt<=2`，暴露首批 payload/scale 到达时间。",
        1150: "下一 task prologue 等待前一 task 的 output TDM 完成，保护即将复用的 LDS，随后才进入 input TDM/WG/cluster 同步。",
        6243: "steady hotloop 的 LDS read drain，等待下一组 WMMA 所需 operand。",
        6056: "steady hotloop 的另一处 LDS read drain。",
        7395: "第一半 output LDS stores 完成；pad8 后已很短。",
        8280: "第二半 output LDS stores 完成；pad8 后已很短。",
        8312: "persistent task 边界的 cluster barrier。",
        5641: "显式 `s_nop 0` hazard spacing；固定成本。",
    }
    for stall_value, code_idx, hits, max_latency, isa in sorted(ranked, reverse=True)[:12]:
        row = analyses[0].rows[code_idx]
        lines.append(
            f"| `0x{row.vaddr:x}` | `{isa}` | {stall_value:,.1f} | "
            f"{stall_value / median_task_mean * 100:.2f}% | "
            f"{explanations.get(code_idx, f'{hits:.1f} 次/task；单次最大 latency 中位数 {max_latency:,.0f} cycles。')} |"
        )

    lines.extend(
        [
            "",
            "其中 `0xac44 s_barrier_wait 0xfffd` 与 `0x36fc s_wait_tensorcnt 0x2`",
            "是最大的两个稳定热点。二者合计约占一个 steady task 的 `13%`。",
            "`0x2dfc s_wait_tensorcnt 0x0` 再占约 `3.5%`。这三处已经解释约六分之一",
            "的 task cycle。需要注意，`0x2dfc` 等待的是前一 persistent task 的",
            "output TDM，而不是当前 task 的 input TDM。",
            "",
            "## issue timeline 构成",
            "",
            "下表把每条动态指令到下一条动态指令的 timestamp 差归到前一条指令。",
            "这是一种互斥的 wave issue-timeline 分解，适合判断时间消耗在哪类工作上；",
            "它不是各执行单元的独占 busy counter。",
            "",
            "| 类别 | cycles/task | 占比 |",
            "|---|---:|---:|",
        ]
    )
    category_names = sorted(
        set().union(*(result.issue_cycles.keys() for result in results)),
        key=lambda name: median(
            result.issue_cycles[name] / len(result.steady_intervals)
            for result in results
        ),
        reverse=True,
    )
    for name in category_names:
        value = median(
            result.issue_cycles[name] / len(result.steady_intervals)
            for result in results
        )
        lines.append(f"| {name} | {value:,.1f} | {value / median_task_mean * 100:.2f}% |")

    lines.extend(
        [
            "",
            "WMMA issue timeline 约占三分之一，但单条 WMMA 没有形成百 cycle 级的",
            "serial stall；它是必要计算吞吐。`EXP/RCP` 和 packed SiLU 的成本已经被",
            "软件流水覆盖在约 `12.5%` 的完整 epilogue 中，优先级低于 input TDM/cluster",
            "同步。",
            "",
            "## 其他长 latency 指令",
            "",
            "除 wait/barrier 外，超过 100 cycles 的事件几乎全部是 `ds_load_b128`。",
            "它们的单次最坏 latency 可达到约 `0.6–1.3k cycles`，但同一静态 PC 的",
            "平均 stall 通常只有几十 cycles/task，说明它们是间歇性的 LDS queue/bank",
            "冲突，而不是像 TDM/barrier 那样每个 task 都稳定暴露的大气泡。",
            "",
            "| PC | 指令 | 单次最大 latency 中位数 | stall cycles/task |",
            "|---|---|---:|---:|",
        ]
    )
    non_wait_rows = []
    for code_idx in all_indices:
        isa = analyses[0].rows[code_idx].isa
        if wait_category(isa) is not None or opcode(isa) == "s_nop":
            continue
        max_latency_value = median(
            max(result.pc_stats[code_idx][2]) for result in results
        )
        if max_latency_value < 100:
            continue
        stall_value = median(
            result.pc_stats[code_idx][0] / len(result.steady_intervals)
            for result in results
        )
        non_wait_rows.append((max_latency_value, stall_value, code_idx, isa))
    for max_latency_value, stall_value, code_idx, isa in sorted(
        non_wait_rows, reverse=True
    )[:8]:
        row = analyses[0].rows[code_idx]
        lines.append(
            f"| `0x{row.vaddr:x}` | `{isa}` | {max_latency_value:,.0f} | "
            f"{stall_value:,.1f} |"
        )
    lines.extend(
        [
            "",
            "`s_nop 0` 的显式 hazard spacing 约 `168 cycles/task`，约占 `0.6%`。它可以",
            "作为低优先级调度微调目标，但删除前必须重新验证 SCHED_MODE 2 下的",
            "RAW/WAR hazard；其收益上限远小于 TDM/barrier。",
            "",
            "## output-pad8 已解决的部分",
            "",
            "用同三轮 baseline trace 比较 output 区域的三个 `s_wait_dscnt 0`：",
            "",
            "| 版本 | output wait stall cycles/task |",
            "|---|---:|",
        ]
    )
    candidate_output_wait = median(
        sum(
            event.stall
            for interval in result.steady_intervals
            for event in active_events(interval)
            if analyses[results.index(result)].rows[event.code_idx].isa == "s_wait_dscnt 0x0"
        )
        / len(result.steady_intervals)
        for result in results
    )
    baseline_value = median(baseline_waits)
    lines.extend(
        [
            f"| `persistent_overlap` | {baseline_value:,.1f} |",
            f"| `persistent_overlap_pad8` | {candidate_output_wait:,.1f} |",
            "",
            f"steady task 中 output wait 从约 `{baseline_value:,.1f}` 降到",
            f"`{candidate_output_wait:,.1f} cycles/task`，下降",
            f"`{(baseline_value - candidate_output_wait) / baseline_value * 100:.2f}%`。",
            "这说明当前继续优化 output LDS bank mapping 的边际收益已经明显下降。",
            "这里减少的是 LDS store drain；前一 task 的 output TDM 写回仍会在下一",
            "task 的 `0x2dfc s_wait_tensorcnt 0x0` 暴露约 `998 cycles/task`。",
            "",
            "## 下一步优化优先级",
            "",
            "1. 优先处理 `0x36fc s_wait_tensorcnt 0x2`。它是 input TDM ready 的",
            "   最大单点，应把更多独立 WMMA/DS/address work 移到 wait 前，或改变",
            "   input TDM descriptor 的发出时机；必须保持每 wave 最多 3 个、每 SIMD",
            "   最多 6 个 in-flight TDM descriptor。",
            "2. 缩小 `0xac44 s_barrier_wait 0xfffd` 的 peer 到达偏差。该 wait 每 task",
            "   动态执行 6 次，是最大的单一稳定热点。应比较四个 wave 的 TDM owner",
            "   工作量和 DS-read tail，而不是删除 cluster barrier。",
            "3. 继续隐藏 `0x2dfc s_wait_tensorcnt 0x0` 的 previous-output TDM drain。",
            "   当前 task/address setup 已经覆盖一部分延迟；后续只能在不提前覆盖",
            "   320 KiB LDS、且不破坏 TDM in-order 约束的前提下再前移独立工作。",
            "4. 针对 `0xa6c0`、`0x9f88` 等 `s_wait_dscnt 0x8` 前的 DS burst 重新排程，",
            "   把不依赖目标 VGPR 的 WMMA/SALU 穿插到 wait 前。",
            "5. 最后才考虑 SiLU epilogue 和显式 NOP。当前整个 output/Silu 区间约占",
            "   `14%`，而 output drain 本身已经被 pad8 大幅压低。",
            "",
            "## 硬件依据",
            "",
            "- MI400 Shader Programming Guide §4.3.7（第 84–88 页）：`S_WAIT_*CNT`",
            "  等待期间 wave 处于 inactive 状态；`DScnt` 跟踪 LDS，`TENSORcnt` 跟踪",
            "  TDM transfer；`S_WAIT_*CNT N` 同时也执行 `S_WAIT_XCNT N`，因此观测到的",
            "  latency 还可能包含地址转换完成时间。",
            "- MI400 Shader Programming Guide §4.3.6：`S_BARRIER_WAIT` 必须等待对应",
            "  workgroup/cluster 的所有成员完成 signal。",
            "- MI400 Shader Programming Guide §4.10.8（第 205–206 页）：每 wave 最多",
            "  3 个、每 SIMD 最多 6 个 TDM descriptor 在 XACK 前处于 in flight；TDM",
            "  可以在多个 wave/descriptor 间切换，并可能被 VMEM arbitration 阻塞。",
            "- CDNA5 ISA §7.10/§7.12：`EXP/RCP` 属于 transcendental pipeline，WMMA",
            "  属于 XDL pipeline，两类完成可与普通 VALU 乱序并行。",
            "",
            "## 复现命令",
            "",
            "```bash",
            "python3 my_code/moe_gemm1_act1_optimized/trace_segment_cycles.py \\",
            "  my_code/moe_gemm1_act1_optimized/pad8_trace_full_task.json",
            "",
            "python3 my_code/moe_gemm1_act1_optimized/trace_segment_cycles.py \\",
            "  my_code/moe_gemm1_act1_optimized/pad8_trace_full_task.json \\",
            "  --specific-part-representative-trace",
            "",
            "python3 my_code/moe_gemm1_act1_optimized/analyze_pad8_thread_trace.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def output_wait_per_task(tool, run_id: str) -> float:
    trace_dir = resolve_ui_dir(run_id, "persistent_overlap")
    analysis, _full, steady = all_task_analysis(tool, f"baseline_{run_id}", trace_dir)
    total = 0
    for interval in steady:
        for event in active_events(interval):
            if analysis.rows[event.code_idx].isa == "s_wait_dscnt 0x0":
                total += event.stall
    return total / len(steady)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "PERSISTENT_OVERLAP_PAD8_THREAD_TRACE_ANALYSIS.md",
    )
    args = parser.parse_args()

    tool = load_trace_tool()
    results = []
    analyses = []
    for index, run_id in enumerate(CANDIDATE_RUNS, 1):
        result, analysis = summarize_capture(
            tool,
            f"pad8_att_{index}",
            resolve_ui_dir(run_id, "candidate"),
        )
        results.append(result)
        analyses.append(analysis)

    baseline_waits = [output_wait_per_task(tool, run_id) for run_id in CANDIDATE_RUNS]
    report = render_report(results, analyses, baseline_waits)
    args.output.write_text(report, encoding="utf-8", newline="\n")
    print(report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
