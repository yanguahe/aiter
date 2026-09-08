#!/usr/bin/env python3
"""Plot interval-average GFX clock frequency from ATT realtime samples."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SIMD_IDS = (0, 1, 2, 3)
EXPECTED_SE_NAMES = ("SE0", "SE1", "SE2", "SE3")
COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red")
LINE_STYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D")
Sample = tuple[float, float]


@dataclass(frozen=True)
class RealtimeCapture:
    path: Path
    reference_hz: float
    metadata_hz: Optional[float]
    reference_source: str
    samples_by_se: dict[str, tuple[Sample, ...]]


@dataclass(frozen=True)
class SeriesSummary:
    se_name: str
    sample_count: int
    valid_intervals: int
    dropped_nonpositive_realtime: int
    reference_hz: float
    elapsed_us: tuple[float, ...]
    interval_mhz: tuple[float, ...]
    total_shader_ticks: float
    total_realtime_ticks: float
    weighted_mean_mhz: float
    min_interval_mhz: float
    max_interval_mhz: float


@dataclass(frozen=True)
class CaptureSummary:
    simd_id: Optional[int]
    capture: RealtimeCapture
    series: tuple[SeriesSummary, ...]
    valid_intervals: int
    dropped_nonpositive_realtime: int
    total_shader_ticks: float
    total_realtime_ticks: float
    weighted_mean_mhz: float


def default_input_path(script_dir: Path) -> Path:
    return (
        script_dir
        / "f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps.asm.d7-2-gpu1.att"
        / "thread_trace"
        / "simd0"
        / "kernel"
        / "rpf_v3"
        / "ui_output_agent_13454_dispatch_44"
        / "realtime.json"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-SE Fgfx derived from ATT realtime clock pairs. With --dir, "
            "strictly select one kernel/rpf_v3/realtime.json from each "
            "thread_trace/simd0..simd3 capture."
        ),
        epilog=(
            "Directory-mode output defaults to the ATT directory's parent and is "
            "named <ATT_BASENAME>.simd0.png through .simd3.png. Examples:\n"
            "  python plot_fgfx.py --dir ./capture.att\n"
            "  python my_code/plot_fgfx.py --dir my_code/capture.att"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--input",
        type=Path,
        help="Legacy mode: input realtime.json path",
    )
    mode.add_argument(
        "--dir",
        dest="att_dir",
        type=Path,
        help=(
            "Directory mode: ATT root containing thread_trace/simd0..simd3; "
            "writes four derived PNGs outside the trace root"
        ),
    )
    mode.add_argument(
        "--self-test",
        action="store_true",
        help="Run lightweight pure-function and discovery tests, then exit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Legacy mode output image path "
            "(default: my_code/fgfx_all_se.png)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory mode output directory (default: parent of --dir); "
            "must be outside the ATT trace root"
        ),
    )
    parser.add_argument(
        "--reference-hz",
        type=float,
        default=None,
        help=(
            "Override metadata.frequency for every selected realtime.json; "
            "must be greater than zero"
        ),
    )
    return parser


def se_sort_key(name: str) -> int:
    match = re.fullmatch(r"SE(\d+)", name)
    if match is None:
        raise ValueError(f"Invalid SE key: {name!r}")
    return int(match.group(1))


def _finite_sample_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return converted


def _optional_metadata_frequency(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def load_and_validate_realtime(
    input_path: Path,
    reference_hz_override: Optional[float] = None,
    required_se_names: Optional[Sequence[str]] = None,
) -> RealtimeCapture:
    """Load one realtime.json and validate its metadata and sample pairs."""
    path = input_path.expanduser()
    if not path.is_file():
        raise ValueError(f"Input realtime.json does not exist: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSON value must be an object: {path}")

    metadata = data.get("metadata")
    metadata_value = (
        metadata.get("frequency") if isinstance(metadata, dict) else None
    )
    metadata_hz = _optional_metadata_frequency(metadata_value)

    if reference_hz_override is None:
        if metadata_hz is None or metadata_hz <= 0:
            raise ValueError(
                f"{path}: metadata.frequency must be greater than zero; "
                "use --reference-hz only when the reference clock is known"
            )
        reference_hz = metadata_hz
        reference_source = "metadata"
    else:
        reference_hz = float(reference_hz_override)
        if not math.isfinite(reference_hz) or reference_hz <= 0:
            raise ValueError("--reference-hz must be finite and greater than zero")
        if metadata_hz is None:
            metadata_description = "missing or invalid"
        else:
            metadata_description = f"{metadata_hz:g} Hz"
        reference_source = (
            f"CLI override; metadata={metadata_description}"
        )

    malformed_se_names = sorted(
        name
        for name in data
        if name.startswith("SE") and re.fullmatch(r"SE\d+", name) is None
    )
    if malformed_se_names:
        raise ValueError(
            f"{path}: malformed SE keys: {', '.join(malformed_se_names)}"
        )
    se_names = sorted(
        (
            name
            for name in data
            if re.fullmatch(r"SE\d+", name) is not None
        ),
        key=se_sort_key,
    )
    if not se_names:
        raise ValueError(f"{path}: no SE samples found")

    if required_se_names is not None:
        expected = set(required_se_names)
        actual = set(se_names)
        missing = sorted(expected - actual, key=se_sort_key)
        unexpected = sorted(actual - expected, key=se_sort_key)
        problems = []
        if missing:
            problems.append(f"missing {', '.join(missing)}")
        if unexpected:
            problems.append(f"unexpected {', '.join(unexpected)}")
        if problems:
            raise ValueError(
                f"{path}: expected exactly {', '.join(required_se_names)}; "
                + "; ".join(problems)
            )

    samples_by_se: dict[str, tuple[Sample, ...]] = {}
    for se_name in se_names:
        raw_samples = data[se_name]
        if not isinstance(raw_samples, list):
            raise ValueError(f"{path}: {se_name} samples must be a list")
        if not raw_samples:
            raise ValueError(f"{path}: {se_name} samples are empty")
        if len(raw_samples) < 2:
            raise ValueError(
                f"{path}: {se_name} needs at least two samples, "
                f"found {len(raw_samples)}"
            )

        samples: list[Sample] = []
        for sample_index, raw_sample in enumerate(raw_samples):
            if not isinstance(raw_sample, list) or len(raw_sample) != 2:
                raise ValueError(
                    f"{path}: {se_name} sample {sample_index} must be "
                    "[gfx_clock, realtime_clock]"
                )
            shader_clock = _finite_sample_number(
                raw_sample[0], f"{path}: {se_name} sample {sample_index} gfx_clock"
            )
            realtime_clock = _finite_sample_number(
                raw_sample[1],
                f"{path}: {se_name} sample {sample_index} realtime_clock",
            )
            samples.append((shader_clock, realtime_clock))
        samples_by_se[se_name] = tuple(samples)

    return RealtimeCapture(
        path=path,
        reference_hz=reference_hz,
        metadata_hz=metadata_hz,
        reference_source=reference_source,
        samples_by_se=samples_by_se,
    )


def compute_interval_summary(
    se_name: str,
    samples: Sequence[Sample],
    reference_hz: float,
    realtime_origin: float,
) -> SeriesSummary:
    """Compute interval values and a REALTIME-tick-weighted mean."""
    elapsed_us: list[float] = []
    interval_mhz: list[float] = []
    total_shader_ticks = 0.0
    total_realtime_ticks = 0.0
    dropped_nonpositive_realtime = 0

    for interval_index, (sample_0, sample_1) in enumerate(
        zip(samples, samples[1:])
    ):
        shader_0, realtime_0 = sample_0
        shader_1, realtime_1 = sample_1
        delta_shader = shader_1 - shader_0
        delta_realtime = realtime_1 - realtime_0

        if delta_shader < 0:
            raise ValueError(
                f"{se_name}: gfx_clock decreases at adjacent sample pair "
                f"{interval_index}->{interval_index + 1}"
            )
        if delta_realtime <= 0:
            dropped_nonpositive_realtime += 1
            continue

        midpoint_realtime = (realtime_0 + realtime_1) / 2.0
        elapsed_us.append(
            (midpoint_realtime - realtime_origin) / reference_hz * 1.0e6
        )
        interval_mhz.append(
            reference_hz * delta_shader / delta_realtime / 1.0e6
        )
        total_shader_ticks += delta_shader
        total_realtime_ticks += delta_realtime

    if not interval_mhz:
        raise ValueError(
            f"{se_name}: no valid intervals remain after filtering "
            f"{dropped_nonpositive_realtime} delta_realtime<=0 pairs"
        )

    weighted_mean_mhz = (
        reference_hz
        * total_shader_ticks
        / total_realtime_ticks
        / 1.0e6
    )
    return SeriesSummary(
        se_name=se_name,
        sample_count=len(samples),
        valid_intervals=len(interval_mhz),
        dropped_nonpositive_realtime=dropped_nonpositive_realtime,
        reference_hz=reference_hz,
        elapsed_us=tuple(elapsed_us),
        interval_mhz=tuple(interval_mhz),
        total_shader_ticks=total_shader_ticks,
        total_realtime_ticks=total_realtime_ticks,
        weighted_mean_mhz=weighted_mean_mhz,
        min_interval_mhz=min(interval_mhz),
        max_interval_mhz=max(interval_mhz),
    )


def summarize_capture(
    capture: RealtimeCapture, simd_id: Optional[int] = None
) -> CaptureSummary:
    """Summarize all SE clock series in one SIMD-select ATT capture."""
    realtime_origin = min(
        samples[0][1] for samples in capture.samples_by_se.values()
    )
    series = tuple(
        compute_interval_summary(
            se_name,
            capture.samples_by_se[se_name],
            capture.reference_hz,
            realtime_origin,
        )
        for se_name in sorted(capture.samples_by_se, key=se_sort_key)
    )
    total_shader_ticks = sum(item.total_shader_ticks for item in series)
    total_realtime_ticks = sum(item.total_realtime_ticks for item in series)
    weighted_mean_mhz = (
        capture.reference_hz
        * total_shader_ticks
        / total_realtime_ticks
        / 1.0e6
    )
    return CaptureSummary(
        simd_id=simd_id,
        capture=capture,
        series=series,
        valid_intervals=sum(item.valid_intervals for item in series),
        dropped_nonpositive_realtime=sum(
            item.dropped_nonpositive_realtime for item in series
        ),
        total_shader_ticks=total_shader_ticks,
        total_realtime_ticks=total_realtime_ticks,
        weighted_mean_mhz=weighted_mean_mhz,
    )


def tick_weighted_mean(series: Sequence[SeriesSummary]) -> float:
    total_ticks = sum(item.total_realtime_ticks for item in series)
    if total_ticks <= 0:
        raise ValueError("Cannot compute a mean without positive REALTIME ticks")
    return (
        sum(
            item.weighted_mean_mhz * item.total_realtime_ticks
            for item in series
        )
        / total_ticks
    )


def _candidate_list(att_root: Path, paths: Sequence[Path]) -> str:
    return "\n".join(f"  - {path.relative_to(att_root)}" for path in paths)


def discover_realtime_files(att_root: Path) -> dict[int, Path]:
    """Strictly discover one kernel/rpf_v3 realtime.json per SIMD capture."""
    root = att_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"ATT root does not exist or is not a directory: {root}")

    thread_trace = root / "thread_trace"
    if not thread_trace.is_dir():
        raise ValueError(f"Missing ATT thread_trace directory: {thread_trace}")

    discovered: dict[int, Path] = {}
    for simd_id in SIMD_IDS:
        simd_dir = thread_trace / f"simd{simd_id}"
        if not simd_dir.is_dir():
            raise ValueError(f"Missing SIMD capture directory: {simd_dir}")

        search_root = simd_dir / "kernel" / "rpf_v3"
        candidates = (
            sorted(
                path
                for path in search_root.rglob("realtime.json")
                if path.is_file()
            )
            if search_root.is_dir()
            else []
        )
        if len(candidates) == 1:
            discovered[simd_id] = candidates[0]
            continue

        alternate_candidates = sorted(
            path
            for path in simd_dir.rglob("realtime.json")
            if path.is_file() and path not in candidates
        )
        rule = (
            f"{simd_dir.relative_to(root)}/kernel/rpf_v3/**/realtime.json"
        )
        if not candidates:
            message = (
                f"SIMD{simd_id}: expected exactly one realtime.json under "
                f"{rule}, found none"
            )
            if alternate_candidates:
                message += (
                    "; files outside kernel/rpf_v3 are intentionally ignored:\n"
                    + _candidate_list(root, alternate_candidates)
                )
            raise ValueError(message)

        raise ValueError(
            f"SIMD{simd_id}: multiple candidates under {rule}; "
            "refusing to choose silently:\n"
            + _candidate_list(root, candidates)
        )

    return discovered


def build_directory_output_paths(
    att_root: Path, output_dir: Optional[Path] = None
) -> dict[int, Path]:
    root = att_root.expanduser().resolve()
    destination = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else root.parent
    )
    return {
        simd_id: destination / f"{root.name}.simd{simd_id}.png"
        for simd_id in SIMD_IDS
    }


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def validate_directory_output_paths(
    att_root: Path,
    output_paths: Mapping[int, Path],
    source_paths: Sequence[Path],
) -> None:
    """Permit repeat overwrites only for the four derived PNG destinations."""
    root = att_root.expanduser().resolve()
    resolved_sources = {path.resolve() for path in source_paths}
    resolved_outputs = [path.resolve() for path in output_paths.values()]
    if len(set(resolved_outputs)) != len(SIMD_IDS):
        raise ValueError("Derived SIMD output paths are not unique")

    for simd_id in SIMD_IDS:
        output = output_paths[simd_id]
        resolved = output.resolve()
        expected_name = f"{root.name}.simd{simd_id}.png"
        if output.name != expected_name:
            raise ValueError(
                f"Unsafe derived output name {output.name!r}; "
                f"expected {expected_name!r}"
            )
        if resolved in resolved_sources:
            raise ValueError(f"Output would overwrite trace source file: {output}")
        if _path_is_within(resolved, root):
            raise ValueError(
                f"Directory-mode output must stay outside the ATT trace root: "
                f"{output}"
            )
        if output.exists() and not output.is_file():
            raise ValueError(
                f"Cannot overwrite non-file output destination: {output}"
            )


def shared_y_limits(
    captures: Sequence[CaptureSummary],
) -> tuple[float, float]:
    maximum = max(
        value
        for capture in captures
        for series in capture.series
        for value in series.interval_mhz
    )
    return (0.0, max(1.0, maximum * 1.05))


def _plot_series(ax: object, series: Sequence[SeriesSummary]) -> None:
    for index, item in enumerate(series):
        ax.plot(
            item.elapsed_us,
            item.interval_mhz,
            color=COLORS[index % len(COLORS)],
            linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            marker=MARKERS[index % len(MARKERS)],
            markevery=20,
            markersize=3,
            linewidth=1.35,
            alpha=0.85,
            label=(
                f"{item.se_name} "
                f"(time-weighted mean {item.weighted_mean_mhz:.1f} MHz)"
            ),
        )


def plot_simd_capture(
    summary: CaptureSummary,
    att_basename: str,
    output: Path,
    y_limits: tuple[float, float],
) -> None:
    """Plot the four SE series observed in one SIMD-select ATT run."""
    if summary.simd_id is None:
        raise ValueError("SIMD id is required for directory-mode plotting")

    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    try:
        _plot_series(ax, summary.series)
        ax.set_title(
            f"{att_basename}\n"
            f"SIMD{summary.simd_id}-select ATT capture × SE clock series\n"
            "per-SE realtime-derived Fgfx\n"
            f"REALTIME reference: "
            f"{summary.capture.reference_hz / 1.0e6:g} MHz "
            f"({summary.capture.reference_source})"
        )
        ax.set_xlabel("Elapsed REALTIME (µs)")
        ax.set_ylabel("Interval-average Fgfx (MHz)")
        ax.set_ylim(*y_limits)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.savefig(output, dpi=180)
    finally:
        plt.close(fig)


def plot_legacy_capture(summary: CaptureSummary, output: Path) -> None:
    """Preserve the original single-input plot layout and naming behavior."""
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    try:
        _plot_series(ax, summary.series)
        ax.set_title(
            "ATT-Derived GFX Clock Frequency by Shader Engine\n"
            f"REALTIME reference: "
            f"{summary.capture.reference_hz / 1.0e6:g} MHz; "
            f"{summary.capture.reference_source}; "
            "one value per adjacent sample pair"
        )
        ax.set_xlabel("Elapsed REALTIME (µs)")
        ax.set_ylabel("Interval-average Fgfx (MHz)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
    finally:
        plt.close(fig)


def _metadata_text(capture: RealtimeCapture) -> str:
    if capture.metadata_hz is None:
        return "missing/invalid"
    return f"{capture.metadata_hz / 1.0e6:.3f} MHz"


def print_directory_summary(
    summaries: Mapping[int, CaptureSummary],
) -> None:
    print("Selected realtime.json files (strict kernel/rpf_v3 rule):")
    for simd_id in SIMD_IDS:
        print(f"  SIMD{simd_id}: {summaries[simd_id].capture.path.resolve()}")

    print("\nReference clocks by SIMD-select ATT capture:")
    for simd_id in SIMD_IDS:
        capture = summaries[simd_id].capture
        print(
            f"  SIMD{simd_id}: {capture.reference_hz / 1.0e6:.3f} MHz "
            f"({capture.reference_source}; "
            f"metadata={_metadata_text(capture)})"
        )
    distinct_references = {
        summary.capture.reference_hz for summary in summaries.values()
    }
    if len(distinct_references) > 1:
        print(
            "  Note: reference frequencies differ across captures; each series "
            "uses its own reference, and pooled results use raw REALTIME-tick "
            "weights."
        )

    print(
        "\nSIMD capture × SE clock series "
        "(delta_realtime<=0 pairs are filtered):"
    )
    print(
        f"{'SIMD':<7} {'SE':<4} {'Samples':>8} {'Valid':>7} "
        f"{'Dropped':>8} {'Ref MHz':>10} {'Mean MHz':>11} "
        f"{'Min MHz':>10} {'Max MHz':>10}"
    )
    print("-" * 94)
    all_series: list[SeriesSummary] = []
    for simd_id in SIMD_IDS:
        for item in summaries[simd_id].series:
            all_series.append(item)
            print(
                f"{f'SIMD{simd_id}':<7} {item.se_name:<4} "
                f"{item.sample_count:>8} {item.valid_intervals:>7} "
                f"{item.dropped_nonpositive_realtime:>8} "
                f"{item.reference_hz / 1.0e6:>10.3f} "
                f"{item.weighted_mean_mhz:>11.3f} "
                f"{item.min_interval_mhz:>10.3f} "
                f"{item.max_interval_mhz:>10.3f}"
            )

    print("\nPer-SIMD capture means (REALTIME-tick weighted across 4 SE series):")
    for simd_id in SIMD_IDS:
        summary = summaries[simd_id]
        print(
            f"  SIMD{simd_id}: {summary.weighted_mean_mhz:.3f} MHz; "
            f"{summary.valid_intervals} valid intervals; "
            f"{summary.dropped_nonpositive_realtime} dropped"
        )

    overall_mean_mhz = tick_weighted_mean(all_series)
    minimum = min(all_series, key=lambda item: item.weighted_mean_mhz)
    maximum = max(all_series, key=lambda item: item.weighted_mean_mhz)
    minimum_simd = next(
        simd_id
        for simd_id in SIMD_IDS
        if minimum in summaries[simd_id].series
    )
    maximum_simd = next(
        simd_id
        for simd_id in SIMD_IDS
        if maximum in summaries[simd_id].series
    )
    spread_mhz = maximum.weighted_mean_mhz - minimum.weighted_mean_mhz
    print(
        "\nOverall (16 SIMD capture × SE clock series, "
        f"REALTIME-tick weighted): {overall_mean_mhz:.3f} MHz"
    )
    print(
        f"Spread of the 16 series means: {spread_mhz:.3f} MHz "
        f"(min SIMD{minimum_simd} {minimum.se_name} "
        f"{minimum.weighted_mean_mhz:.3f}; "
        f"max SIMD{maximum_simd} {maximum.se_name} "
        f"{maximum.weighted_mean_mhz:.3f})"
    )


def run_directory_mode(args: argparse.Namespace) -> list[Path]:
    att_root = args.att_dir.expanduser().resolve()
    realtime_files = discover_realtime_files(att_root)
    summaries = {
        simd_id: summarize_capture(
            load_and_validate_realtime(
                realtime_files[simd_id],
                reference_hz_override=args.reference_hz,
                required_se_names=EXPECTED_SE_NAMES,
            ),
            simd_id=simd_id,
        )
        for simd_id in SIMD_IDS
    }
    output_paths = build_directory_output_paths(att_root, args.output_dir)
    validate_directory_output_paths(
        att_root,
        output_paths,
        list(realtime_files.values()),
    )
    y_limits = shared_y_limits(list(summaries.values()))

    print_directory_summary(summaries)
    output_paths[0].parent.mkdir(parents=True, exist_ok=True)
    for simd_id in SIMD_IDS:
        plot_simd_capture(
            summaries[simd_id],
            att_root.name,
            output_paths[simd_id],
            y_limits,
        )
        if (
            not output_paths[simd_id].is_file()
            or output_paths[simd_id].stat().st_size == 0
        ):
            raise RuntimeError(
                f"Plot was not created as a non-empty file: "
                f"{output_paths[simd_id]}"
            )

    print("\nSaved 4 SIMD-select capture plots (shared y-axis):")
    for simd_id in SIMD_IDS:
        print(f"  {output_paths[simd_id]}")
    return [output_paths[simd_id] for simd_id in SIMD_IDS]


def run_legacy_mode(args: argparse.Namespace) -> Path:
    script_dir = Path(__file__).resolve().parent
    input_path = args.input or default_input_path(script_dir)
    output = args.output or script_dir / "fgfx_all_se.png"
    summary = summarize_capture(
        load_and_validate_realtime(
            input_path,
            reference_hz_override=args.reference_hz,
        )
    )
    plot_legacy_capture(summary, output)
    print(f"Saved plot to {output}")
    return output


def _write_self_test_realtime(path: Path) -> None:
    payload = {
        se_name: [[0, 0], [10, 5]]
        for se_name in EXPECTED_SE_NAMES
    }
    payload["metadata"] = {
        "descriptor": "[gfx_clock, realtime_clock]",
        "frequency": 100,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def run_self_test() -> None:
    weighted = compute_interval_summary(
        "SE0",
        ((0.0, 0.0), (20.0, 2.0), (50.0, 8.0)),
        reference_hz=100.0,
        realtime_origin=0.0,
    )
    if not math.isclose(weighted.weighted_mean_mhz, 0.000625):
        raise AssertionError(
            f"weighted mean mismatch: {weighted.weighted_mean_mhz}"
        )
    print("[PASS] weighted mean uses summed ticks")

    bad_timestamps = compute_interval_summary(
        "SE0",
        (
            (0.0, 10.0),
            (10.0, 20.0),
            (20.0, 20.0),
            (30.0, 15.0),
            (50.0, 25.0),
        ),
        reference_hz=100.0,
        realtime_origin=10.0,
    )
    if (
        bad_timestamps.valid_intervals != 2
        or bad_timestamps.dropped_nonpositive_realtime != 2
        or not math.isclose(
            bad_timestamps.weighted_mean_mhz, 0.000150
        )
    ):
        raise AssertionError(
            "bad timestamp filtering or accounting produced the wrong result"
        )
    print("[PASS] bad timestamps are filtered and counted")

    with tempfile.TemporaryDirectory(prefix="plot_fgfx_self_test_") as temp:
        temp_root = Path(temp)
        att_root = temp_root / "capture.att"
        for simd_id in SIMD_IDS:
            _write_self_test_realtime(
                att_root
                / "thread_trace"
                / f"simd{simd_id}"
                / "kernel"
                / "rpf_v3"
                / f"ui_output_agent_{simd_id}_dispatch_1"
                / "realtime.json"
            )
        discovered = discover_realtime_files(att_root)
        if tuple(discovered) != SIMD_IDS:
            raise AssertionError("unique directory discovery failed")

        output_paths = build_directory_output_paths(att_root)
        expected_names = {
            simd_id: f"capture.att.simd{simd_id}.png"
            for simd_id in SIMD_IDS
        }
        if any(
            output_paths[simd_id].parent != temp_root
            or output_paths[simd_id].name != expected_names[simd_id]
            for simd_id in SIMD_IDS
        ):
            raise AssertionError("derived output naming failed")
        print("[PASS] directory output naming is deterministic")

        _write_self_test_realtime(
            att_root
            / "thread_trace"
            / "simd0"
            / "kernel"
            / "rpf_v3"
            / "ui_output_agent_extra_dispatch_2"
            / "realtime.json"
        )
        try:
            discover_realtime_files(att_root)
        except ValueError as error:
            if "multiple candidates" not in str(error):
                raise AssertionError(
                    f"unexpected candidate-conflict error: {error}"
                ) from error
        else:
            raise AssertionError("candidate conflict was not rejected")
        print("[PASS] directory candidate conflicts are rejected")

    print("Self-test passed: 4 checks")


def main() -> None:
    if sys.platform == "win32":
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.self_test:
            if (
                args.output is not None
                or args.output_dir is not None
                or args.reference_hz is not None
            ):
                parser.error(
                    "--self-test cannot be combined with output or reference "
                    "options"
                )
            run_self_test()
            return

        if args.att_dir is not None:
            if args.output is not None:
                parser.error("--output is only valid with legacy --input mode")
            run_directory_mode(args)
            return

        if args.output_dir is not None:
            parser.error("--output-dir requires --dir")
        run_legacy_mode(args)
    except (AssertionError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
