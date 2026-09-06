#!/usr/bin/env python3
"""Measure dispatch time, active cycles, and effective GPU clock with rocprofv3.

The script deliberately performs two independent workload runs:

1. dispatch timing only (``--kernel-trace``);
2. one PMC counter plus the PMC run's own dispatch timestamps.

It never enables ATT, advanced thread trace, or instruction trace.  The value
reported as GHz is always computed from one PMC dispatch's counter value and
that same dispatch's duration.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


DEFAULT_COUNTER = "GRBM_GUI_ACTIVE"
KNOWN_ROCPROFV3_PATHS = (
    Path("/data/yanguahe/code/wk_sp1/rocprof-install/bin/rocprofv3"),
    Path("/opt/rocm/bin/rocprofv3"),
    Path("/opt/rocm/libexec/rocprofiler-sdk/rocprofv3"),
)
ISA_RUNNER_RE = re.compile(r"^gemm(?:_[A-Za-z0-9]+)*_isa_runner\.py$")
SHELL_CONTROL_CHARS = frozenset("|&;<>()")


class MeasurementError(RuntimeError):
    """A user-actionable measurement or parsing failure."""


@dataclass(frozen=True)
class DispatchKey:
    dispatch_id: str
    agent_id: str
    queue_id: str
    kernel_name: str


@dataclass(frozen=True)
class TimingRecord:
    key: DispatchKey
    start_ns: int
    end_ns: int

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True)
class CounterRow:
    key: DispatchKey
    start_ns: int
    end_ns: int
    value: float
    xcc_id: str | None


@dataclass(frozen=True)
class AggregatedCounter:
    key: DispatchKey
    start_ns: int
    end_ns: int
    raw_cycles: float
    per_xcc_cycles: float


@dataclass(frozen=True)
class MetricRecord:
    key: DispatchKey
    start_ns: int
    duration_ns: int
    raw_cycles: float
    per_xcc_cycles: float

    @property
    def ghz(self) -> float:
        return self.per_xcc_cycles / self.duration_ns


@dataclass(frozen=True)
class CommandAdjustment:
    original: str
    profiled: str
    isa_runner: bool
    clang_wrapper: Path | None
    clang_target: Path | None


@dataclass(frozen=True)
class CsvInventory:
    timing_files: tuple[Path, ...]
    counter_files: tuple[Path, ...]
    agent_files: tuple[Path, ...]
    other_csv_files: tuple[Path, ...]
    database_files: tuple[Path, ...]


@dataclass(frozen=True)
class ScratchSnapshot:
    root_existed: bool
    files: Mapping[str, tuple[int, int, int]]
    directories: frozenset[str]


DISPATCH_ALIASES = ("dispatchid", "dispatch")
AGENT_ALIASES = ("agentid", "agent")
QUEUE_ALIASES = ("queueid", "queue")
KERNEL_ALIASES = ("kernelname",)
START_ALIASES = ("starttimestamp", "starttime", "startns")
END_ALIASES = ("endtimestamp", "endtime", "endns")
COUNTER_NAME_ALIASES = ("countername", "counter")
COUNTER_VALUE_ALIASES = ("countervalue", "value")
NUM_XCC_ALIASES = ("numxcc", "numberofxcc", "xcccount")


def normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def normalized_schema(fieldnames: Sequence[str] | None) -> dict[str, str]:
    if not fieldnames:
        return {}
    result: dict[str, str] = {}
    for field in fieldnames:
        normalized = normalize_column(field)
        if normalized and normalized not in result:
            result[normalized] = field
    return result


def schema_field(
    schema: Mapping[str, str],
    aliases: Sequence[str],
    *,
    required: bool = True,
) -> str | None:
    for alias in aliases:
        if alias in schema:
            return schema[alias]
    if required:
        raise MeasurementError(
            f"CSV schema is missing one of {list(aliases)}; "
            f"available columns={list(schema.values())}"
        )
    return None


def _has_fields(schema: Mapping[str, str], alias_groups: Sequence[Sequence[str]]) -> bool:
    return all(any(alias in schema for alias in aliases) for aliases in alias_groups)


def classify_csv_schema(fieldnames: Sequence[str] | None) -> str:
    schema = normalized_schema(fieldnames)
    dispatch_groups = (
        DISPATCH_ALIASES,
        AGENT_ALIASES,
        QUEUE_ALIASES,
        KERNEL_ALIASES,
        START_ALIASES,
        END_ALIASES,
    )
    if _has_fields(
        schema,
        (*dispatch_groups, COUNTER_NAME_ALIASES, COUNTER_VALUE_ALIASES),
    ):
        return "counter"
    if _has_fields(schema, dispatch_groups):
        return "timing"
    if _has_fields(schema, (NUM_XCC_ALIASES,)):
        return "agent"
    return "other"


def read_csv_header(path: Path) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            return tuple(next(reader))
    except StopIteration:
        return ()
    except (OSError, UnicodeError, csv.Error) as exc:
        raise MeasurementError(f"cannot read CSV header {path}: {exc}") from exc


def inventory_outputs(root: Path) -> CsvInventory:
    timing: list[Path] = []
    counter: list[Path] = []
    agent: list[Path] = []
    other: list[Path] = []
    databases: list[Path] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        suffix = path.suffix.lower()
        if suffix == ".csv":
            kind = classify_csv_schema(read_csv_header(path))
            {
                "timing": timing,
                "counter": counter,
                "agent": agent,
                "other": other,
            }[kind].append(path)
        elif suffix in {".db", ".sqlite", ".sqlite3"}:
            databases.append(path)
    return CsvInventory(
        timing_files=tuple(timing),
        counter_files=tuple(counter),
        agent_files=tuple(agent),
        other_csv_files=tuple(other),
        database_files=tuple(databases),
    )


def parse_integral(value: object, description: str) -> int:
    text = str(value).strip()
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise MeasurementError(f"{description} is not numeric: {value!r}") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise MeasurementError(f"{description} is not an integer: {value!r}")
    return int(number)


def parse_float(value: object, description: str) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise MeasurementError(f"{description} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise MeasurementError(f"{description} is not finite: {value!r}")
    return result


def normalize_identifier(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise MeasurementError("encountered an empty dispatch identity field")
    try:
        if text.lower().startswith(("0x", "+0x", "-0x")):
            return str(int(text, 0))
        return str(parse_integral(text, "identity"))
    except MeasurementError:
        return text


def row_key(row: Mapping[str, str], schema: Mapping[str, str]) -> DispatchKey:
    return DispatchKey(
        dispatch_id=normalize_identifier(
            row[schema_field(schema, DISPATCH_ALIASES)]
        ),
        agent_id=normalize_identifier(row[schema_field(schema, AGENT_ALIASES)]),
        queue_id=normalize_identifier(row[schema_field(schema, QUEUE_ALIASES)]),
        kernel_name=row[schema_field(schema, KERNEL_ALIASES)].strip(),
    )


def make_kernel_matcher(
    kernel_name: str, use_regex: bool
) -> tuple[Callable[[str], bool], str]:
    if use_regex:
        try:
            pattern = re.compile(kernel_name)
        except re.error as exc:
            raise MeasurementError(f"invalid --kernel-regex pattern: {exc}") from exc
        return (lambda candidate: pattern.search(candidate) is not None), kernel_name
    exact = kernel_name
    return (lambda candidate: candidate == exact), f"^{re.escape(exact)}$"


def load_timing_records(
    files: Sequence[Path],
    matcher: Callable[[str], bool],
) -> list[TimingRecord]:
    if not files:
        raise MeasurementError("no dispatch timing CSV with the required schema was found")
    records: dict[DispatchKey, TimingRecord] = {}
    distinct_kernel_names: set[str] = set()
    for path in files:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                schema = normalized_schema(reader.fieldnames)
                start_field = schema_field(schema, START_ALIASES)
                end_field = schema_field(schema, END_ALIASES)
                for row_number, row in enumerate(reader, start=2):
                    key = row_key(row, schema)
                    if not matcher(key.kernel_name):
                        continue
                    distinct_kernel_names.add(key.kernel_name)
                    start_ns = parse_integral(
                        row[start_field], f"{path}:{row_number} start timestamp"
                    )
                    end_ns = parse_integral(
                        row[end_field], f"{path}:{row_number} end timestamp"
                    )
                    if end_ns <= start_ns:
                        raise MeasurementError(
                            f"{path}:{row_number} has non-positive dispatch duration"
                        )
                    record = TimingRecord(key, start_ns, end_ns)
                    previous = records.get(key)
                    if previous is not None and previous != record:
                        raise MeasurementError(
                            f"conflicting timing rows for dispatch key {key}"
                        )
                    records[key] = record
        except (OSError, UnicodeError, csv.Error) as exc:
            raise MeasurementError(f"cannot parse timing CSV {path}: {exc}") from exc
    if not records:
        raise MeasurementError("no dispatch matched the requested kernel")
    if len(distinct_kernel_names) != 1:
        raise MeasurementError(
            "kernel expression matched multiple distinct names; refine it to avoid "
            f"mixing kernels: {sorted(distinct_kernel_names)}"
        )
    return sorted(records.values(), key=timing_sort_key)


def _find_xcc_field(schema: Mapping[str, str]) -> str | None:
    candidates = [
        original
        for normalized, original in schema.items()
        if "xcc" in normalized and normalized not in NUM_XCC_ALIASES
    ]
    if len(candidates) > 1:
        raise MeasurementError(
            f"counter CSV has multiple possible XCC dimension columns: {candidates}"
        )
    return candidates[0] if candidates else None


def load_counter_rows(
    files: Sequence[Path],
    matcher: Callable[[str], bool],
    counter_name: str,
) -> list[CounterRow]:
    if not files:
        raise MeasurementError("no counter collection CSV with the required schema was found")
    records: set[CounterRow] = set()
    distinct_kernel_names: set[str] = set()
    observed_counters: set[str] = set()
    for path in files:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                schema = normalized_schema(reader.fieldnames)
                start_field = schema_field(schema, START_ALIASES)
                end_field = schema_field(schema, END_ALIASES)
                counter_field = schema_field(schema, COUNTER_NAME_ALIASES)
                value_field = schema_field(schema, COUNTER_VALUE_ALIASES)
                xcc_field = _find_xcc_field(schema)
                for row_number, row in enumerate(reader, start=2):
                    key = row_key(row, schema)
                    if not matcher(key.kernel_name):
                        continue
                    distinct_kernel_names.add(key.kernel_name)
                    observed = row[counter_field].strip()
                    observed_counters.add(observed)
                    if observed != counter_name:
                        continue
                    start_ns = parse_integral(
                        row[start_field], f"{path}:{row_number} start timestamp"
                    )
                    end_ns = parse_integral(
                        row[end_field], f"{path}:{row_number} end timestamp"
                    )
                    value = parse_float(
                        row[value_field], f"{path}:{row_number} counter value"
                    )
                    if value < 0:
                        raise MeasurementError(
                            f"{path}:{row_number} has a negative counter value"
                        )
                    xcc_id = None
                    if xcc_field is not None and row.get(xcc_field, "").strip():
                        xcc_id = row[xcc_field].strip()
                    records.add(CounterRow(key, start_ns, end_ns, value, xcc_id))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise MeasurementError(f"cannot parse counter CSV {path}: {exc}") from exc
    if not records:
        raise MeasurementError(
            f"no {counter_name!r} rows matched the kernel; observed counters="
            f"{sorted(observed_counters)}"
        )
    if len(distinct_kernel_names) != 1:
        raise MeasurementError(
            "kernel expression matched multiple distinct names in PMC output: "
            f"{sorted(distinct_kernel_names)}"
        )
    return sorted(
        records,
        key=lambda row: (
            row.start_ns,
            sortable_identifier(row.key.dispatch_id),
            row.xcc_id or "",
        ),
    )


def _agent_identity_values(
    row: Mapping[str, str], schema: Mapping[str, str]
) -> set[str]:
    identity_columns = {
        "agentid",
        "id",
        "nodeid",
        "logicalnodeid",
        "logicalid",
    }
    result: set[str] = set()
    for normalized, original in schema.items():
        if normalized in identity_columns and row.get(original, "").strip():
            result.add(normalize_identifier(row[original]))
    return result


def load_num_xcc(
    files: Sequence[Path], target_agent_ids: set[str]
) -> tuple[int, str]:
    if not files:
        raise MeasurementError("no agent_info CSV containing Num_Xcc was found")
    candidates: list[tuple[int, set[str], str]] = []
    for path in files:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                schema = normalized_schema(reader.fieldnames)
                num_field = schema_field(schema, NUM_XCC_ALIASES)
                type_field = schema.get("agenttype") or schema.get("type")
                for row_number, row in enumerate(reader, start=2):
                    if type_field is not None:
                        agent_type = row[type_field].strip().lower()
                        if agent_type and "gpu" not in agent_type:
                            continue
                    num_xcc = parse_integral(
                        row[num_field], f"{path}:{row_number} Num_Xcc"
                    )
                    if num_xcc <= 0:
                        continue
                    candidates.append(
                        (
                            num_xcc,
                            _agent_identity_values(row, schema),
                            f"{path}:{row_number}",
                        )
                    )
        except (OSError, UnicodeError, csv.Error) as exc:
            raise MeasurementError(f"cannot parse agent CSV {path}: {exc}") from exc
    if not candidates:
        raise MeasurementError("agent_info contains no GPU row with positive Num_Xcc")
    direct = [
        item for item in candidates if item[1] and item[1].intersection(target_agent_ids)
    ]
    if direct:
        values = {item[0] for item in direct}
        if len(values) != 1:
            raise MeasurementError(
                f"matching agent_info rows disagree on Num_Xcc: {direct}"
            )
        locations = ", ".join(item[2] for item in direct)
        value = next(iter(values))
        return value, f"matched target Agent_Id in {locations}"
    if len(candidates) == 1:
        value, _, location = candidates[0]
        return value, f"used the only GPU agent row at {location}"
    raise MeasurementError(
        "multiple GPU agent_info rows exist, but none can be mapped uniquely to "
        f"target Agent_Id values {sorted(target_agent_ids)}"
    )


def aggregate_counter_rows(
    rows: Sequence[CounterRow], num_xcc: int
) -> tuple[dict[DispatchKey, AggregatedCounter], str]:
    if num_xcc <= 0:
        raise MeasurementError(f"Num_Xcc must be positive, got {num_xcc}")
    grouped: dict[DispatchKey, list[CounterRow]] = {}
    for row in rows:
        grouped.setdefault(row.key, []).append(row)
    counts = {len(group) for group in grouped.values()}
    result: dict[DispatchKey, AggregatedCounter] = {}
    if counts == {1}:
        method = (
            "counter CSV has exactly one value per dispatch and no per-XCC row "
            f"multiplicity; treating it as a cross-XCC aggregate and dividing by "
            f"Num_Xcc={num_xcc}"
        )
        for key, group in grouped.items():
            row = group[0]
            result[key] = AggregatedCounter(
                key,
                row.start_ns,
                row.end_ns,
                row.value,
                row.value / num_xcc,
            )
        return result, method
    if counts != {num_xcc}:
        raise MeasurementError(
            "counter row multiplicity is ambiguous: expected either one aggregate "
            f"row or Num_Xcc={num_xcc} rows per dispatch, observed {sorted(counts)}"
        )
    for key, group in grouped.items():
        if any(row.xcc_id is None for row in group):
            raise MeasurementError(
                f"dispatch {key} has multiple counter rows but no explicit XCC IDs"
            )
        xcc_ids = {row.xcc_id for row in group}
        if len(xcc_ids) != num_xcc:
            raise MeasurementError(
                f"dispatch {key} has duplicate/missing XCC IDs: {sorted(xcc_ids)}"
            )
        timestamps = {(row.start_ns, row.end_ns) for row in group}
        if len(timestamps) != 1:
            raise MeasurementError(
                f"per-XCC rows disagree on timestamps for dispatch {key}"
            )
        start_ns, end_ns = next(iter(timestamps))
        raw_cycles = sum(row.value for row in group)
        result[key] = AggregatedCounter(
            key,
            start_ns,
            end_ns,
            raw_cycles,
            raw_cycles / num_xcc,
        )
    return (
        result,
        f"counter CSV has {num_xcc} explicitly identified per-XCC rows per "
        "dispatch; summing once for raw cycles and averaging once for per-XCC cycles",
    )


def join_timing_and_counters(
    timings: Sequence[TimingRecord],
    counters: Mapping[DispatchKey, AggregatedCounter],
) -> list[MetricRecord]:
    timing_map = {record.key: record for record in timings}
    if len(timing_map) != len(timings):
        raise MeasurementError("timing records contain duplicate dispatch keys")
    timing_keys = set(timing_map)
    counter_keys = set(counters)
    if timing_keys != counter_keys:
        missing = sorted(
            (key.dispatch_id for key in timing_keys - counter_keys),
            key=sortable_identifier,
        )
        extra = sorted(
            (key.dispatch_id for key in counter_keys - timing_keys),
            key=sortable_identifier,
        )
        raise MeasurementError(
            "PMC timing/counter keys do not match exactly; "
            f"missing counter dispatches={missing}, extra counter dispatches={extra}"
        )
    result: list[MetricRecord] = []
    for timing in sorted(timings, key=timing_sort_key):
        counter = counters[timing.key]
        if (timing.start_ns, timing.end_ns) != (
            counter.start_ns,
            counter.end_ns,
        ):
            raise MeasurementError(
                "PMC counter and kernel trace timestamps disagree for "
                f"dispatch {timing.key.dispatch_id}: "
                f"trace=({timing.start_ns},{timing.end_ns}), "
                f"counter=({counter.start_ns},{counter.end_ns})"
            )
        result.append(
            MetricRecord(
                timing.key,
                timing.start_ns,
                timing.duration_ns,
                counter.raw_cycles,
                counter.per_xcc_cycles,
            )
        )
    return result


def sortable_identifier(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value, 0))
    except ValueError:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)


def timing_sort_key(record: TimingRecord) -> tuple[int, tuple[int, int | str]]:
    return record.start_ns, sortable_identifier(record.key.dispatch_id)


def infer_last_iters(command: str) -> int | None:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as exc:
        raise MeasurementError(f"cannot parse --workload-cmd for --iters: {exc}") from exc
    result: int | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        value: str | None = None
        if token == "--iters" and index + 1 < len(tokens):
            value = tokens[index + 1]
            index += 1
        elif token.startswith("--iters="):
            value = token.split("=", 1)[1]
        if value is not None:
            try:
                parsed = int(value, 10)
            except ValueError:
                parsed = 0
            if parsed > 0:
                result = parsed
        index += 1
    return result


def select_formal_records(
    records: Sequence[TimingRecord] | Sequence[MetricRecord],
    formal_count: int | None,
) -> list[TimingRecord] | list[MetricRecord]:
    ordered = sorted(
        records,
        key=lambda record: (
            record.start_ns,
            sortable_identifier(record.key.dispatch_id),
        ),
    )
    if formal_count is None:
        return list(ordered)
    if formal_count <= 0:
        raise MeasurementError("--formal-count must be positive")
    if len(ordered) < formal_count:
        raise MeasurementError(
            f"requested {formal_count} formal dispatches, but only "
            f"{len(ordered)} target dispatches were captured"
        )
    return list(ordered[-formal_count:])


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    materialized = [float(value) for value in values]
    if not materialized:
        raise MeasurementError("cannot summarize an empty value sequence")
    if any(not math.isfinite(value) for value in materialized):
        raise MeasurementError("cannot summarize non-finite values")
    return {
        "count": len(materialized),
        "min": min(materialized),
        "median": statistics.median(materialized),
        "mean": statistics.fmean(materialized),
        "max": max(materialized),
        "sum": sum(materialized),
    }


def should_cleanup(keep: bool) -> bool:
    return not keep


def _scratch_state(root: Path) -> tuple[dict[str, tuple[int, int, int]], set[str]]:
    files: dict[str, tuple[int, int, int]] = {}
    directories: set[str] = set()
    if not root.exists():
        return files, directories
    if root.is_symlink() or not root.is_dir():
        raise MeasurementError(f"unsafe profiler scratch path: {root}")
    for path in root.rglob("*"):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            raise MeasurementError(f"refusing symlink in profiler scratch: {path}")
        if path.is_dir():
            directories.add(relative)
        elif path.is_file():
            stat = path.stat()
            files[relative] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
    return files, directories


def snapshot_profiler_scratch(workload_cwd: Path) -> ScratchSnapshot:
    root = workload_cwd / ".rocprofv3"
    files, directories = _scratch_state(root)
    return ScratchSnapshot(
        root_existed=root.exists(),
        files=files,
        directories=frozenset(directories),
    )


def quarantine_new_profiler_scratch(
    workload_cwd: Path,
    snapshot: ScratchSnapshot,
    output_root: Path,
) -> list[str]:
    """Move only this run's new rocprof counter scratch under the output root."""

    root = workload_cwd / ".rocprofv3"
    current_files, current_directories = _scratch_state(root)
    for relative, original_stat in snapshot.files.items():
        current_stat = current_files.get(relative)
        if current_stat is None:
            raise MeasurementError(
                f"pre-existing profiler scratch disappeared during the run: {relative}"
            )
        if current_stat != original_stat:
            raise MeasurementError(
                f"pre-existing profiler scratch changed during the run: {relative}"
            )
    new_files = sorted(set(current_files) - set(snapshot.files))
    for relative in new_files:
        source = root / relative
        if not re.fullmatch(r"(?:.*/)?[0-9]+-[0-9]+-counter_values\.dat", relative):
            raise MeasurementError(
                "refusing to move an unrecognized new profiler scratch file: "
                f"{source}"
            )
        destination = output_root / "repo_scratch_quarantine" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    for relative in sorted(
        current_directories - set(snapshot.directories),
        key=lambda value: len(Path(value).parts),
        reverse=True,
    ):
        directory = root / relative
        try:
            directory.rmdir()
        except OSError:
            pass
    if not snapshot.root_existed and root.exists():
        try:
            root.rmdir()
        except OSError:
            pass
    if new_files and not snapshot.root_existed and root.exists():
        raise MeasurementError(
            f"new profiler scratch remains in workload directory: {root}"
        )
    return new_files


def _shell_has_control_operators(command: str) -> bool:
    try:
        lexer = shlex.shlex(
            command,
            posix=True,
            punctuation_chars="".join(sorted(SHELL_CONTROL_CHARS)),
        )
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError as exc:
        raise MeasurementError(f"cannot parse ISA runner command: {exc}") from exc
    return any(token and set(token).issubset(SHELL_CONTROL_CHARS) for token in tokens)


def _extract_default_clang(runner_token: str, workload_cwd: Path) -> Path:
    runner_path = Path(os.path.expandvars(os.path.expanduser(runner_token)))
    if not runner_path.is_absolute():
        runner_path = workload_cwd / runner_path
    source_path = runner_path.resolve().with_name("gemm_isa_runner.py")
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise MeasurementError(
            f"cannot read gemm_isa_runner.DEFAULT_CLANG from {source_path}: {exc}"
        ) from exc
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if not any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_CLANG"
            for target in targets
        ):
            continue
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "Path"
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Constant)
            and isinstance(value.args[0].value, str)
        ):
            return Path(value.args[0].value)
        raise MeasurementError(
            f"{source_path} DEFAULT_CLANG is not a simple Path string literal"
        )
    raise MeasurementError(f"{source_path} does not define DEFAULT_CLANG")


def _write_clang_wrapper(wrapper: Path, clang: Path) -> None:
    content = (
        "#!/usr/bin/env bash\n"
        "# Keep rocprof's preload in the workload, but not in the clang child.\n"
        "unset LD_PRELOAD\n"
        f"exec {shlex.quote(str(clang))} \"$@\"\n"
    )
    wrapper.write_text(content, encoding="utf-8", newline="\n")
    wrapper.chmod(0o700)


def adjust_workload_command(
    command: str,
    workload_cwd: Path,
    output_root: Path,
    *,
    enabled: bool,
) -> CommandAdjustment:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as exc:
        raise MeasurementError(f"cannot parse --workload-cmd: {exc}") from exc
    runner_tokens = [
        token for token in tokens if ISA_RUNNER_RE.fullmatch(Path(token).name)
    ]
    if not runner_tokens or not enabled:
        return CommandAdjustment(command, command, bool(runner_tokens), None, None)
    if len(runner_tokens) != 1:
        raise MeasurementError(
            f"expected one ISA runner in --workload-cmd, found {runner_tokens}"
        )
    if _shell_has_control_operators(command):
        raise MeasurementError(
            "automatic ISA-runner adjustment requires one simple shell command, "
            "not a pipeline/redirection/control expression; restructure the command "
            "or use --no-runner-adjustments only after making it rocprof-safe"
        )
    if "--inmoe" in tokens or "--cudagh" in tokens:
        raise MeasurementError(
            "this ISA runner mode is incompatible with --timing-method cuda-event; "
            "external rocprof cannot safely replace its nested profiler automatically"
        )
    if any(
        token == "--clang" or token.startswith("--clang=") for token in tokens
    ):
        raise MeasurementError(
            "the ISA runner command already supplies --clang; remove it so this "
            "script can wrap gemm_isa_runner.DEFAULT_CLANG, or explicitly use "
            "--no-runner-adjustments if the supplied clang is already preload-safe"
        )
    clang_target = _extract_default_clang(runner_tokens[0], workload_cwd)
    wrapper = output_root / "clang_no_rocprof"
    _write_clang_wrapper(wrapper, clang_target)
    profiled = (
        command.rstrip()
        + " --timing-method cuda-event"
        + f" --clang {shlex.quote(str(wrapper))}"
    )
    return CommandAdjustment(command, profiled, True, wrapper, clang_target)


def resolve_executable(value: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if expanded.is_file():
        return expanded.absolute()
    found = shutil.which(value)
    if found:
        return Path(found).absolute()
    raise MeasurementError(f"executable was not found: {value}")


def discover_rocprofv3(override: str | None) -> Path:
    if override:
        return resolve_executable(override)
    # Prefer the validated workspace/ROCm installations over an unrelated
    # virtual-environment shim.  This also lets us find the workspace's
    # rocprof_env.sh and preserve clang's runtime-library dependencies.
    for candidate in KNOWN_ROCPROFV3_PATHS:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.absolute()
    found = shutil.which("rocprofv3")
    if found:
        return Path(found).absolute()
    raise MeasurementError(
        "rocprofv3 was not found on PATH or at known ROCm/workspace paths; "
        "pass --rocprofv3"
    )


def infer_rocprof_env_script(rocprofv3: Path) -> Path | None:
    if (
        rocprofv3.parent.name == "bin"
        and rocprofv3.parent.parent.name == "rocprof-install"
    ):
        candidate = rocprofv3.parent.parent.parent / "rocprof_env.sh"
        if candidate.is_file():
            return candidate
    return None


def load_sourced_environment(script: Path, base: Mapping[str, str]) -> dict[str, str]:
    bash = shutil.which("bash")
    if not bash:
        raise MeasurementError("bash is required to source the rocprof environment")
    command = [
        bash,
        "-c",
        'set -e; set -a; source "$1" >/dev/null 2>&1; env -0',
        "_",
        str(script),
    ]
    process = subprocess.run(
        command,
        cwd=str(script.parent),
        env=dict(base),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise MeasurementError(
            f"failed to source {script} (exit {process.returncode}): "
            f"{process.stderr.decode(errors='replace')}"
        )
    result: dict[str, str] = {}
    for item in process.stdout.split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode(errors="surrogateescape")] = value.decode(
            errors="surrogateescape"
        )
    if not result:
        raise MeasurementError(f"sourcing {script} produced no environment")
    return result


def prepare_environment(
    rocprofv3: Path, explicit_env_script: str | None, temp_root: Path
) -> tuple[dict[str, str], Path | None]:
    environment = dict(os.environ)
    env_script: Path | None
    if explicit_env_script:
        env_script = Path(
            os.path.expandvars(os.path.expanduser(explicit_env_script))
        ).resolve()
        if not env_script.is_file():
            raise MeasurementError(f"--rocprof-env is not a file: {env_script}")
    else:
        env_script = infer_rocprof_env_script(rocprofv3)
    if env_script is not None:
        environment = load_sourced_environment(env_script, environment)
    for name in tuple(environment):
        upper = name.upper()
        if upper.startswith("ROCPROF") and "ATT" in upper:
            environment.pop(name, None)
    temp_dir = temp_root / "tmp"
    temp_dir.mkdir(mode=0o700)
    environment["TMPDIR"] = str(temp_dir)
    return environment, env_script


def _format_command(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def run_process(
    label: str,
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_root: Path,
    verbose: bool,
    echo_workload_summary: bool = False,
) -> subprocess.CompletedProcess[str]:
    print(f"\n[{label}] command:\n{_format_command(command)}")
    try:
        process = subprocess.run(
            [str(part) for part in command],
            cwd=str(cwd),
            env=dict(environment),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise MeasurementError(f"cannot start {label}: {exc}") from exc
    stdout_path = log_root / f"{label}.stdout.log"
    stderr_path = log_root / f"{label}.stderr.log"
    stdout_path.write_text(process.stdout, encoding="utf-8", newline="\n")
    stderr_path.write_text(process.stderr, encoding="utf-8", newline="\n")
    if verbose:
        if process.stdout:
            print(f"[{label}] stdout:\n{process.stdout}", end="")
        if process.stderr:
            print(f"[{label}] stderr:\n{process.stderr}", end="", file=sys.stderr)
    elif echo_workload_summary:
        print_workload_timing_excerpt(label, process.stdout, process.stderr)
    if process.returncode != 0:
        if not verbose:
            if process.stdout:
                print(f"[{label}] stdout:\n{process.stdout}", end="")
            if process.stderr:
                print(
                    f"[{label}] stderr:\n{process.stderr}",
                    end="",
                    file=sys.stderr,
                )
        raise MeasurementError(
            f"{label} failed with exit code {process.returncode}; "
            f"stdout={stdout_path}, stderr={stderr_path}"
        )
    return process


def print_workload_timing_excerpt(label: str, stdout: str, stderr: str) -> None:
    interesting = re.compile(
        r"cuda-event|profiler count|device_time|timing source|"
        r"captured kernel|tb/s|kernel name",
        re.IGNORECASE,
    )
    stdout_lines = [line for line in stdout.splitlines() if interesting.search(line)]
    stderr_lines = [line for line in stderr.splitlines() if interesting.search(line)]
    print(
        f"[{label}] workload-reported timing excerpt "
        "(informational; never joined across runs):"
    )
    if not stdout_lines and not stderr_lines:
        print("  <no recognized runner timing lines; use --verbose for full output>")
        return
    for line in stdout_lines:
        print(f"  {line}")
    for line in stderr_lines:
        print(f"  stderr: {line}")


def collect_counter_names(root: Path, process: subprocess.CompletedProcess[str]) -> set[str]:
    names: set[str] = set()
    text_chunks = [process.stdout, process.stderr]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
            continue
        if path.suffix.lower() not in {".csv", ".json", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        text_chunks.append(text)
        if path.suffix.lower() == ".csv":
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    for row in csv.reader(handle):
                        for value in row:
                            value = value.strip()
                            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]*", value):
                                names.add(value)
            except (OSError, csv.Error):
                pass
    for text in text_chunks:
        names.update(re.findall(r"\b[A-Za-z][A-Za-z0-9_]{2,}\b", text))
    return names


def verify_counter_available(
    rocprofv3: Path,
    counter_name: str,
    *,
    root: Path,
    environment: Mapping[str, str],
    verbose: bool,
) -> None:
    availability = root / "availability"
    availability.mkdir()
    process = run_process(
        "counter-availability",
        [
            str(rocprofv3),
            "--list-avail",
            "--output-directory",
            str(availability),
            "--output-format",
            "csv",
        ],
        cwd=availability,
        environment=environment,
        log_root=root,
        verbose=verbose,
    )
    names = collect_counter_names(availability, process)
    if counter_name not in names:
        requested_terms = {
            term for term in re.split(r"[^A-Z0-9]+", counter_name.upper()) if term
        }
        candidates = sorted(
            name
            for name in names
            if any(term in name.upper() for term in requested_terms)
            or any(term in name.upper() for term in ("ACTIVE", "CYCLE", "GRBM", "GUI"))
        )
        raise MeasurementError(
            f"counter {counter_name!r} is unavailable; related available "
            f"candidates={candidates[:80]}"
        )
    related = sorted(
        name
        for name in names
        if any(term in name.upper() for term in ("ACTIVE", "CYCLE", "GRBM", "GUI"))
    )
    print(f"Counter availability: exact {counter_name!r} found")
    if verbose:
        print(f"Related available names: {related[:80]}")


def rocprof_command(
    rocprofv3: Path,
    output_directory: Path,
    output_file: str,
    kernel_filter: str,
    shell_command: str,
    *,
    counter_name: str | None,
) -> list[str]:
    command = [str(rocprofv3)]
    if counter_name is not None:
        command.extend(["--pmc", counter_name])
    command.extend(
        [
            "--kernel-trace",
            "--stats",
            "--kernel-include-regex",
            kernel_filter,
            "--output-directory",
            str(output_directory),
            "--output-file",
            output_file,
            "--output-format",
            "csv",
            "--",
            "bash",
            "-lc",
            shell_command,
        ]
    )
    return command


def assert_no_forbidden_artifacts(root: Path) -> None:
    forbidden: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if (
            name.endswith(".att")
            or "advanced_thread_trace" in name
            or "thread_trace" in name
            or name.endswith((".tar", ".tar.gz", ".tgz"))
        ):
            forbidden.append(path)
    if forbidden:
        raise MeasurementError(
            "unexpected ATT/thread-trace/archive artifacts were found: "
            f"{[str(path) for path in forbidden]}"
        )


def _display_inventory(label: str, inventory: CsvInventory) -> None:
    print(
        f"{label} output discovery: timing CSV={len(inventory.timing_files)}, "
        f"counter CSV={len(inventory.counter_files)}, "
        f"agent_info CSV={len(inventory.agent_files)}, "
        f"other CSV={len(inventory.other_csv_files)}, "
        f"DB={len(inventory.database_files)}"
    )
    for kind, files in (
        ("timing", inventory.timing_files),
        ("counter", inventory.counter_files),
        ("agent", inventory.agent_files),
    ):
        for path in files:
            print(f"  {kind}: {path}")


def _print_numeric_summary(
    label: str, summary: Mapping[str, float | int], unit: str
) -> None:
    print(
        f"{label}: count={summary['count']}, "
        f"min={summary['min']:.6f}{unit}, "
        f"median={summary['median']:.6f}{unit}, "
        f"mean={summary['mean']:.6f}{unit}, "
        f"max={summary['max']:.6f}{unit}"
    )


def timing_summary(records: Sequence[TimingRecord]) -> dict[str, float | int]:
    return summarize(record.duration_ns / 1000.0 for record in records)


def metric_summaries(
    records: Sequence[MetricRecord],
) -> dict[str, dict[str, float | int]]:
    return {
        "duration_us": summarize(record.duration_ns / 1000.0 for record in records),
        "raw_cycles": summarize(record.raw_cycles for record in records),
        "per_xcc_cycles": summarize(record.per_xcc_cycles for record in records),
        "ghz": summarize(record.ghz for record in records),
    }


def print_timing_results(
    all_records: Sequence[TimingRecord],
    formal_records: Sequence[TimingRecord],
) -> None:
    print("\nDispatch timing-only results")
    print(
        f"Matched target dispatches: all={len(all_records)}, "
        f"formal(last N)={len(formal_records)}"
    )
    print(
        "Formal dispatch IDs: "
        + ", ".join(record.key.dispatch_id for record in formal_records)
    )
    _print_numeric_summary("All duration", timing_summary(all_records), " us")
    _print_numeric_summary("Formal duration", timing_summary(formal_records), " us")


def print_pmc_results(
    all_records: Sequence[MetricRecord],
    formal_records: Sequence[MetricRecord],
    *,
    num_xcc: int,
    aggregation_method: str,
) -> None:
    print("\nPMC results (duration and cycles are from the same PMC run)")
    print(
        f"Matched target dispatches: all={len(all_records)}, "
        f"formal(last N)={len(formal_records)}"
    )
    print(f"Num_Xcc={num_xcc}")
    print(f"Counter aggregation decision: {aggregation_method}")
    print(
        f"{'Dispatch_Id':>14} {'duration_us':>14} {'raw_cycles':>16} "
        f"{'per_XCC_cycles':>18} {'GHz':>10}"
    )
    for record in formal_records:
        print(
            f"{record.key.dispatch_id:>14} "
            f"{record.duration_ns / 1000.0:14.6f} "
            f"{record.raw_cycles:16.3f} "
            f"{record.per_xcc_cycles:18.3f} "
            f"{record.ghz:10.6f}"
        )
    all_summary = metric_summaries(all_records)
    formal_summary = metric_summaries(formal_records)
    print("\nAll-dispatch PMC summary")
    for key, unit in (
        ("duration_us", " us"),
        ("raw_cycles", " cycles"),
        ("per_xcc_cycles", " cycles"),
        ("ghz", " GHz"),
    ):
        _print_numeric_summary(key, all_summary[key], unit)
    print(f"raw_cycles total={all_summary['raw_cycles']['sum']:.3f} cycles")
    print("\nFormal-dispatch PMC summary")
    for key, unit in (
        ("duration_us", " us"),
        ("raw_cycles", " cycles"),
        ("per_xcc_cycles", " cycles"),
        ("ghz", " GHz"),
    ):
        _print_numeric_summary(key, formal_summary[key], unit)
    print(f"raw_cycles total={formal_summary['raw_cycles']['sum']:.3f} cycles")


def _serializable_metric(record: MetricRecord) -> dict[str, object]:
    return {
        "dispatch_id": record.key.dispatch_id,
        "agent_id": record.key.agent_id,
        "queue_id": record.key.queue_id,
        "kernel_name": record.key.kernel_name,
        "duration_us": record.duration_ns / 1000.0,
        "raw_cycles": record.raw_cycles,
        "per_xcc_cycles": record.per_xcc_cycles,
        "ghz": record.ghz,
    }


def write_requested_outputs(
    args: argparse.Namespace,
    payload: Mapping[str, object],
    formal_metrics: Sequence[MetricRecord],
    output_root: Path,
) -> None:
    for requested in (args.json_output, args.csv_output):
        if not requested:
            continue
        destination = Path(requested).expanduser().resolve()
        try:
            destination.relative_to(output_root.resolve())
        except ValueError:
            pass
        else:
            if not args.keep:
                raise MeasurementError(
                    f"requested output {destination} is inside temporary --outdir "
                    "and would be deleted; use --keep or choose another path"
                )
        destination.parent.mkdir(parents=True, exist_ok=True)
    if args.json_output:
        destination = Path(args.json_output).expanduser().resolve()
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"JSON summary: {destination}")
    if args.csv_output:
        destination = Path(args.csv_output).expanduser().resolve()
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "dispatch_id",
                    "agent_id",
                    "queue_id",
                    "kernel_name",
                    "duration_us",
                    "raw_cycles",
                    "per_xcc_cycles",
                    "ghz",
                ],
            )
            writer.writeheader()
            for record in formal_metrics:
                writer.writerow(_serializable_metric(record))
        print(f"Formal-dispatch CSV: {destination}")


def create_output_root(requested: str | None) -> Path:
    if requested is None:
        return Path(tempfile.mkdtemp(prefix="measure_kernel_cycles_", dir="/tmp"))
    root = Path(os.path.expandvars(os.path.expanduser(requested))).resolve()
    if root.exists():
        raise MeasurementError(
            f"--outdir already exists; refusing to delete or mix with it: {root}"
        )
    root.mkdir(parents=True, mode=0o700)
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workload-cmd",
        help=(
            "shell command to profile; it is intentionally executed by "
            "/bin/bash -lc and may contain environment assignments and spaces"
        ),
    )
    parser.add_argument(
        "--kernel-name",
        help="exact kernel name by default; see --kernel-regex",
    )
    parser.add_argument(
        "--kernel-regex",
        action="store_true",
        help="interpret --kernel-name as a regex (default: exact escaped match)",
    )
    parser.add_argument(
        "--counter",
        default=DEFAULT_COUNTER,
        help=f"active-cycle counter (default: {DEFAULT_COUNTER})",
    )
    parser.add_argument(
        "--outdir",
        help=(
            "new output directory (default: a unique /tmp directory); an existing "
            "path is rejected for safe cleanup"
        ),
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="retain the output directory and diagnostic logs",
    )
    parser.add_argument(
        "--formal-count",
        type=int,
        help=(
            "take the final N matched dispatches as formal; default: infer the "
            "last --iters value, or use all matches when no literal value exists"
        ),
    )
    parser.add_argument(
        "--rocprofv3",
        help="rocprofv3 executable override",
    )
    parser.add_argument(
        "--rocprof-env",
        help=(
            "shell environment file to source before rocprof; by default, "
            "rocprof_env.sh is auto-detected beside a workspace rocprof-install"
        ),
    )
    parser.add_argument(
        "--workdir",
        help="workload working directory (default: current directory)",
    )
    parser.add_argument(
        "--no-runner-adjustments",
        action="store_true",
        help=(
            "do not add cuda-event timing and the clang preload-isolation wrapper "
            "for recognized gemm_*_isa_runner.py commands"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print complete subprocess stdout/stderr and counter candidates",
    )
    parser.add_argument(
        "--json-output",
        help="write a persistent JSON summary to this explicit path",
    )
    parser.add_argument(
        "--csv-output",
        help="write persistent formal-dispatch metrics to this explicit CSV path",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run CPU-only pure-function/parser tests and exit",
    )
    return parser


def _self_test() -> None:
    assert (
        infer_last_iters(
            "AITER_LOG_MORE=1 python runner.py --iters 7 --shape '64, 64, 64' "
            "--iters=20"
        )
        == 20
    )
    exact, exact_filter = make_kernel_matcher("foo.bar+1", False)
    assert exact("foo.bar+1")
    assert not exact("prefix_foo.bar+1")
    assert exact_filter == r"^foo\.bar\+1$"
    regex_matcher, _ = make_kernel_matcher(r"foo_[0-9]+", True)
    assert regex_matcher("prefix foo_12 suffix")
    assert not regex_matcher("foo_x")

    with tempfile.TemporaryDirectory(prefix="measure_cycles_selftest_") as temp:
        root = Path(temp)
        timing_path = root / "arbitrary_timing_name.csv"
        counter_path = root / "nested" / "arbitrary_counter_name.csv"
        agent_path = root / "nested" / "arbitrary_agent_name.csv"
        counter_path.parent.mkdir()
        timing_header = [
            "Dispatch_Id",
            "Agent_Id",
            "Queue_Id",
            "Kernel_Name",
            "Start_Timestamp",
            "End_Timestamp",
        ]
        with timing_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=timing_header)
            writer.writeheader()
            writer.writerow(
                {
                    "Dispatch_Id": "9",
                    "Agent_Id": "1",
                    "Queue_Id": "2",
                    "Kernel_Name": "target",
                    "Start_Timestamp": "1000",
                    "End_Timestamp": "2000",
                }
            )
        counter_header = timing_header + ["Counter_Name", "Counter_Value"]
        with counter_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=counter_header)
            writer.writeheader()
            writer.writerow(
                {
                    "Dispatch_Id": "9",
                    "Agent_Id": "1",
                    "Queue_Id": "2",
                    "Kernel_Name": "target",
                    "Start_Timestamp": "1000",
                    "End_Timestamp": "2000",
                    "Counter_Name": DEFAULT_COUNTER,
                    "Counter_Value": "8000",
                }
            )
        with agent_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["Agent_Type", "Agent_Id", "Num_Xcc"]
            )
            writer.writeheader()
            writer.writerow(
                {"Agent_Type": "GPU", "Agent_Id": "1", "Num_Xcc": "8"}
            )
        inventory = inventory_outputs(root)
        assert inventory.timing_files == (timing_path,)
        assert inventory.counter_files == (counter_path,)
        assert inventory.agent_files == (agent_path,)
        matcher, _ = make_kernel_matcher("target", False)
        timings = load_timing_records(inventory.timing_files, matcher)
        counter_rows = load_counter_rows(
            inventory.counter_files, matcher, DEFAULT_COUNTER
        )
        num_xcc, _ = load_num_xcc(inventory.agent_files, {"1"})
        aggregated, method = aggregate_counter_rows(counter_rows, num_xcc)
        assert "one value per dispatch" in method
        metrics = join_timing_and_counters(timings, aggregated)
        assert len(metrics) == 1
        assert metrics[0].raw_cycles == 8000
        assert metrics[0].per_xcc_cycles == 1000
        assert metrics[0].ghz == 1.0

        bad_timing = [TimingRecord(timings[0].key, 1001, 2000)]
        try:
            join_timing_and_counters(bad_timing, aggregated)
        except MeasurementError:
            pass
        else:
            raise AssertionError("timestamp mismatch must fail")

        workload = root / "workload"
        output = root / "output"
        workload.mkdir()
        output.mkdir()
        scratch = workload / ".rocprofv3"
        scratch.mkdir()
        preexisting = scratch / "preexisting.dat"
        preexisting.write_bytes(b"keep")
        snapshot = snapshot_profiler_scratch(workload)
        generated = scratch / "10-20-counter_values.dat"
        generated.write_bytes(b"move")
        moved = quarantine_new_profiler_scratch(workload, snapshot, output)
        assert moved == ["10-20-counter_values.dat"]
        assert preexisting.read_bytes() == b"keep"
        assert not generated.exists()
        assert (
            output
            / "repo_scratch_quarantine"
            / "10-20-counter_values.dat"
        ).read_bytes() == b"move"

    key = DispatchKey("10", "1", "2", "target")
    per_xcc_rows = [
        CounterRow(key, 100, 200, 100.0 + index, str(index))
        for index in range(8)
    ]
    per_xcc, method = aggregate_counter_rows(per_xcc_rows, 8)
    assert "explicitly identified per-XCC" in method
    assert per_xcc[key].raw_cycles == sum(100.0 + index for index in range(8))
    assert per_xcc[key].per_xcc_cycles == statistics.fmean(
        100.0 + index for index in range(8)
    )

    values = summarize([1.0, 2.0, 9.0])
    assert values == {
        "count": 3,
        "min": 1.0,
        "median": 2.0,
        "mean": 4.0,
        "max": 9.0,
        "sum": 12.0,
    }
    assert should_cleanup(False)
    assert not should_cleanup(True)
    print(
        "SELF_TEST_OK: command/last-iters parsing, exact/regex kernel matching, "
        "recursive CSV schema discovery, key+timestamp join, aggregate/per-XCC "
        "cycle handling, statistics, cleanup decision, and scoped repo-scratch "
        "quarantine"
    )


def run_measurement(args: argparse.Namespace) -> int:
    if not args.workload_cmd:
        raise MeasurementError("--workload-cmd is required unless --self-test is used")
    if not args.kernel_name:
        raise MeasurementError("--kernel-name is required unless --self-test is used")
    if args.formal_count is not None and args.formal_count <= 0:
        raise MeasurementError("--formal-count must be positive")
    workload_cwd = (
        Path(args.workdir).expanduser().resolve()
        if args.workdir
        else Path.cwd().resolve()
    )
    if not workload_cwd.is_dir():
        raise MeasurementError(f"--workdir is not a directory: {workload_cwd}")

    output_root = create_output_root(args.outdir)
    scratch_snapshot = snapshot_profiler_scratch(workload_cwd)
    print(f"Temporary output root: {output_root}")
    print(
        "SECURITY BOUNDARY: --workload-cmd is user-supplied shell code and will "
        "execute with current-user privileges via /bin/bash -lc."
    )
    try:
        matcher, kernel_filter = make_kernel_matcher(
            args.kernel_name, args.kernel_regex
        )
        inferred_count = infer_last_iters(args.workload_cmd)
        formal_count = (
            args.formal_count if args.formal_count is not None else inferred_count
        )
        count_source = (
            "explicit --formal-count"
            if args.formal_count is not None
            else "last literal --iters"
            if inferred_count is not None
            else "all matched dispatches (no literal --iters)"
        )
        adjustment = adjust_workload_command(
            args.workload_cmd,
            workload_cwd,
            output_root,
            enabled=not args.no_runner_adjustments,
        )
        print(f"Kernel matching mode: {'regex' if args.kernel_regex else 'exact'}")
        print(f"rocprof kernel filter: {kernel_filter}")
        print(f"Formal selection: {formal_count or 'all'} from {count_source}")
        print(f"Original workload command:\n{adjustment.original}")
        print(f"Actual profiled workload command:\n{adjustment.profiled}")
        if adjustment.clang_wrapper is not None:
            print(
                "ISA-runner adjustment: appended --timing-method cuda-event and "
                f"--clang {adjustment.clang_wrapper}"
            )
            print(
                "Clang wrapper behavior: unset LD_PRELOAD only in the clang child, "
                f"then exec gemm_isa_runner.DEFAULT_CLANG={adjustment.clang_target}"
            )
        elif adjustment.isa_runner and args.no_runner_adjustments:
            print(
                "WARNING: ISA-runner adjustments were explicitly disabled; nested "
                "profiler/preload conflicts are the user's responsibility."
            )
        else:
            print("Non-ISA-runner workload: no unknown arguments were added.")

        rocprofv3 = discover_rocprofv3(args.rocprofv3)
        environment, env_script = prepare_environment(
            rocprofv3, args.rocprof_env, output_root
        )
        print(f"rocprofv3: {rocprofv3}")
        print(f"rocprof environment: {env_script or '<inherited environment>'}")
        version = run_process(
            "rocprof-version",
            [str(rocprofv3), "--version"],
            cwd=output_root,
            environment=environment,
            log_root=output_root,
            verbose=args.verbose,
        )
        version_text = (version.stdout + "\n" + version.stderr).strip()
        print(f"rocprofv3 version output: {version_text}")
        verify_counter_available(
            rocprofv3,
            args.counter,
            root=output_root,
            environment=environment,
            verbose=args.verbose,
        )

        shell_command = (
            f"cd -- {shlex.quote(str(workload_cwd))} && {adjustment.profiled}"
        )
        timing_dir = output_root / "timing"
        timing_dir.mkdir()
        run_process(
            "timing-run",
            rocprof_command(
                rocprofv3,
                timing_dir,
                "timing",
                kernel_filter,
                shell_command,
                counter_name=None,
            ),
            cwd=timing_dir,
            environment=environment,
            log_root=output_root,
            verbose=args.verbose,
            echo_workload_summary=True,
        )
        timing_inventory = inventory_outputs(timing_dir)
        _display_inventory("Timing run", timing_inventory)
        timing_records = load_timing_records(
            timing_inventory.timing_files, matcher
        )
        formal_timing = select_formal_records(timing_records, formal_count)
        print_timing_results(timing_records, formal_timing)

        pmc_dir = output_root / "pmc"
        pmc_dir.mkdir()
        run_process(
            "pmc-run",
            rocprof_command(
                rocprofv3,
                pmc_dir,
                "pmc",
                kernel_filter,
                shell_command,
                counter_name=args.counter,
            ),
            cwd=pmc_dir,
            environment=environment,
            log_root=output_root,
            verbose=args.verbose,
            echo_workload_summary=True,
        )
        pmc_inventory = inventory_outputs(pmc_dir)
        _display_inventory("PMC run", pmc_inventory)
        pmc_timings = load_timing_records(pmc_inventory.timing_files, matcher)
        counter_rows = load_counter_rows(
            pmc_inventory.counter_files, matcher, args.counter
        )
        target_agents = {record.key.agent_id for record in pmc_timings}
        if len(target_agents) != 1:
            raise MeasurementError(
                f"target dispatches span multiple agents: {sorted(target_agents)}"
            )
        num_xcc, agent_decision = load_num_xcc(
            pmc_inventory.agent_files, target_agents
        )
        counters, aggregation_method = aggregate_counter_rows(
            counter_rows, num_xcc
        )
        metrics = join_timing_and_counters(pmc_timings, counters)
        formal_metrics = select_formal_records(metrics, formal_count)
        print(f"Num_Xcc source: {agent_decision}")
        print_pmc_results(
            metrics,
            formal_metrics,
            num_xcc=num_xcc,
            aggregation_method=aggregation_method,
        )

        assert_no_forbidden_artifacts(output_root)
        print(
            "\nArtifact check: no .att, advanced/thread-trace, or trace archive "
            "was produced."
        )
        payload: dict[str, object] = {
            "rocprofv3": str(rocprofv3),
            "rocprofv3_version_output": version_text,
            "counter": args.counter,
            "kernel_name_argument": args.kernel_name,
            "kernel_match_mode": "regex" if args.kernel_regex else "exact",
            "original_workload_command": adjustment.original,
            "profiled_workload_command": adjustment.profiled,
            "formal_count": len(formal_metrics),
            "formal_count_source": count_source,
            "timing_only": {
                "all_count": len(timing_records),
                "formal_count": len(formal_timing),
                "all_duration_us": timing_summary(timing_records),
                "formal_duration_us": timing_summary(formal_timing),
                "formal_dispatch_ids": [
                    record.key.dispatch_id for record in formal_timing
                ],
            },
            "pmc": {
                "all_count": len(metrics),
                "formal_count": len(formal_metrics),
                "num_xcc": num_xcc,
                "num_xcc_source": agent_decision,
                "aggregation_method": aggregation_method,
                "all_summary": metric_summaries(metrics),
                "formal_summary": metric_summaries(formal_metrics),
                "formal_dispatches": [
                    _serializable_metric(record) for record in formal_metrics
                ],
            },
            "att_or_thread_trace_enabled": False,
        }
        write_requested_outputs(args, payload, formal_metrics, output_root)
        print(
            "\nMeasurement complete: timing-only data remains separate; every GHz "
            "value used cycles and duration from the same PMC dispatch."
        )
        return 0
    finally:
        quarantined = quarantine_new_profiler_scratch(
            workload_cwd, scratch_snapshot, output_root
        )
        if quarantined:
            print(
                "Repo scratch cleanup: moved this run's generated files under "
                f"the output root: {quarantined}"
            )
        if should_cleanup(args.keep):
            try:
                shutil.rmtree(output_root, ignore_errors=False)
            except OSError as exc:
                raise MeasurementError(
                    f"failed to remove temporary output root {output_root}: {exc}"
                ) from exc
            print(f"Cleanup confirmed: removed {output_root}")
        else:
            print(f"Diagnostics retained by --keep: {output_root}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    try:
        return run_measurement(args)
    except MeasurementError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if not args.keep:
            print(
                "Diagnostics were cleaned by default; rerun with --keep to retain "
                "rocprof output and subprocess logs.",
                file=sys.stderr,
            )
        return 1
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
