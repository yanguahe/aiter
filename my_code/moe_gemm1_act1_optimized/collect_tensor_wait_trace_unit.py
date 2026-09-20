#!/usr/bin/env python3
"""Register the first complete, uncontended SIMD capture without altering raw data."""

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--simd", type=int, choices=range(4), required=True)
    parser.add_argument("--isa-sha", required=True)
    parser.add_argument("--scale-pairs-per-task", type=int, required=True)
    parser.add_argument("--b128-loads-per-task", type=int, default=1824)
    parser.add_argument("--trace-tool", type=Path, required=True)
    args = parser.parse_args()
    for value in (args.tag, args.unit, args.case):
        if not re.fullmatch(r"[A-Za-z0-9_]+", value):
            parser.error("Tags, units, and cases must be simple identifiers")
    here = Path(__file__).resolve().parent
    root = args.root.resolve()
    if not root.is_relative_to(here / "history_runs"):
        parser.error("Evidence root must be inside this task's history_runs")
    monitor = json.loads((root / f"{args.unit}_monitor.json").read_text())
    if monitor.get("kfd_gate_passed") is not True or monitor.get("completion_exit_code") != 0:
        raise SystemExit("Unit did not pass the contention/completion gate")
    log = (root / f"{args.unit}.log").read_text(encoding="utf-8")
    dates = re.findall(r"^date_utc=(\d{8}T\d{6}Z)$", log, re.M)
    if len(dates) != 1 or log.rstrip().splitlines()[-1] != "EXIT_CODE=0":
        raise SystemExit("Ambiguous or unfinished capture log")
    run = f"heliosr-1b114-a07-3_{dates[0]}_att"
    run_root = here / "history_runs" / run
    with (run_root / "att.tsv").open(newline="") as handle:
        entries = [row for row in csv.DictReader(handle, delimiter="\t") if row["case"] == args.case]
    if len(entries) != 1 or entries[0]["return_code"] != "0" or entries[0]["isa_sha256"] != args.isa_sha:
        raise SystemExit("Case return code or ISA checksum does not match")
    case_root = run_root / "att" / args.case
    yaml = (case_root / f"input_simd{args.simd}.yaml").read_text()
    if not re.search(r"att_target_cu:\s*0\b", yaml) or not re.search(r'att_shader_engine_mask:\s*"?0xf\b', yaml):
        raise SystemExit("Unexpected sampling point or shader-engine mask")
    integrity = root / f"{args.unit}_integrity.json"
    with integrity.open("w", encoding="utf-8") as handle:
        result = subprocess.run([sys.executable, str(here / "audit_tensor_wait_capture_integrity.py"),
                                 str(case_root), "--trace-tool", str(args.trace_tool),
                                 "--expected-simds", str(args.simd), "--scale-pairs-per-task",
                                 str(args.scale_pairs_per_task), "--b128-loads-per-task",
                                 str(args.b128_loads_per_task)], stdout=handle)
    integrity.chmod(0o666)
    if result.returncode:
        raise SystemExit("Decoded unit is incomplete; raw data retained and view not changed")
    view = root / f"{args.tag}_complete_trace_view" / args.case
    (view / "thread_trace").mkdir(parents=True, exist_ok=True)
    for relative in (Path("thread_trace") / f"simd{args.simd}", Path(f"input_simd{args.simd}.yaml")):
        link, source = view / relative, case_root / relative
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != source.resolve():
                raise SystemExit(f"A different first qualifying unit is already registered: {link}")
        else:
            link.symlink_to(source, target_is_directory=source.is_dir())
    manifest = root / f"{args.tag}_capture_units.json"
    payload = json.loads(manifest.read_text()) if manifest.exists() else {
        "method": "First uncontended complete case/SIMD unit; raw wave fragments are never spliced.",
        "sampling": {"target_cu": 0, "shader_engine_mask": "0xf"}, "units": []}
    unit = {"case": args.case, "simd": args.simd, "run": run,
            "unit_log": f"{args.unit}.log", "monitor": f"{args.unit}_monitor.json",
            "integrity": integrity.name, "complete_waves": 4,
            "expected_scale_pairs_per_task": args.scale_pairs_per_task,
            "expected_b128_loads_per_task": args.b128_loads_per_task, "isa_sha256": args.isa_sha}
    matches = [row for row in payload["units"] if (row["case"], row["simd"]) == (args.case, args.simd)]
    if matches and any(row["run"] != run for row in matches):
        raise SystemExit("Manifest already selects a different unit")
    if not matches:
        payload["units"].append(unit)
    payload["units"].sort(key=lambda row: (row["case"], row["simd"]))
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    manifest.chmod(0o666)
    print(json.dumps(unit, indent=2))


if __name__ == "__main__":
    main()
