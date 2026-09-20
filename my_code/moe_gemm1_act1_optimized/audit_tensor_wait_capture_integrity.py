#!/usr/bin/env python3
"""Report every decoded wave's completeness without accepting partial traces."""

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_root", type=Path)
    parser.add_argument("--trace-tool", type=Path, required=True)
    parser.add_argument("--expected-simds", default="0,1,2,3",
                        help="Exact SIMD-select cohort expected in this capture unit")
    parser.add_argument("--scale-pairs-per-task", type=int,
                        help="Optional fixed-workload LDS count check (parent=0, V2-A=224)")
    parser.add_argument("--b128-loads-per-task", type=int, default=1824,
                        help="Expected B128 payload loads per task (parent=1824, V3-A=1796)")
    args = parser.parse_args()
    expected_simds = [int(value) for value in args.expected_simds.split(",")]
    if not expected_simds or len(set(expected_simds)) != len(expected_simds) or not set(expected_simds) <= {0, 1, 2, 3}:
        parser.error("expected-simds must contain unique IDs from 0,1,2,3")
    spec = importlib.util.spec_from_file_location("integrity_trace_tool", args.trace_tool)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    results = []
    for capture in sorted((args.case_root / "thread_trace").glob("simd[0-3]")):
        ui_dirs = list((capture / "kernel/rpf_v3").glob("ui_output_agent_*"))
        if len(ui_dirs) != 1:
            raise RuntimeError(f"Expected one UI directory: {capture}")
        rows = module.load_code(ui_dirs[0])
        starts = [i for i, row in enumerate(rows) if row.isa == "s_add_co_u32 s28, s28, 16"]
        if len(starts) != 1:
            raise RuntimeError(f"Ambiguous boundary: {starts}")
        marker = [row.idx for row in rows[starts[0]:starts[0] + 3]]
        for path in sorted(ui_dirs[0].glob("se*_sm*_sl*_wv*.json")):
            wave = json.loads(path.read_text(encoding="utf-8"))["wave"]
            events = wave["instructions"]
            indices = [int(event[4]) for event in events]
            counts = Counter(rows[index].isa.split()[0] for index in indices)
            boundaries = sum(indices[i:i + 3] == marker for i in range(len(indices) - 2))
            first_markers = indices.count(marker[0])
            wmma = counts["v_wmma_scale_f32_32x16x128_f4"]
            exp = sum(n for op, n in counts.items() if op.startswith("v_exp_f32"))
            rcp = sum(n for op, n in counts.items() if op.startswith("v_rcp_f32"))
            lds_counts = {op: counts[op] for op in
                          ("ds_load_b32", "ds_load_2addr_b32", "ds_load_b128", "ds_store_b64")}
            lds_complete = True
            if args.scale_pairs_per_task is not None:
                paired = 36 * args.scale_pairs_per_task
                lds_complete = lds_counts == {
                    "ds_load_b32": 16416 - 2 * paired, "ds_load_2addr_b32": paired,
                    "ds_load_b128": 36 * args.b128_loads_per_task, "ds_store_b64": 2304}
            results.append({
                "capture": capture.name, "wave": path.name,
                "shader_engine": int(re.match(r"se(\d+)_", path.name)[1]),
                "boundary_triples": boundaries, "boundary_first_instruction": first_markers,
                "wmma": wmma, "exp": exp, "rcp": rcp,
                "lds_counts": lds_counts, "lds_count_check_passed": lds_complete,
                "instruction_count": len(events),
                "first_timestamp": int(events[0][0]),
                "last_timestamp": int(events[-1][0]) + int(events[-1][3]),
                "metadata_begin": wave.get("begin"), "metadata_end": wave.get("end"),
                "complete": boundaries == 36 and first_markers == 36
                and wmma == 64512 and exp == 9216 and rcp == 9216 and lds_complete,
            })
    expected_captures = {f"simd{value}" for value in expected_simds}
    complete = (len(results) == 4 * len(expected_simds)
                and {row["capture"] for row in results} == expected_captures
                and all(row["complete"] for row in results)
                and all(sorted(row["shader_engine"] for row in results if row["capture"] == capture)
                        == [0, 1, 2, 3] for capture in expected_captures))
    print(json.dumps({"case_root": str(args.case_root), "complete": complete,
                      "expected_simd_selects": expected_simds,
                      "wave_count": len(results), "waves": results}, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
