#!/usr/bin/env python3
"""Summarize original profiler rows with per-run ownership qualification."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = []
    for group in sorted(args.root.iterdir()):
        manifest = group / "manifest.json"
        if not manifest.is_file() or json.loads(manifest.read_text()).get("diagnostic"):
            continue
        for result in sorted(group.glob("result_*.json")):
            run = int(result.stem.split("_")[-1])
            status = json.loads(result.read_text())
            logfile = group / f"run_{run}.log"
            log = logfile.read_text(errors="replace")
            rows = [line for line in log.splitlines()
                    if line.startswith("| a4w4") and "random" in line]
            gemm1 = gemm2 = e2e = None
            if rows:
                cells = [cell.strip() for cell in rows[-1].split("|")]
                e2e, gemm1, gemm2 = (float(cells[i]) for i in (13, 14, 15))
            records.append({"group": group.name, "run": run,
                            "rc": status["returncode"],
                            "qualified": status["ownership_qualified"],
                            "gemm1_us": gemm1, "gemm2_us": gemm2,
                            "e2e_us": e2e,
                            "log": str(logfile.relative_to(args.root)),
                            "live_post_owners": status.get("live_post_owners", [])})
    if args.output:
        args.output.write_text(json.dumps(records, indent=2) + "\n")
    for group in sorted({row["group"] for row in records}):
        rows = [row for row in records if row["group"] == group]
        full_group_qualified = (len(rows) == 3 and all(
            row["qualified"] and row["gemm1_us"] is not None for row in rows
        ))
        print(json.dumps({"group": group,
                          "samples": [{k: row[k] for k in ("run", "rc", "qualified",
                                      "gemm1_us", "gemm2_us", "e2e_us")} for row in rows],
                          "full_group_qualified": full_group_qualified,
                          "qualified_gemm1_median": statistics.median(
                              row["gemm1_us"] for row in rows
                          ) if full_group_qualified else None}))


if __name__ == "__main__":
    main()
