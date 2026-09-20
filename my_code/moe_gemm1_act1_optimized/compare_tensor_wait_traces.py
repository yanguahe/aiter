#!/usr/bin/env python3
"""Compare complete canonical-parser results and independently validate CSV waits."""

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path


def summarize(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    whole = data["whole_wave"]
    waves = data["waves"]
    assert whole["wave_count"] == len(waves) == 16
    assert data["task_distribution"]["count"] == 560
    assert len(data["anchors"]) == len(data["occupancy"]) == 4
    assert all(w["task_count"] == 36 and w["wmma_count"] == 64512
               and w["exp_count"] == w["rcp_count"] == 9216 for w in waves)
    assert all(owner["waves"] == 4 for owner in data["owners"])
    forms, families = Counter(), Counter()
    for anchor in data["anchors"]:
        with Path(anchor["stats_csv"]).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                isa = " ".join(row["Instruction"].split())
                if isa.startswith("s_wait") or isa.startswith("s_barrier_wait"):
                    stall = int(row["Stall"])
                    forms[isa] += stall
                    families[isa.split()[0]] += stall
    decoded = {op: round(values["cycles_per_wave"] * len(waves))
               for op, values in whole["wait_stall"].items()
               if values["cycles_per_wave"]}
    assert {op: n for op, n in families.items() if n} == decoded, (families, decoded)
    total = whole["total_observed_wave_cycles"]
    assert sum(whole["issue_timeline"][key]["cycles_per_wave"]
               for key in whole["issue_timeline"]) == whole["mean_cycles_per_wave"]
    numbers = {"whole_wave_cycles": whole["mean_cycles_per_wave"],
               "body_task_mean": data["task_distribution"]["mean"],
               "body_task_median": data["task_distribution"]["median"],
               **data["partition_mean_cycles"]}
    for prefix, field in (("wait", "wait_stall"), ("issue", "issue_timeline")):
        numbers.update({f"{prefix}:{name}": value["cycles_per_wave"]
                        for name, value in whole[field].items()})
    completion = [row for cap in data["occupancy"] for row in cap["completion"]]
    configs = []
    for simd in range(4):
        text = (Path(data["case_root"]) / f"input_simd{simd}.yaml").read_text()
        configs.append({"target_cu": int(re.search(r"att_target_cu:\s*(\d+)", text)[1]),
                        "se_mask": re.search(r'att_shader_engine_mask:\s*"?(0x[0-9a-fA-F]+)', text)[1]})
    assert all(config == configs[0] for config in configs)
    return {
        "metrics_source": str(path), "case_root": data["case_root"],
        "sampling": configs[0], "wave_count": len(waves),
        "csv_wait_validation": "exact match", "numbers": numbers,
        "wait_shares": whole["wait_stall"],
        "wait_forms": {isa: {"cycles_per_wave": cycles / len(waves),
                              "percent": 100 * cycles / total}
                       for isa, cycles in forms.most_common() if cycles},
        "lds_instructions_per_wave": {
            op: whole["opcode_counts"].get(op, 0) / len(waves)
            for op in ("ds_load_b32", "ds_load_2addr_b32", "ds_load_b128", "ds_store_b64")},
        "owner_body_cycles": {owner["role"]: owner["body_mean_cycles_per_task"]
                              for owner in data["owners"]},
        "completion_imbalance_percent": {
            "mean": 100 * statistics.mean(row["completion_imbalance"] for row in completion),
            "max": 100 * max(row["completion_imbalance"] for row in completion)},
        "physical_wgps_per_capture": [len(cap["physical_wgps"]) for cap in data["occupancy"]],
        "mean_gfx_mhz": statistics.mean(cap["weighted_mean_gfx_mhz"]
                                         for cap in data["occupancy"]),
        "top_wait_sites": whole["top_wait_sites"][:8],
        "top_nonwait_max_latency_sites": whole["top_nonwait_max_latency_sites"][:6],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent, candidate = summarize(args.parent), summarize(args.candidate)
    assert parent["sampling"] == candidate["sampling"]
    changes = {}
    for key in sorted(parent["numbers"].keys() | candidate["numbers"].keys()):
        before = parent["numbers"].get(key, 0)
        after = candidate["numbers"].get(key, 0)
        changes[key] = {"parent": before, "candidate": after,
                        "change_percent": 100 * (after / before - 1) if before else None}
    result = {"method": "Complete per-wave exposures; instruction latency is not additive.",
              "parent": parent, "candidate": candidate, "changes": changes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(changes, indent=2))


if __name__ == "__main__":
    main()
