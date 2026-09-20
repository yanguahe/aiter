#!/usr/bin/env python3
"""Conservatively audit B128 payload load use in a complete decoded wave.

WMMA conservatively uses all payload registers in its SRC0/SRC1 banks, so
operand-cache/reuse behavior cannot be mistaken for an unused explicit tuple.
Other vector mentions are treated as reads in every bank. Both choices can
hide dead loads, but do not treat an unknown read as evidence of deadness.
This reports candidates for static verification, not permission to remove ISA.
"""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import sys


REG = re.compile(r"\bv\[(\d+):(\d+)\]|\bv(\d+)\b")


def operands(text):
    return [list(range(int(m[1]), int(m[2]) + 1)) if m[3] is None else [int(m[3])]
            for m in REG.finditer(text)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_root", type=Path)
    parser.add_argument("--trace-tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("payload_trace_tool", args.trace_tool)
    tool = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tool
    spec.loader.exec_module(tool)
    summaries, code_table = [], None
    for simd in sorted((args.case_root / "thread_trace").glob("simd[0-3]")):
        ui = list((simd / "kernel/rpf_v3").glob("ui_output_agent_*"))
        if len(ui) != 1:
            raise RuntimeError("Ambiguous UI directory")
        rows = tool.load_code(ui[0])
        banks, bank = {}, 0
        for row in rows:
            if row.isa.startswith("s_set_vgpr_msb "):
                bank = int(row.isa.split()[1], 0) & 255
            banks[row.idx] = bank
        table = [{"isa": row.isa, "bank": banks[row.idx], "pc": f"0x{row.vaddr:x}"}
                 for row in rows if row.isa.startswith("ds_load_b128 ")]
        if code_table is None:
            code_table = table
        elif [(row["isa"], row["bank"]) for row in table] != [(row["isa"], row["bank"]) for row in code_table]:
            raise RuntimeError("Static LDS signatures differ between captures")
        ordinals = {row.idx: i for i, row in enumerate(row for row in rows if row.isa.startswith("ds_load_b128 "))}
        for path in sorted(ui[0].glob("se*_sm*_sl*_wv*.json")):
            events = json.loads(path.read_text())["wave"]["instructions"]
            loads, definitions = [], {}
            task = missing = wmma_count = 0
            for pos, event in enumerate(events):
                row = rows[int(event[4])]
                isa, bank = row.isa, banks[row.idx]
                regs = operands(isa)
                if isa == "s_add_co_u32 s28, s28, 16":
                    task += 1
                if isa.startswith("v_wmma_scale_f32_32x16x128_f4"):
                    wmma_count += 1
                    if len(regs) != 6:
                        raise RuntimeError(f"Unexpected WMMA operand form: {isa}")
                    for values, shift in ((regs[1], 0), (regs[2], 2)):
                        for logical in range(8, 72):
                            physical = logical + 256 * ((bank >> shift) & 3)
                            producer = definitions.get(physical)
                            if producer is not None:
                                loads[producer]["used"] = True
                        missing += sum(definitions.get(logical + 256 * ((bank >> shift) & 3)) is None
                                       for logical in values)
                elif isa.startswith("ds_load_b128 "):
                    if len(regs) != 2:
                        raise RuntimeError(f"Unexpected B128 form: {isa}")
                    load = {"pc": f"0x{row.vaddr:x}", "isa": isa, "task": task,
                            "position": pos, "bank": bank, "ordinal": ordinals[row.idx], "used": False}
                    index = len(loads)
                    loads.append(load)
                    for logical in regs[0]:
                        definitions[logical + 256 * (bank >> 6)] = index
                else:
                    for values in regs:
                        for logical in values:
                            for possible_bank in range(4):
                                producer = definitions.get(logical + 256 * possible_bank)
                                if producer is not None:
                                    loads[producer]["used"] = True
            if task != 36 or wmma_count != 64512 or missing:
                raise RuntimeError(f"Incomplete/ambiguous payload use in {path}: tasks={task}, wmma={wmma_count}, missing={missing}")
            dead = [load for load in loads if not load["used"]]
            by_pc = Counter(load["pc"] for load in dead)
            last_position, occurrences = {}, Counter()
            for load in loads:
                key = (load["task"], load["pc"])
                last_position[key] = load["position"]
                occurrences[key] += 1
            summaries.append({"capture": simd.name, "wave": path.name,
                              "payload_loads": len(loads), "unused_candidates": len(dead),
                              "all_unused_are_last_occurrence_in_task": all(
                                  load["position"] == last_position[(load["task"], load["pc"])]
                                  for load in dead),
                              "unused_site_occurrences_per_task": sorted({
                                  occurrences[(load["task"], load["pc"])] for load in dead}),
                              "unused_per_task": dict(Counter(load["task"] for load in dead)),
                              "unused_sites": [{"pc": pc, "count": count,
                                                **{key: next(load[key] for load in dead if load["pc"] == pc)
                                                   for key in ("isa", "bank", "ordinal")}}
                                               for pc, count in by_pc.items()]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"method": __doc__, "case_root": str(args.case_root.resolve()),
                                      "code_lds_b128": code_table, "waves": summaries}, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in row.items() if k != "unused_sites"} for row in summaries], indent=2))


if __name__ == "__main__":
    main()
