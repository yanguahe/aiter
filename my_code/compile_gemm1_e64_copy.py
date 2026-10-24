#!/usr/bin/env python3
"""Compile the independent E64 GEMM1 with production layouts and no dispatch."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

os.environ["COMPILE_ONLY"] = "1"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
os.environ["FLYDSL_GPU_ARCH"] = "gfx1250"
os.environ["AITER_MOE_EXPERT_BALANCE"] = "false"
os.environ.setdefault("ENABLE_CK", "0")
os.environ.setdefault("AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
import flydsl.compiler as flyc
import flydsl.expr as fx
from my_code.run_gemm1_e64_copy import TUNING, elf_bytes, install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", choices=("baseline", "copy"), required=True)
    parser.add_argument("--tdm-b-th", type=int, default=6, choices=range(7))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    launcher, _, manifest = install(args.variant)
    m = 28672
    tuning = TUNING if args.variant == "copy" else dict(
        balanced_m192=0, partial_m_direct_scale=0, partial_m_drain_output=0)
    with FakeTensorMode():
        output = torch.empty((1, m, 1024), dtype=torch.uint8, device="cpu")
        sa = torch.empty((1, m // 8, 448), dtype=torch.int32, device="cpu")
        sb = torch.empty((64 * 4096 * 7168 // 128,), dtype=torch.int32, device="cpu")
        qs = torch.empty((1, m // 8, 512), dtype=torch.uint8, device="cpu")
        layouts = {name: dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                             dtype=str(tensor.dtype),
                             storage_bytes=tensor.numel() * tensor.element_size())
                   for name, tensor in (("output", output), ("scale_a", sa),
                                        ("scale_b", sb), ("quant_scale", qs))}
        pointer = flyc.from_c_void_p(fx.Uint8, 0)
        launcher(output, pointer, pointer, sa, sb, m, fx.Stream(None),
                 4096, 7168, 256, 256, 256, 2, 4, 0, 4, 1, pointer, 64, 1, 0,
                 pointer, 7.0, 1, 8, qs, 4, 1, 1, 2, args.tdm_b_th, **tuning)
    cached = list(launcher._mem_cache.values())
    if not cached and launcher._last_compiled:
        cached = [launcher._last_compiled[1]]
    assert len(cached) == 1, len(cached)
    compiled = cached[0] if hasattr(cached[0], "ir") else cached[0][0]
    data = elf_bytes(compiled.ir)
    elf = args.output / "kernel.elf"
    elf.write_bytes(data)
    (args.output / "launcher.mlir").write_text(compiled.ir)
    llvm = Path("/data/yanguahe/code/wk_sp1/llvm-project/mlir_install/bin")
    if not (llvm / "llvm-objdump").is_file():
        llvm = Path("/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/llvm/bin")
    notes = subprocess.check_output([str(llvm / "llvm-readelf"), "-n", str(elf)], text=True)
    isa = subprocess.check_output([str(llvm / "llvm-objdump"), "-d", "--mcpu=gfx1250", str(elf)], text=True)
    (args.output / "kernel.notes").write_text(notes)
    (args.output / "kernel.isa").write_text(isa)
    resources = {key: int(re.search(r"\." + key + r":\s+(\d+)", notes).group(1))
                 for key in ("group_segment_fixed_size", "private_segment_fixed_size", "vgpr_count",
                             "sgpr_count", "vgpr_spill_count", "sgpr_spill_count")}
    assert resources["private_segment_fixed_size"] == resources["vgpr_spill_count"] == resources["sgpr_spill_count"] == 0
    instructions = [line.strip().split("//")[0].strip() for line in isa.splitlines()
                    if re.search(r"// [0-9A-F]{12}:", line)
                    and not line.strip().startswith("s_code_end")]
    record = dict(passed=True, variant=args.variant, source_manifest=manifest,
                  capacity_rows=m, experts=64, tuning=tuning, tdm_b_th=args.tdm_b_th,
                  layouts=layouts,
                  elf_sha256=hashlib.sha256(data).hexdigest(), elf_bytes=len(data),
                  resources=resources, instruction_count=len(instructions),
                  opcode_counts=dict(Counter(line.split()[0] for line in instructions)),
                  no_dispatch=True)
    (args.output / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "opcode_counts"}, indent=2))


if __name__ == "__main__":
    main()
