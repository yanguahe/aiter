#!/usr/bin/env python3
"""Compile the G1 launcher with production tensor layouts, without dispatch."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

os.environ["COMPILE_ONLY"] = "1"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
os.environ["FLYDSL_GPU_ARCH"] = "gfx1250"
os.environ.setdefault("ENABLE_CK", "0")
os.environ.setdefault("AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.mxfp4_preshuffle_gfx1250_tdm_prefill_persistent import (
    launch_gemm_a8w4_tdm_fused_persistent as launcher,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--expert-scheduling-mode", type=int, choices=(0, 1), default=0)
    parser.add_argument("--staged-quant-scale", type=int, choices=(0, 1), default=0)
    parser.add_argument("--row-tail-predication", type=int, choices=(0, 1, 3), default=0)
    parser.add_argument("--half-a-prefetch", type=int, choices=(0, 1), default=0)
    parser.add_argument("--row-tail-a-groups", type=int, choices=(0, 2, 4), default=0)
    parser.add_argument("--half-scale-prefetch", type=int, choices=(0, 1), default=0)
    parser.add_argument("--pair-a-prefetch", type=int, choices=(0, 1), default=0)
    parser.add_argument("--task-m-order", type=int, choices=(0, 1), default=0)
    parser.add_argument("--task-m-group", type=int, choices=(8, 16), default=16)
    parser.add_argument("--tdm-b-th", type=int, choices=(0, 1, 2, 4, 5, 6), default=0)
    parser.add_argument("--tensor-layout", choices=("production", "legacy"),
                        default="production",
                        help="Use production ScaleA dimensions or reproduce the old CPU audit.")
    parser.add_argument("--original-head", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selected_launcher = launcher
    source_path = Path(__file__).resolve().parents[1] / (
        "aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py"
    )
    if args.original_head:
        if vars(args) != dict(vars(args), tasks=4, expert_scheduling_mode=0,
                             staged_quant_scale=0, row_tail_predication=0,
                             half_a_prefetch=0, row_tail_a_groups=0, half_scale_prefetch=0,
                             pair_a_prefetch=0,
                             task_m_order=0,
                             task_m_group=16,
                             tdm_b_th=0):
            parser.error("--original-head cannot be combined with candidate switches")
        snapshot = Path(__file__).resolve().parents[1] / (
            ".codex_tmp/dsr1_e65_gemm1_20261017/baseline_settled/sources/"
            "aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py"
        )
        expected_sha = "b1f0ff87c886915b737cebbb77f61ca67bd213abde41f913c7e9400a251bf6b4"
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected_sha:
            raise RuntimeError("original HEAD snapshot SHA256 mismatch")
        name = "aiter.ops.flydsl.kernels.mxfp4_preshuffle_gfx1250_tdm_prefill_persistent"
        spec = importlib.util.spec_from_file_location(name, snapshot)
        original = importlib.util.module_from_spec(spec)
        sys.modules[name] = original
        spec.loader.exec_module(original)
        selected_launcher = original.launch_gemm_a8w4_tdm_fused_persistent
        source_path = snapshot
    if args.expert_scheduling_mode:
        selected_launcher.compile_hints["llvm_options"]["amdgpu-expert-scheduling-mode"] = True
    candidate_kwargs = {} if args.original_head else dict(
        persistent_n_tasks=args.tasks,
        staged_quant_scale=args.staged_quant_scale,
        row_tail_predication=args.row_tail_predication,
        half_a_prefetch=args.half_a_prefetch,
        row_tail_a_groups=args.row_tail_a_groups,
        half_scale_prefetch=args.half_scale_prefetch,
        pair_a_prefetch=args.pair_a_prefetch,
        task_m_order=args.task_m_order,
        task_m_group=args.task_m_group,
    )
    m = 30464
    with FakeTensorMode():
        output = torch.empty((1, m, 1024), dtype=torch.uint8, device="cpu")
        # The producer allocates uint8 (1, M / 8, (7168 / 32) * 8).
        # The grouped GEMM wrapper views that exact storage as int32.
        scale_a_shape = ((1, m // 8, (7168 // 32) * 8 // 4)
                         if args.tensor_layout == "production"
                         else (1, m // 32, 7168))
        sa = torch.empty(scale_a_shape, dtype=torch.int32, device="cpu")
        sb = torch.empty((65 * 4096 * 7168 // 128,), dtype=torch.int32, device="cpu")
        qs = torch.empty((1, m // 8, 512), dtype=torch.uint8, device="cpu")
        tensor_layouts = {
            name: {"shape": list(tensor.shape), "stride": list(tensor.stride()),
                   "dtype": str(tensor.dtype),
                   "storage_bytes": tensor.numel() * tensor.element_size()}
            for name, tensor in (("output", output), ("scale_a", sa),
                                 ("scale_b", sb), ("quant_scale", qs))
        }
        pointer = flyc.from_c_void_p(fx.Uint8, 0)
        selected_launcher(output, pointer, pointer, sa, sb, m, fx.Stream(None),
                 4096, 7168, 256, 256, 256, 2, 4, 0, 4, 1, pointer, 65, 1, 0,
                 pointer, 7.0, 1, 8, qs, 4, 1, 1, 2, args.tdm_b_th,
                 balanced_m192=0, partial_m_direct_scale=0,
                 partial_m_drain_output=0, **candidate_kwargs)
    values = list(selected_launcher._mem_cache.values())
    assert len(values) == 1, len(values)
    value = values[0]
    compiled = value if hasattr(value, "ir") else value[0]
    ir_text = compiled.ir
    literals = [match.group(1) for match in re.finditer(
        r'"((?:\\[0-9A-Fa-f]{2}|[^"\\]|\\.)*)"', ir_text)
        if match.group(1).startswith("\\7FELF")]
    assert literals and len(set(literals)) == 1
    literal = literals[0]
    data = bytearray()
    index = 0
    while index < len(literal):
        if literal[index] != "\\":
            data.append(ord(literal[index]))
            index += 1
        elif re.fullmatch(r"[0-9A-Fa-f]{2}", literal[index + 1:index + 3]):
            data.append(int(literal[index + 1:index + 3], 16))
            index += 3
        else:
            data.append({"n": 10, "t": 9, "r": 13}.get(literal[index + 1], ord(literal[index + 1])))
            index += 2
    elf = args.output / "kernel.elf"
    elf.write_bytes(data)
    (args.output / "launcher.mlir").write_text(ir_text)
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
    instructions = [line.strip().split("//")[0].strip() for line in isa.splitlines()
                    if re.search(r"// [0-9A-F]{12}:", line)
                    and not line.strip().startswith("s_code_end")]
    counts = Counter(line.split()[0] for line in instructions)
    record = {"tasks": args.tasks,
              "expert_scheduling_mode": args.expert_scheduling_mode,
              "staged_quant_scale": args.staged_quant_scale,
              "row_tail_predication": args.row_tail_predication,
              "half_a_prefetch": args.half_a_prefetch,
              "row_tail_a_groups": args.row_tail_a_groups,
              "half_scale_prefetch": args.half_scale_prefetch,
              "pair_a_prefetch": args.pair_a_prefetch,
              "task_m_order": args.task_m_order,
              "task_m_group": args.task_m_group,
              "tdm_b_th": args.tdm_b_th,
              "tensor_layout": args.tensor_layout, "tensor_layouts": tensor_layouts,
              "original_head": args.original_head,
              "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "elf_sha256": hashlib.sha256(data).hexdigest(),
              "elf_bytes": len(data), "resources": resources,
              "instruction_count": len(instructions), "opcode_counts": dict(counts),
              "no_dispatch": True}
    (args.output / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "opcode_counts"}, indent=2))


if __name__ == "__main__":
    main()
