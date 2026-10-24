#!/usr/bin/env python3
"""Run the unchanged E64 benchmark with an isolated copy of accepted GEMM1."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import sys

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / ".codex_tmp/side_e64_gemm1_20261018"
KERNEL = "aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_gemm1_e64.py"
TUNING = dict(
    persistent_n_tasks=1,
    staged_quant_scale=1,
    row_tail_predication=3,
    half_a_prefetch=1,
    half_scale_prefetch=1,
    row_tail_a_groups=2,
    pair_a_prefetch=1,
    task_m_order=1,
    task_m_group=8,
    balanced_m192=0,
    partial_m_direct_scale=0,
    partial_m_drain_output=0,
)
PARAMETERS = (
    "arg_c", "arg_a", "arg_b", "arg_scale_a", "arg_scale_b", "i32_m", "stream",
    "N", "K", "tile_m", "tile_n", "tile_k", "m_warp", "n_warp", "out_is_f16",
    "num_buffers", "a_is_fp4", "arg_m_tile_map", "n_experts", "stage1_act",
    "has_bias", "arg_bias", "f32_swiglu_limit", "stage1_quant_out",
    "quant_wmma_rep", "arg_quant_scale", "cluster_n", "cluster_m",
    "next_stage_prefetch", "num_waves_per_tensor_tdm", "tdm_b_th", "balanced_m192",
    "partial_m_direct_scale", "partial_m_drain_output", "persistent_n_tasks",
    "staged_quant_scale", "row_tail_predication", "half_a_prefetch", "row_tail_a_groups",
    "half_scale_prefetch", "pair_a_prefetch", "task_m_order", "task_m_group",
)


def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    parent, attr = name.rsplit(".", 1)
    setattr(importlib.import_module(parent), attr, module)
    return module


def elf_bytes(ir):
    literals = [m.group(1) for m in re.finditer(
        r'"((?:\\[0-9A-Fa-f]{2}|[^"\\]|\\.)*)"', ir)
        if m.group(1).startswith("\\7FELF")]
    if not literals or len(set(literals)) != 1:
        raise RuntimeError("expected one embedded GEMM1 ELF")
    literal, data, index = literals[0], bytearray(), 0
    while index < len(literal):
        if literal[index] != "\\":
            data.append(ord(literal[index]))
            index += 1
        elif re.fullmatch(r"[0-9A-Fa-f]{2}", literal[index + 1:index + 3]):
            data.append(int(literal[index + 1:index + 3], 16))
            index += 3
        else:
            data.append({"n": 10, "t": 9, "r": 13}.get(
                literal[index + 1], ord(literal[index + 1])))
            index += 2
    return bytes(data)


def install(variant):
    sys.path.insert(0, str(REPO))
    manifest = json.loads((ARTIFACTS / "source_manifest.json").read_text())
    for relative, expected in manifest.items():
        actual = hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"source SHA256 mismatch: {relative}")
    import aiter.ops.flydsl.kernels
    prefix = ".codex_tmp/side_e64_gemm1_20261018/production_snapshot/"
    persistent = load_source(
        "aiter.ops.flydsl.kernels.mxfp4_preshuffle_gfx1250_tdm_prefill_persistent",
        prefix + "aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py",
    )
    load_source("aiter.ops.flydsl.grouped_gemm_mxfp4",
                prefix + "aiter/ops/flydsl/grouped_gemm_mxfp4.py")
    load_source("aiter.ops.flydsl.grouped_moe_gfx1250",
                prefix + "aiter/ops/flydsl/grouped_moe_gfx1250.py")
    baseline = persistent.launch_gemm_a8w4_tdm_fused_persistent
    copied = importlib.import_module(
        "aiter.ops.flydsl.kernels.mxfp4_preshuffle_gfx1250_tdm_prefill_gemm1_e64"
    ).launch_gemm_a8w4_tdm_fused_persistent
    state = {"calls": 0, "capacity_rows": None, "layouts": None, "tdm_b_th": None}

    def selected(*positional, **keywords):
        arguments = dict(zip(PARAMETERS, positional))
        arguments.update(keywords)
        expected = dict(N=4096, K=7168, tile_m=256, tile_n=256, tile_k=256,
                        m_warp=2, n_warp=4, num_buffers=4, n_experts=64,
                        stage1_act=1, stage1_quant_out=1, has_bias=0,
                        cluster_n=4, cluster_m=1, a_is_fp4=1,
                        next_stage_prefetch=1, num_waves_per_tensor_tdm=2)
        if not all(arguments.get(key) == value for key, value in expected.items()):
            return baseline(*positional, **keywords)
        if arguments["i32_m"] != 28672:
            raise RuntimeError(f"unexpected E64 capacity: {arguments['i32_m']}")
        state["calls"] += 1
        state["capacity_rows"] = arguments["i32_m"]
        state["tdm_b_th"] = arguments["tdm_b_th"]
        if state["layouts"] is None:
            state["layouts"] = {
                key: dict(shape=list(arguments[key].shape),
                          stride=list(arguments[key].stride()), dtype=str(arguments[key].dtype))
                for key in ("arg_c", "arg_scale_a", "arg_scale_b", "arg_quant_scale")
            }
        if variant == "copy":
            if len(positional) > 31:
                raise RuntimeError("tuning overrides require keyword arguments")
            return copied(*positional, **dict(keywords, **TUNING))
        return baseline(*positional, **keywords)

    persistent.launch_gemm_a8w4_tdm_fused_persistent = selected
    return (copied if variant == "copy" else baseline), state, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "copy"), default="copy")
    parser.add_argument("--script", choices=("bench", "verify"), required=True)
    parser.add_argument("--binding-output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    remaining = remaining[1:] if remaining[:1] == ["--"] else remaining
    for flag, value in (("--experts", "64"), ("--tokens", "1536"), ("--topk", "8"),
                        ("--shared-experts", "0"), ("--model-dim", "7168"),
                        ("--inter-dim", "2048")):
        if remaining.count(flag) != 1 or remaining[remaining.index(flag) + 1] != value:
            parser.error(f"this isolated entry requires {flag} {value}")
    if "--const-init" in remaining or "--original-head" in remaining:
        parser.error("this entry requires random inputs and its pinned production snapshot")
    if os.environ.get("AITER_MOE_EXPERT_BALANCE", "").lower() not in ("false", "0"):
        parser.error("AITER_MOE_EXPERT_BALANCE=false is required")
    if any(key.startswith("AITER_GFX1250_GEMM1_") for key in os.environ):
        parser.error("unrelated GEMM1 tuning environment variables are not allowed")
    launcher, state, manifest = install(args.variant)
    print("[E64 independent GEMM1] " + json.dumps(dict(
        variant=args.variant, tuning=TUNING if args.variant == "copy" else "production E64 defaults",
        numerical_kernel_changes=False, shared_experts=0, valid_routes=12288)), flush=True)
    script = REPO / "my_code" / ("test_flydsl_grouped_gemm_gfx1250.py"
                                if args.script == "bench" else "verify_dsr1_e65_gemm1.py")
    sys.argv = [str(script), *remaining]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exit_status:
        if exit_status.code not in (None, 0):
            raise
    if not state["calls"]:
        raise RuntimeError("target GEMM1 was not dispatched")
    cached = list(launcher._mem_cache.values())
    if not cached and launcher._last_compiled:
        cached = [launcher._last_compiled[1]]
    records = []
    for entry in cached:
        compiled = entry if hasattr(entry, "ir") else entry[0]
        data = elf_bytes(compiled.ir)
        records.append(dict(elf_sha256=hashlib.sha256(data).hexdigest(), elf_bytes=len(data)))
        args.binding_output.parent.mkdir(parents=True, exist_ok=True)
        artifact = args.binding_output.parent / (args.binding_output.stem + f"_elf{len(records)}")
        artifact.with_suffix(".elf").write_bytes(data)
        artifact.with_suffix(".mlir").write_text(compiled.ir)
    if not records:
        raise RuntimeError("dispatched GEMM1 has no compiled ELF")
    output = dict(passed=True, variant=args.variant, script=args.script,
                  source_manifest=manifest, calls=state["calls"],
                  capacity_rows=state["capacity_rows"], layouts=state["layouts"],
                  valid_routes=12288, tdm_b_th=state["tdm_b_th"], records=records)
    args.binding_output.parent.mkdir(parents=True, exist_ok=True)
    with args.binding_output.open("x") as destination:
        destination.write(json.dumps(output, indent=2) + "\n")
    print("[actual GEMM1 binding] " + json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
