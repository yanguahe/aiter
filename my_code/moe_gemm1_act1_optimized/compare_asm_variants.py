#!/usr/bin/env python3
"""Interleave timing and correctness checks for ASM variants on one input set."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import statistics
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SYMBOL = "moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("isa", nargs="+")
    parser.add_argument("--const-init", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--launches-per-sample", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--zero-fast-flag", action="store_true")
    parser.add_argument("--validation-repeats", type=int, default=1)
    parser.add_argument("--grid-x", type=int)
    parser.add_argument("--grid-y", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(HERE))
    import torch

    import gemm_batch_isa_runner as runner

    workload = runner.reference_moe_workload(dense_256=True)
    device = torch.device(f"cuda:{args.device}")
    stream = torch.cuda.current_stream(device)
    tensors = runner.build_moe_inputs(
        torch,
        workload=workload,
        device=device,
        seed=args.seed,
        const_init=args.const_init,
        a_preshuffle=True,
    )
    contiguous_m = int(tensors["contiguous_m"])
    n = int(workload.raw_n)
    scale_a_desc = tensors["scale_a"].view(torch.int32)
    scale_b_desc = tensors["scale_b"].view(torch.int32)
    c_shape = (1, contiguous_m, n // 2)
    c_strides = (contiguous_m * (n // 2), n // 2)
    payload = runner.pack_moe_kernargs(
        ptr_c=int(tensors["out"].data_ptr()),
        ptr_a=int(tensors["a"].data_ptr()),
        ptr_b=int(tensors["b"].data_ptr()),
        ptr_scale_a=int(tensors["scale_a"].data_ptr()),
        ptr_scale_b=int(tensors["scale_b"].data_ptr()),
        ptr_m_tile_map=int(tensors["m_tile_map"].data_ptr()),
        ptr_bias=int(tensors["a"].data_ptr()),
        ptr_quant_scale=int(tensors["out"].data_ptr()),
        c_shape=c_shape,
        c_strides=c_strides,
        sa_shape=tuple(int(v) for v in scale_a_desc.shape),
        sa_strides=tuple(int(v) for v in scale_a_desc.stride()[:2]),
        sb_size0=int(scale_b_desc.shape[0]),
        qs_shape=c_shape,
        qs_strides=c_strides,
        i32_m=contiguous_m,
        i32_n=n,
        situ_beta=(
            0.0
            if args.zero_fast_flag and args.const_init == 0.0
            else 1.0
        ),
    )
    geometry = runner.make_moe_launch_geometry(workload, cluster_m=4)
    if args.grid_x is not None or args.grid_y is not None:
        geometry = dataclasses.replace(
            geometry,
            grid=(
                args.grid_x if args.grid_x is not None else geometry.grid[0],
                args.grid_y if args.grid_y is not None else geometry.grid[1],
                geometry.grid[2],
            ),
        )
    clang = runner.single._resolve_clang(None)
    if runner.single._clang_uses_default_runtime_libraries(clang):
        runner.single._prepend_default_clang_runtime_libraries()

    paths = [Path(name).resolve() for name in args.isa]
    names = [path.stem for path in paths]
    samples: dict[str, list[float]] = {name: [] for name in names}

    with contextlib.ExitStack() as stack:
        modules = []
        for path, name in zip(paths, names):
            temp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            result = runner.single.compile_isa(path, clang, temp, SYMBOL)
            module = stack.enter_context(
                runner.single._LoadedClusterKernel(result.code_object, SYMBOL, args.device)
            )
            module.configure(
                payload,
                geometry,
                int(stream.cuda_stream),
                runner.MOE_KERNARG_SIZE,
            )
            modules.append(module)
            print(f"loaded {name}", flush=True)

        for module in modules:
            for _ in range(args.warmup):
                module.launch()
        torch.cuda.synchronize()
        for module in modules:
            module.mark_stream_synchronized()

        if not args.skip_validation:
            valid = tensors["valid_rows"]
            ref = tensors["reference"][valid]
            for name, module in zip(names, modules):
                for validation_idx in range(args.validation_repeats):
                    # A nonzero sentinel proves that a zero fast path really
                    # writes the output rather than relying on initialization.
                    tensors["out"].fill_(1.0)
                    module.launch()
                    module.synchronize()
                    got = tensors["out_view"][valid].float()
                    exact_mismatch = int((got != ref).sum().item())
                    zero_count = int((got == 0).sum().item())
                    err = runner.single._load_dependencies(args.device).check_allclose(
                        ref,
                        got,
                        rtol=1e-1,
                        atol=1.0,
                        msg=name,
                    )
                    max_err, rel_l2 = runner.single.float32_error_metrics(ref, got)
                    print(
                        f"validation {name} [{validation_idx + 1}/"
                        f"{args.validation_repeats}]: err={err} "
                        f"exact_mismatch={exact_mismatch} "
                        f"zero_count={zero_count}/{got.numel()} rel_l2={rel_l2} "
                        f"max_err_info={max_err}",
                        flush=True,
                    )
                if args.const_init == 0.0 and exact_mismatch:
                    full = tensors["out_view"][: int(tensors["routed_m"])]
                    nz = full != 0
                    row_counts = nz.sum(dim=1)
                    col_counts = nz.sum(dim=0)
                    nz_rows = (row_counts != 0).nonzero().flatten()
                    nz_cols = (col_counts != 0).nonzero().flatten()
                    print(
                        "  residual extent: "
                        f"rows={int(nz_rows.numel())} "
                        f"row_min={int(nz_rows.min()) if nz_rows.numel() else -1} "
                        f"row_max={int(nz_rows.max()) if nz_rows.numel() else -1}; "
                        f"cols={int(nz_cols.numel())} "
                        f"col_min={int(nz_cols.min()) if nz_cols.numel() else -1} "
                        f"col_max={int(nz_cols.max()) if nz_cols.numel() else -1}",
                        flush=True,
                    )
                    print(
                        "  residual row-count histogram: "
                        f"{torch.unique(row_counts, return_counts=True)}",
                        flush=True,
                    )
                    for row_half in range(2):
                        for col_half in range(2):
                            block = full[
                                row_half * 128 :: 256,
                                col_half * 64 :: 128,
                            ]
                            print(
                                "  residual quadrant "
                                f"m{row_half}n{col_half}: "
                                f"nonzero={int((block != 0).sum().item())}/"
                                f"{block.numel()}",
                                flush=True,
                            )

        for round_idx in range(args.rounds):
            order = list(range(len(modules)))
            if round_idx & 1:
                order.reverse()
            for idx in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                for _ in range(args.launches_per_sample):
                    modules[idx].launch()
                end.record(stream)
                end.synchronize()
                modules[idx].mark_stream_synchronized()
                us = (
                    float(start.elapsed_time(end))
                    * 1000.0
                    / args.launches_per_sample
                )
                samples[names[idx]].append(us)
            print(f"round {round_idx + 1}/{args.rounds} complete", flush=True)

    for name in names:
        values = samples[name]
        print(
            f"RESULT {name}: median={statistics.median(values):.3f} us "
            f"mean={statistics.fmean(values):.3f} us "
            f"min={min(values):.3f} us max={max(values):.3f} us "
            f"samples={','.join(f'{v:.3f}' for v in values)}"
        )


if __name__ == "__main__":
    main()
