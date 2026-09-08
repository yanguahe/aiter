#!/usr/bin/env python3
"""Launch the precompiled optimized kernel once for rocprof ATT capture."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gemm_batch_isa_runner as batch  # noqa: E402


def main() -> None:
    deps = batch.single._load_dependencies(0)
    torch = deps.torch
    device = torch.device("cuda:0")
    stream = torch.cuda.current_stream(device)
    workload = batch.reference_moe_workload(dense_256=True)
    tensors = batch.build_moe_inputs(
        torch,
        workload=workload,
        device=device,
        seed=0,
        const_init=0.0,
        a_preshuffle=True,
    )
    n = workload.raw_n
    k = workload.model_dim
    contiguous_m = tensors["contiguous_m"]
    scale_a_desc = tensors["scale_a"].view(torch.int32)
    scale_b_desc = tensors["scale_b"].view(torch.int32)
    c_shape = (1, contiguous_m, n // 2)
    c_strides = (contiguous_m * (n // 2), n // 2)

    payload = batch.pack_moe_kernargs(
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
        sa_shape=tuple(int(value) for value in scale_a_desc.shape),
        sa_strides=tuple(int(value) for value in scale_a_desc.stride()[:2]),
        sb_size0=int(scale_b_desc.shape[0]),
        qs_shape=c_shape,
        qs_strides=c_strides,
        i32_m=contiguous_m,
        i32_n=n,
    )
    geometry = batch.make_moe_launch_geometry(workload, cluster_m=4)
    code_object = HERE / "act1_opt.co"
    symbol = batch.MOE_ACT1_256_KERNEL_SYMBOL

    torch.cuda.synchronize(device)
    with batch.single._LoadedClusterKernel(code_object, symbol, 0) as module:
        module.configure(
            payload,
            geometry,
            int(stream.cuda_stream),
            batch.MOE_KERNARG_SIZE,
        )
        module.launch()
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
