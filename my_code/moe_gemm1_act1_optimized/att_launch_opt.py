#!/usr/bin/env python3
"""Launch the precompiled optimized kernel once for rocprof ATT capture."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SNAPSHOT_ROOT = HERE / "repo_snapshot"
if not (SNAPSHOT_ROOT / "aiter" / "__init__.py").is_file():
    raise SystemExit(
        "self-contained HEAD snapshot is missing; run on the host/local checkout: "
        f"{HERE / 'sync_head_repo_snapshot.py'}"
    )
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SNAPSHOT_ROOT))
os.environ["AITER_META_DIR"] = str(SNAPSHOT_ROOT)

import aiter  # noqa: E402
import gemm_batch_isa_runner as batch  # noqa: E402

aiter_path = Path(aiter.__file__).resolve()
if SNAPSHOT_ROOT.resolve() not in aiter_path.parents:
    raise RuntimeError(
        f"aiter resolved outside the self-contained snapshot: {aiter_path}"
    )


def _audit_repository_module_paths() -> None:
    repo = REPO.resolve()
    allowed = (REPO / "my_code").resolve()
    violations: list[Path] = []
    for module in tuple(sys.modules.values()):
        raw = getattr(module, "__file__", None)
        if not raw:
            continue
        path = Path(raw).resolve()
        if not path.exists():
            continue
        if (path == repo or repo in path.parents) and not (
            path == allowed or allowed in path.parents
        ):
            violations.append(path)
    if violations:
        rendered = "\n".join(f"  {path}" for path in sorted(set(violations)))
        raise RuntimeError(
            "repository modules resolved outside my_code:\n" + rendered
        )


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
    grid_x_raw = os.environ.get("AITER_ATT_GRID_X", "")
    grid_y_raw = os.environ.get("AITER_ATT_GRID_Y", "")
    if bool(grid_x_raw) != bool(grid_y_raw):
        raise SystemExit("AITER_ATT_GRID_X and AITER_ATT_GRID_Y must be set together")
    if grid_x_raw:
        grid_x = int(grid_x_raw)
        grid_y = int(grid_y_raw)
        if grid_x <= 0 or grid_y <= 0:
            raise SystemExit("persistent ATT grid dimensions must be positive")
        geometry = replace(geometry, grid=(grid_x, grid_y, geometry.grid[2]))
    code_object = Path(
        os.environ.get("AITER_ATT_CODE_OBJECT", str(HERE / "act1_opt.co"))
    ).resolve()
    if not code_object.is_file():
        raise SystemExit(f"ATT code object does not exist: {code_object}")
    print(f"[att_launch_opt] code object: {code_object}", flush=True)
    print(f"[att_launch_opt] grid: {geometry.grid}", flush=True)
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
    _audit_repository_module_paths()
    print("[att_launch_opt] repository module audit passed", flush=True)


if __name__ == "__main__":
    main()
