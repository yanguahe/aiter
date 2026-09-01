#!/usr/bin/env python3
r"""Compile and benchmark batched gfx1250 MXFP4 GEMM assembly.

The batched kernel uses one physical grid-Z plane per independent matrix while
retaining the original 64x256 workgroup tile and persistent X/Y task traversal
in every plane.  The default physical layout uses N on X and M on Y with a 4x1
cluster; an explicitly marked ISA variant supports M on X and N on Y with a
1x4 cluster.  It launches one physical cluster per logical task up to 64; small
non-power-of-two grids encode a separate power-of-two recurrence stride.
Inputs are contiguous and batch-major:

    A[batch,M,K/2], B[batch,N,K/2], sA[batch,M,K/32],
    sB[batch,N,K/32], C[batch,M,N]

The batch-specific 120-byte ABI is the original 80-byte MXFP4 preload ABI
followed by five uint64 byte strides for C, A, B, sA, and sB.  One timed launch
is one kernel dispatch for the complete batch.

For compatibility, ``--batch 1`` also accepts the original non-batch
``..._64x256_1x4_ps`` source and uses its exact 80-byte ABI and launch path.
The original symbol is rejected for ``--batch > 1``; use the
``..._64x256_1x4_batch_ps`` source instead.

The independent ``mab_tdm_gemm`` mode is intentionally separate from those
MXFP4 paths.  It uses FP16 inputs/output, a 1x4 workgroup cluster, and the
launcher-observed 112-byte preload ABI for shapes where M is divisible by 16,
N by 1024, and K by 512.  The symbol
``mab_tdm_gemm_full_batch`` uses the same ABI and instruction body with
contiguous stride1 values and one grid-Z plane per batch item.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from . import gemm_isa_runner as single
except ImportError:
    import gemm_isa_runner as single


BATCH_KERNEL_SYMBOL = (
    "f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps"
)
LEGACY_KERNEL_SYMBOL = single.KERNEL_SYMBOL_64X256
# HIP's maximum grid-Z dimension is 65535 workgroups.
MAX_BATCH = (1 << 16) - 1
MAX_PERSISTENT_CLUSTERS = 64
MAX_PERSISTENT_CLUSTER_GRID = (16, 4)
GRID_LAYOUT_N_ON_X_M_ON_Y = "n-on-x-m-on-y"
GRID_LAYOUT_M_ON_X_N_ON_Y = "m-on-x-n-on-y"
GRID_LAYOUT_CHOICES = (
    GRID_LAYOUT_N_ON_X_M_ON_Y,
    GRID_LAYOUT_M_ON_X_N_ON_Y,
)
DEFAULT_GRID_LAYOUT = GRID_LAYOUT_N_ON_X_M_ON_Y
MXNY_GRID_LAYOUT_MARKER = "__aiter_grid_layout_mxny"
UINT64_MAX = (1 << 64) - 1
BATCH_KERNARG_SIZE = 120
BATCH_KERNARG_LAYOUT = (
    *single.KERNARG_LAYOUT,
    ("batch_stride_D", 80, "Q"),
    ("batch_stride_A", 88, "Q"),
    ("batch_stride_B", 96, "Q"),
    ("batch_stride_ScaleA", 104, "Q"),
    ("batch_stride_ScaleB", 112, "Q"),
)
BATCH_PROFILE = replace(
    single.KERNEL_PROFILE_64X256,
    name="bf16-mxfp4-wg64x256-wave64-1x4-persistent-batch-z",
    primary_symbol=BATCH_KERNEL_SYMBOL,
    symbols=(BATCH_KERNEL_SYMBOL,),
    abi_name="bf16-mxfp4-preload-batch-z-v1",
    kernarg_size=BATCH_KERNARG_SIZE,
    kernarg_layout=BATCH_KERNARG_LAYOUT,
)

MAB_KERNEL_SYMBOL = "mab_tdm_gemm"
MAB_FULL_KERNEL_SYMBOL = "mab_tdm_gemm_full"
MAB_FULL_BATCH_KERNEL_SYMBOL = "mab_tdm_gemm_full_batch"
MAB_FULL_BATCH_LOADONLY_KERNEL_SYMBOL = "mab_tdm_gemm_full_batch_loadonly"
MAB_FULL_BATCH_LOADONLY_WV23_256K_KERNEL_SYMBOL = (
    "mab_tdm_gemm_full_batch_loadonly_wv23_same_lds_256k"
)
MAB_FULL_BATCH_LOADONLY_SYMBOLS = (
    MAB_FULL_BATCH_LOADONLY_KERNEL_SYMBOL,
    MAB_FULL_BATCH_LOADONLY_WV23_256K_KERNEL_SYMBOL,
)
MAB_BATCH = 1
MAB_INIT = "UniformRandom"
MAB_RANDOM_RANGE = (-1.0, 1.0)
MAB_M_MULTIPLE = 16
MAB_N_MULTIPLE = 1024
MAB_K_MULTIPLE = 512
# The immutable V# descriptors set num_records to 0x1000000 bytes.  Keep the
# output span strictly below it because gfx1250 range checking clamps on >=.
MAB_BUFFER_RANGE_BYTES = 0x1000000
MAB_KERNARG_SIZE = 112
MAB_WV23_ALIAS_B_MAX_EXCLUSIVE = 0x264E0
MAB_WV23_RETAINED_A_MAX_EXCLUSIVE = 0x485E0
MAB_BLOCK = (128, 1, 1)
MAB_CLUSTER = (1, 4, 1)
MAB_REFERENCE_CHUNK_N = 2048
MAB_KERNARG_LAYOUT = (
    ("sizeC", 0, "Q"),
    ("sizeA", 8, "Q"),
    ("D", 16, "Q"),
    ("C", 24, "Q"),
    ("A", 32, "Q"),
    ("B", 40, "Q"),
    ("alpha", 48, "f"),
    ("beta", 52, "f"),
    ("strideD0", 56, "I"),
    ("strideD1", 60, "I"),
    ("strideC0", 64, "I"),
    ("strideC1", 68, "I"),
    ("strideA0", 72, "I"),
    ("strideA1", 76, "I"),
    ("strideB0", 80, "I"),
    ("strideB1", 84, "I"),
    ("SizesFree0", 88, "I"),
    ("SizesFree1", 92, "I"),
    ("SizesFree2", 96, "I"),
    ("SizesSum0", 100, "I"),
    ("NumWorkGroups0", 104, "I"),
    ("NumWorkGroups1", 108, "I"),
)
MAB_PROFILE = single.KernelProfile(
    name="fp16-mab-tdm-wg16x256-wave16x64-1x4",
    primary_symbol=MAB_KERNEL_SYMBOL,
    symbols=(MAB_KERNEL_SYMBOL,),
    wg_tile=(16, 256),
    wave_tile=(16, 64),
    output_quadrants=(1, 4),
    cluster=MAB_CLUSTER,
    block=MAB_BLOCK,
    k_multiple=MAB_K_MULTIPLE,
    persistent_tg=256,
    persistent_grid_y=256,
    apre=1,
    abi_name="mab-tdm-preload-v1-observed-112",
    kernarg_size=MAB_KERNARG_SIZE,
    kernarg_layout=MAB_KERNARG_LAYOUT,
)
MAB_FULL_BATCH_PROFILE = single.KernelProfile(
    name="fp16-mab-tdm-full-batch-z-wg16x256-wave16x64-1x4",
    primary_symbol=MAB_FULL_BATCH_KERNEL_SYMBOL,
    symbols=(MAB_FULL_BATCH_KERNEL_SYMBOL,),
    wg_tile=MAB_PROFILE.wg_tile,
    wave_tile=MAB_PROFILE.wave_tile,
    output_quadrants=MAB_PROFILE.output_quadrants,
    cluster=MAB_PROFILE.cluster,
    block=MAB_PROFILE.block,
    k_multiple=MAB_PROFILE.k_multiple,
    persistent_tg=MAB_PROFILE.persistent_tg,
    persistent_grid_y=MAB_PROFILE.persistent_grid_y,
    apre=MAB_PROFILE.apre,
    abi_name=MAB_PROFILE.abi_name,
    kernarg_size=MAB_PROFILE.kernarg_size,
    kernarg_layout=MAB_PROFILE.kernarg_layout,
)
MAB_FULL_BATCH_LOADONLY_PROFILE = replace(
    MAB_FULL_BATCH_PROFILE,
    name="fp16-mab-tdm-full-batch-loadonly-z-wg16x256-wave16x64-1x4",
    primary_symbol=MAB_FULL_BATCH_LOADONLY_KERNEL_SYMBOL,
    symbols=(MAB_FULL_BATCH_LOADONLY_KERNEL_SYMBOL,),
)
MAB_FULL_BATCH_LOADONLY_WV23_256K_PROFILE = replace(
    MAB_FULL_BATCH_PROFILE,
    name="fp16-mab-tdm-full-batch-loadonly-wv23-same-lds-256k",
    primary_symbol=MAB_FULL_BATCH_LOADONLY_WV23_256K_KERNEL_SYMBOL,
    symbols=(MAB_FULL_BATCH_LOADONLY_WV23_256K_KERNEL_SYMBOL,),
)

@dataclass(frozen=True)
class BatchStrides:
    """Byte strides between adjacent batch-major matrices."""

    d: int
    a: int
    b: int
    scale_a: int
    scale_b: int


@dataclass(frozen=True)
class BatchValidation:
    """Correctness result for the complete dispatch."""

    reference_hash: str
    output_hash: str
    error: float
    max_abs: float
    max_err_info: single.MaxErrorInfo
    rel_l2: float


@dataclass(frozen=True)
class MabValidation:
    """Whole-tensor validation for the fixed MAB dispatch."""

    reference_hash: str
    output_hash: str
    max_abs: float
    max_err_info: single.MaxErrorInfo
    rel_l2: float
    mismatch_count: int
    element_count: int
    rtol: float
    atol: float
    passed: bool
    dense_reference_hash: str
    dense_max_abs: float
    dense_rel_l2: float
    dense_mismatch_count: int
    batch_mismatch_counts: tuple[int, ...]


def _validate_batch(batch: int) -> int:
    if isinstance(batch, bool) or not isinstance(batch, int):
        raise single.GemmIsaRunnerError(
            f"batch must be an integer, got {batch!r}"
        )
    if batch < 1:
        raise single.GemmIsaRunnerError(
            f"batch must be at least one, got {batch}"
        )
    if batch > MAX_BATCH:
        raise single.GemmIsaRunnerError(
            f"batch={batch} exceeds the HIP grid-Z/TTMP7 maximum {MAX_BATCH}"
        )
    return batch


def _validate_grid_layout(grid_layout: str) -> str:
    if grid_layout not in GRID_LAYOUT_CHOICES:
        raise single.GemmIsaRunnerError(
            f"grid layout must be one of {GRID_LAYOUT_CHOICES}, got "
            f"{grid_layout!r}"
        )
    return grid_layout


def _physical_cluster_for_layout(
    grid_layout: str,
) -> tuple[int, int, int]:
    grid_layout = _validate_grid_layout(grid_layout)
    if grid_layout == GRID_LAYOUT_N_ON_X_M_ON_Y:
        return BATCH_PROFILE.cluster
    cluster_x, cluster_y, cluster_z = BATCH_PROFILE.cluster
    return (cluster_y, cluster_x, cluster_z)


def _checked_batch_stride(name: str, stride: int, batch: int) -> int:
    batch = _validate_batch(batch)
    stride = single._checked_unsigned(name, stride, 64)
    if stride == 0:
        raise single.GemmIsaRunnerError(
            f"{name} must be a positive byte stride"
        )
    if stride > UINT64_MAX // batch:
        raise single.GemmIsaRunnerError(
            f"{name}={stride} with batch={batch} spans more than uint64 "
            "addressing can represent"
        )
    return stride


def make_contiguous_batch_strides(
    m: int,
    n: int,
    k: int,
    batch: int,
) -> BatchStrides:
    """Return exact byte strides for the documented contiguous layouts."""

    batch = _validate_batch(batch)
    for name, value in (("M", m), ("N", n), ("K", k)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise single.GemmIsaRunnerError(
                f"{name} must be a positive integer, got {value!r}"
            )
    if k % single.MXFP4_SCALE_BLOCK:
        raise single.GemmIsaRunnerError(
            f"K={k} must be divisible by {single.MXFP4_SCALE_BLOCK}"
        )

    values = BatchStrides(
        d=m * n * 2,
        a=m * (k // 2),
        b=n * (k // 2),
        scale_a=m * (k // single.MXFP4_SCALE_BLOCK),
        scale_b=n * (k // single.MXFP4_SCALE_BLOCK),
    )
    for name, value in (
        ("batch_stride_D", values.d),
        ("batch_stride_A", values.a),
        ("batch_stride_B", values.b),
        ("batch_stride_ScaleA", values.scale_a),
        ("batch_stride_ScaleB", values.scale_b),
    ):
        _checked_batch_stride(name, value, batch)
    return values


def _batch_scheduler_layout(
    logical_tasks: int,
    grid_layout: str = DEFAULT_GRID_LAYOUT,
) -> tuple[int, int, tuple[int, int], int]:
    """Return physical cluster X/Y, encoded log2 X/Y, and recurrence stride."""

    grid_layout = _validate_grid_layout(grid_layout)
    if logical_tasks < 1:
        raise single.GemmIsaRunnerError(
            f"logical cluster-task count must be positive, got {logical_tasks}"
        )
    if logical_tasks <= MAX_PERSISTENT_CLUSTERS:
        persistent_stride = 1 << (logical_tasks - 1).bit_length()
        layout = (
            logical_tasks,
            1,
            (persistent_stride.bit_length() - 1, 0),
            persistent_stride,
        )
    else:
        cluster_grid_x, cluster_grid_y = MAX_PERSISTENT_CLUSTER_GRID
        layout = (
            cluster_grid_x,
            cluster_grid_y,
            (
                cluster_grid_x.bit_length() - 1,
                cluster_grid_y.bit_length() - 1,
            ),
            MAX_PERSISTENT_CLUSTERS,
        )
    if grid_layout == GRID_LAYOUT_N_ON_X_M_ON_Y:
        return layout
    cluster_grid_x, cluster_grid_y, log2_grid, persistent_stride = layout
    return (
        cluster_grid_y,
        cluster_grid_x,
        (log2_grid[1], log2_grid[0]),
        persistent_stride,
    )


def make_batched_launch_geometry(
    m: int,
    n: int,
    k: int,
    batch: int,
    grid_layout: str = DEFAULT_GRID_LAYOUT,
) -> single.LaunchGeometry:
    """Build an adaptive persistent X/Y launch and replicate it in grid Z."""

    batch = _validate_batch(batch)
    grid_layout = _validate_grid_layout(grid_layout)
    base = single.make_launch_geometry(m, n, k, profile=BATCH_PROFILE)
    # Fail before compilation/allocation if either the appended u64 batch
    # address span or an inherited u32 row-stride field would overflow.
    make_contiguous_batch_strides(m, n, k, batch)
    for name, value in (
        ("strideD0", n * 2),
        ("strideA0", k // 2),
        ("strideB0", k // 2),
        ("ScaleA_stride0", k // single.MXFP4_SCALE_BLOCK),
        ("ScaleB_stride0", k // single.MXFP4_SCALE_BLOCK),
    ):
        single._checked_unsigned(name, value, 32)

    (
        cluster_grid_x,
        cluster_grid_y,
        log2_grid,
        persistent_stride,
    ) = _batch_scheduler_layout(base.logical_cluster_tasks, grid_layout)
    physical_clusters = cluster_grid_x * cluster_grid_y
    if physical_clusters != min(
        base.logical_cluster_tasks,
        MAX_PERSISTENT_CLUSTERS,
    ):
        raise single.GemmIsaRunnerError(
            f"adaptive physical cluster count {physical_clusters} does not "
            f"equal min(T_XY,64) for T_XY={base.logical_cluster_tasks}"
        )
    if persistent_stride < physical_clusters:
        raise single.GemmIsaRunnerError(
            f"adaptive persistent stride {persistent_stride} is smaller than "
            f"physical seed count {physical_clusters}"
        )
    if persistent_stride != 1 << sum(log2_grid):
        raise single.GemmIsaRunnerError(
            f"adaptive persistent stride {persistent_stride} does not match "
            f"encoded log2 grid {log2_grid}"
        )
    cluster_x, cluster_y, cluster_z = _physical_cluster_for_layout(
        grid_layout
    )
    return replace(
        base,
        grid=(
            cluster_grid_x * cluster_x,
            cluster_grid_y * cluster_y,
            batch,
        ),
        cluster=(cluster_x, cluster_y, cluster_z),
        cluster_grid=(cluster_grid_x, cluster_grid_y),
        log2_grid=log2_grid,
        persistent_stride=persistent_stride,
    )


def pack_batched_mxfp4_kernargs(
    *,
    ptr_d: int,
    ptr_a: int,
    ptr_b: int,
    ptr_scale_a: int,
    ptr_scale_b: int,
    m: int,
    n: int,
    k: int,
    batch: int,
    batch_strides: BatchStrides,
    geometry: single.LaunchGeometry,
    grid_layout: str = DEFAULT_GRID_LAYOUT,
) -> bytes:
    """Pack the exact 120-byte batch-Z preload ABI."""

    batch = _validate_batch(batch)
    grid_layout = _validate_grid_layout(grid_layout)
    if geometry.grid[2] != batch:
        raise single.GemmIsaRunnerError(
            f"grid Z={geometry.grid[2]} does not match batch={batch}"
        )
    if geometry.block != BATCH_PROFILE.block:
        raise single.GemmIsaRunnerError(
            f"launch block {geometry.block} does not match "
            f"{BATCH_PROFILE.block}"
        )
    expected_geometry = make_batched_launch_geometry(
        m,
        n,
        k,
        batch,
        grid_layout,
    )
    if geometry != expected_geometry:
        raise single.GemmIsaRunnerError(
            "launch geometry does not match the exact batch scheduler layout: "
            f"got {geometry}, expected {expected_geometry} for "
            f"{grid_layout!r}"
        )

    stride_values = {
        "batch_stride_D": batch_strides.d,
        "batch_stride_A": batch_strides.a,
        "batch_stride_B": batch_strides.b,
        "batch_stride_ScaleA": batch_strides.scale_a,
        "batch_stride_ScaleB": batch_strides.scale_b,
    }
    for name, value in stride_values.items():
        _checked_batch_stride(name, value, batch)

    return single._pack_kernarg_fields(
        {
            "ptr_D": ptr_d,
            "ptr_A": ptr_a,
            "ptr_B": ptr_b,
            "ptr_ScaleA": ptr_scale_a,
            "ptr_ScaleB": ptr_scale_b,
            "strideD0": n * 2,
            "strideA0": k // 2,
            "strideB0": k // 2,
            "ScaleA_stride0": k // single.MXFP4_SCALE_BLOCK,
            "ScaleB_stride0": k // single.MXFP4_SCALE_BLOCK,
            "M": m,
            "N": n,
            "K": k,
            "log2_grid_x": geometry.log2_grid[0],
            "log2_grid_y": geometry.log2_grid[1],
            **stride_values,
        },
        layout=BATCH_KERNARG_LAYOUT,
        size=BATCH_KERNARG_SIZE,
    )


def _validate_mab_shape(
    m: int,
    n: int,
    k: int,
    *,
    require_output_range: bool = True,
) -> tuple[int, int, int]:
    """Validate the MAB tensor, ABI-stride, and fixed buffer-range contract."""

    shape = (m, n, k)
    for name, value in zip(("M", "N", "K"), shape):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise single.GemmIsaRunnerError(
                f"{name} must be a positive integer, got {value!r}"
            )
        single._checked_unsigned(name, value, 32)
    for name, value, multiple in (
        ("M", m, MAB_M_MULTIPLE),
        ("N", n, MAB_N_MULTIPLE),
        ("K", k, MAB_K_MULTIPLE),
    ):
        if value % multiple:
            raise single.GemmIsaRunnerError(
                f"MAB requires {name} % {multiple} == 0, got {name}={value}"
            )
    for name, value in (
        ("M*K", m * k),
        ("N*K", n * k),
        ("M*N", m * n),
    ):
        if value > 0xFFFFFFFF:
            raise single.GemmIsaRunnerError(
                f"MAB u32 element stride {name}={value} exceeds UINT32_MAX"
            )
    output_bytes = 2 * m * n
    if require_output_range and output_bytes >= MAB_BUFFER_RANGE_BYTES:
        raise single.GemmIsaRunnerError(
            f"MAB D/C byte span {output_bytes} must be below the ISA buffer "
            f"resource range {MAB_BUFFER_RANGE_BYTES}"
        )
    return shape


def make_mab_launch_geometry(
    m: int,
    n: int,
    k: int,
    batch: int,
    *,
    symbol: str = MAB_KERNEL_SYMBOL,
) -> single.LaunchGeometry:
    """Return the symbol-scoped MAB launch geometry."""

    loadonly = symbol in MAB_FULL_BATCH_LOADONLY_SYMBOLS
    _validate_mab_shape(m, n, k, require_output_range=not loadonly)
    batch = _validate_batch(batch)
    if symbol not in (
        MAB_FULL_BATCH_KERNEL_SYMBOL,
        *MAB_FULL_BATCH_LOADONLY_SYMBOLS,
    ) and batch != MAB_BATCH:
        raise single.GemmIsaRunnerError(
            f"{symbol} supports only batch={MAB_BATCH}, got {batch}"
        )
    wg_m = m // MAB_M_MULTIPLE
    wg_n = n // 256
    cluster_m = wg_m
    cluster_n = n // MAB_N_MULTIPLE
    cluster_tasks = cluster_m * cluster_n
    return single.LaunchGeometry(
        grid=(wg_m, wg_n, batch),
        block=MAB_BLOCK,
        cluster=MAB_CLUSTER,
        tiles=(wg_m, wg_n),
        cluster_grid=(cluster_m, cluster_n),
        log2_grid=(
            (cluster_m - 1).bit_length(),
            (cluster_n - 1).bit_length(),
        ),
        logical_cluster_grid=(cluster_m, cluster_n),
        logical_wg_tasks=wg_m * wg_n,
        logical_cluster_tasks=cluster_tasks,
        persistent_stride=cluster_tasks,
    )


def pack_mab_kernargs(
    *,
    ptr_d: int,
    ptr_c: int,
    ptr_a: int,
    ptr_b: int,
    m: int,
    n: int,
    k: int,
    batch: int,
    geometry: single.LaunchGeometry,
    symbol: str = MAB_KERNEL_SYMBOL,
) -> bytes:
    """Pack the MAB 112-byte preload ABI.

    The batch-capable symbol uses contiguous element strides in the existing
    stride1 fields.  NumWorkGroups1 is the 64-cluster Y grid, not the
    256-workgroup physical Y grid.
    """

    expected_geometry = make_mab_launch_geometry(
        m,
        n,
        k,
        batch,
        symbol=symbol,
    )
    if geometry != expected_geometry:
        raise single.GemmIsaRunnerError(
            f"MAB launch geometry is {geometry}, expected {expected_geometry}"
        )
    pointer_values = {
        "D": ptr_d,
        "C": ptr_c,
        "A": ptr_a,
        "B": ptr_b,
    }
    for name, value in pointer_values.items():
        single._checked_unsigned(name, value, 64)

    batched = symbol in (
        MAB_FULL_BATCH_KERNEL_SYMBOL,
        *MAB_FULL_BATCH_LOADONLY_SYMBOLS,
    )
    values: dict[str, int | float] = {
        "sizeC": 0,
        "sizeA": 0,
        **pointer_values,
        "alpha": 1.0,
        "beta": 0.0,
        "strideD0": m,
        "strideD1": m * n if batched else 0,
        "strideC0": m,
        "strideC1": m * n if batched else 0,
        "strideA0": k,
        "strideA1": m * k if batched else 0,
        "strideB0": k,
        "strideB1": n * k if batched else 0,
        "SizesFree0": m,
        "SizesFree1": n,
        "SizesFree2": batch,
        "SizesSum0": k,
        "NumWorkGroups0": geometry.cluster_grid[0],
        "NumWorkGroups1": geometry.cluster_grid[1],
    }
    payload = bytearray(MAB_KERNARG_SIZE)
    for name, offset, kind in MAB_KERNARG_LAYOUT:
        value = values[name]
        if kind == "f":
            struct.pack_into("<f", payload, offset, float(value))
        else:
            bits = 64 if kind == "Q" else 32
            checked = single._checked_unsigned(name, int(value), bits)
            struct.pack_into(f"<{kind}", payload, offset, checked)
    return bytes(payload)


def detect_batch_isa_grid_layout(source: str) -> str:
    """Return the ISA's explicit layout contract, defaulting to legacy NX/MY."""

    marker_lines: list[str] = []
    for line in source.splitlines():
        clean = single._strip_asm_comment(line)
        if MXNY_GRID_LAYOUT_MARKER in clean:
            marker_lines.append(re.sub(r"\s+", " ", clean))
    if not marker_lines:
        return DEFAULT_GRID_LAYOUT
    expected = f".set {MXNY_GRID_LAYOUT_MARKER}, 1"
    if marker_lines != [expected]:
        raise single.GemmIsaRunnerError(
            f"malformed or duplicated {MXNY_GRID_LAYOUT_MARKER} marker: "
            f"{marker_lines}; expected exactly {expected!r}"
        )
    return GRID_LAYOUT_M_ON_X_N_ON_Y


def validate_batch_isa_grid_layout(
    source: str,
    requested_grid_layout: str,
) -> str:
    """Reject a CLI/ISA axis mismatch before compilation or GPU launch."""

    requested_grid_layout = _validate_grid_layout(requested_grid_layout)
    isa_grid_layout = detect_batch_isa_grid_layout(source)
    if isa_grid_layout != requested_grid_layout:
        raise single.GemmIsaRunnerError(
            f"requested --grid-layout {requested_grid_layout!r}, but the ISA "
            f"declares {isa_grid_layout!r}; the M-on-X/N-on-Y layout requires "
            f"exactly one '.set {MXNY_GRID_LAYOUT_MARKER}, 1' marker and the "
            "default N-on-X/M-on-Y layout rejects that marker"
        )
    return isa_grid_layout


def select_kernel_mode(
    symbol: str,
    batch: int,
) -> tuple[str, single.KernelProfile]:
    """Select one exact, symbol-scoped ABI and execution path."""

    batch = _validate_batch(batch)
    if symbol == MAB_FULL_BATCH_LOADONLY_WV23_256K_KERNEL_SYMBOL:
        return (
            "mab-full-batch-loadonly-wv23-256k",
            MAB_FULL_BATCH_LOADONLY_WV23_256K_PROFILE,
        )
    if symbol == MAB_FULL_BATCH_LOADONLY_KERNEL_SYMBOL:
        return "mab-full-batch-loadonly", MAB_FULL_BATCH_LOADONLY_PROFILE
    if symbol == MAB_FULL_BATCH_KERNEL_SYMBOL:
        return "mab-full-batch", MAB_FULL_BATCH_PROFILE
    if symbol in (MAB_KERNEL_SYMBOL, MAB_FULL_KERNEL_SYMBOL):
        if batch != MAB_BATCH:
            raise single.GemmIsaRunnerError(
                f"{symbol!r} supports only --batch {MAB_BATCH}"
            )
        return (
            "mab-full" if symbol == MAB_FULL_KERNEL_SYMBOL else "mab",
            MAB_PROFILE,
        )
    if symbol == LEGACY_KERNEL_SYMBOL:
        if batch != 1:
            raise single.GemmIsaRunnerError(
                f"the original kernel {LEGACY_KERNEL_SYMBOL!r} has no batch "
                f"addressing and cannot run --batch {batch}; use "
                f"{BATCH_KERNEL_SYMBOL!r} from the _batch_ps.s source"
            )
        return "legacy", single.KERNEL_PROFILE_64X256
    if symbol == BATCH_KERNEL_SYMBOL:
        return "batch-z", BATCH_PROFILE
    raise single.GemmIsaRunnerError(
        f"unsupported kernel {symbol!r}; this runner accepts "
        f"{MAB_KERNEL_SYMBOL!r}, {MAB_FULL_KERNEL_SYMBOL!r}, and "
        f"{LEGACY_KERNEL_SYMBOL!r} for batch=1; "
        f"{MAB_FULL_BATCH_KERNEL_SYMBOL!r} and "
        f"{MAB_FULL_BATCH_LOADONLY_KERNEL_SYMBOL!r} and "
        f"{MAB_FULL_BATCH_LOADONLY_WV23_256K_KERNEL_SYMBOL!r} "
        "for supported batches; or "
        f"{BATCH_KERNEL_SYMBOL!r} for any supported batch"
    )


def _validate_mab_mode(args: argparse.Namespace, symbol: str) -> None:
    if args.intype != "fp16":
        raise single.GemmIsaRunnerError(
            f"{symbol} requires --intype fp16"
        )
    if args.dtype != "fp16":
        raise single.GemmIsaRunnerError(
            f"{symbol} writes FP16 and requires --dtype fp16"
        )
    if args.apre != 1:
        raise single.GemmIsaRunnerError(
            f"{symbol} has no A-preshuffle variant; keep --apre 1"
        )
    if args.init != MAB_INIT:
        raise single.GemmIsaRunnerError(
            f"{symbol} supports only --init {MAB_INIT}"
        )
    make_mab_launch_geometry(*args.shape, args.batch, symbol=symbol)


def _validate_batch_mode(
    intype: str,
    apre: int,
    dtype: str,
) -> None:
    if intype != "mxfp4":
        raise single.GemmIsaRunnerError(
            "the batched ISA supports MXFP4 only"
        )
    if apre != 1:
        raise single.GemmIsaRunnerError(
            "the batched ABpreShuffle ISA requires --apre 1"
        )
    if dtype != "bf16":
        raise single.GemmIsaRunnerError(
            "the batched ISA writes BF16 output only"
        )


def _runtime_batch_strides(
    inputs: Mapping[str, Any],
    output: Any,
    *,
    torch_module: Any,
    batch: int,
    m: int,
    n: int,
    k: int,
) -> BatchStrides:
    expected_shapes = {
        "A": (batch, m, k // 2),
        "B": (batch, n, k // 2),
        "sA": (batch, m, k // single.MXFP4_SCALE_BLOCK),
        "sB": (batch, n, k // single.MXFP4_SCALE_BLOCK),
    }
    for name, expected_shape in expected_shapes.items():
        tensor = inputs[name]
        if tuple(tensor.shape) != expected_shape:
            raise single.GemmIsaRunnerError(
                f"{name} has shape {tuple(tensor.shape)}, expected "
                f"{expected_shape}"
            )
        if tensor.dtype != torch_module.uint8:
            raise single.GemmIsaRunnerError(
                f"{name} has dtype {tensor.dtype}, expected uint8"
            )
        if not tensor.is_contiguous():
            raise single.GemmIsaRunnerError(
                f"{name} must be contiguous batch-major"
            )
    if tuple(output.shape) != (batch, m, n):
        raise single.GemmIsaRunnerError(
            f"output has shape {tuple(output.shape)}, expected {(batch, m, n)}"
        )
    if output.dtype != torch_module.bfloat16 or not output.is_contiguous():
        raise single.GemmIsaRunnerError(
            "output must be contiguous batch-major BF16"
        )

    actual = BatchStrides(
        d=int(output.stride(0)) * int(output.element_size()),
        a=int(inputs["A"].stride(0)) * int(inputs["A"].element_size()),
        b=int(inputs["B"].stride(0)) * int(inputs["B"].element_size()),
        scale_a=int(inputs["sA"].stride(0))
        * int(inputs["sA"].element_size()),
        scale_b=int(inputs["sB"].stride(0))
        * int(inputs["sB"].element_size()),
    )
    expected = make_contiguous_batch_strides(m, n, k, batch)
    if actual != expected:
        raise single.GemmIsaRunnerError(
            f"runtime byte strides {actual} do not match contiguous ABI "
            f"{expected}"
        )
    return actual


def _prepare_batched_inputs(
    torch_module: Any,
    *,
    batch: int,
    m: int,
    n: int,
    k: int,
    apre: int,
    dtype: Any,
    init: str,
) -> tuple[dict[str, Any], Any]:
    names = ("A", "B", "sA", "sB")
    first_inputs, first_reference = single.prepare_mxfp4_inputs_and_reference(
        m,
        n,
        k,
        apre,
        dtype,
        init,
    )
    if batch == 1:
        inputs = {
            name: first_inputs[name].unsqueeze(0)
            for name in names
        }
        return inputs, first_reference.unsqueeze(0)

    # Allocate the final contiguous batch once and copy one prepared matrix at
    # a time.  This keeps peak input memory near batch storage + one matrix,
    # rather than retaining every matrix and then duplicating all of them in
    # torch.stack (important for the 1 GiB-per-B example shape).
    inputs = {
        name: torch_module.empty(
            (batch, *first_inputs[name].shape),
            dtype=first_inputs[name].dtype,
            device=first_inputs[name].device,
        )
        for name in names
    }
    reference = torch_module.empty(
        (batch, *first_reference.shape),
        dtype=first_reference.dtype,
        device=first_reference.device,
    )
    for name in names:
        inputs[name][0].copy_(first_inputs[name])
    reference[0].copy_(first_reference)
    del first_inputs, first_reference

    for index in range(1, batch):
        inputs_i, reference_i = single.prepare_mxfp4_inputs_and_reference(
            m,
            n,
            k,
            apre,
            dtype,
            init,
        )
        for name in names:
            inputs[name][index].copy_(inputs_i[name])
        reference[index].copy_(reference_i)
    return inputs, reference


def _tensor_blake2b128(tensor: Any, torch_module: Any) -> str:
    """Hash one complete tensor in its logical, contiguous element order."""

    byte_view = tensor.detach().contiguous().view(torch_module.uint8).cpu()
    return hashlib.blake2b(
        byte_view.numpy().tobytes(),
        digest_size=16,
    ).hexdigest()


def _build_mab_dense_reference(
    torch_module: Any,
    a: Any,
    b: Any,
) -> Any:
    """Build one dense FP32-accumulated, FP16 MAB reference in N-major storage."""

    m, k = tuple(a.shape)
    n, b_k = tuple(b.shape)
    _validate_mab_shape(m, n, k)
    if b_k != k:
        raise single.GemmIsaRunnerError(
            f"MAB dense reference got A={tuple(a.shape)}, B={tuple(b.shape)}"
        )
    storage = torch_module.empty(
        (n, m),
        dtype=torch_module.float16,
        device=a.device,
    )
    old_precision = torch_module.get_float32_matmul_precision()
    allow_tf32 = getattr(torch_module.backends.cuda.matmul, "allow_tf32", None)
    try:
        torch_module.set_float32_matmul_precision("highest")
        if allow_tf32 is not None:
            torch_module.backends.cuda.matmul.allow_tf32 = False
        a_fp32 = a.float()
        for start in range(0, n, MAB_REFERENCE_CHUNK_N):
            stop = min(start + MAB_REFERENCE_CHUNK_N, n)
            tile = torch_module.mm(
                a_fp32,
                b[start:stop].float().transpose(0, 1),
            )
            storage[start:stop].copy_(
                tile.transpose(0, 1).to(torch_module.float16)
            )
            del tile
        del a_fp32
    finally:
        torch_module.set_float32_matmul_precision(old_precision)
        if allow_tf32 is not None:
            torch_module.backends.cuda.matmul.allow_tf32 = allow_tf32
    return storage


def _mab_wv23_lds_note(metadata_lds_bytes: int) -> str:
    static_lds = single._format_lds_kb(metadata_lds_bytes)
    if metadata_lds_bytes >= MAB_WV23_RETAINED_A_MAX_EXCLUSIVE:
        return (
            f"the parsed {static_lds} KB static LDS keeps retained A and "
            "aliased B destinations in range"
        )
    if metadata_lds_bytes >= MAB_WV23_ALIAS_B_MAX_EXCLUSIVE:
        return (
            f"the parsed {static_lds} KB static LDS keeps aliased B "
            "destinations in range, but retained A high-plane/stage "
            "destinations exceed it, so some A Global requests may be "
            "suppressed"
        )
    return (
        f"the parsed {static_lds} KB static LDS is smaller than both the "
        "retained A and aliased B maximum destinations, so Global requests "
        "may be suppressed"
    )


def _run_mab_gemm(
    args: argparse.Namespace,
    code_object: Path,
    geometry: single.LaunchGeometry,
    symbol: str,
    metadata_lds_bytes: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Callable[[], tuple[MabValidation, bool]],
]:
    """Prepare, time, and defer validation for the fixed FP16 MAB kernel."""

    m, n, k = args.shape
    dependencies = single._load_dependencies(args.device)
    torch = dependencies.torch
    gfx = dependencies.get_gfx()
    if gfx != single.ARCH:
        raise single.GemmIsaRunnerError(
            f"the supplied kernel targets {single.ARCH}, but the active "
            f"device reports {gfx}"
        )

    device = torch.device("cuda", args.device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    low, high = MAB_RANDOM_RANGE
    batch = args.batch
    a = torch.empty((batch, m, k), dtype=torch.float16, device=device)
    b = torch.empty((batch, n, k), dtype=torch.float16, device=device)
    loadonly = symbol in MAB_FULL_BATCH_LOADONLY_SYMBOLS
    c_storage = torch.empty(
        (batch, n, m),
        dtype=torch.float16,
        device=device,
    ) if not loadonly else None
    for index in range(batch):
        tensors = (a[index], b[index]) if loadonly else (
            a[index],
            b[index],
            c_storage[index],
        )
        for tensor in tensors:
            tensor.uniform_(low, high, generator=generator)
    if loadonly:
        torch.cuda.synchronize()
        stream = torch.cuda.current_stream(args.device)
        payload = pack_mab_kernargs(
            ptr_d=int(a.data_ptr()),
            ptr_c=int(a.data_ptr()),
            ptr_a=int(a.data_ptr()),
            ptr_b=int(b.data_ptr()),
            m=m,
            n=n,
            k=k,
            batch=batch,
            geometry=geometry,
            symbol=symbol,
        )
        with single._LoadedClusterKernel(
            code_object,
            symbol,
            args.device,
        ) as module:
            module.configure(
                payload,
                geometry,
                int(stream.cuda_stream),
                MAB_KERNARG_SIZE,
            )

            def launch_loadonly() -> None:
                module.launch()

            _, microseconds = single._run_batched_cuda_event_timing(
                torch,
                launch_loadonly,
                stream,
                num_warmup=args.warmup,
                num_iters=args.iters,
            )
            module.mark_stream_synchronized()
            event_timing = single.make_cuda_event_timing_row(
                symbol,
                args.iters,
                args.warmup,
                microseconds,
                args.device,
            )
        us = float(event_timing["device_time_avg"])
        logical_bytes = int(a.nbytes + b.nbytes)
        not_validated = "n/a (not validated)"
        if symbol == MAB_FULL_BATCH_LOADONLY_WV23_256K_KERNEL_SYMBOL:
            lds_note = _mab_wv23_lds_note(metadata_lds_bytes)
            print(
                f"[gemm_batch_isa_runner] MAB TDM load-only nominal logical "
                f"A+B bytes={logical_bytes} = A ({a.nbytes}) + B ({b.nbytes}); "
                f"{lds_note}; actual HBM bytes also depend on multicast/cache "
                "behavior, and descriptor payload is not claimed as HBM bytes"
            )
        else:
            print(
                f"[gemm_batch_isa_runner] MAB TDM load-only logical unique "
                f"A+B bytes={logical_bytes} = A ({a.nbytes}) + B ({b.nbytes}); "
                "descriptor payload=n/a (not claimed as HBM bytes)"
            )
        row = {
            "intype": args.intype,
            "batch": batch,
            "M": m,
            "N": n,
            "K": k,
            "apre": args.apre,
            "init": args.init,
            "seed": args.seed,
            "dtype": args.dtype,
            "gfx": gfx,
            "logical cluster tasks/plane": geometry.logical_cluster_tasks,
            "physical clusters/plane": (
                geometry.cluster_grid[0] * geometry.cluster_grid[1]
            ),
            "physical WGs/plane": geometry.grid[0] * geometry.grid[1],
            "encoded recurrence stride/plane": geometry.persistent_stride,
            "ref hash128": not_validated,
            "gemm_a4w4 us": round(us, 3),
            "gemm_a4w4 TFLOPS": "n/a (no store)",
            "gemm_a4w4 TB/s": round(
                single.bytes_per_microsecond_to_tbps(logical_bytes, us), 3
            ),
            "gemm_a4w4 err": not_validated,
            "gemm_a4w4 out hash128": not_validated,
            "gemm_a4w4 max_err_info": not_validated,
            "gemm_a4w4 rel_l2": not_validated,
        }

        def validate_loadonly() -> tuple[None, bool]:
            return None, True

        return row, event_timing, validate_loadonly

    assert c_storage is not None
    output_storage = torch.full(
        (batch, n, m),
        float("nan"),
        dtype=torch.float16,
        device=device,
    )
    expected_nbytes = {
        "A": batch * m * k * 2,
        "B": batch * n * k * 2,
        "C": batch * n * m * 2,
        "D": batch * n * m * 2,
    }
    actual_nbytes = {
        "A": int(a.nbytes),
        "B": int(b.nbytes),
        "C": int(c_storage.nbytes),
        "D": int(output_storage.nbytes),
    }
    if actual_nbytes != expected_nbytes:
        raise single.GemmIsaRunnerError(
            f"MAB allocation sizes {actual_nbytes} do not match "
            f"{expected_nbytes}"
        )

    if symbol == MAB_KERNEL_SYMBOL:
        print(
            "[gemm_batch_isa_runner] correctness expected false: "
            "mab_tdm_gemm is a nondeterministic B-LDS-alias experiment with "
            "cross-wave WAW; dense comparison is diagnostic only"
        )

    # The dense reference and digest finish before any timed launch.
    dense_storage = torch.empty_like(output_storage)
    for index in range(batch):
        dense_storage[index].copy_(
            _build_mab_dense_reference(torch, a[index], b[index])
        )
    dense_reference = dense_storage.transpose(1, 2)
    dense_reference_hash = _tensor_blake2b128(dense_reference, torch)
    torch.cuda.synchronize()

    stream = torch.cuda.current_stream(args.device)
    payload = pack_mab_kernargs(
        ptr_d=int(output_storage.data_ptr()),
        ptr_c=int(c_storage.data_ptr()),
        ptr_a=int(a.data_ptr()),
        ptr_b=int(b.data_ptr()),
        m=m,
        n=n,
        k=k,
        batch=args.batch,
        geometry=geometry,
        symbol=symbol,
    )
    with single._LoadedClusterKernel(
        code_object,
        symbol,
        args.device,
    ) as module:
        module.configure(
            payload,
            geometry,
            int(stream.cuda_stream),
            MAB_KERNARG_SIZE,
        )

        def launch() -> Any:
            module.launch()
            return output_storage

        timed_output_storage, microseconds = (
            single._run_batched_cuda_event_timing(
                torch,
                launch,
                stream,
                num_warmup=args.warmup,
                num_iters=args.iters,
            )
        )
        module.mark_stream_synchronized()
        event_timing = single.make_cuda_event_timing_row(
            symbol,
            args.iters,
            args.warmup,
            microseconds,
            args.device,
        )
    us = float(event_timing["device_time_avg"])
    logical_bytes = int(a.nbytes + b.nbytes + output_storage.nbytes)
    flops = 2 * batch * m * n * k
    print(
        f"[gemm_batch_isa_runner] MAB logical_bytes={logical_bytes} = "
        f"A ({a.nbytes}) + B ({b.nbytes}) + D ({output_storage.nbytes}); "
        f"C ({c_storage.nbytes}) is allocated but beta=0 and is not read"
    )
    row = {
        "intype": args.intype,
        "batch": args.batch,
        "M": m,
        "N": n,
        "K": k,
        "apre": args.apre,
        "init": args.init,
        "seed": args.seed,
        "dtype": args.dtype,
        "gfx": gfx,
        "logical cluster tasks/plane": geometry.logical_cluster_tasks,
        "physical clusters/plane": geometry.cluster_grid[0] * geometry.cluster_grid[1],
        "physical WGs/plane": geometry.grid[0] * geometry.grid[1],
        "encoded recurrence stride/plane": geometry.persistent_stride,
        "ref hash128": dense_reference_hash,
        "gemm_a4w4 us": round(us, 3),
        "gemm_a4w4 TFLOPS": round(flops / us / 1e6, 3),
        "gemm_a4w4 TB/s": round(
            single.bytes_per_microsecond_to_tbps(logical_bytes, us), 3
        ),
    }

    def validate() -> tuple[MabValidation, bool]:
        output = timed_output_storage.transpose(1, 2)
        reference_fp32 = dense_reference.float()
        output_fp32 = output.float()
        difference = output_fp32 - reference_fp32
        max_err_info, rel_l2 = single.float32_error_metrics(
            reference_fp32,
            output_fp32,
            difference=difference,
            clamp_rel_l2=True,
        )
        max_abs = max_err_info[2]
        # FP16 output contributes about 4.88e-4 relative rounding error.  The
        # one-ULP absolute allowance covers a different FP32 reduction tree.
        rtol = 1.0e-3
        atol = 1.25e-1
        close = torch.isclose(
            output_fp32,
            reference_fp32,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )
        mismatch_count = int((~close).sum().item())
        batch_mismatch_counts = tuple(
            int(value)
            for value in (~close)
            .reshape(batch, -1)
            .sum(dim=1)
            .tolist()
        )
        output_hash = _tensor_blake2b128(output, torch)
        validation = MabValidation(
            reference_hash=dense_reference_hash,
            output_hash=output_hash,
            max_abs=max_abs,
            max_err_info=max_err_info,
            rel_l2=rel_l2,
            mismatch_count=mismatch_count,
            element_count=output.numel(),
            rtol=rtol,
            atol=atol,
            passed=mismatch_count == 0,
            dense_reference_hash=dense_reference_hash,
            dense_max_abs=max_abs,
            dense_rel_l2=rel_l2,
            dense_mismatch_count=mismatch_count,
            batch_mismatch_counts=batch_mismatch_counts,
        )
        row["gemm_a4w4 err"] = mismatch_count / output.numel()
        row["gemm_a4w4 out hash128"] = output_hash
        row["gemm_a4w4 max_err_info"] = max_err_info
        row["gemm_a4w4 rel_l2"] = rel_l2
        return validation, validation.passed or symbol == MAB_KERNEL_SYMBOL

    return row, event_timing, validate


def _run_batch_gemm(
    args: argparse.Namespace,
    code_object: Path,
    geometry: single.LaunchGeometry,
    writes_output: bool,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Callable[[], tuple[BatchValidation, bool]],
]:
    m, n, k = args.shape
    batch = args.batch
    dependencies = single._load_dependencies(args.device)
    torch = dependencies.torch
    gfx = dependencies.get_gfx()
    if gfx != single.ARCH:
        raise single.GemmIsaRunnerError(
            f"the supplied kernel targets {single.ARCH}, but the active "
            f"device reports {gfx}"
        )

    dtype = torch.bfloat16
    with single._LoadedClusterKernel(
        code_object,
        BATCH_KERNEL_SYMBOL,
        args.device,
    ) as module:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        inputs, reference = _prepare_batched_inputs(
            torch,
            batch=batch,
            m=m,
            n=n,
            k=k,
            apre=args.apre,
            dtype=dtype,
            init=args.init,
        )
        reference_hash = single.tensor_blake2b128(reference)
        output = torch.empty(
            (batch, m, n),
            dtype=dtype,
            device=inputs["A"].device,
        )
        batch_strides = _runtime_batch_strides(
            inputs,
            output,
            torch_module=torch,
            batch=batch,
            m=m,
            n=n,
            k=k,
        )
        stream = torch.cuda.current_stream(args.device)
        payload = pack_batched_mxfp4_kernargs(
            ptr_d=int(output.data_ptr()),
            ptr_a=int(inputs["A"].data_ptr()),
            ptr_b=int(inputs["B"].data_ptr()),
            ptr_scale_a=int(inputs["sA"].data_ptr()),
            ptr_scale_b=int(inputs["sB"].data_ptr()),
            m=m,
            n=n,
            k=k,
            batch=batch,
            batch_strides=batch_strides,
            geometry=geometry,
            grid_layout=args.grid_layout,
        )
        module.configure(
            payload,
            geometry,
            int(stream.cuda_stream),
            BATCH_KERNARG_SIZE,
        )

        def launch() -> Any:
            module.launch()
            return output

        timed_output, microseconds = single._run_batched_cuda_event_timing(
            torch,
            launch,
            stream,
            num_warmup=args.warmup,
            num_iters=args.iters,
        )
        module.mark_stream_synchronized()
        event_timing = single.make_cuda_event_timing_row(
            BATCH_KERNEL_SYMBOL,
            args.iters,
            args.warmup,
            microseconds,
            args.device,
        )
        us = float(event_timing["device_time_avg"])

    flops = 2 * batch * m * n * k
    # Distinct bytes each matrix contributes once; A/B/sA/sB are always read,
    # but D is only touched by kernels that actually store.
    input_bytes = (
        inputs["A"].nbytes
        + inputs["B"].nbytes
        + inputs["sA"].nbytes
        + inputs["sB"].nbytes
    )
    logical_bytes = input_bytes + (output.nbytes if writes_output else 0)
    if writes_output:
        print(
            f"[gemm_batch_isa_runner] logical_bytes={logical_bytes} = "
            f"A+B+sA+sB ({input_bytes}) + D ({output.nbytes})"
        )
    else:
        print(
            f"[gemm_batch_isa_runner] logical_bytes={logical_bytes} = "
            f"A+B+sA+sB ({input_bytes}); excludes D ({output.nbytes}) "
            f"(no store found in ISA); TFLOPS is not reported"
        )
    row = {
        "intype": args.intype,
        "batch": batch,
        "M": m,
        "N": n,
        "K": k,
        "apre": args.apre,
        "init": args.init,
        "seed": args.seed,
        "dtype": args.dtype,
        "gfx": gfx,
        "ref hash128": reference_hash,
        "gemm_a4w4 us": round(us, 2),
        "gemm_a4w4 TFLOPS": (
            round(flops / us / 1e6, 1) if writes_output else "n/a (no store)"
        ),
        "gemm_a4w4 TB/s": round(
            single.bytes_per_microsecond_to_tbps(logical_bytes, us), 2
        ),
    }

    # Deferred so the timing rows are printed before the single whole-batch
    # comparison, which is allowed to fail without hiding them.
    def validate() -> tuple[BatchValidation, bool]:
        error = float(
            dependencies.check_allclose(
                reference,
                timed_output,
                rtol=1e-1,
                atol=1.0,
                msg="mxfp4 batched GEMM ISA runner",
            )
        )
        max_err_info, rel_l2 = single.float32_error_metrics(
            reference,
            timed_output,
        )
        validation = BatchValidation(
            reference_hash=reference_hash,
            output_hash=single.tensor_blake2b128(timed_output),
            error=error,
            max_abs=max_err_info[2],
            max_err_info=max_err_info,
            rel_l2=rel_l2,
        )
        row["gemm_a4w4 err"] = validation.error
        row["gemm_a4w4 out hash128"] = validation.output_hash
        row["gemm_a4w4 max_err_info"] = validation.max_err_info
        row["gemm_a4w4 rel_l2"] = validation.rel_l2
        return validation, validation.error == 0.0

    return row, event_timing, validate


def _print_mab_validation(item: MabValidation) -> None:
    error = item.mismatch_count / item.element_count
    dense_error = item.dense_mismatch_count / item.element_count
    print(
        f"[gemm_batch_isa_runner] MAB dense diagnostic: "
        f"primary ref hash128={item.reference_hash}; out hash128="
        f"{item.output_hash}; mismatches={item.mismatch_count}/"
        f"{item.element_count}; err={error}; max_abs={item.max_abs}; "
        f"rel_l2={item.rel_l2}; allclose={item.passed}; dense ref hash128="
        f"{item.dense_reference_hash}; dense err={dense_error}; "
        f"dense max_abs={item.dense_max_abs}; dense rel_l2="
        f"{item.dense_rel_l2}; batch mismatches="
        f"{item.batch_mismatch_counts}"
    )


def _keep_code_object_if_requested(
    args: argparse.Namespace,
    isa: Path,
    symbol: str,
    code_object: Path,
) -> None:
    if not (args.co_out or args.keep_co):
        return
    destination = (
        Path(os.path.expandvars(args.co_out)).expanduser().resolve()
        if args.co_out
        else isa.with_name(f"{symbol}.co")
    )
    if destination == isa:
        raise single.GemmIsaRunnerError(
            "--co-out must not overwrite the input ISA source"
        )
    if destination.suffix.lower() != ".co":
        raise single.GemmIsaRunnerError(
            f"--co-out must use a .co suffix, got: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(code_object, destination)
    print(f"[gemm_batch_isa_runner] kept code object: {destination}")


def _build_parser() -> argparse.ArgumentParser:
    parser = single._build_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == "isa":
            action.help = (
                "complete gfx1250 AMDGPU assembly source: the original MAB "
                "symbols require --batch 1; mab_tdm_gemm_full_batch and "
                "64x256_1x4_batch_ps support batched dispatch"
            )
        elif action.dest == "shape":
            action.help = (
                "per-matrix GEMM shape M,N,K; MAB modes require M%%16=0, "
                "N%%1024=0, and K%%512=0 "
                "(default for other modes: 18432,2048,7168)"
            )
        elif action.dest == "intype":
            action.choices = (*action.choices, "fp16")
            action.help = (
                "input format; MXFP4 modes require mxfp4 and mab_tdm_gemm "
                "requires fp16"
            )
        elif action.dest == "dtype":
            action.choices = (*action.choices, "fp16")
            action.help = (
                "output dtype; MXFP4 modes require bf16 and mab_tdm_gemm "
                "requires fp16"
            )
        elif action.dest == "init":
            action.choices = (*action.choices, MAB_INIT)
            action.help = (
                "input initialization; UniformRandom is reserved for "
                "mab_tdm_gemm and uses its documented deterministic range"
            )
        elif action.dest == "clang":
            action.help = (
                "AMDGPU clang executable (explicit --clang path/name; "
                f"otherwise fixed {single.DEFAULT_CLANG}; no ROCm/PATH fallback)"
            )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help=(
            f"number of independent batch-major GEMMs in one dispatch "
            f"(default: 1; maximum: {MAX_BATCH})"
        ),
    )
    parser.add_argument(
        "--grid-layout",
        choices=GRID_LAYOUT_CHOICES,
        default=DEFAULT_GRID_LAYOUT,
        help=(
            "physical grid/cluster axis order: "
            "'n-on-x-m-on-y' keeps the legacy physical X=logical N and "
            "physical Y=logical M layout; 'm-on-x-n-on-y' requires the "
            f"permanent ISA variant marked by '.set "
            f"{MXNY_GRID_LAYOUT_MARKER}, 1' (default: "
            f"{DEFAULT_GRID_LAYOUT})"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iters < 1:
        parser.error("--iters must be at least 1")
    if args.device < 0:
        parser.error("--device must be non-negative")
    if args.batch < 1:
        parser.error("--batch must be at least 1")
    if args.batch > MAX_BATCH:
        parser.error(f"--batch must not exceed {MAX_BATCH}")

    try:
        isa = single._resolve_isa(args.isa)
        source = single._read_isa_source(isa)
        symbol = single.resolve_kernel_symbol_from_text(source, args.symbol)
        mode, profile = select_kernel_mode(symbol, args.batch)
        writes_output = single.isa_writes_output(source)

        if mode in (
            "mab",
            "mab-full",
            "mab-full-batch",
            "mab-full-batch-loadonly",
            "mab-full-batch-loadonly-wv23-256k",
        ):
            _validate_mab_mode(args, symbol)
            resources = single.parse_assembly_resources(source, symbol)
            if mode in (
                "mab-full-batch-loadonly",
                "mab-full-batch-loadonly-wv23-256k",
            ) and writes_output:
                raise single.GemmIsaRunnerError(
                    f"{symbol} load-only contract requires stores to D=False"
                )
            if mode not in (
                "mab-full-batch-loadonly",
                "mab-full-batch-loadonly-wv23-256k",
            ) and not writes_output:
                raise single.GemmIsaRunnerError(
                    f"{symbol} contract requires FP16 D stores"
                )
            geometry = make_mab_launch_geometry(
                *args.shape,
                args.batch,
                symbol=symbol,
            )
        elif mode == "legacy":
            if args.grid_layout != DEFAULT_GRID_LAYOUT:
                raise single.GemmIsaRunnerError(
                    "the original non-batch compatibility kernel supports "
                    f"only --grid-layout {DEFAULT_GRID_LAYOUT!r}"
                )
            profile = single._validate_mode(
                args.intype,
                args.apre,
                args.dtype,
                symbol,
            )
            resources = single.parse_assembly_resources(source, symbol)
            geometry = single.make_launch_geometry(
                *args.shape,
                profile=profile,
            )
            print(
                "[gemm_batch_isa_runner] compatibility mode: batch=1 uses "
                "the original symbol, 80-byte ABI, and exact non-batch "
                "execution path"
            )
        else:
            _validate_batch_mode(args.intype, args.apre, args.dtype)
            validate_batch_isa_grid_layout(source, args.grid_layout)
            resources = single.parse_assembly_resources(source, symbol)
            geometry = make_batched_launch_geometry(
                *args.shape,
                batch=args.batch,
                grid_layout=args.grid_layout,
            )

        single.print_contract(
            single.ContractReport(
                runner=Path(__file__).stem,
                profile=profile,
                symbol=symbol,
                shape=args.shape,
                batch=args.batch,
                geometry=geometry,
                resources=resources,
                dynamic_lds_bytes=0,
                stores_to_d=writes_output,
            )
        )

        clang = single._resolve_clang(args.clang)
        if single._clang_uses_default_runtime_libraries(clang):
            single._prepend_default_clang_runtime_libraries()
        print(f"[gemm_batch_isa_runner] selected clang: {clang}")
        with tempfile.TemporaryDirectory(
            prefix="gemm_batch_isa_runner_"
        ) as temp:
            result = single.compile_isa(
                isa,
                clang,
                Path(temp),
                symbol,
            )
            for command in result.commands:
                print(
                    f"[gemm_batch_isa_runner] "
                    f"{single._format_command(command)}"
                )
            for patch in result.patches:
                print(f"[gemm_batch_isa_runner] {patch}")
            print(
                f"[gemm_batch_isa_runner] loading kernel symbol: {symbol}"
            )
            _keep_code_object_if_requested(
                args,
                isa,
                symbol,
                result.code_object,
            )

            if mode in (
                "mab",
                "mab-full",
                "mab-full-batch",
                "mab-full-batch-loadonly",
                "mab-full-batch-loadonly-wv23-256k",
            ):
                row, event_timing, validate = _run_mab_gemm(
                    args,
                    result.code_object,
                    geometry,
                    symbol,
                    resources.metadata_lds,
                )
            elif mode == "legacy":
                legacy_row, event_timing, legacy_passed = single._run_gemm(
                    args,
                    result.code_object,
                    symbol,
                    profile,
                    geometry,
                )
                row = dict(legacy_row)
                row["batch"] = 1
                max_err_info = row["gemm_a4w4 max_err_info"]
                legacy_validation = BatchValidation(
                    reference_hash=str(row["ref hash128"]),
                    output_hash=str(row["gemm_a4w4 out hash128"]),
                    error=float(row["gemm_a4w4 err"]),
                    max_abs=float(max_err_info[2]),
                    max_err_info=max_err_info,
                    rel_l2=float(row["gemm_a4w4 rel_l2"]),
                )

                def validate() -> tuple[BatchValidation, bool]:
                    return legacy_validation, legacy_passed

            else:
                row, event_timing, validate = _run_batch_gemm(
                    args,
                    result.code_object,
                    geometry,
                    writes_output,
                )

            row["logical cluster tasks/plane"] = geometry.logical_cluster_tasks
            row["physical clusters/plane"] = (
                geometry.cluster_grid[0] * geometry.cluster_grid[1]
            )
            row["physical WGs/plane"] = geometry.grid[0] * geometry.grid[1]
            row["encoded recurrence stride/plane"] = geometry.persistent_stride
            single._print_cuda_event_timing(event_timing)
            validation, passed = validate()
            if mode in ("mab", "mab-full", "mab-full-batch"):
                _print_mab_validation(validation)
            row = {column: row[column] for column in single.SUMMARY_COLUMNS}
            single.print_summary(row, symbol=symbol, profile=profile)
            return 0 if passed else 3
    except single.CompileError as exc:
        print(
            f"[gemm_batch_isa_runner] compile failed:\n{exc}",
            file=sys.stderr,
        )
        return 2
    except single.GemmIsaRunnerError as exc:
        print(
            f"[gemm_batch_isa_runner] error: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
