#!/usr/bin/env python3
r"""Compile, validate, and benchmark batched gfx1250 MXFP4 GEMM assembly.

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
MXFP4 paths.  It reproduces the original MAB launch for exactly
M=16,N=65536,K=16384,batch=1 with FP16 inputs/output, a 1x4 workgroup cluster,
and the launcher-observed 112-byte preload ABI.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import io
import os
import re
import shutil
import struct
import sys
import tempfile
from collections import Counter
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
# gfx1250 shares one 384 KiB SRAM per WGP between LDS and the vector cache and
# caps the LDS partition at 320 KiB, so this bounds one workgroup's LDS and
# also fixes how many workgroups the declared size lets a WGP hold.
MAX_LDS_PER_WGP = 320 * 1024
# Mnemonic prefixes that write memory outside LDS.  ds_* is deliberately absent:
# it targets LDS, not the D buffer.
STORE_MNEMONIC_PREFIXES = (
    "tensor_store",
    "global_store",
    "global_atomic",
    "buffer_store",
    "buffer_atomic",
    "flat_store",
    "flat_atomic",
    "scratch_store",
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
MAB_SHAPE = (16, 65536, 16384)
MAB_BATCH = 1
MAB_INIT = "UniformRandom"
MAB_RANDOM_RANGE = (-1.0, 1.0)
MAB_KERNARG_SIZE = 112
MAB_METADATA_GROUP_SEGMENT_SIZE = 256 * 1024
MAB_GRID = (1, 256, 1)
MAB_BLOCK = (128, 1, 1)
MAB_CLUSTER = (1, 4, 1)
MAB_CLUSTER_GRID = (1, 64)
MAB_REFERENCE_CHUNK_N = 2048
# CDNA5 ISA section 7.1 requires an unused VOP3/VOP3P SRC field to be 0x80
# (inline zero).  LLVM 23 canonicalizes the unused SRC2[8:0] in these 36
# two-source instructions from the original object's 0x00 (SGPR0) to 0x80.
# SRC2 is not consumed, so this encoding difference is semantically inert and
# is unrelated to S_SET_VGPR_MSB.
MAB_CANONICAL_TEXT_SIZE = 3992
MAB_CANONICAL_TEXT_SHA256 = (
    "6fa26b0e9e5b20a9f67185a55eb9a68eac25f7c604adf9c939fff2fa79aef200"
)
MAB_RESTORED_MNEMONICS = (
    "v_mul_lo_u32 v0, s63, v0",
    "v_mul_lo_u32 v5, s63, v5",
    "v_mul_lo_u32 v2, s63, v2",
    "v_mul_lo_u32 v4, s90, v4",
    "v_mul_lo_u32 v1, s90, v1",
    "v_add_co_u32 v221, vcc_lo, 0x880, v221",
    "v_add_nc_u32_e64 v221, v221, s94",
    "v_mul_lo_u32 v6, s63, v6",
    "v_mul_lo_u32 v4, s63, v4",
    "v_add_co_u32 v193, vcc_lo, 0x880, v193",
    "v_sub_nc_u32_e64 v219, v219, 0x880",
    "v_add_nc_u32_e64 v218, v220, s61",
    "v_add_nc_u32_e64 v221, v223, s61",
    "v_add_nc_u32_e64 v218, v220, s61",
    "v_add_nc_u32_e64 v221, v223, s61",
    "v_lshlrev_b32_e64 v6, 5, s67",
    "v_mul_u32_u24_e64 v6, v6, s16",
    "v_mul_lo_u32 v4, v1, 1",
    "v_mul_lo_u32 v0, 16, v0",
    "v_mul_u32_u24_e64 v6, v6, s16",
    "v_cvt_pk_f16_f32 v4, v0 /*v256*/, v1 /*v257*/",
    "v_cvt_pk_f16_f32 v5, v2 /*v258*/, v3 /*v259*/",
    "v_cvt_pk_f16_f32 v6, v4 /*v260*/, v5 /*v261*/",
    "v_cvt_pk_f16_f32 v7, v6 /*v262*/, v7 /*v263*/",
    "v_cvt_pk_f16_f32 v16, v16 /*v272*/, v17 /*v273*/",
    "v_cvt_pk_f16_f32 v17, v18 /*v274*/, v19 /*v275*/",
    "v_cvt_pk_f16_f32 v18, v20 /*v276*/, v21 /*v277*/",
    "v_cvt_pk_f16_f32 v19, v22 /*v278*/, v23 /*v279*/",
    "v_cvt_pk_f16_f32 v32, v32 /*v288*/, v33 /*v289*/",
    "v_cvt_pk_f16_f32 v33, v34 /*v290*/, v35 /*v291*/",
    "v_cvt_pk_f16_f32 v34, v36 /*v292*/, v37 /*v293*/",
    "v_cvt_pk_f16_f32 v35, v38 /*v294*/, v39 /*v295*/",
    "v_cvt_pk_f16_f32 v48, v48 /*v304*/, v49 /*v305*/",
    "v_cvt_pk_f16_f32 v49, v50 /*v306*/, v51 /*v307*/",
    "v_cvt_pk_f16_f32 v50, v52 /*v308*/, v53 /*v309*/",
    "v_cvt_pk_f16_f32 v51, v54 /*v310*/, v55 /*v311*/",
)
MAB_DEFAULT_CLANG = single.DEFAULT_CLANG
MAB_LLVM_RUNTIME_LIBRARIES = single.DEFAULT_CLANG_RUNTIME_LIBRARIES
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
    k_multiple=16384,
    persistent_tg=256,
    persistent_grid_y=256,
    apre=1,
    abi_name="mab-tdm-preload-v1-observed-112",
    # These are the 64-byte descriptor values.  The metadata-only fixed LDS is
    # checked separately because the original object deliberately differs.
    kernarg_size=MAB_KERNARG_SIZE,
    kernarg_layout=MAB_KERNARG_LAYOUT,
    group_segment_fixed_size=0,
    next_free_vgpr=1024,
    next_free_sgpr=98,
    metadata_vgpr_count=1024,
    metadata_sgpr_count=97,
)

BATCH_SUMMARY_COLUMNS = (
    "intype",
    "batch",
    "M",
    "N",
    "K",
    "apre",
    "init",
    "seed",
    "dtype",
    "gfx",
    "logical cluster tasks/plane",
    "physical clusters/plane",
    "physical WGs/plane",
    "encoded recurrence stride/plane",
    "ref hash128",
    "gemm_a4w4 us",
    "gemm_a4w4 TFLOPS",
    "gemm_a4w4 TB/s",
    "gemm_a4w4 err",
    "gemm_a4w4 out hash128",
    "gemm_a4w4 max_abs",
    "gemm_a4w4 rel_l2",
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
    rel_l2: float


@dataclass(frozen=True)
class MabValidation:
    """Whole-tensor validation for the fixed MAB dispatch."""

    reference_hash: str
    output_hash: str
    max_abs: float
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


def _grid_layout_description(grid_layout: str) -> str:
    grid_layout = _validate_grid_layout(grid_layout)
    if grid_layout == GRID_LAYOUT_N_ON_X_M_ON_Y:
        return "physical X=logical N, physical Y=logical M"
    return "physical X=logical M, physical Y=logical N"


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


def make_mab_launch_geometry(
    m: int,
    n: int,
    k: int,
    batch: int,
) -> single.LaunchGeometry:
    """Return the one launch geometry proven from the original AQL packets."""

    shape = (m, n, k)
    if shape != MAB_SHAPE:
        raise single.GemmIsaRunnerError(
            f"{MAB_KERNEL_SYMBOL} supports only shape={MAB_SHAPE}, got {shape}"
        )
    if batch != MAB_BATCH:
        raise single.GemmIsaRunnerError(
            f"{MAB_KERNEL_SYMBOL} supports only batch={MAB_BATCH}, got {batch}"
        )
    return single.LaunchGeometry(
        grid=MAB_GRID,
        block=MAB_BLOCK,
        cluster=MAB_CLUSTER,
        tiles=(1, 256),
        cluster_grid=MAB_CLUSTER_GRID,
        log2_grid=(0, 6),
        logical_cluster_grid=MAB_CLUSTER_GRID,
        logical_wg_tasks=256,
        logical_cluster_tasks=64,
        persistent_stride=64,
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
) -> bytes:
    """Pack the exact 112 bytes intercepted from ``gemm_launcher``.

    The first two legacy size slots are zero in the launcher.  Stride1 is zero
    because this fixed mode has no batched dimension.  NumWorkGroups1 is the
    64-cluster Y grid, not the 256-workgroup physical Y grid.
    """

    expected_geometry = make_mab_launch_geometry(m, n, k, batch)
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

    values: dict[str, int | float] = {
        "sizeC": 0,
        "sizeA": 0,
        **pointer_values,
        "alpha": 1.0,
        "beta": 0.0,
        "strideD0": m,
        "strideD1": 0,
        "strideC0": m,
        "strideC1": 0,
        "strideA0": k,
        "strideA1": 0,
        "strideB0": k,
        "strideB1": 0,
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


def isa_writes_output(source: str) -> bool:
    """Report whether the assembly stores anything outside LDS."""

    for line in source.splitlines():
        clean = re.split(r";|//|#", line, maxsplit=1)[0].strip()
        if not clean or clean.startswith(".") or clean.endswith(":"):
            continue
        if clean.split(maxsplit=1)[0].startswith(STORE_MNEMONIC_PREFIXES):
            return True
    return False


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


def _non_comment_significant_lines(source: str) -> list[str]:
    lines: list[str] = []
    for line in source.splitlines():
        clean = single._strip_asm_comment(line)
        if not clean:
            continue
        normalized = re.sub(r"\s+", " ", clean)
        if normalized == f".set {MXNY_GRID_LAYOUT_MARKER}, 1":
            continue
        lines.append(normalized)
    return lines


def validate_mxny_isa_variant(
    legacy_source: str,
    mxny_source: str,
) -> int:
    """Prove the permanent MX/NY ISA differs only in four TTMP6 decodes."""

    validate_batch_isa_grid_layout(
        legacy_source,
        GRID_LAYOUT_N_ON_X_M_ON_Y,
    )
    validate_batch_isa_grid_layout(
        mxny_source,
        GRID_LAYOUT_M_ON_X_N_ON_Y,
    )
    legacy_lds = validate_batch_assembly_contract_from_text(legacy_source)
    mxny_lds = validate_batch_assembly_contract_from_text(mxny_source)
    if legacy_lds != 131072 or mxny_lds != legacy_lds:
        raise single.GemmIsaRunnerError(
            f"MX/NY comparison requires matching 128 KiB LDS variants, got "
            f"legacy={legacy_lds}, mxny={mxny_lds}"
        )

    legacy_lines = _non_comment_significant_lines(legacy_source)
    mxny_lines = _non_comment_significant_lines(mxny_source)
    if len(legacy_lines) != len(mxny_lines):
        raise single.GemmIsaRunnerError(
            "MX/NY variant changed the number of non-comment ISA/directive "
            f"lines: legacy={len(legacy_lines)}, mxny={len(mxny_lines)}"
        )
    differences = [
        (legacy, mxny)
        for legacy, mxny in zip(legacy_lines, mxny_lines)
        if legacy != mxny
    ]
    expected_differences = [
        (
            "s_bfe_u32 s52, ttmp6, 0x40010",
            "s_bfe_u32 s52, ttmp6, 0x4000c",
        ),
        (
            "s_bfe_u32 s51, ttmp6, 0x4000c",
            "s_bfe_u32 s51, ttmp6, 0x40010",
        ),
        (
            "s_bfe_u32 s50, ttmp6, 0x40004",
            "s_bfe_u32 s50, ttmp6, 0x40000",
        ),
        (
            "s_bfe_u32 s49, ttmp6, 0x40000",
            "s_bfe_u32 s49, ttmp6, 0x40004",
        ),
    ]
    if differences != expected_differences:
        raise single.GemmIsaRunnerError(
            "MX/NY variant must change only the four ordered TTMP6 physical-"
            f"to-logical axis decodes; got differences: {differences}"
        )

    tdm_load_count = sum(
        line.startswith("tensor_load_to_lds ")
        for line in legacy_lines
    )
    if tdm_load_count != 56:
        raise single.GemmIsaRunnerError(
            f"expected 56 unchanged tensor_load_to_lds instructions, got "
            f"{tdm_load_count}"
        )
    return tdm_load_count


def validate_batch_assembly_contract_from_text(source: str) -> int:
    """Validate symbol, resources, ABI, Z ID, and pointer-offset prologue.

    Returns the declared LDS size, which varies between the real kernel and the
    reduced-LDS bandwidth variants.
    """

    symbol = single.detect_kernel_symbol_from_text(source)
    if symbol != BATCH_KERNEL_SYMBOL:
        raise single.GemmIsaRunnerError(
            f"batch ISA symbol is {symbol!r}, expected "
            f"{BATCH_KERNEL_SYMBOL!r}"
        )

    descriptor = single._descriptor_body(source, symbol)
    metadata = single._metadata_body(source)
    descriptor_expected = {
        "amdhsa_private_segment_fixed_size": 0,
        "amdhsa_kernarg_size": BATCH_KERNARG_SIZE,
        "amdhsa_user_sgpr_count": 32,
        "amdhsa_user_sgpr_kernarg_segment_ptr": 1,
        "amdhsa_user_sgpr_kernarg_preload_length": 30,
        "amdhsa_user_sgpr_kernarg_preload_offset": 0,
        "amdhsa_system_sgpr_workgroup_id_x": 1,
        "amdhsa_system_sgpr_workgroup_id_y": 1,
        "amdhsa_system_sgpr_workgroup_id_z": 1,
        "amdhsa_wavefront_size32": 1,
        "amdhsa_next_free_vgpr": 384,
        "amdhsa_next_free_sgpr": 104,
        "amdhsa_named_barrier_count": 0,
    }
    for directive, expected in descriptor_expected.items():
        actual = single._descriptor_int(descriptor, directive)
        if actual != expected:
            raise single.GemmIsaRunnerError(
                f"{symbol}: .{directive} is {actual}, expected {expected}"
            )

    metadata_expected = {
        "kernarg_segment_align": 8,
        "kernarg_segment_size": BATCH_KERNARG_SIZE,
        "max_flat_workgroup_size": 128,
        "private_segment_fixed_size": 0,
        "sgpr_count": 106,
        "vgpr_count": 384,
        "wavefront_size": 32,
    }
    for field, expected in metadata_expected.items():
        actual = single._metadata_int(metadata, field)
        if actual != expected:
            raise single.GemmIsaRunnerError(
                f"{symbol}: metadata .{field} is {actual}, expected {expected}"
            )

    # The LDS size is the one resource the bandwidth variants are allowed to
    # shrink, so require descriptor/metadata agreement and a WGP-legal size
    # instead of one fixed value.
    group_segment = single._descriptor_int(
        descriptor,
        "amdhsa_group_segment_fixed_size",
    )
    metadata_group_segment = single._metadata_int(
        metadata,
        "group_segment_fixed_size",
    )
    if metadata_group_segment != group_segment:
        raise single.GemmIsaRunnerError(
            f"{symbol}: metadata .group_segment_fixed_size is "
            f"{metadata_group_segment}, but .amdhsa_group_segment_fixed_size "
            f"is {group_segment}"
        )
    if not 0 < group_segment <= MAX_LDS_PER_WGP:
        raise single.GemmIsaRunnerError(
            f"{symbol}: .amdhsa_group_segment_fixed_size is {group_segment}, "
            f"expected a positive size at most {MAX_LDS_PER_WGP}"
        )

    expected_args = tuple(
        (offset, struct.calcsize(f"<{kind}"))
        for _name, offset, kind in BATCH_KERNARG_LAYOUT
    )
    actual_args = single._metadata_arg_layout(metadata)
    if actual_args != expected_args:
        raise single.GemmIsaRunnerError(
            f"{symbol}: metadata kernarg layout {actual_args} does not match "
            f"batch ABI {expected_args}"
        )
    if not re.search(
        rf"(?m)^amdhsa\.target:[ \t]+"
        rf"amdgcn-amd-amdhsa--{re.escape(single.ARCH)}[ \t]*$",
        metadata,
    ):
        raise single.GemmIsaRunnerError(
            f"{symbol}: metadata target is not gfx1250"
        )

    required_prologue = (
        "s_lshr_b32 s32, ttmp7, 16",
        "s_mul_hi_u32 s35, s32, s22",
        "s_mul_i32 s34, s32, s23",
        "s_add_co_u32 s35, s35, s34",
        "s_mul_i32 s34, s32, s22",
        "s_add_nc_u64 s[2:3], s[2:3], s[34:35]",
        "s_mul_hi_u32 s35, s32, s24",
        "s_mul_i32 s34, s32, s25",
        "s_add_co_u32 s35, s35, s34",
        "s_mul_i32 s34, s32, s24",
        "s_add_nc_u64 s[4:5], s[4:5], s[34:35]",
        "s_mul_hi_u32 s35, s32, s26",
        "s_mul_i32 s34, s32, s27",
        "s_add_co_u32 s35, s35, s34",
        "s_mul_i32 s34, s32, s26",
        "s_add_nc_u64 s[6:7], s[6:7], s[34:35]",
        "s_mul_hi_u32 s35, s32, s28",
        "s_mul_i32 s34, s32, s29",
        "s_add_co_u32 s35, s35, s34",
        "s_mul_i32 s34, s32, s28",
        "s_add_nc_u64 s[8:9], s[8:9], s[34:35]",
        "s_mul_hi_u32 s35, s32, s30",
        "s_mul_i32 s34, s32, s31",
        "s_add_co_u32 s35, s35, s34",
        "s_mul_i32 s34, s32, s30",
        "s_add_nc_u64 s[10:11], s[10:11], s[34:35]",
    )
    entry = re.search(
        rf"(?m)^[ \t]*{re.escape(symbol)}:[ \t]*(?:[;#].*)?$",
        source,
    )
    if entry is None:
        raise single.GemmIsaRunnerError(
            f"{symbol}: kernel entry label is missing"
        )
    descriptor_start = re.search(
        rf"(?m)^[ \t]*\.amdhsa_kernel[ \t]+{re.escape(symbol)}[ \t]*$",
        source[entry.end() :],
    )
    if descriptor_start is None:
        raise single.GemmIsaRunnerError(
            f"{symbol}: kernel descriptor does not follow the entry body"
        )
    body_end = entry.end() + descriptor_start.start()
    instruction_lines: list[str] = []
    for line in source[entry.end() : body_end].splitlines():
        clean = re.split(r";|//|#", line, maxsplit=1)[0].strip()
        if not clean or clean.startswith(".") or clean.endswith(":"):
            continue
        instruction_lines.append(re.sub(r"\s+", " ", clean))

    # Exact global multiplicity catches duplicated adjustment instructions.
    # The high-limb add is intentionally present five times; every other
    # instruction is unique, so compare against the expected multiplicity
    # rather than reducing either sequence to a set.
    for instruction in dict.fromkeys(required_prologue):
        expected_count = required_prologue.count(instruction)
        actual_count = instruction_lines.count(instruction)
        if actual_count != expected_count:
            raise single.GemmIsaRunnerError(
                f"{symbol}: batch-Z prologue instruction {instruction!r} "
                f"occurs {actual_count} times, expected {expected_count}"
            )

    prologue_start = instruction_lines.index(required_prologue[0])
    actual_prologue = tuple(
        instruction_lines[
            prologue_start : prologue_start + len(required_prologue)
        ]
    )
    if actual_prologue != required_prologue:
        mismatch = next(
            (
                offset
                for offset, (actual, expected) in enumerate(
                    zip(actual_prologue, required_prologue)
                )
                if actual != expected
            ),
            min(len(actual_prologue), len(required_prologue)),
        )
        raise single.GemmIsaRunnerError(
            f"{symbol}: batch-Z prologue order/offset mismatch at normalized "
            f"instruction {mismatch}; got {actual_prologue}, expected "
            f"{required_prologue}"
        )

    prologue_end = prologue_start + len(required_prologue)
    original_pointer_capture = "s_mov_b32 s44, s2"
    capture_indices = [
        index
        for index, instruction in enumerate(instruction_lines)
        if instruction == original_pointer_capture
    ]
    if capture_indices != [prologue_end]:
        raise single.GemmIsaRunnerError(
            f"{symbol}: batch-Z adjustment must end immediately before the "
            f"first original D-pointer capture {original_pointer_capture!r}; "
            f"found at normalized offsets {capture_indices}, expected "
            f"[{prologue_end}]"
        )

    memory_prefixes = (
        "global_",
        "tensor_load",
        "tensor_store",
        "buffer_",
        "flat_",
        "s_load_",
    )
    first_memory_access = next(
        (
            index
            for index, instruction in enumerate(instruction_lines)
            if instruction.split(maxsplit=1)[0].startswith(memory_prefixes)
        ),
        None,
    )
    if first_memory_access is None or prologue_end > first_memory_access:
        raise single.GemmIsaRunnerError(
            f"{symbol}: batch-Z adjustment at normalized offsets "
            f"{prologue_start}:{prologue_end} must precede the first original "
            f"descriptor/global memory access; found memory offset "
            f"{first_memory_access}"
        )

    scheduler_instruction_counts = {
        "s_lshl_b32 s24, s24, s20": 1,
        "s_add_co_u32 s28, s24, ttmp9": 1,
        "s_add_co_i32 s24, s20, s21": 2,
        "s_lshl_b32 s24, 1, s24": 2,
        "s_add_co_u32 s28, s28, s24": 2,
    }
    for instruction, expected_count in scheduler_instruction_counts.items():
        actual_count = instruction_lines.count(instruction)
        if actual_count != expected_count:
            raise single.GemmIsaRunnerError(
                f"{symbol}: adaptive scheduler instruction {instruction!r} "
                f"occurs {actual_count} times, expected {expected_count}"
            )

    return group_segment


def validate_mab_assembly_contract_from_text(source: str) -> None:
    """Validate the fixed MAB source before clang or a GPU allocation."""

    symbol = single.detect_kernel_symbol_from_text(source)
    if symbol != MAB_KERNEL_SYMBOL:
        raise single.GemmIsaRunnerError(
            f"MAB ISA symbol is {symbol!r}, expected {MAB_KERNEL_SYMBOL!r}"
        )
    if re.search(r"(?m)^[ \t]*(?:MAB_TDMs:|\.amdhsa_kernel[ \t]+MAB_TDMs)\b", source):
        raise single.GemmIsaRunnerError(
            "the source still defines the obsolete MAB_TDMs kernel"
        )

    descriptor = single._descriptor_body(source, symbol)
    metadata = single._metadata_body(source)
    descriptor_expected = {
        "amdhsa_group_segment_fixed_size": 0,
        "amdhsa_private_segment_fixed_size": 0,
        "amdhsa_kernarg_size": MAB_KERNARG_SIZE,
        "amdhsa_user_sgpr_count": 30,
        "amdhsa_user_sgpr_dispatch_ptr": 0,
        "amdhsa_user_sgpr_queue_ptr": 0,
        "amdhsa_user_sgpr_kernarg_segment_ptr": 1,
        "amdhsa_user_sgpr_dispatch_id": 0,
        "amdhsa_user_sgpr_kernarg_preload_length": 28,
        "amdhsa_user_sgpr_kernarg_preload_offset": 0,
        "amdhsa_user_sgpr_private_segment_size": 0,
        "amdhsa_wavefront_size32": 1,
        "amdhsa_uses_dynamic_stack": 0,
        "amdhsa_enable_private_segment": 0,
        "amdhsa_system_sgpr_workgroup_id_x": 1,
        "amdhsa_system_sgpr_workgroup_id_y": 1,
        "amdhsa_system_sgpr_workgroup_id_z": 1,
        "amdhsa_system_sgpr_workgroup_info": 0,
        "amdhsa_system_vgpr_workitem_id": 2,
        "amdhsa_next_free_vgpr": 1024,
        "amdhsa_next_free_sgpr": 98,
        "amdhsa_named_barrier_count": 0,
        "amdhsa_reserve_vcc": 1,
        "amdhsa_float_round_mode_32": 0,
        "amdhsa_float_round_mode_16_64": 0,
        "amdhsa_float_denorm_mode_32": 0,
        "amdhsa_float_denorm_mode_16_64": 3,
        "amdhsa_fp16_overflow": 0,
        "amdhsa_memory_ordered": 1,
        "amdhsa_forward_progress": 1,
        "amdhsa_inst_pref_size": 30,
        "amdhsa_round_robin_scheduling": 0,
    }
    for directive, expected in descriptor_expected.items():
        actual = single._descriptor_int(descriptor, directive)
        if actual != expected:
            raise single.GemmIsaRunnerError(
                f"{symbol}: .{directive} is {actual}, expected {expected}"
            )

    metadata_expected = {
        "group_segment_fixed_size": MAB_METADATA_GROUP_SEGMENT_SIZE,
        "kernarg_segment_align": 8,
        "kernarg_segment_size": MAB_KERNARG_SIZE,
        "max_flat_workgroup_size": 128,
        "private_segment_fixed_size": 0,
        "sgpr_count": 97,
        "vgpr_count": 1024,
        "wavefront_size": 32,
    }
    for field, expected in metadata_expected.items():
        actual = single._metadata_int(metadata, field)
        if actual != expected:
            raise single.GemmIsaRunnerError(
                f"{symbol}: metadata .{field} is {actual}, expected {expected}"
            )
    expected_args = tuple(
        (offset, struct.calcsize(f"<{kind}"))
        for _name, offset, kind in MAB_KERNARG_LAYOUT
    )
    actual_args = single._metadata_arg_layout(metadata)
    if actual_args != expected_args:
        raise single.GemmIsaRunnerError(
            f"{symbol}: metadata kernarg layout {actual_args} does not match "
            f"the launcher-observed ABI {expected_args}"
        )
    if not re.search(
        rf"(?m)^amdhsa\.target:[ \t]+"
        rf"amdgcn-amd-amdhsa--{re.escape(single.ARCH)}[ \t]*$",
        metadata,
    ):
        raise single.GemmIsaRunnerError(
            f"{symbol}: metadata target is not gfx1250"
        )

    entry = re.search(
        rf"(?m)^[ \t]*{re.escape(symbol)}:[ \t]*(?://.*)?$",
        source,
    )
    descriptor_start = re.search(
        rf"(?m)^[ \t]*\.amdhsa_kernel[ \t]+{re.escape(symbol)}[ \t]*$",
        source,
    )
    if entry is None or descriptor_start is None or entry.end() >= descriptor_start.start():
        raise single.GemmIsaRunnerError(
            f"{symbol}: entry/body/descriptor ordering is invalid"
        )
    instruction_lines: list[str] = []
    uncommented_body: list[str] = []
    for line in source[entry.end() : descriptor_start.start()].splitlines():
        clean = re.split(r";|//|#", line, maxsplit=1)[0].strip()
        if not clean:
            continue
        uncommented_body.append(clean)
        if clean.startswith(".") or clean.endswith(":"):
            continue
        instruction_lines.append(re.sub(r"\s+", " ", clean))

    counts = {
        "s_version ": 1,
        "tensor_load_to_lds ": 5,
        "ds_load_b128 ": 120,
        "v_wmma_f32_16x16x32_f16 ": 48,
        "buffer_store_b128 ": 4,
        "s_set_vgpr_msb ": 34,
        "s_barrier_signal -3": 3,
        "s_endpgm": 1,
        "s_code_end": 5,
    }
    for prefix, expected in counts.items():
        actual = sum(
            line == prefix or line.startswith(prefix)
            for line in instruction_lines
        )
        if actual != expected:
            raise single.GemmIsaRunnerError(
                f"{symbol}: body pattern {prefix!r} occurs {actual} times, "
                f"expected {expected}"
            )
    raw_dword_lines = sum(
        line.startswith(".long ")
        for line in uncommented_body
    )
    if raw_dword_lines != 0:
        raise single.GemmIsaRunnerError(
            f"{symbol}: canonical mnemonic source must contain no raw .long "
            f"instructions, got {raw_dword_lines}"
        )
    expected_restored = Counter(MAB_RESTORED_MNEMONICS)
    actual_instructions = Counter(instruction_lines)
    restored_mismatches = {
        mnemonic: (actual_instructions[mnemonic], expected)
        for mnemonic, expected in expected_restored.items()
        if actual_instructions[mnemonic] != expected
    }
    if restored_mismatches:
        raise single.GemmIsaRunnerError(
            f"{symbol}: restored canonical mnemonic counts changed: "
            f"{restored_mismatches}"
        )

    expected_branches = (
        "s_cbranch_scc0 1",
        "s_cbranch_scc0 1",
        "s_cbranch_scc1 91",
        "s_cbranch_scc1 36",
        "s_branch 38",
        "s_cbranch_scc1 9",
        "s_cbranch_scc0 1",
        "s_branch 573",
        "s_cbranch_scc1 313",
        "s_cbranch_scc1 163",
        "s_cbranch_scc0 1",
        "s_cbranch_scc0 65375",
        "s_cbranch_scc0 1",
        "s_cbranch_scc0 203",
        "s_cbranch_scc0 201",
        "s_cbranch_scc1 195",
        "s_cbranch_scc1 188",
        "s_cbranch_scc0 184",
        "s_branch 0",
    )
    actual_branches = tuple(
        line
        for line in instruction_lines
        if line.startswith("s_cbranch_") or line.startswith("s_branch ")
    )
    if actual_branches != expected_branches:
        raise single.GemmIsaRunnerError(
            f"{symbol}: numeric branch sequence changed: {actual_branches}"
        )

    body_text = "\n".join(uncommented_body)
    required_abi_uses = (
        "s_mov_b32 s44, s6",
        "s_mov_b32 s48, s8",
        "s_add_co_u32 s36, s10, s94",
        "s_add_co_u32 s40, s12, s94",
        "s_cmp_eq_f32 s14, 1.0",
        "s_cmp_eq_f32 s15, 0",
        "s_mul_i32 s92, s16, 32",
        "s_mul_i32 s92, s94, s18",
        "s_mul_i32 s89, s89, s22",
        "s_and_b32 s92, 15, s24",
        "s_and_b32 s92, 0x7f, s25",
        "s_and_b32 s93, 0x7f, s27",
        "s_add_co_u32 s93, -1, s28",
        "s_add_co_u32 s93, -1, s29",
    )
    for instruction in required_abi_uses:
        if instruction not in body_text:
            raise single.GemmIsaRunnerError(
                f"{symbol}: required ABI-use pattern is missing: {instruction}"
            )
    for sgpr in range(2, 6):
        if re.search(rf"\bs{sgpr}\b", body_text):
            raise single.GemmIsaRunnerError(
                f"{symbol}: launcher-zero legacy size slot unexpectedly uses s{sgpr}"
            )


def _elf_named_section_bytes(data: bytes, name: str, label: str) -> bytes:
    sections = single._elf64_sections(data, label)
    (shstrndx,) = struct.unpack_from("<H", data, 0x3E)
    if shstrndx >= len(sections):
        raise single.GemmIsaRunnerError(
            f"{label}: invalid section-name string table {shstrndx}"
        )
    names = sections[shstrndx]
    matches = [
        section
        for section in sections
        if single._elf64_string(
            data,
            names,
            section.name_offset,
            label,
        )
        == name
    ]
    if len(matches) != 1:
        raise single.GemmIsaRunnerError(
            f"{label}: expected one {name} section, found {len(matches)}"
        )
    section = matches[0]
    return data[section.offset : section.offset + section.size]


def verify_mab_code_object(code_object: Path) -> None:
    """Check linked body bytes and every non-relocatable descriptor field."""

    label = str(code_object)
    data = code_object.read_bytes()
    text = _elf_named_section_bytes(data, ".text", label)
    digest = hashlib.sha256(text).hexdigest()
    if len(text) != MAB_CANONICAL_TEXT_SIZE or digest != MAB_CANONICAL_TEXT_SHA256:
        raise single.GemmIsaRunnerError(
            f"{label}: MAB .text is {len(text)} bytes with sha256={digest}; "
            f"expected {MAB_CANONICAL_TEXT_SIZE} and "
            f"{MAB_CANONICAL_TEXT_SHA256}"
        )

    sections = single._elf64_sections(data, label)
    descriptor_offset, _origin = single._find_kernel_descriptor_offset(
        data,
        sections,
        f"{MAB_KERNEL_SYMBOL}.kd",
        label,
    )
    descriptor = data[descriptor_offset : descriptor_offset + 64]
    group, private, kernarg = struct.unpack_from("<III", descriptor, 0)
    rsrc3, rsrc1, rsrc2 = struct.unpack_from("<III", descriptor, 44)
    properties, preload = struct.unpack_from("<HH", descriptor, 56)
    actual = (
        group,
        private,
        kernarg,
        rsrc3,
        rsrc1,
        rsrc2,
        properties,
        preload,
    )
    expected = (
        0,
        0,
        MAB_KERNARG_SIZE,
        0x000001E0,
        0xC00C003F,
        0x000013BC,
        0x0408,
        0x001C,
    )
    if actual != expected:
        raise single.GemmIsaRunnerError(
            f"{label}: MAB kernel descriptor fields {actual} do not match "
            f"the original descriptor {expected}"
        )
    if descriptor[12:16] != bytes(4) or descriptor[24:44] != bytes(20):
        raise single.GemmIsaRunnerError(
            f"{label}: MAB descriptor reserved fields are nonzero"
        )
    if descriptor[60:64] != bytes(4):
        raise single.GemmIsaRunnerError(
            f"{label}: MAB descriptor trailing reserved field is nonzero"
        )


def select_kernel_mode(
    symbol: str,
    batch: int,
) -> tuple[str, single.KernelProfile]:
    """Select one exact, symbol-scoped ABI and execution path."""

    batch = _validate_batch(batch)
    if symbol == MAB_KERNEL_SYMBOL:
        if batch != MAB_BATCH:
            raise single.GemmIsaRunnerError(
                f"{MAB_KERNEL_SYMBOL!r} supports only --batch {MAB_BATCH}"
            )
        return "mab", MAB_PROFILE
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
        f"{MAB_KERNEL_SYMBOL!r} and {LEGACY_KERNEL_SYMBOL!r} only for "
        f"batch=1, or {BATCH_KERNEL_SYMBOL!r} for any supported batch"
    )


def _validate_mab_mode(args: argparse.Namespace) -> None:
    if args.intype != "fp16":
        raise single.GemmIsaRunnerError(
            f"{MAB_KERNEL_SYMBOL} requires --intype fp16"
        )
    if args.dtype != "fp16":
        raise single.GemmIsaRunnerError(
            f"{MAB_KERNEL_SYMBOL} writes FP16 and requires --dtype fp16"
        )
    if args.apre != 1:
        raise single.GemmIsaRunnerError(
            f"{MAB_KERNEL_SYMBOL} has no A-preshuffle variant; keep --apre 1"
        )
    if args.init != MAB_INIT:
        raise single.GemmIsaRunnerError(
            f"{MAB_KERNEL_SYMBOL} supports only --init {MAB_INIT}"
        )
    make_mab_launch_geometry(*args.shape, args.batch)


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


def _assembly_contract_fixture() -> str:
    args = "\n".join(
        (
            f"      - .offset:         {offset}\n"
            f"        .size:           {struct.calcsize(f'<{kind}')}\n"
            "        .value_kind:     by_value"
        )
        for _name, offset, kind in BATCH_KERNARG_LAYOUT
    )
    prologue = "\n".join(
        (
            "    s_lshr_b32 s32, ttmp7, 16",
            "    s_mul_hi_u32 s35, s32, s22",
            "    s_mul_i32 s34, s32, s23",
            "    s_add_co_u32 s35, s35, s34",
            "    s_mul_i32 s34, s32, s22",
            "    s_add_nc_u64 s[2:3], s[2:3], s[34:35]",
            "    s_mul_hi_u32 s35, s32, s24",
            "    s_mul_i32 s34, s32, s25",
            "    s_add_co_u32 s35, s35, s34",
            "    s_mul_i32 s34, s32, s24",
            "    s_add_nc_u64 s[4:5], s[4:5], s[34:35]",
            "    s_mul_hi_u32 s35, s32, s26",
            "    s_mul_i32 s34, s32, s27",
            "    s_add_co_u32 s35, s35, s34",
            "    s_mul_i32 s34, s32, s26",
            "    s_add_nc_u64 s[6:7], s[6:7], s[34:35]",
            "    s_mul_hi_u32 s35, s32, s28",
            "    s_mul_i32 s34, s32, s29",
            "    s_add_co_u32 s35, s35, s34",
            "    s_mul_i32 s34, s32, s28",
            "    s_add_nc_u64 s[8:9], s[8:9], s[34:35]",
            "    s_mul_hi_u32 s35, s32, s30",
            "    s_mul_i32 s34, s32, s31",
            "    s_add_co_u32 s35, s35, s34",
            "    s_mul_i32 s34, s32, s30",
            "    s_add_nc_u64 s[10:11], s[10:11], s[34:35]",
        )
    )
    return f"""
.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl {BATCH_KERNEL_SYMBOL}
.type {BATCH_KERNEL_SYMBOL},@function
{BATCH_KERNEL_SYMBOL}:
{prologue}
    s_mov_b32 s44, s2
    global_prefetch_b8 v4, s[24:25]
    s_lshl_b32 s24, s24, s20
    s_add_co_u32 s28, s24, ttmp9
    s_add_co_i32 s24, s20, s21
    s_lshl_b32 s24, 1, s24
    s_add_co_u32 s28, s28, s24
    s_add_co_i32 s24, s20, s21
    s_lshl_b32 s24, 1, s24
    s_add_co_u32 s28, s28, s24
    s_endpgm
.amdhsa_kernel {BATCH_KERNEL_SYMBOL}
    .amdhsa_group_segment_fixed_size 206848
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 120
    .amdhsa_user_sgpr_count 32
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_user_sgpr_kernarg_preload_length 30
    .amdhsa_user_sgpr_kernarg_preload_offset 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr 384
    .amdhsa_next_free_sgpr 104
    .amdhsa_named_barrier_count 0
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
{args}
    .group_segment_fixed_size: 206848
    .kernarg_segment_align: 8
    .kernarg_segment_size: 120
    .max_flat_workgroup_size: 128
    .name: {BATCH_KERNEL_SYMBOL}
    .private_segment_fixed_size: 0
    .sgpr_count: 106
    .symbol: {BATCH_KERNEL_SYMBOL}.kd
    .vgpr_count: 384
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...
.end_amdgpu_metadata
"""


def run_static_contract_checks(
    actual_batch_source: str | None = None,
    legacy_candidate_source: str | None = None,
    mxny_candidate_source: str | None = None,
    mab_source: str | None = None,
) -> None:
    """CPU-only checks for contracts, compatibility, and clang selection."""

    single.run_static_contract_checks()
    if MAB_DEFAULT_CLANG != single.DEFAULT_CLANG:
        raise AssertionError(
            "batch and single runners use different fixed default clang paths"
        )
    mab_geometry = make_mab_launch_geometry(*MAB_SHAPE, MAB_BATCH)
    if (
        mab_geometry.grid != MAB_GRID
        or mab_geometry.block != MAB_BLOCK
        or mab_geometry.cluster != MAB_CLUSTER
        or mab_geometry.cluster_grid != MAB_CLUSTER_GRID
    ):
        raise AssertionError(f"unexpected fixed MAB geometry: {mab_geometry}")
    mab_attributes, mab_config = single._make_hip_launch_config(
        mab_geometry,
        ctypes.c_void_p(0x1234),
    )
    mab_cluster = mab_attributes[0].value.clusterDim
    if (
        (
            mab_config.gridDimX,
            mab_config.gridDimY,
            mab_config.gridDimZ,
        )
        != MAB_GRID
        or (
            mab_config.blockDimX,
            mab_config.blockDimY,
            mab_config.blockDimZ,
        )
        != MAB_BLOCK
        or (mab_cluster.x, mab_cluster.y, mab_cluster.z) != MAB_CLUSTER
        or mab_config.sharedMemBytes != 0
    ):
        raise AssertionError("HIP launch config changed the fixed MAB contract")
    mab_payload = pack_mab_kernargs(
        ptr_d=0x1111111122222222,
        ptr_c=0x3333333344444444,
        ptr_a=0x5555555566666666,
        ptr_b=0x7777777788888888,
        m=MAB_SHAPE[0],
        n=MAB_SHAPE[1],
        k=MAB_SHAPE[2],
        batch=MAB_BATCH,
        geometry=mab_geometry,
    )
    expected_mab_values: dict[str, int | float] = {
        "sizeC": 0,
        "sizeA": 0,
        "D": 0x1111111122222222,
        "C": 0x3333333344444444,
        "A": 0x5555555566666666,
        "B": 0x7777777788888888,
        "alpha": 1.0,
        "beta": 0.0,
        "strideD0": 16,
        "strideD1": 0,
        "strideC0": 16,
        "strideC1": 0,
        "strideA0": 16384,
        "strideA1": 0,
        "strideB0": 16384,
        "strideB1": 0,
        "SizesFree0": 16,
        "SizesFree1": 65536,
        "SizesFree2": 1,
        "SizesSum0": 16384,
        "NumWorkGroups0": 1,
        "NumWorkGroups1": 64,
    }
    if len(mab_payload) != MAB_KERNARG_SIZE:
        raise AssertionError("MAB kernarg size changed")
    for name, offset, kind in MAB_KERNARG_LAYOUT:
        actual = struct.unpack_from(f"<{kind}", mab_payload, offset)[0]
        expected = expected_mab_values[name]
        if actual != expected:
            raise AssertionError(
                f"MAB field {name} packed as {actual}, expected {expected}"
            )
    if select_kernel_mode(MAB_KERNEL_SYMBOL, 1) != ("mab", MAB_PROFILE):
        raise AssertionError("MAB symbol did not select its independent mode")
    if mab_source is not None:
        validate_mab_assembly_contract_from_text(mab_source)

    contract_fixture = _assembly_contract_fixture()
    validate_batch_assembly_contract_from_text(contract_fixture)

    first_instruction = "    s_lshr_b32 s32, ttmp7, 16"
    second_instruction = "    s_mul_hi_u32 s35, s32, s22"
    comment_tolerant_fixture = contract_fixture.replace(
        f"{first_instruction}\n{second_instruction}",
        f"{first_instruction} ; normalized comment\n\n"
        f"    // normalized comment-only line\n{second_instruction}",
        1,
    )
    validate_batch_assembly_contract_from_text(comment_tolerant_fixture)

    def expect_contract_rejection(
        source: str,
        label: str,
        expected_message: str,
    ) -> None:
        try:
            validate_batch_assembly_contract_from_text(source)
        except single.GemmIsaRunnerError as exc:
            if expected_message not in str(exc):
                raise AssertionError(
                    f"{label} produced unexpected contract rejection: {exc}"
                ) from exc
        else:
            raise AssertionError(f"batch assembly contract accepted {label}")

    reordered_fixture = contract_fixture.replace(
        f"{first_instruction}\n{second_instruction}",
        f"{second_instruction}\n{first_instruction}",
        1,
    )
    expect_contract_rejection(
        reordered_fixture,
        "reordered prologue instructions",
        "order/offset mismatch",
    )
    duplicated_fixture = contract_fixture.replace(
        first_instruction,
        f"{first_instruction}\n{first_instruction}",
        1,
    )
    expect_contract_rejection(
        duplicated_fixture,
        "duplicated prologue instruction",
        "occurs 2 times, expected 1",
    )
    pre_access_fixture = contract_fixture.replace(
        first_instruction,
        "    global_prefetch_b8 v4, s[24:25]\n"
        f"{first_instruction}",
        1,
    )
    expect_contract_rejection(
        pre_access_fixture,
        "global memory access before the batch prologue",
        "must precede the first original descriptor/global memory access",
    )

    if isa_writes_output(contract_fixture):
        raise AssertionError(
            "store detection treated the store-free fixture as writing D"
        )
    if isa_writes_output("\ts_nop 0 ; tensor_store_from_lds was removed\n"):
        raise AssertionError("store detection matched a comment")
    for store in (
        "tensor_store_from_lds s[32:35], s[36:43]",
        "global_store_b128 v[4:5], v[8:11], off",
        "buffer_store_dwordx4 v[8:11], v4, s[24:27], 0 offen",
    ):
        if not isa_writes_output(
            contract_fixture.replace(
                "    s_endpgm",
                f"    {store}\n    s_endpgm",
                1,
            )
        ):
            raise AssertionError(f"store detection missed {store!r}")

    if actual_batch_source is not None:
        if validate_batch_assembly_contract_from_text(
            actual_batch_source
        ) != 206848:
            raise AssertionError(
                "the repository batch ISA no longer declares 206848 LDS bytes"
            )
        if not isa_writes_output(actual_batch_source):
            raise AssertionError(
                "the repository batch ISA should store its D tiles"
            )
    if (legacy_candidate_source is None) != (mxny_candidate_source is None):
        raise AssertionError(
            "the permanent ISA comparison requires both legacy and MX/NY "
            "candidate sources"
        )
    if legacy_candidate_source is not None and mxny_candidate_source is not None:
        if validate_mxny_isa_variant(
            legacy_candidate_source,
            mxny_candidate_source,
        ) != 56:
            raise AssertionError("the permanent ISA pair changed its TDM count")

    marked_fixture = contract_fixture.replace(
        ".text",
        f".set {MXNY_GRID_LAYOUT_MARKER}, 1\n.text",
        1,
    )
    validate_batch_isa_grid_layout(
        contract_fixture,
        GRID_LAYOUT_N_ON_X_M_ON_Y,
    )
    validate_batch_isa_grid_layout(
        marked_fixture,
        GRID_LAYOUT_M_ON_X_N_ON_Y,
    )
    for source, requested_layout in (
        (contract_fixture, GRID_LAYOUT_M_ON_X_N_ON_Y),
        (marked_fixture, GRID_LAYOUT_N_ON_X_M_ON_Y),
    ):
        try:
            validate_batch_isa_grid_layout(source, requested_layout)
        except single.GemmIsaRunnerError as exc:
            if "requested --grid-layout" not in str(exc):
                raise AssertionError(
                    f"unexpected ISA/layout mismatch error: {exc}"
                ) from exc
        else:
            raise AssertionError(
                f"accepted ISA mismatch for {requested_layout!r}"
            )
    malformed_marker = marked_fixture.replace(
        f".set {MXNY_GRID_LAYOUT_MARKER}, 1",
        f".set {MXNY_GRID_LAYOUT_MARKER}, 0",
        1,
    )
    try:
        detect_batch_isa_grid_layout(malformed_marker)
    except single.GemmIsaRunnerError as exc:
        if "malformed or duplicated" not in str(exc):
            raise AssertionError(
                f"unexpected malformed marker error: {exc}"
            ) from exc
    else:
        raise AssertionError("accepted a malformed MX/NY ISA marker")

    if BATCH_PROFILE.wg_tile != single.KERNEL_PROFILE_64X256.wg_tile:
        raise AssertionError("batched profile changed the 64x256 tile")
    if (
        BATCH_PROFILE.cluster != (4, 1, 1)
        or BATCH_PROFILE.next_free_vgpr != 384
        or BATCH_PROFILE.next_free_sgpr != 104
        or BATCH_PROFILE.group_segment_fixed_size != 206848
    ):
        raise AssertionError("batched profile changed cluster/resource contracts")

    legacy_geometry = single.make_launch_geometry(
        64,
        1024,
        256,
        profile=single.KERNEL_PROFILE_64X256,
    )
    if (
        legacy_geometry.grid != (64, 4, 1)
        or legacy_geometry.cluster_grid != (16, 4)
        or legacy_geometry.log2_grid != (4, 2)
        or legacy_geometry.persistent_stride != 64
    ):
        raise AssertionError(
            f"legacy fixed geometry changed unexpectedly: {legacy_geometry}"
        )

    test_task_counts = (1, 2, 3, 6, 8, 9, 16, 17, 32, 33, 63, 64, 65, 576)
    geometry_by_layout_and_tasks: dict[
        tuple[str, int],
        single.LaunchGeometry,
    ] = {}
    for grid_layout in GRID_LAYOUT_CHOICES:
        for logical_tasks in test_task_counts:
            geometry = make_batched_launch_geometry(
                64,
                logical_tasks * 1024,
                256,
                7,
                grid_layout,
            )
            geometry_by_layout_and_tasks[(grid_layout, logical_tasks)] = (
                geometry
            )
            expected_clusters = min(
                logical_tasks,
                MAX_PERSISTENT_CLUSTERS,
            )
            if logical_tasks <= MAX_PERSISTENT_CLUSTERS:
                canonical_grid_x = logical_tasks
                canonical_grid_y = 1
                expected_stride = 1 << (logical_tasks - 1).bit_length()
                canonical_log2 = (
                    expected_stride.bit_length() - 1,
                    0,
                )
            else:
                canonical_grid_x, canonical_grid_y = (
                    MAX_PERSISTENT_CLUSTER_GRID
                )
                expected_stride = MAX_PERSISTENT_CLUSTERS
                canonical_log2 = (
                    canonical_grid_x.bit_length() - 1,
                    canonical_grid_y.bit_length() - 1,
                )
            if grid_layout == GRID_LAYOUT_N_ON_X_M_ON_Y:
                expected_grid_x, expected_grid_y = (
                    canonical_grid_x,
                    canonical_grid_y,
                )
                expected_log2 = canonical_log2
            else:
                expected_grid_x, expected_grid_y = (
                    canonical_grid_y,
                    canonical_grid_x,
                )
                expected_log2 = (canonical_log2[1], canonical_log2[0])
            expected_cluster = _physical_cluster_for_layout(grid_layout)
            expected_grid = (
                expected_grid_x * expected_cluster[0],
                expected_grid_y * expected_cluster[1],
                7,
            )
            if (
                geometry.grid != expected_grid
                or geometry.block != (128, 1, 1)
                or geometry.cluster != expected_cluster
                or geometry.cluster_grid
                != (expected_grid_x, expected_grid_y)
                or geometry.log2_grid != expected_log2
                or geometry.logical_cluster_tasks != logical_tasks
                or geometry.persistent_stride != expected_stride
            ):
                raise AssertionError(
                    f"unexpected {grid_layout} adaptive geometry for "
                    f"T_XY={logical_tasks}: {geometry}"
                )

            initial_tasks = [
                cluster_x + (cluster_y << geometry.log2_grid[0])
                for cluster_y in range(geometry.cluster_grid[1])
                for cluster_x in range(geometry.cluster_grid[0])
            ]
            if initial_tasks != list(range(expected_clusters)):
                raise AssertionError(
                    f"{grid_layout} T_XY={logical_tasks} initial p values "
                    f"are not [0,C): {initial_tasks}"
                )
            if (
                logical_tasks <= MAX_PERSISTENT_CLUSTERS
                and any(
                    seed + geometry.persistent_stride < logical_tasks
                    for seed in initial_tasks
                )
            ):
                raise AssertionError(
                    f"{grid_layout} T_XY={logical_tasks} exact small grid "
                    "entered a second persistent round"
                )
            visited = [
                task
                for initial_task in initial_tasks
                for task in range(
                    initial_task,
                    logical_tasks,
                    geometry.persistent_stride,
                )
            ]
            if len(visited) != logical_tasks or sorted(visited) != list(
                range(logical_tasks)
            ):
                raise AssertionError(
                    f"{grid_layout} T_XY={logical_tasks} adaptive recurrence "
                    f"has duplicates or misses: {visited}"
                )

    geometry_one = make_batched_launch_geometry(
        64,
        1024,
        256,
        1,
        GRID_LAYOUT_N_ON_X_M_ON_Y,
    )
    geometry_seven = geometry_by_layout_and_tasks[
        (GRID_LAYOUT_N_ON_X_M_ON_Y, 1)
    ]

    attributes, config = single._make_hip_launch_config(
        geometry_seven,
        ctypes.c_void_p(0x1234),
    )
    cluster = attributes[0].value.clusterDim
    if (
        (config.gridDimX, config.gridDimY, config.gridDimZ) != (4, 1, 7)
        or (cluster.x, cluster.y, cluster.z) != (4, 1, 1)
    ):
        raise AssertionError("HIP launch config did not preserve batch grid Z")

    documented_old = make_batched_launch_geometry(
        64,
        65536,
        32768,
        1,
        GRID_LAYOUT_N_ON_X_M_ON_Y,
    )
    documented_new = make_batched_launch_geometry(
        64,
        65536,
        32768,
        1,
        GRID_LAYOUT_M_ON_X_N_ON_Y,
    )
    if (
        documented_old.grid != (256, 1, 1)
        or documented_old.cluster != (4, 1, 1)
        or documented_old.cluster_grid != (64, 1)
        or documented_new.grid != (1, 256, 1)
        or documented_new.cluster != (1, 4, 1)
        or documented_new.cluster_grid != (1, 64)
        or documented_old.logical_cluster_tasks != 64
        or documented_new.logical_cluster_tasks != 64
    ):
        raise AssertionError(
            "documented 64-cluster old/new axis tuples changed: "
            f"old={documented_old}, new={documented_new}"
        )

    observed_geometry = make_batched_launch_geometry(
        64,
        6144,
        7168,
        96,
        GRID_LAYOUT_N_ON_X_M_ON_Y,
    )
    observed_mxny_geometry = make_batched_launch_geometry(
        64,
        6144,
        7168,
        96,
        GRID_LAYOUT_M_ON_X_N_ON_Y,
    )
    if (
        observed_geometry.logical_cluster_tasks != 6
        or observed_geometry.cluster_grid != (6, 1)
        or observed_geometry.grid != (24, 1, 96)
        or observed_geometry.log2_grid != (3, 0)
        or observed_geometry.persistent_stride != 8
    ):
        raise AssertionError(
            f"unexpected M64/N6144 adaptive geometry: {observed_geometry}"
        )
    if (
        observed_mxny_geometry.logical_cluster_tasks != 6
        or observed_mxny_geometry.logical_cluster_grid
        != observed_geometry.logical_cluster_grid
        or observed_mxny_geometry.logical_wg_tasks
        != observed_geometry.logical_wg_tasks
        or observed_mxny_geometry.cluster_grid != (1, 6)
        or observed_mxny_geometry.grid != (1, 24, 96)
        or observed_mxny_geometry.cluster != (1, 4, 1)
        or observed_mxny_geometry.log2_grid != (0, 3)
        or observed_mxny_geometry.persistent_stride != 8
    ):
        raise AssertionError(
            "unexpected M64/N6144 M-on-X/N-on-Y geometry: "
            f"{observed_mxny_geometry}"
        )
    for grid_layout, geometry in (
        (GRID_LAYOUT_N_ON_X_M_ON_Y, observed_geometry),
        (GRID_LAYOUT_M_ON_X_N_ON_Y, observed_mxny_geometry),
    ):
        visited_wg_tiles: list[tuple[int, int]] = []
        for task in range(geometry.logical_cluster_tasks):
            cluster_n = task % geometry.logical_cluster_grid[0]
            cluster_m = task // geometry.logical_cluster_grid[0]
            for physical_wg_y in range(geometry.cluster[1]):
                for physical_wg_x in range(geometry.cluster[0]):
                    if grid_layout == GRID_LAYOUT_N_ON_X_M_ON_Y:
                        logical_wg_n = physical_wg_x
                        logical_wg_m = physical_wg_y
                    else:
                        logical_wg_n = physical_wg_y
                        logical_wg_m = physical_wg_x
                    visited_wg_tiles.append(
                        (
                            cluster_m * BATCH_PROFILE.cluster[1]
                            + logical_wg_m,
                            cluster_n * BATCH_PROFILE.cluster[0]
                            + logical_wg_n,
                        )
                    )
        expected_wg_tiles = [
            (0, logical_n)
            for logical_n in range(24)
        ]
        if (
            len(visited_wg_tiles) != 24
            or sorted(visited_wg_tiles) != expected_wg_tiles
        ):
            raise AssertionError(
                f"{grid_layout} physical-to-logical WG mapping has duplicate, "
                f"missing, or out-of-bounds tiles: {visited_wg_tiles}"
            )
    observed_attributes, observed_config = single._make_hip_launch_config(
        observed_geometry,
        ctypes.c_void_p(0x1234),
    )
    observed_cluster = observed_attributes[0].value.clusterDim
    if (
        (
            observed_config.gridDimX,
            observed_config.gridDimY,
            observed_config.gridDimZ,
        )
        != (24, 1, 96)
        or (
            observed_cluster.x,
            observed_cluster.y,
            observed_cluster.z,
        )
        != (4, 1, 1)
    ):
        raise AssertionError(
            "observed HIP launch config did not preserve exact grid/batch Z"
        )
    mxny_attributes, mxny_config = single._make_hip_launch_config(
        observed_mxny_geometry,
        ctypes.c_void_p(0x1234),
    )
    mxny_cluster = mxny_attributes[0].value.clusterDim
    if (
        (
            mxny_config.gridDimX,
            mxny_config.gridDimY,
            mxny_config.gridDimZ,
        )
        != (1, 24, 96)
        or (mxny_cluster.x, mxny_cluster.y, mxny_cluster.z) != (1, 4, 1)
    ):
        raise AssertionError(
            "M-on-X/N-on-Y HIP launch config changed grid, cluster, or batch Z"
        )
    observed_strides = make_contiguous_batch_strides(
        64,
        6144,
        7168,
        96,
    )
    observed_payload = pack_batched_mxfp4_kernargs(
        ptr_d=1,
        ptr_a=2,
        ptr_b=3,
        ptr_scale_a=4,
        ptr_scale_b=5,
        m=64,
        n=6144,
        k=7168,
        batch=96,
        batch_strides=observed_strides,
        geometry=observed_geometry,
    )
    if struct.unpack_from("<II", observed_payload, 72) != (3, 0):
        raise AssertionError(
            "observed batch ABI did not encode stride-8 log2 grid (3,0)"
        )
    observed_mxny_payload = pack_batched_mxfp4_kernargs(
        ptr_d=1,
        ptr_a=2,
        ptr_b=3,
        ptr_scale_a=4,
        ptr_scale_b=5,
        m=64,
        n=6144,
        k=7168,
        batch=96,
        batch_strides=observed_strides,
        geometry=observed_mxny_geometry,
        grid_layout=GRID_LAYOUT_M_ON_X_N_ON_Y,
    )
    if struct.unpack_from("<II", observed_mxny_payload, 72) != (0, 3):
        raise AssertionError(
            "MX/NY batch ABI did not encode physical stride-8 log2 grid (0,3)"
        )

    strides = make_contiguous_batch_strides(64, 1024, 256, 1)
    batch_payload = pack_batched_mxfp4_kernargs(
        ptr_d=0x1010101010101010,
        ptr_a=0x2020202020202020,
        ptr_b=0x3030303030303030,
        ptr_scale_a=0x4040404040404040,
        ptr_scale_b=0x5050505050505050,
        m=64,
        n=1024,
        k=256,
        batch=1,
        batch_strides=strides,
        geometry=geometry_one,
    )
    legacy_payload = single.pack_mxfp4_kernargs(
        ptr_d=0x1010101010101010,
        ptr_a=0x2020202020202020,
        ptr_b=0x3030303030303030,
        ptr_scale_a=0x4040404040404040,
        ptr_scale_b=0x5050505050505050,
        m=64,
        n=1024,
        k=256,
        geometry=legacy_geometry,
        profile=single.KERNEL_PROFILE_64X256,
    )
    if len(batch_payload) != 120 or batch_payload[:72] != legacy_payload[:72]:
        raise AssertionError(
            "batch ABI does not preserve legacy fields before adaptive log2 "
            "grid values"
        )
    if struct.unpack_from("<II", batch_payload, 72) != geometry_one.log2_grid:
        raise AssertionError("batch ABI did not pack adaptive log2 grid X/Y")
    expected_strides = {
        "batch_stride_D": strides.d,
        "batch_stride_A": strides.a,
        "batch_stride_B": strides.b,
        "batch_stride_ScaleA": strides.scale_a,
        "batch_stride_ScaleB": strides.scale_b,
    }
    for name, offset, kind in BATCH_KERNARG_LAYOUT:
        if name not in expected_strides:
            continue
        actual = struct.unpack_from(f"<{kind}", batch_payload, offset)[0]
        if actual != expected_strides[name]:
            raise AssertionError(
                f"{name} packed as {actual}, expected {expected_strides[name]}"
            )

    geometry_two = make_batched_launch_geometry(64, 1024, 256, 2)
    wide_strides = BatchStrides(
        d=0x100000010,
        a=0x200000020,
        b=0x300000030,
        scale_a=0x400000040,
        scale_b=0x500000050,
    )
    wide_payload = pack_batched_mxfp4_kernargs(
        ptr_d=1,
        ptr_a=2,
        ptr_b=3,
        ptr_scale_a=4,
        ptr_scale_b=5,
        m=64,
        n=1024,
        k=256,
        batch=2,
        batch_strides=wide_strides,
        geometry=geometry_two,
    )
    for name, offset, kind in BATCH_KERNARG_LAYOUT:
        if not name.startswith("batch_stride_"):
            continue
        expected = {
            "batch_stride_D": wide_strides.d,
            "batch_stride_A": wide_strides.a,
            "batch_stride_B": wide_strides.b,
            "batch_stride_ScaleA": wide_strides.scale_a,
            "batch_stride_ScaleB": wide_strides.scale_b,
        }[name]
        actual = struct.unpack_from(f"<{kind}", wide_payload, offset)[0]
        if actual != expected:
            raise AssertionError(
                f"64-bit {name} packed as 0x{actual:x}, expected "
                f"0x{expected:x}"
            )

    # Mirror the ISA's two-limb multiply and verify it beyond 32-bit offsets.
    batch_index = MAX_BATCH - 1
    stride = 0x0000100000001234
    stride_lo = stride & 0xFFFFFFFF
    stride_hi = stride >> 32
    low_product = batch_index * stride_lo
    product_lo = low_product & 0xFFFFFFFF
    product_hi = (
        (low_product >> 32) + batch_index * stride_hi
    ) & 0xFFFFFFFF
    limb_product = product_lo | (product_hi << 32)
    if limb_product != batch_index * stride:
        raise AssertionError(
            f"u64 prologue limb product 0x{limb_product:x} differs from "
            f"0x{batch_index * stride:x}"
        )

    if select_kernel_mode(LEGACY_KERNEL_SYMBOL, 1)[0] != "legacy":
        raise AssertionError("legacy batch=1 compatibility was not selected")
    if select_kernel_mode(BATCH_KERNEL_SYMBOL, 1)[0] != "batch-z":
        raise AssertionError("batch symbol did not select batch-Z mode")
    if MAX_BATCH != 65535 or _validate_batch(65535) != 65535:
        raise AssertionError("HIP grid-Z maximum batch 65535 was not accepted")

    for invalid_batch in (0, -1, 65536):
        try:
            _validate_batch(invalid_batch)
        except single.GemmIsaRunnerError:
            pass
        else:
            raise AssertionError(f"accepted invalid batch {invalid_batch}")
    try:
        select_kernel_mode(LEGACY_KERNEL_SYMBOL, 2)
    except single.GemmIsaRunnerError as exc:
        if "has no batch addressing" not in str(exc):
            raise AssertionError(f"unexpected legacy rejection: {exc}") from exc
    else:
        raise AssertionError("legacy symbol accepted batch > 1")
    try:
        make_contiguous_batch_strides(
            0xFFFFFFFF,
            0xFFFFFFFF,
            256,
            1,
        )
    except single.GemmIsaRunnerError as exc:
        if "64-bit" not in str(exc) and "uint64" not in str(exc):
            raise AssertionError(f"unexpected overflow rejection: {exc}") from exc
    else:
        raise AssertionError("accepted a batch stride wider than uint64")
    try:
        make_batched_launch_geometry(64, 0x80000000, 256, 1)
    except single.GemmIsaRunnerError as exc:
        if "strideD0" not in str(exc):
            raise AssertionError(
                f"unexpected row-stride overflow rejection: {exc}"
            ) from exc
    else:
        raise AssertionError("accepted a row stride wider than uint32")

    parser = _build_parser()
    defaults = parser.parse_args(["--isa", "fixture.s"])
    if (
        defaults.batch != 1
        or defaults.grid_layout != DEFAULT_GRID_LAYOUT
        or defaults.clang is not None
    ):
        raise AssertionError(
            f"unexpected default batch/grid layout/clang: {defaults}"
        )
    clang_action = next(
        action for action in parser._actions if action.dest == "clang"
    )
    clang_help = clang_action.help or ""
    if (
        str(MAB_DEFAULT_CLANG) not in clang_help
        or "no ROCm/PATH fallback" not in clang_help
    ):
        raise AssertionError(
            f"batch --clang help does not describe the fixed-only policy: "
            f"{clang_help}"
        )
    explicit = parser.parse_args(
        [
            "--isa",
            "fixture.s",
            "--batch",
            "7",
            "--shape",
            "64,1024,256",
            "--grid-layout",
            GRID_LAYOUT_M_ON_X_N_ON_Y,
        ]
    )
    if (
        explicit.batch != 7
        or explicit.shape != (64, 1024, 256)
        or explicit.grid_layout != GRID_LAYOUT_M_ON_X_N_ON_Y
    ):
        raise AssertionError(f"unexpected parsed batch arguments: {explicit}")
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            parser.parse_args(
                [
                    "--isa",
                    "fixture.s",
                    "--grid-layout",
                    "ambiguous-axis-order",
                ]
            )
        except SystemExit as exc:
            if exc.code != 2:
                raise AssertionError(
                    f"argparse rejected bad grid layout with {exc.code}"
                ) from exc
        else:
            raise AssertionError("argparse accepted an invalid grid layout")


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
    dependencies: Any,
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
    first_inputs, first_reference = dependencies.f4_test._prep_mxfp4(
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
        inputs_i, reference_i = dependencies.f4_test._prep_mxfp4(
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


def _build_mab_references(
    torch_module: Any,
    a: Any,
    b: Any,
) -> tuple[Any, Any]:
    """Build both the dense target and the exact observed ISA result.

    Hardware probes against the original code object establish that its TDM
    pipeline omits repeatable K128 phases for part of every N256 tile.  The
    dense result remains a separately reported ``A @ B.T`` oracle; the primary
    pass/fail oracle reproduces the immutable instruction stream exactly.
    """

    m, k = tuple(a.shape)
    n, b_k = tuple(b.shape)
    if (m, n, k) != MAB_SHAPE or b_k != k:
        raise single.GemmIsaRunnerError(
            f"MAB reference got A={tuple(a.shape)}, B={tuple(b.shape)}"
        )
    dense_storage = torch_module.empty(
        (n, m),
        dtype=torch_module.float16,
        device=a.device,
    )
    isa_storage = torch_module.empty(
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
        k_index = torch_module.arange(k, device=a.device)
        phase_mod4 = (k_index // 128) % 4
        selectors = tuple(phase_mod4 == residue for residue in range(4))
        special_selector = (phase_mod4 == 1) & ((k_index % 128) < 32)
        a_parts = tuple(a[:, selector].float() for selector in selectors)
        a_special = a[:, special_selector].float()
        for start in range(0, n, MAB_REFERENCE_CHUNK_N):
            stop = min(start + MAB_REFERENCE_CHUNK_N, n)
            b_chunk = b[start:stop]
            parts = tuple(
                torch_module.mm(
                    a_parts[residue],
                    b_chunk[:, selector].float().transpose(0, 1),
                )
                for residue, selector in enumerate(selectors)
            )
            dense_tile = parts[0] + parts[1] + parts[2] + parts[3]
            isa_tile = dense_tile.clone()
            n_mod_256 = (
                torch_module.arange(start, stop, device=a.device) % 256
            )

            half_all = (n_mod_256 >= 192) & (n_mod_256 <= 209)
            isa_tile[:, half_all] = (parts[0] + parts[1])[:, half_all]

            special = n_mod_256 == 210
            if bool(special.any()):
                special_part = torch_module.mm(
                    a_special,
                    b_chunk[:, special_selector].float().transpose(0, 1),
                )
                isa_tile[:, special] = (parts[0] + special_part)[:, special]
                del special_part

            quarter_all = n_mod_256 >= 211
            isa_tile[:, quarter_all] = parts[0][:, quarter_all]

            half_lower_m = (n_mod_256 >= 64) & (n_mod_256 <= 127)
            isa_tile[8:, half_lower_m] = (
                parts[0] + parts[1]
            )[8:, half_lower_m]

            dense_storage[start:stop].copy_(
                dense_tile.transpose(0, 1).to(torch_module.float16)
            )
            isa_storage[start:stop].copy_(
                isa_tile.transpose(0, 1).to(torch_module.float16)
            )
            del parts, dense_tile, isa_tile
        del a_parts, a_special, selectors, special_selector, phase_mod4, k_index
    finally:
        torch_module.set_float32_matmul_precision(old_precision)
        if allow_tf32 is not None:
            torch_module.backends.cuda.matmul.allow_tf32 = allow_tf32
    return dense_storage, isa_storage


def _run_mab_gemm(
    args: argparse.Namespace,
    code_object: Path,
    geometry: single.LaunchGeometry,
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
    a = torch.empty((m, k), dtype=torch.float16, device=device)
    b = torch.empty((n, k), dtype=torch.float16, device=device)
    c_storage = torch.empty((n, m), dtype=torch.float16, device=device)
    for tensor in (a, b, c_storage):
        tensor.uniform_(low, high, generator=generator)
    output_storage = torch.full(
        (n, m),
        float("nan"),
        dtype=torch.float16,
        device=device,
    )
    expected_nbytes = {
        "A": 524288,
        "B": 2147483648,
        "C": 2097152,
        "D": 2097152,
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

    # Both references and their digests finish before any timed launch.
    dense_storage, reference_storage = _build_mab_references(torch, a, b)
    dense_reference = dense_storage.transpose(0, 1)
    reference = reference_storage.transpose(0, 1)
    dense_reference_hash = _tensor_blake2b128(dense_reference, torch)
    reference_hash = _tensor_blake2b128(reference, torch)
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
    )
    with single._LoadedClusterKernel(
        code_object,
        MAB_KERNEL_SYMBOL,
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
            MAB_KERNEL_SYMBOL,
            args.iters,
            args.warmup,
            microseconds,
            args.device,
        )
    us = float(event_timing["device_time_avg"])
    logical_bytes = int(a.nbytes + b.nbytes + output_storage.nbytes)
    flops = 2 * m * n * k
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
        "init": args.init,
        "seed": args.seed,
        "range": f"[{low},{high})",
        "dtype": args.dtype,
        "gfx": gfx,
        "ref hash128": reference_hash,
        "dense ref hash128": dense_reference_hash,
        "mab_tdm_gemm us": round(us, 3),
        "mab_tdm_gemm TFLOPS": round(flops / us / 1e6, 3),
        "mab_tdm_gemm TB/s": round(logical_bytes / us / 1e6, 3),
    }

    def validate() -> tuple[MabValidation, bool]:
        output = timed_output_storage.transpose(0, 1)
        reference_fp32 = reference.float()
        output_fp32 = output.float()
        difference = output_fp32 - reference_fp32
        max_abs = float(difference.abs().max().item())
        denominator = torch.linalg.vector_norm(reference_fp32)
        rel_l2 = float(
            (
                torch.linalg.vector_norm(difference)
                / torch.clamp(denominator, min=torch.finfo(torch.float32).tiny)
            ).item()
        )
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
        dense_reference_fp32 = dense_reference.float()
        dense_difference = output_fp32 - dense_reference_fp32
        dense_max_abs = float(dense_difference.abs().max().item())
        dense_rel_l2 = float(
            (
                torch.linalg.vector_norm(dense_difference)
                / torch.clamp(
                    torch.linalg.vector_norm(dense_reference_fp32),
                    min=torch.finfo(torch.float32).tiny,
                )
            ).item()
        )
        dense_mismatch_count = int(
            (
                ~torch.isclose(
                    output_fp32,
                    dense_reference_fp32,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=False,
                )
            )
            .sum()
            .item()
        )
        output_hash = _tensor_blake2b128(output, torch)
        validation = MabValidation(
            reference_hash=reference_hash,
            output_hash=output_hash,
            max_abs=max_abs,
            rel_l2=rel_l2,
            mismatch_count=mismatch_count,
            element_count=output.numel(),
            rtol=rtol,
            atol=atol,
            passed=mismatch_count == 0,
            dense_reference_hash=dense_reference_hash,
            dense_max_abs=dense_max_abs,
            dense_rel_l2=dense_rel_l2,
            dense_mismatch_count=dense_mismatch_count,
        )
        row["out hash128"] = output_hash
        row["max_abs"] = max_abs
        row["rel_l2"] = rel_l2
        row["allclose"] = validation.passed
        return validation, validation.passed

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
            dependencies,
            torch,
            batch=batch,
            m=m,
            n=n,
            k=k,
            apre=args.apre,
            dtype=dtype,
            init=args.init,
        )
        reference_hash = dependencies.f4_test._tensor_blake2b128(reference)
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
        "gemm_a4w4 TB/s": round(logical_bytes / us / 1e6, 2),
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
        max_abs, rel_l2 = dependencies.f4_test._float32_error_metrics(
            reference,
            timed_output,
        )
        validation = BatchValidation(
            reference_hash=reference_hash,
            output_hash=dependencies.f4_test._tensor_blake2b128(timed_output),
            error=error,
            max_abs=max_abs,
            rel_l2=rel_l2,
        )
        row["gemm_a4w4 err"] = validation.error
        row["gemm_a4w4 out hash128"] = validation.output_hash
        row["gemm_a4w4 max_abs"] = validation.max_abs
        row["gemm_a4w4 rel_l2"] = validation.rel_l2
        return validation, validation.error == 0.0

    return row, event_timing, validate


def _print_mab_contract(
    isa: Path,
    geometry: single.LaunchGeometry,
) -> None:
    print(
        f"[gemm_batch_isa_runner] selected profile: {MAB_PROFILE.name}; "
        f"exact shape={MAB_SHAPE}; batch={MAB_BATCH}; "
        "dense target=FP16(A @ B.T with FP32 accumulation); primary "
        "validation models the original ISA's measured K128-phase omissions"
    )
    print(
        f"[gemm_batch_isa_runner] physical launch={geometry.grid}; "
        f"block={geometry.block}; cluster={geometry.cluster}; physical "
        f"cluster grid={(*geometry.cluster_grid, 1)}; sharedMemBytes=0"
    )
    print(
        f"[gemm_batch_isa_runner] ABI={MAB_PROFILE.abi_name}; "
        f"kernarg={MAB_KERNARG_SIZE} bytes; descriptor fixed LDS=0; "
        f"metadata/AQL fixed group segment={MAB_METADATA_GROUP_SEGMENT_SIZE}; "
        "next-free VGPR/SGPR=1024/98; dynamic VGPR=0; inst-pref=30; "
        "round-robin=0"
    )
    print(
        "[gemm_batch_isa_runner] MAB ABI scalars: "
        "sizeC=0 sizeA=0 alpha=1 beta=0; "
        "strideD=(16,0) strideC=(16,0) strideA=(16384,0) "
        "strideB=(16384,0); SizesFree=(16,65536,1) SizesSum0=16384; "
        "NumWorkGroups=(1,64)"
    )
    print(f"[gemm_batch_isa_runner] source: {isa}")
    print(f"[gemm_batch_isa_runner] symbol: {MAB_KERNEL_SYMBOL}")


def _print_mab_validation(item: MabValidation) -> None:
    print(
        f"[gemm_batch_isa_runner] MAB full tensor: "
        f"ISA ref hash128={item.reference_hash}; "
        f"out hash128={item.output_hash}; max_abs={item.max_abs}; "
        f"rel_l2={item.rel_l2}; allclose={item.passed} "
        f"(rtol={item.rtol}, atol={item.atol}, mismatches="
        f"{item.mismatch_count}/{item.element_count}); dense A@B.T "
        f"hash128={item.dense_reference_hash}; dense max_abs="
        f"{item.dense_max_abs}; dense rel_l2={item.dense_rel_l2}; "
        f"dense mismatches={item.dense_mismatch_count}/{item.element_count}"
    )


def _print_mab_summary(row: Mapping[str, Any]) -> None:
    print(
        "[gemm_batch_isa_runner] MAB summary: "
        f"shape=({row['M']},{row['N']},{row['K']}); "
        f"batch={row['batch']}; init={row['init']}; seed={row['seed']}; "
        f"range={row['range']}; dtype={row['dtype']}; "
        f"time={row['mab_tdm_gemm us']} us; "
        f"TFLOPS={row['mab_tdm_gemm TFLOPS']}; "
        f"TB/s(A+B+D)={row['mab_tdm_gemm TB/s']}"
    )


def _print_batch_contract(
    isa: Path,
    geometry: single.LaunchGeometry,
    batch: int,
    group_segment: int,
    writes_output: bool,
    grid_layout: str,
) -> None:
    per_plane_clusters = (
        geometry.cluster_grid[0] * geometry.cluster_grid[1]
    )
    per_plane_wgs = geometry.grid[0] * geometry.grid[1]
    print(
        f"[gemm_batch_isa_runner] selected profile: {BATCH_PROFILE.name}; "
        f"WG tile={BATCH_PROFILE.wg_tile}; wave tile="
        f"{BATCH_PROFILE.wave_tile}; batch={batch}"
    )
    if grid_layout != DEFAULT_GRID_LAYOUT:
        print(
            f"[gemm_batch_isa_runner] grid layout={grid_layout}; "
            f"{_grid_layout_description(grid_layout)}"
        )
    print(
        f"[gemm_batch_isa_runner] logical WG grid="
        f"{(*geometry.tiles, batch)}; WG tasks total="
        f"{geometry.logical_wg_tasks * batch}; logical cluster grid="
        f"{(*geometry.logical_cluster_grid, batch)}; cluster tasks/plane="
        f"{geometry.logical_cluster_tasks}; cluster tasks total="
        f"{geometry.logical_cluster_tasks * batch}"
    )
    print(
        f"[gemm_batch_isa_runner] physical launch={geometry.grid}; "
        f"block={geometry.block}; cluster={geometry.cluster}; physical "
        f"cluster grid={(*geometry.cluster_grid, batch)}; clusters/plane="
        f"{per_plane_clusters}; clusters total={per_plane_clusters * batch}; "
        f"physical WGs/plane={per_plane_wgs}; physical WGs total="
        f"{per_plane_wgs * batch}; encoded recurrence stride/plane="
        f"{geometry.persistent_stride}; log2 X/Y grid={geometry.log2_grid}"
    )
    print(
        f"[gemm_batch_isa_runner] ABI={BATCH_PROFILE.abi_name}; "
        f"kernarg=120 bytes (legacy 80-byte prefix + five u64 byte "
        f"strides); fixed LDS={group_segment}; WGs/WGP by LDS="
        f"{MAX_LDS_PER_WGP // group_segment} (LDS budget "
        f"{MAX_LDS_PER_WGP}); VGPR/SGPR metadata=384/106; stores to D="
        f"{writes_output}"
    )
    print(f"[gemm_batch_isa_runner] source: {isa}")
    print(f"[gemm_batch_isa_runner] symbol: {BATCH_KERNEL_SYMBOL}")


def _print_validation(item: BatchValidation) -> None:
    print(
        f"[gemm_batch_isa_runner] full batch: "
        f"ref hash128={item.reference_hash}; "
        f"out hash128={item.output_hash}; err={item.error}; "
        f"max_abs={item.max_abs}; rel_l2={item.rel_l2}"
    )


def _print_summary(
    row: Mapping[str, Any],
    *,
    symbol: str,
    profile: single.KernelProfile,
) -> None:
    pandas = single._load_pandas_from_row_context()
    frame = pandas.DataFrame([row], columns=BATCH_SUMMARY_COLUMNS)
    try:
        markdown = frame.to_markdown(index=False)
    except ImportError:
        markdown = single._plain_markdown(row, BATCH_SUMMARY_COLUMNS)
    print(
        "gemm_a4w4 (batched F4GEMM ISA runner) summary "
        f"(markdown; profile={profile.name}; symbol={symbol}):"
    )
    print(markdown)


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
            action.required = False
            action.help = (
                "complete gfx1250 AMDGPU assembly source: mab_tdm_gemm and "
                "the original 64x256_1x4_ps symbol require --batch 1; the "
                "64x256_1x4_batch_ps symbol supports true batched dispatch"
            )
        elif action.dest == "shape":
            action.help = (
                "per-matrix GEMM shape M,N,K; mab_tdm_gemm requires exactly "
                "16,65536,16384 (default for other modes: 18432,2048,7168)"
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
                f"otherwise fixed {MAB_DEFAULT_CLANG}; no ROCm/PATH fallback)"
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
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "run CPU-only ABI/geometry/overflow checks and validate the "
            "repository batch and MAB ISA sources, without compiling or "
            "using a GPU"
        ),
    )
    return parser


def _select_clang(override: str | None) -> tuple[Path, bool]:
    """Apply the single-runner clang policy and report runtime-library needs."""

    clang = single._resolve_clang(override)
    return clang, single._clang_uses_default_runtime_libraries(clang)


def _prepend_llvm_runtime_libraries() -> None:
    """Prepend LLVM runtime dependencies without dropping user entries."""

    requested = [str(path) for path in MAB_LLVM_RUNTIME_LIBRARIES]
    existing = [
        entry
        for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if entry
    ]
    combined: list[str] = []
    for entry in (*requested, *existing):
        if entry not in combined:
            combined.append(entry)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(combined)


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
        if args.self_test:
            repository_batch_isa = (
                single._REPO
                / "my_code"
                / "f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.s"
            )
            candidate_directory = (
                single._REPO
                / "my_code"
                / "f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps"
            )
            legacy_candidate_isa = candidate_directory / (
                "f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps_"
                "loadonly_lds128k.s"
            )
            mxny_candidate_isa = candidate_directory / (
                "f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps_"
                "loadonly_lds128k_mxny.s"
            )
            mab_isa = single._REPO / "my_code" / "MAB" / "mab_tdm_gemm.s"
            source = single._read_isa_source(repository_batch_isa)
            legacy_candidate_source = single._read_isa_source(
                legacy_candidate_isa
            )
            mxny_candidate_source = single._read_isa_source(mxny_candidate_isa)
            mab_source = single._read_isa_source(mab_isa)
            run_static_contract_checks(
                source,
                legacy_candidate_source,
                mxny_candidate_source,
                mab_source,
            )
            print(
                "[gemm_batch_isa_runner] self-test passed: batch ABI, "
                "batch=1 compatibility, both physical axis layouts, adaptive "
                "geometry/task coverage, strict ISA markers, 56-instruction "
                "TDM equivalence, overflow gates, HIP launch config, fixed "
                "clang policy, MAB ABI/geometry/body, and repository ISA "
                "contracts"
            )
            return 0

        if args.isa is None:
            parser.error("--isa is required unless --self-test is used")
        run_static_contract_checks()
        isa = single._resolve_isa(args.isa)
        source = single._read_isa_source(isa)
        symbol = single.resolve_kernel_symbol_from_text(source, args.symbol)
        mode, profile = select_kernel_mode(symbol, args.batch)
        writes_output = isa_writes_output(source)

        if mode == "mab":
            _validate_mab_mode(args)
            validate_mab_assembly_contract_from_text(source)
            if not writes_output:
                raise single.GemmIsaRunnerError(
                    f"{MAB_KERNEL_SYMBOL} contract requires FP16 D stores"
                )
            geometry = make_mab_launch_geometry(*args.shape, args.batch)
            _print_mab_contract(isa, geometry)
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
            single.validate_assembly_contract_from_text(
                source,
                symbol,
                profile,
            )
            geometry = single.make_launch_geometry(
                *args.shape,
                profile=profile,
            )
            single._print_selected_contract(
                isa,
                symbol,
                profile,
                geometry,
            )
            print(
                "[gemm_batch_isa_runner] compatibility mode: batch=1 uses "
                "the original symbol, 80-byte ABI, and exact non-batch "
                "execution path"
            )
        else:
            _validate_batch_mode(args.intype, args.apre, args.dtype)
            validate_batch_isa_grid_layout(source, args.grid_layout)
            group_segment = validate_batch_assembly_contract_from_text(source)
            geometry = make_batched_launch_geometry(
                *args.shape,
                batch=args.batch,
                grid_layout=args.grid_layout,
            )
            _print_batch_contract(
                isa,
                geometry,
                args.batch,
                group_segment,
                writes_output,
                args.grid_layout,
            )

        clang, uses_in_tree_llvm = _select_clang(args.clang)
        if uses_in_tree_llvm:
            _prepend_llvm_runtime_libraries()
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
            if mode == "mab":
                verify_mab_code_object(result.code_object)
                print(
                    f"[gemm_batch_isa_runner] verified canonical mnemonic .text: "
                    f"{MAB_CANONICAL_TEXT_SIZE} bytes, sha256="
                    f"{MAB_CANONICAL_TEXT_SHA256}"
                )
            print(
                f"[gemm_batch_isa_runner] loading kernel symbol: {symbol}"
            )
            _keep_code_object_if_requested(
                args,
                isa,
                symbol,
                result.code_object,
            )

            if mode == "mab":
                row, event_timing, validate = _run_mab_gemm(
                    args,
                    result.code_object,
                    geometry,
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
                legacy_validation = BatchValidation(
                    reference_hash=str(row["ref hash128"]),
                    output_hash=str(row["gemm_a4w4 out hash128"]),
                    error=float(row["gemm_a4w4 err"]),
                    max_abs=float(row["gemm_a4w4 max_abs"]),
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

            row["logical cluster tasks/plane"] = (
                geometry.logical_cluster_tasks
            )
            row["physical clusters/plane"] = (
                geometry.cluster_grid[0] * geometry.cluster_grid[1]
            )
            row["physical WGs/plane"] = geometry.grid[0] * geometry.grid[1]
            row["encoded recurrence stride/plane"] = (
                geometry.persistent_stride
            )
            single._print_cuda_event_timing(event_timing)
            validation, passed = validate()
            if mode == "mab":
                _print_mab_validation(validation)
                _print_mab_summary(row)
            else:
                _print_validation(validation)
                _print_summary(row, symbol=symbol, profile=profile)
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
