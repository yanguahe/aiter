#!/usr/bin/env python3
r"""Compile, validate, and benchmark batched gfx1250 MXFP4 GEMM assembly.

The batched kernel uses one physical grid-Z plane per independent matrix while
retaining the original 64x256 workgroup tile, 4x1 cluster, and persistent X/Y
task traversal in every plane.  Inputs are contiguous and batch-major:

    A[batch,M,K/2], B[batch,N,K/2], sA[batch,M,K/32],
    sB[batch,N,K/32], C[batch,M,N]

The batch-specific 120-byte ABI is the original 80-byte MXFP4 preload ABI
followed by five uint64 byte strides for C, A, B, sA, and sB.  One timed launch
is one kernel dispatch for the complete batch.

For compatibility, ``--batch 1`` also accepts the original non-batch
``..._64x256_1x4_ps`` source and uses its exact 80-byte ABI and launch path.
The original symbol is rejected for ``--batch > 1``; use the
``..._64x256_1x4_batch_ps`` source instead.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    """Correctness result for one matrix in the dispatch."""

    index: int
    reference_hash: str
    output_hash: str
    error: float
    max_abs: float
    rel_l2: float


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


def make_batched_launch_geometry(
    m: int,
    n: int,
    k: int,
    batch: int,
) -> single.LaunchGeometry:
    """Replicate the unchanged persistent X/Y launch in grid Z."""

    batch = _validate_batch(batch)
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
    return replace(base, grid=(base.grid[0], base.grid[1], batch))


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
) -> bytes:
    """Pack the exact 120-byte batch-Z preload ABI."""

    batch = _validate_batch(batch)
    if geometry.grid[2] != batch:
        raise single.GemmIsaRunnerError(
            f"grid Z={geometry.grid[2]} does not match batch={batch}"
        )
    if (
        geometry.block != BATCH_PROFILE.block
        or geometry.cluster != BATCH_PROFILE.cluster
    ):
        raise single.GemmIsaRunnerError(
            f"launch block/cluster {geometry.block}/{geometry.cluster} does "
            f"not match {BATCH_PROFILE.block}/{BATCH_PROFILE.cluster}"
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


def validate_batch_assembly_contract_from_text(source: str) -> None:
    """Validate symbol, resources, ABI, Z ID, and pointer-offset prologue."""

    symbol = single.detect_kernel_symbol_from_text(source)
    if symbol != BATCH_KERNEL_SYMBOL:
        raise single.GemmIsaRunnerError(
            f"batch ISA symbol is {symbol!r}, expected "
            f"{BATCH_KERNEL_SYMBOL!r}"
        )

    descriptor = single._descriptor_body(source, symbol)
    metadata = single._metadata_body(source)
    descriptor_expected = {
        "amdhsa_group_segment_fixed_size": 206848,
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
        "group_segment_fixed_size": 206848,
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


def select_kernel_mode(
    symbol: str,
    batch: int,
) -> tuple[str, single.KernelProfile]:
    """Select the legacy batch=1 path or the true batch-Z path."""

    batch = _validate_batch(batch)
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
        f"{LEGACY_KERNEL_SYMBOL!r} only for batch=1, or "
        f"{BATCH_KERNEL_SYMBOL!r} for any supported batch"
    )


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
) -> None:
    """CPU-only checks for ABI, geometry, overflow, and compatibility."""

    single.run_static_contract_checks()
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

    if actual_batch_source is not None:
        validate_batch_assembly_contract_from_text(actual_batch_source)

    if BATCH_PROFILE.wg_tile != single.KERNEL_PROFILE_64X256.wg_tile:
        raise AssertionError("batched profile changed the 64x256 tile")
    if (
        BATCH_PROFILE.cluster != (4, 1, 1)
        or BATCH_PROFILE.next_free_vgpr != 384
        or BATCH_PROFILE.next_free_sgpr != 104
        or BATCH_PROFILE.group_segment_fixed_size != 206848
    ):
        raise AssertionError("batched profile changed cluster/resource contracts")

    geometry_one = make_batched_launch_geometry(64, 1024, 256, 1)
    legacy_geometry = single.make_launch_geometry(
        64,
        1024,
        256,
        profile=single.KERNEL_PROFILE_64X256,
    )
    if geometry_one != legacy_geometry:
        raise AssertionError(
            f"batch=1 geometry {geometry_one} differs from legacy "
            f"{legacy_geometry}"
        )
    geometry_seven = make_batched_launch_geometry(64, 1024, 256, 7)
    if (
        geometry_seven.grid != (64, 4, 7)
        or geometry_seven.block != (128, 1, 1)
        or geometry_seven.cluster != (4, 1, 1)
        or geometry_seven.persistent_stride != 64
    ):
        raise AssertionError(f"unexpected batch geometry: {geometry_seven}")

    attributes, config = single._make_hip_launch_config(
        geometry_seven,
        ctypes.c_void_p(0x1234),
    )
    cluster = attributes[0].value.clusterDim
    if (
        (config.gridDimX, config.gridDimY, config.gridDimZ) != (64, 4, 7)
        or (cluster.x, cluster.y, cluster.z) != (4, 1, 1)
    ):
        raise AssertionError("HIP launch config did not preserve batch grid Z")

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
    if len(batch_payload) != 120 or batch_payload[:80] != legacy_payload:
        raise AssertionError(
            "batch ABI does not preserve the legacy 80-byte prefix"
        )
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
    if defaults.batch != 1:
        raise AssertionError(f"default batch is {defaults.batch}, expected 1")
    explicit = parser.parse_args(
        ["--isa", "fixture.s", "--batch", "7", "--shape", "64,1024,256"]
    )
    if explicit.batch != 7 or explicit.shape != (64, 1024, 256):
        raise AssertionError(f"unexpected parsed batch arguments: {explicit}")


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


def _run_batch_gemm(
    args: argparse.Namespace,
    code_object: Path,
    geometry: single.LaunchGeometry,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[BatchValidation, ...],
    bool,
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

    validations: list[BatchValidation] = []
    for index in range(batch):
        error = dependencies.check_allclose(
            reference[index],
            timed_output[index],
            rtol=1e-1,
            atol=1.0,
            msg=f"mxfp4 batched GEMM ISA runner batch={index}",
        )
        max_abs, rel_l2 = dependencies.f4_test._float32_error_metrics(
            reference[index],
            timed_output[index],
        )
        validations.append(
            BatchValidation(
                index=index,
                reference_hash=dependencies.f4_test._tensor_blake2b128(
                    reference[index]
                ),
                output_hash=dependencies.f4_test._tensor_blake2b128(
                    timed_output[index]
                ),
                error=float(error),
                max_abs=max_abs,
                rel_l2=rel_l2,
            )
        )

    output_hash = dependencies.f4_test._tensor_blake2b128(timed_output)
    aggregate_max_abs, aggregate_rel_l2 = (
        dependencies.f4_test._float32_error_metrics(reference, timed_output)
    )
    aggregate_error = max(item.error for item in validations)
    flops = 2 * batch * m * n * k
    logical_bytes = (
        inputs["A"].nbytes
        + inputs["B"].nbytes
        + inputs["sA"].nbytes
        + inputs["sB"].nbytes
        + output.nbytes
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
        "gemm_a4w4 TFLOPS": round(flops / us / 1e6, 1),
        "gemm_a4w4 TB/s": round(logical_bytes / us / 1e6, 2),
        "gemm_a4w4 err": aggregate_error,
        "gemm_a4w4 out hash128": output_hash,
        "gemm_a4w4 max_abs": aggregate_max_abs,
        "gemm_a4w4 rel_l2": aggregate_rel_l2,
    }
    passed = all(item.error == 0.0 for item in validations)
    return row, event_timing, tuple(validations), passed


def _print_batch_contract(
    isa: Path,
    geometry: single.LaunchGeometry,
    batch: int,
) -> None:
    per_plane_clusters = (
        geometry.cluster_grid[0] * geometry.cluster_grid[1]
    )
    print(
        f"[gemm_batch_isa_runner] selected profile: {BATCH_PROFILE.name}; "
        f"WG tile={BATCH_PROFILE.wg_tile}; wave tile="
        f"{BATCH_PROFILE.wave_tile}; batch={batch}"
    )
    print(
        f"[gemm_batch_isa_runner] logical WG grid="
        f"{(*geometry.tiles, batch)}; tasks="
        f"{geometry.logical_wg_tasks * batch}; logical cluster grid="
        f"{(*geometry.logical_cluster_grid, batch)}; cluster tasks="
        f"{geometry.logical_cluster_tasks * batch}"
    )
    print(
        f"[gemm_batch_isa_runner] physical launch={geometry.grid}; "
        f"block={geometry.block}; cluster={geometry.cluster}; physical "
        f"cluster grid={(*geometry.cluster_grid, batch)}; clusters="
        f"{per_plane_clusters * batch}; persistent WGs/plane="
        f"{BATCH_PROFILE.persistent_tg}; persistent stride/plane="
        f"{geometry.persistent_stride}; log2 X/Y grid={geometry.log2_grid}"
    )
    print(
        f"[gemm_batch_isa_runner] ABI={BATCH_PROFILE.abi_name}; "
        f"kernarg=120 bytes (legacy 80-byte prefix + five u64 byte "
        f"strides); fixed LDS=206848; VGPR/SGPR metadata=384/106"
    )
    print(f"[gemm_batch_isa_runner] source: {isa}")
    print(f"[gemm_batch_isa_runner] symbol: {BATCH_KERNEL_SYMBOL}")


def _print_validations(
    validations: Sequence[BatchValidation],
) -> None:
    for item in validations:
        print(
            f"[gemm_batch_isa_runner] batch {item.index}: "
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
                "complete gfx1250 AMDGPU assembly source: the original "
                "64x256_1x4_ps symbol is accepted only with --batch 1; the "
                "64x256_1x4_batch_ps symbol supports true batched dispatch"
            )
        elif action.dest == "shape":
            action.help = (
                "per-matrix GEMM shape M,N,K "
                "(default: 18432,2048,7168)"
            )
        elif action.dest == "intype":
            action.help = (
                "input format; both supported ISA paths require mxfp4"
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
        "--self-test",
        action="store_true",
        help=(
            "run CPU-only ABI/geometry/overflow checks and validate the "
            "repository batch ISA, without compiling or using a GPU"
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
        if args.self_test:
            repository_batch_isa = (
                single._REPO
                / "my_code"
                / "f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.s"
            )
            source = single._read_isa_source(repository_batch_isa)
            run_static_contract_checks(source)
            print(
                "[gemm_batch_isa_runner] self-test passed: batch ABI, "
                "batch=1 compatibility, geometry, overflow gates, HIP launch "
                "config, and repository ISA contract"
            )
            return 0

        if args.isa is None:
            parser.error("--isa is required unless --self-test is used")
        run_static_contract_checks()
        isa = single._resolve_isa(args.isa)
        source = single._read_isa_source(isa)
        symbol = single.resolve_kernel_symbol_from_text(source, args.symbol)
        mode, profile = select_kernel_mode(symbol, args.batch)

        if mode == "legacy":
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
            validate_batch_assembly_contract_from_text(source)
            geometry = make_batched_launch_geometry(
                *args.shape,
                batch=args.batch,
            )
            _print_batch_contract(isa, geometry, args.batch)

        clang = single._resolve_clang(args.clang)
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
            print(
                f"[gemm_batch_isa_runner] loading kernel symbol: {symbol}"
            )
            _keep_code_object_if_requested(
                args,
                isa,
                symbol,
                result.code_object,
            )

            if mode == "legacy":
                legacy_row, event_timing, passed = single._run_gemm(
                    args,
                    result.code_object,
                    symbol,
                    profile,
                    geometry,
                )
                row = dict(legacy_row)
                row["batch"] = 1
                validations = (
                    BatchValidation(
                        index=0,
                        reference_hash=str(row["ref hash128"]),
                        output_hash=str(row["gemm_a4w4 out hash128"]),
                        error=float(row["gemm_a4w4 err"]),
                        max_abs=float(row["gemm_a4w4 max_abs"]),
                        rel_l2=float(row["gemm_a4w4 rel_l2"]),
                    ),
                )
            else:
                row, event_timing, validations, passed = _run_batch_gemm(
                    args,
                    result.code_object,
                    geometry,
                )

            single._print_cuda_event_timing(event_timing)
            _print_validations(validations)
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
