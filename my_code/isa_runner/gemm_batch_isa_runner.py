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

The exact ``moe_gemm1_a4w4_v0`` full/load-only symbols, the v1 direct-scale
full symbol, and the audited v21 load-only symbol use the production 184-byte
grouped-MoE ABI.  Their preferred CLI is
``--experts/--tokens/--topk/--model-dim/--inter-dim``; shape, balanced valid
rows, contiguous-M capacity, logical 24x144 grid, and physical 3456-WG launch
are derived and checked as one contract.  The former exact
``--shape 64,6144,7168 --batch 96`` form remains as an explicit compatibility
path, but mixing old and new forms is rejected.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import math
import os
import re
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# aiter announces every JIT module it loads at INFO
# ("[aiter] import [module_aiter_core] under ..."); that is noise in a runner
# whose output is a benchmark table, so quiet it before aiter is first imported.
import logging as _logging

_logging.getLogger("aiter").setLevel(_logging.WARNING)

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


# ---------------------------------------------------------------------------
# MoE grouped GEMM stage-1 (a8w4, TDM), gfx1250
#
# ABI mirrors launch_gemm_a8w4_tdm's kargs tuple, lowered by FlyDSL as
#   (ptr, packed{3xi32,2xi64})  for every fx.Tensor
#   (ptr,)                      for every fx.Pointer
# giving this exact 184-byte layout.  Every offset below is confirmed against
# the s_load_* offsets the ISA itself issues (0x28/0x30/0x38/0x60/0x70/0xa4).
# ---------------------------------------------------------------------------
MOE_GEMM1_KERNEL_SYMBOL = "a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1"
MOE_GEMM1_LOADONLY_KERNEL_SYMBOL = (
    "a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4_loadonly"
)
MOE_V21_LOADONLY_KERNEL_SYMBOL = (
    "f4gemm_bf16_mxfp4_ABpreShuffle_64x256_moe_loadonly_v21"
)
# Same launcher, same kernarg ABI and tile geometry; only
# waves_per_tensor_tdm differs (4 instead of the default 2), which is what the
# `_wpt4` suffix encodes.  This is the variant the production config actually
# selects, so both are benchmarked in the same harness to isolate that knob.
MOE_GEMM1_WPT4_KERNEL_SYMBOL = (
    "a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4"
)
MOE_GEMM1_V1_KERNEL_SYMBOL = (
    "a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_"
    "wpt4_directscale_v1"
)
MOE_GEMM1_SYMBOLS = (
    MOE_GEMM1_KERNEL_SYMBOL,
    MOE_GEMM1_WPT4_KERNEL_SYMBOL,
    MOE_GEMM1_V1_KERNEL_SYMBOL,
    MOE_GEMM1_LOADONLY_KERNEL_SYMBOL,
    MOE_V21_LOADONLY_KERNEL_SYMBOL,
)
MOE_LOADONLY_SYMBOLS = (
    MOE_GEMM1_LOADONLY_KERNEL_SYMBOL,
    MOE_V21_LOADONLY_KERNEL_SYMBOL,
)
MOE_MODES = (
    "moe-gemm1",
    "moe-gemm1-loadonly",
    "moe-v21-loadonly",
)
LAUNCH_BACKEND_PYTHON = "python"
LAUNCH_BACKEND_CPP = "cpp"
TIMING_CONTEXT_STANDALONE = "standalone"
TIMING_CONTEXT_MOE_PIPELINE = "moe-pipeline"
MOE_CPP_ISA_BASENAME = "moe_gemm1_a4w4_v0.s"
MOE_V21_CONTRACT_MARKER = "__aiter_moe_v21_contract"
TIMING_METHOD_PROFILER = single.TIMING_METHOD_PROFILER
TIMING_METHOD_CUDA_EVENT = single.TIMING_METHOD_CUDA_EVENT
TIMING_METHOD_CHOICES = single.TIMING_METHOD_CHOICES
RUN_PERFTEST_TIMING_SOURCE = single.RUN_PERFTEST_TIMING_SOURCE
CUDA_EVENT_TIMING_SOURCE = single.CUDA_EVENT_TIMING_SOURCE


def moe_pipeline_timing_source(test_graph: bool) -> str:
    return (
        f"{single.run_perftest_timing_source(test_graph)}; "
        "moe-pipeline context; target-only warmup"
    )

MOE_KERNARG_SIZE = 184
MOE_KERNARG_LAYOUT: tuple[tuple[str, int, str], ...] = (
    ("ptr_c", 0, "Q"),
    ("c_size0", 8, "I"),
    ("c_size1", 12, "I"),
    ("c_size2", 16, "I"),
    ("c_stride0", 20, "Q"),
    ("c_stride1", 28, "Q"),
    ("ptr_a", 40, "Q"),
    ("ptr_b", 48, "Q"),
    ("ptr_scale_a", 56, "Q"),
    ("sa_size0", 64, "I"),
    ("sa_size1", 68, "I"),
    ("sa_size2", 72, "I"),
    ("sa_stride0", 76, "Q"),
    ("sa_stride1", 84, "Q"),
    ("ptr_scale_b", 96, "Q"),
    ("sb_size0", 104, "I"),
    ("ptr_m_tile_map", 112, "Q"),
    ("ptr_bias", 120, "Q"),
    ("ptr_quant_scale", 128, "Q"),
    ("qs_size0", 136, "I"),
    ("qs_size1", 140, "I"),
    ("qs_size2", 144, "I"),
    ("qs_stride0", 148, "Q"),
    ("qs_stride1", 156, "Q"),
    ("i32_m", 164, "I"),
    ("i32_n", 168, "I"),
    ("f32_swiglu_limit", 172, "I"),
    ("f32_situ_beta", 176, "I"),
    ("f32_situ_linear_beta", 180, "I"),
)

# Compile-time geometry baked into the kernel name
# a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1
MOE_TILE_M, MOE_TILE_N, MOE_TILE_K = 64, 256, 256
MOE_M_WARP, MOE_N_WARP = 1, 4
MOE_NUM_BUFFERS = 2
MOE_N_EXPERTS = 96
MOE_BAKED_K = 7168
MOE_STAGE1_ACT = 1  # silu
MOE_BLOCK = (MOE_M_WARP * MOE_N_WARP * 32, 1, 1)  # (128,1,1)
MOE_PIPELINE_SHAPE = (MOE_TILE_M, 6144, MOE_BAKED_K)
MOE_REFERENCE_TOKENS = 512
MOE_REFERENCE_TOPK = 6
MOE_REFERENCE_INTER_DIM = 3072
MOE_REFERENCE_VALID_ROWS_PER_EXPERT = 32
MOE_REFERENCE_GRID = (3456, 1, 1)
MOE_CLI_FIELDS = (
    "experts",
    "tokens",
    "topk",
    "model_dim",
    "inter_dim",
)

# The epilogue clamps gate to <= limit and up to [-limit, limit] before
# silu(gate)*up.  With random fp4 operands accumulated over K=7168 the raw
# values reach the hundreds, so the production limit of 7.0 saturates almost
# every output to +-silu(7)*7 = +-48.955 -- which makes the comparison blind to
# both scale tensors (e8m0 scales are positive, so rescaling a saturated value
# changes nothing) and degrades it into a sign-agreement test.  Validation
# therefore runs with the clamp effectively disabled so the full GEMM is
# compared; the kernel reads this as an f32 kernarg, so it is a pure input.
MOE_VALIDATE_SWIGLU_LIMIT = float(
    os.environ.get("AITER_MOE_SWIGLU_LIMIT", "3.0e38")
)

@dataclass(frozen=True)
class MoeWorkload:
    """User inputs and the exact balanced stage-1 launch derived from them."""

    experts: int
    tokens: int
    topk: int
    model_dim: int
    inter_dim: int
    valid_routes: int
    valid_rows_per_expert: int
    raw_n: int
    contiguous_m: int
    active_m_tiles: int
    total_m_tiles: int
    n_tiles: int
    working_wgs: int
    tail_wgs: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (MOE_TILE_M, self.raw_n, self.model_dim)

    @property
    def grid(self) -> tuple[int, int, int]:
        return (self.total_m_tiles * self.n_tiles, 1, 1)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise single.GemmIsaRunnerError(
            f"alignment must be positive, got {alignment}"
        )
    return ((value + alignment - 1) // alignment) * alignment


def derive_moe_workload(
    *,
    experts: int,
    tokens: int,
    topk: int,
    model_dim: int,
    inter_dim: int,
) -> MoeWorkload:
    """Derive the production balanced-MoE contract from user-level sizes.

    The contiguous-M formula is the one owned by
    ``_grouped_a8w4_tdm_moe`` in ``grouped_moe_gfx1250.py``.  This runner
    accepts only the compile-time specialization represented by the audited
    ISA symbols, and reports every derived value instead of silently coercing
    an unsupported workload.
    """

    values = {
        "experts": experts,
        "tokens": tokens,
        "topk": topk,
        "model-dim": model_dim,
        "inter-dim": inter_dim,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise single.GemmIsaRunnerError(
                f"--{name} must be a positive integer, got {value!r}"
            )

    valid_routes = tokens * topk
    if valid_routes % experts:
        raise single.GemmIsaRunnerError(
            "the current balanced MoE contract requires "
            "tokens*topk to be divisible by experts; got "
            f"{tokens}*{topk}={valid_routes} over {experts} experts"
        )
    valid_rows_per_expert = valid_routes // experts
    raw_n = 2 * inter_dim
    contiguous_m = max(
        MOE_TILE_M,
        _align_up(
            valid_routes + experts * MOE_TILE_M - topk,
            MOE_TILE_M,
        ),
    )
    active_m_tiles = experts
    total_m_tiles = contiguous_m // MOE_TILE_M
    n_tiles = raw_n // MOE_TILE_N if raw_n % MOE_TILE_N == 0 else 0
    working_wgs = active_m_tiles * n_tiles
    tail_wgs = (total_m_tiles - active_m_tiles) * n_tiles

    required = {
        "experts": (experts, MOE_N_EXPERTS),
        "tokens": (tokens, MOE_REFERENCE_TOKENS),
        "topk": (topk, MOE_REFERENCE_TOPK),
        "model-dim": (model_dim, MOE_BAKED_K),
        "inter-dim": (inter_dim, MOE_REFERENCE_INTER_DIM),
        "valid rows/expert": (
            valid_rows_per_expert,
            MOE_REFERENCE_VALID_ROWS_PER_EXPERT,
        ),
    }
    mismatches = [
        f"{name}={actual} (required {expected})"
        for name, (actual, expected) in required.items()
        if actual != expected
    ]
    if mismatches:
        raise single.GemmIsaRunnerError(
            "this MoE ISA family is an exact balanced specialization: "
            + "; ".join(mismatches)
        )
    if raw_n % MOE_TILE_N:
        raise single.GemmIsaRunnerError(
            f"raw GEMM N={raw_n} must be divisible by tile_n={MOE_TILE_N}"
        )

    workload = MoeWorkload(
        experts=experts,
        tokens=tokens,
        topk=topk,
        model_dim=model_dim,
        inter_dim=inter_dim,
        valid_routes=valid_routes,
        valid_rows_per_expert=valid_rows_per_expert,
        raw_n=raw_n,
        contiguous_m=contiguous_m,
        active_m_tiles=active_m_tiles,
        total_m_tiles=total_m_tiles,
        n_tiles=n_tiles,
        working_wgs=working_wgs,
        tail_wgs=tail_wgs,
    )
    if workload.grid != MOE_REFERENCE_GRID:
        raise single.GemmIsaRunnerError(
            f"derived grid {workload.grid} does not match audited "
            f"{MOE_REFERENCE_GRID}"
        )
    return workload


def reference_moe_workload() -> MoeWorkload:
    return derive_moe_workload(
        experts=MOE_N_EXPERTS,
        tokens=MOE_REFERENCE_TOKENS,
        topk=MOE_REFERENCE_TOPK,
        model_dim=MOE_BAKED_K,
        inter_dim=MOE_REFERENCE_INTER_DIM,
    )


def resolve_moe_cli_workload(
    args: argparse.Namespace,
    *,
    shape_explicit: bool,
    batch_explicit: bool,
) -> MoeWorkload:
    """Choose one unambiguous MoE CLI form and reject mixed contracts."""

    supplied = {
        name: getattr(args, name)
        for name in MOE_CLI_FIELDS
        if getattr(args, name) is not None
    }
    if supplied:
        missing = [
            f"--{name.replace('_', '-')}"
            for name in MOE_CLI_FIELDS
            if getattr(args, name) is None
        ]
        if missing:
            raise single.GemmIsaRunnerError(
                "MoE user-level parameters are an all-or-none group; missing "
                + ", ".join(missing)
            )
        conflicts = []
        if shape_explicit:
            conflicts.append("--shape/-mnk")
        if batch_explicit:
            conflicts.append("--batch")
        if conflicts:
            raise single.GemmIsaRunnerError(
                "do not mix MoE user-level parameters with legacy "
                + ", ".join(conflicts)
            )
        return derive_moe_workload(
            experts=args.experts,
            tokens=args.tokens,
            topk=args.topk,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
        )

    if shape_explicit != batch_explicit:
        raise single.GemmIsaRunnerError(
            "legacy MoE compatibility requires both --shape and --batch; "
            "prefer the complete --experts/--tokens/--topk/--model-dim/"
            "--inter-dim group"
        )
    if shape_explicit and batch_explicit:
        expected_shape = (
            MOE_TILE_M,
            2 * MOE_REFERENCE_INTER_DIM,
            MOE_BAKED_K,
        )
        if tuple(args.shape) != expected_shape or args.batch != MOE_N_EXPERTS:
            raise single.GemmIsaRunnerError(
                "legacy MoE compatibility accepts only "
                f"--shape {','.join(map(str, expected_shape))} "
                f"--batch {MOE_N_EXPERTS}; got shape={args.shape}, "
                f"batch={args.batch}"
            )
        print(
            "[gemm_batch_isa_runner] legacy MoE --shape/--batch compatibility "
            "selected; prefer user-level MoE parameters"
        )
        return reference_moe_workload()

    raise single.GemmIsaRunnerError(
        "MoE ISA requires either the complete user-level parameter group "
        "--experts/--tokens/--topk/--model-dim/--inter-dim or the exact "
        "legacy --shape/--batch pair"
    )


def print_moe_workload_contract(workload: MoeWorkload) -> None:
    padded_rows = workload.experts * MOE_TILE_M
    print(
        "[gemm_batch_isa_runner] MoE user parameters: "
        f"experts={workload.experts}; tokens={workload.tokens}; "
        f"topk={workload.topk}; model_dim={workload.model_dim}; "
        f"inter_dim={workload.inter_dim}"
    )
    print(
        "[gemm_batch_isa_runner] MoE derived contract: "
        f"valid_routes={workload.valid_routes}; "
        f"valid_rows/expert={workload.valid_rows_per_expert}; "
        f"raw GEMM shape={workload.shape}; contiguous_m="
        f"{workload.contiguous_m}; logical WG grid="
        f"({workload.n_tiles},{workload.total_m_tiles},1); physical grid="
        f"{workload.grid}; working WGs={workload.working_wgs}; "
        f"tail WGs={workload.tail_wgs}; expert tile padding rows="
        f"{padded_rows - workload.valid_routes}; tail capacity rows="
        f"{workload.contiguous_m - padded_rows}"
    )


def moe_contiguous_m(
    experts: int,
    tile_m: int,
    *,
    tokens: int = MOE_REFERENCE_TOKENS,
    topk: int = MOE_REFERENCE_TOPK,
) -> int:
    """Compatibility wrapper for callers using the former helper signature."""

    if tile_m != MOE_TILE_M:
        raise single.GemmIsaRunnerError(
            f"MoE tile_m must be {MOE_TILE_M}, got {tile_m}"
        )
    return max(
        tile_m,
        _align_up(tokens * topk + experts * tile_m - topk, tile_m),
    )

MOE_GEMM1_PROFILE = single.KernelProfile(
    name="a8w4-moe-gemm1-tdm-t64x256x256-w1x4-b2-e96-act1",
    primary_symbol=MOE_GEMM1_KERNEL_SYMBOL,
    symbols=(MOE_GEMM1_KERNEL_SYMBOL,),
    wg_tile=(MOE_TILE_M, MOE_TILE_N),
    wave_tile=(MOE_TILE_M // MOE_M_WARP, MOE_TILE_N // MOE_N_WARP),
    output_quadrants=(1, 1),
    cluster=(1, 1, 1),
    block=MOE_BLOCK,
    k_multiple=MOE_TILE_K,
    persistent_tg=0,
    persistent_grid_y=0,
    apre=1,
    abi_name="a8w4-moe-tdm-v1",
    kernarg_size=MOE_KERNARG_SIZE,
    kernarg_layout=MOE_KERNARG_LAYOUT,
)
MOE_GEMM1_WPT4_PROFILE = replace(
    MOE_GEMM1_PROFILE,
    name="a4w4-moe-gemm1-tdm-t64x256x256-w1x4-b2-e96-act1-wpt4",
    primary_symbol=MOE_GEMM1_WPT4_KERNEL_SYMBOL,
    symbols=(MOE_GEMM1_WPT4_KERNEL_SYMBOL,),
)
MOE_GEMM1_V1_PROFILE = replace(
    MOE_GEMM1_PROFILE,
    name="a4w4-moe-gemm1-ab-tdm-direct-scale-v1",
    primary_symbol=MOE_GEMM1_V1_KERNEL_SYMBOL,
    symbols=(MOE_GEMM1_V1_KERNEL_SYMBOL,),
)
MOE_GEMM1_LOADONLY_PROFILE = replace(
    MOE_GEMM1_PROFILE,
    name="a8w4-moe-gemm1-tdm-loadonly",
    primary_symbol=MOE_GEMM1_LOADONLY_KERNEL_SYMBOL,
    symbols=(MOE_GEMM1_LOADONLY_KERNEL_SYMBOL,),
)
MOE_V21_LOADONLY_PROFILE = replace(
    MOE_GEMM1_PROFILE,
    name="a4w4-moe-v21-v18v2-load-pattern-loadonly",
    primary_symbol=MOE_V21_LOADONLY_KERNEL_SYMBOL,
    symbols=(MOE_V21_LOADONLY_KERNEL_SYMBOL,),
)


def make_moe_launch_geometry(
    m: int | MoeWorkload,
    n: int | None = None,
    k: int | None = None,
    batch: int | None = None,
) -> single.LaunchGeometry:
    """Grid for the MoE stage-1 kernel.

    New code passes a :class:`MoeWorkload`.  The four-integer form remains for
    internal compatibility and is accepted only for the exact historical
    ``64,6144,7168``/96 specialization.

    launch_gemm_a8w4_tdm uses
        m_tiles = ceil(i32_m / tile_m);  n_tiles = ceil(N / tile_n)
        grid    = (m_tiles * n_tiles, 1, 1)
    with ``i32_m = contiguous_m`` from :func:`derive_moe_workload`.
    """
    if isinstance(m, MoeWorkload):
        if any(value is not None for value in (n, k, batch)):
            raise single.GemmIsaRunnerError(
                "MoeWorkload geometry form does not accept extra dimensions"
            )
        workload = m
    else:
        if n is None or k is None or batch is None:
            raise single.GemmIsaRunnerError(
                "legacy MoE geometry form requires M,N,K,batch"
            )
        expected = (MOE_TILE_M, 2 * MOE_REFERENCE_INTER_DIM, MOE_BAKED_K)
        if (m, n, k) != expected or batch != MOE_N_EXPERTS:
            raise single.GemmIsaRunnerError(
                "legacy MoE geometry accepts only "
                f"shape={expected}, batch={MOE_N_EXPERTS}; got "
                f"shape={(m, n, k)}, batch={batch}"
            )
        workload = reference_moe_workload()

    return single.LaunchGeometry(
        grid=workload.grid,
        block=MOE_BLOCK,
        cluster=(1, 1, 1),
        tiles=(workload.n_tiles, workload.total_m_tiles),
        cluster_grid=(workload.grid[0], 1),
        log2_grid=(0, 0),
        logical_cluster_grid=(workload.n_tiles, workload.total_m_tiles),
        logical_wg_tasks=workload.grid[0],
        logical_cluster_tasks=workload.grid[0],
        persistent_stride=0,
        # The 96 experts are already inside contiguous_m -> m_tiles, so this
        # kernel launches once; --batch must not be applied a second time.
        planes=1,
    )


def _f32_bits(value: float) -> int:
    import struct as _struct

    return int(_struct.unpack("<I", _struct.pack("<f", float(value)))[0])


def pack_moe_kernargs(
    *,
    ptr_c: int,
    ptr_a: int,
    ptr_b: int,
    ptr_scale_a: int,
    ptr_scale_b: int,
    ptr_m_tile_map: int,
    ptr_bias: int,
    ptr_quant_scale: int,
    c_shape: tuple[int, int, int],
    c_strides: tuple[int, int],
    sa_shape: tuple[int, int, int],
    sa_strides: tuple[int, int],
    sb_size0: int,
    qs_shape: tuple[int, int, int],
    qs_strides: tuple[int, int],
    i32_m: int,
    i32_n: int,
    swiglu_limit: float = MOE_VALIDATE_SWIGLU_LIMIT,
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
) -> bytes:
    fields = {
        "ptr_c": ptr_c,
        "c_size0": c_shape[0],
        "c_size1": c_shape[1],
        "c_size2": c_shape[2],
        "c_stride0": c_strides[0],
        "c_stride1": c_strides[1],
        "ptr_a": ptr_a,
        "ptr_b": ptr_b,
        "ptr_scale_a": ptr_scale_a,
        "sa_size0": sa_shape[0],
        "sa_size1": sa_shape[1],
        "sa_size2": sa_shape[2],
        "sa_stride0": sa_strides[0],
        "sa_stride1": sa_strides[1],
        "ptr_scale_b": ptr_scale_b,
        "sb_size0": sb_size0,
        "ptr_m_tile_map": ptr_m_tile_map,
        "ptr_bias": ptr_bias,
        "ptr_quant_scale": ptr_quant_scale,
        "qs_size0": qs_shape[0],
        "qs_size1": qs_shape[1],
        "qs_size2": qs_shape[2],
        "qs_stride0": qs_strides[0],
        "qs_stride1": qs_strides[1],
        "i32_m": i32_m,
        "i32_n": i32_n,
        "f32_swiglu_limit": _f32_bits(swiglu_limit),
        "f32_situ_beta": _f32_bits(situ_beta),
        "f32_situ_linear_beta": _f32_bits(situ_linear_beta),
    }
    return single._pack_kernarg_fields(
        fields,
        layout=MOE_KERNARG_LAYOUT,
        size=MOE_KERNARG_SIZE,
    )


def build_moe_inputs(
    torch,
    *,
    workload: MoeWorkload | None = None,
    m_per_expert: int | None = None,
    n: int | None = None,
    k: int | None = None,
    experts: int | None = None,
    device,
    seed: int,
):
    """Build every tensor the MoE stage-1 kernel dereferences, plus a reference.

    The kernel symbol is ``a8w4_tdm_fp4_...``.  In launch_gemm_a8w4_tdm's name
    template the ``a8w4_tdm_`` prefix is a fixed string; the token that encodes
    the activation dtype is the ``_fp4`` right after it, which means
    ``a_is_fp4=1`` -- activations are MXFP4-packed, *not* fp8.  So ``a`` is
    K//2 bytes per row, not K.  (Cross-check: the A copy_atom in the kernel's
    MLIR is (32,128):(3584,1), and 3584 == K//2.)

    Layouts, all following grouped_moe_gfx1250's a8w4 stage-1 call site:
      a           uint8  [1, M, K//2]                MXFP4 payload
      b           uint8  [E, N, K//2]                MXFP4 weights, GUGU rows,
                                                     16x16 WMMA-tile shuffled
      scale_a     uint8  [1, M//rep_a, K//32*rep_a]  wmma_rep-preshuffled e8m0
      scale_b     uint8  [E, N//32, K//32*32]        n32k4-preshuffled e8m0
      m_tile_map  int32  [E]                         contiguous_psum
      out         bf16   [M, N//2]                   N//2: stage1_act gates it

    Every logical input is drawn by the same helpers, in the same order, from
    the same global RNG as ``_run_grouped_via_fused_moe``
    (test_flydsl_grouped_gemm_gfx1250.py:1085-1108) -- including the ``w2`` and
    bias draws this kernel never touches, because skipping a draw shifts the
    stream and silently changes ``w1_scale_raw`` and ``hidden``.  Input hashes
    are printed so the tensors can be pinned across runs.

    Activations are quantised with ``per_1x32_f4_quant(..., shuffle=False)``,
    the exact call ``_torch_moe_ref`` makes on its a4w4 branch
    (test_flydsl_grouped_gemm_gfx1250.py:897-899), so the reference and the
    kernel consume the same A bytes and the same logical block scales; only the
    scale's *placement* is transformed for the kernel.
    """
    import torch.nn.functional as F
    from aiter.ops.shuffle import moe_shuffle_weight, moe_shuffle_scale
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_fused_quant_preshuffle
    from aiter.utility import dtypes, fp4_utils

    if workload is None:
        expected = (MOE_TILE_M, 2 * MOE_REFERENCE_INTER_DIM, MOE_BAKED_K)
        if (
            m_per_expert,
            n,
            k,
            experts,
        ) != (*expected, MOE_N_EXPERTS):
            raise single.GemmIsaRunnerError(
                "legacy build_moe_inputs form accepts only "
                f"M,N,K,experts={(*expected, MOE_N_EXPERTS)}; got "
                f"{(m_per_expert, n, k, experts)}"
            )
        workload = reference_moe_workload()
    elif any(value is not None for value in (m_per_expert, n, k, experts)):
        raise single.GemmIsaRunnerError(
            "build_moe_inputs(workload=...) does not accept legacy dimensions"
        )

    m_per_expert = MOE_TILE_M
    n = workload.raw_n
    k = workload.model_dim
    experts = workload.experts

    # contiguous_m is the production static upper bound (9216 for this config),
    # so the buffers cover the trailing tiles that map to no expert even though
    # nothing reads or writes them.  routed_m is where the data actually lives.
    contiguous_m = workload.contiguous_m
    routed_m = experts * m_per_expert
    m_tiles = workload.active_m_tiles
    rep_a = (MOE_TILE_M // MOE_M_WARP) // 16   # 64//16 = 4
    k_scale = k // 32
    inter = n // 2

    # --- test helpers, verbatim (test_flydsl_grouped_gemm_gfx1250.py:957-974)
    def _pattern_packed(experts_, rows, k_pack):
        return torch.randint(0, 256, (experts_, rows, k_pack), dtype=torch.uint8)

    def init_weight_scales(experts_, rows, n_blocks):
        r = torch.randint(0, 3, (experts_, rows, n_blocks), dtype=torch.int16)
        return (r + (127 - 1)).to(torch.uint8)   # DEFAULT_SCALE_BYTE = 127

    # --- draw order, verbatim (same file, 1085-1108) -------------------------
    torch.manual_seed(seed)
    w1_logical = _pattern_packed(experts, n, k // 2)
    _w2_logical = _pattern_packed(experts, k, inter // 2)        # unused draw
    w1_scale_raw = init_weight_scales(experts, n, k_scale)
    _w2_scale_raw = init_weight_scales(experts, k, inter // 32)  # unused draw
    _bias1 = (torch.randn((experts, n)) * 1e-3).float()          # unused draw
    _bias2 = (torch.randn((experts, k)) * 1e-3).float()          # unused draw
    hidden = (torch.randn((contiguous_m, k)) * 0.5).to(torch.bfloat16).to(device)

    # ---- kernel-layout weights: GGUU -> GUGU row interleave, then shuffle ----
    b = moe_shuffle_weight(
        w1_logical, experts_cnt=experts, is_guinterleave=True, gate_up=True
    ).to(device)
    scale_b = moe_shuffle_scale(
        w1_scale_raw.contiguous(),
        experts_cnt=experts,
        is_guinterleave=True,
        gate_up=True,
    ).to(device)

    # ---- activations: one quantisation, shared by kernel and reference ------
    a_q, a_scale_logical = per_1x32_f4_quant(
        hidden, quant_dtype=dtypes.fp4x2, shuffle=False
    )
    a_scale_logical = a_scale_logical.view(torch.uint8).contiguous()

    # The kernel's A comes from the production preshuffle, not from a mapping
    # rederived here: a hand-applied wmma_rep scatter was measurably wrong
    # (cross-check below reported scale_match=False), and the placement is the
    # kind of detail that must be taken from the code that owns it.
    a_pre, scale_a = flydsl_moe_fused_quant_preshuffle(
        hidden.reshape(1, contiguous_m, k),
        1,
        contiguous_m,
        wmma_rep=rep_a,
        quant_mode="fp4",
        masked_m=None,
        topids_to_rows=None,
    )
    a = a_pre.view(torch.uint8).contiguous()

    # The reference keeps per_1x32_f4_quant's *logical* scale while the kernel
    # gets the preshuffled one.  That is only sound if both quantisers produced
    # the same fp4 payload -- payload equality pins the block scales, since the
    # payload is round(x / scale).  Checked, not assumed.
    if not torch.equal(a.reshape(-1), a_q.view(torch.uint8).reshape(-1)):
        raise single.GemmIsaRunnerError(
            "per_1x32_f4_quant and flydsl_moe_fused_quant_preshuffle disagree on "
            "the fp4 payload; the reference's logical scale would not describe "
            "the bytes the kernel reads"
        )

    # contiguous_psum(), copied from _psum_ref
    # (test_flydsl_grouped_gemm_gfx1250.py:1477-1486).  Each expert's rows are
    # padded up to a whole tile, so `starts` is the tile-aligned block offset
    # while psum[e] = starts[e] + masked_m[e] is the *exact* exclusive end of
    # its valid rows; the kernel upper-bound-bisects psum to recover the expert.
    #
    # masked_m is deliberately a fraction of a tile, matching the production
    # run: run_gemm_a4w4.sh routes 512 tokens x topk 6 over 96 experts, i.e. 32
    # rows each, which pad to a 64-row tile.  Loads and MMA are tile-granular
    # and so are unaffected, but the epilogue masks its stores at psum, leaving
    # the padding rows untouched -- so those rows must be excluded from the
    # comparison rather than treated as wrong answers.
    masked_m = torch.full(
        (experts,),
        workload.valid_rows_per_expert,
        dtype=torch.int64,
    )
    aligned = ((masked_m + MOE_TILE_M - 1) // MOE_TILE_M) * MOE_TILE_M
    inclusive = torch.cumsum(aligned, 0)
    starts = inclusive - aligned
    m_tile_map = (starts + masked_m).to(torch.int32).to(device)
    if int(inclusive[-1]) != routed_m:
        raise single.GemmIsaRunnerError(
            f"tile-aligned rows {int(inclusive[-1])} do not match routed_m "
            f"{routed_m}"
        )
    # Rows the kernel actually writes: inside an expert's block (row < routed_m)
    # and below that expert's psum.  Everything above routed_m bisects to
    # expert >= n_experts and the workgroup exits at entry.
    row_ids = torch.arange(contiguous_m, device=device)
    block_start = (row_ids // MOE_TILE_M) * MOE_TILE_M
    valid_rows = (row_ids < routed_m) & (
        (row_ids - block_start) < workload.valid_rows_per_expert
    )

    # The C descriptor says N//2 columns, but allocate the full N width as
    # slack: if that reading of stage1_act's gating were wrong the kernel would
    # write N-wide rows, and on this bring-up machine an out-of-bounds store is
    # a page fault, not a wrong number.  Only the first N//2 columns are
    # compared; the rest must stay zero, which is itself a check on the stride.
    # Exactly the production allocation (grouped_moe_gfx1250.py:632:
    # y = torch.empty((1, contiguous_m, inter_dim))).  The earlier 2x-wide
    # buffer was slack against mis-reading stage1_act's gating; the store-width
    # check has confirmed the N//2 stride often enough that the slack now only
    # doubles the output footprint relative to production.
    out = torch.zeros((contiguous_m, inter), dtype=torch.bfloat16, device=device)
    out_view = out
    # bias / quant_scale are unused by this build (has_bias=0, quant_out=0) but
    # must still be valid addresses.
    bias = torch.zeros(n, dtype=torch.float32, device=device)
    quant_scale = torch.zeros(1, dtype=torch.float32, device=device)

    reference = _moe_stage1_reference(
        torch,
        F,
        fp4_utils,
        a_payload=a,
        a_scale_logical=a_scale_logical,
        w1_logical=w1_logical.to(device),
        w1_scale_raw=w1_scale_raw.to(device),
        contiguous_m=contiguous_m,
        n=n,
        k=k,
        experts=experts,
        m_per_expert=m_per_expert,
        device=device,
    )
    return {
        "a": a,
        "b": b,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "m_tile_map": m_tile_map,
        "valid_rows": valid_rows,
        "out": out,
        "out_view": out_view,
        "bias": bias,
        "quant_scale": quant_scale,
        "contiguous_m": contiguous_m,
        "routed_m": routed_m,
        "m_tiles": m_tiles,
        "valid_routed_rows": workload.valid_routes,
        "active_experts": experts,
        "workload": workload,
        "reference": reference,
    }


def _moe_stage1_reference(
    torch,
    F,
    fp4_utils,
    *,
    a_payload,
    a_scale_logical,
    w1_logical,
    w1_scale_raw,
    contiguous_m: int,
    n: int,
    k: int,
    experts: int,
    m_per_expert: int,
    device,
    swiglu_limit: float = MOE_VALIDATE_SWIGLU_LIMIT,
):
    """fp32 reference for one contiguous-grouped a4w4 stage-1 GEMM.

    Same math as ``aiter.fused_moe.torch_moe_stage1`` (fused_moe.py:3556-3605)
    with ``quant_type=per_1x32`` and ``activation=Silu``, specialised to the
    identity routing this runner uses: contiguous row ``r`` belongs to expert
    ``r // m_per_expert``, so the expert mask collapses to a row slice.

    Both operands are dequantised from the *logical* (pre-shuffle) tensors, so
    the reference is independent of every layout transform the kernel relies
    on -- a shuffle bug shows up as a mismatch rather than cancelling out.
    """
    k_scale = k // 32
    inter = n // 2

    def deq(packed, scale_e8m0, rows):
        # mxfp4_to_f32 doubles the last dim (2 nibbles per byte); the e8m0
        # scale then multiplies each 32-element block.
        vals = fp4_utils.mxfp4_to_f32(packed.reshape(rows, k // 2)).view(
            rows, k_scale, 32
        )
        scl = fp4_utils.e8m0_to_f32(scale_e8m0.reshape(rows, k_scale)).view(
            rows, k_scale, 1
        )
        return (vals * scl).view(rows, k).float()

    ref = torch.empty((contiguous_m, inter), dtype=torch.float32, device=device)
    for e in range(experts):
        lo, hi = e * m_per_expert, (e + 1) * m_per_expert
        a_deq = deq(a_payload.reshape(contiguous_m, k // 2)[lo:hi], a_scale_logical[lo:hi], hi - lo)
        w_deq = deq(w1_logical[e], w1_scale_raw[e], n)
        acc = a_deq @ w_deq.transpose(0, 1)          # (rows, N), logical GGUU
        gate, up = acc.split([inter, inter], dim=-1)
        # torch_moe_stage1's non-swiglu branch clamps both halves before the
        # gate, matching the kernel's fused_silu_swiglu_elem which always
        # applies the limit.
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        ref[lo:hi] = F.silu(gate) * up
    return ref

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
    if symbol == MOE_V21_LOADONLY_KERNEL_SYMBOL:
        return "moe-v21-loadonly", MOE_V21_LOADONLY_PROFILE
    if symbol == MOE_GEMM1_LOADONLY_KERNEL_SYMBOL:
        return "moe-gemm1-loadonly", MOE_GEMM1_LOADONLY_PROFILE
    if symbol == MOE_GEMM1_V1_KERNEL_SYMBOL:
        return "moe-gemm1", MOE_GEMM1_V1_PROFILE
    if symbol == MOE_GEMM1_WPT4_KERNEL_SYMBOL:
        return "moe-gemm1", MOE_GEMM1_WPT4_PROFILE
    if symbol == MOE_GEMM1_KERNEL_SYMBOL:
        return "moe-gemm1", MOE_GEMM1_PROFILE
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
        f"{BATCH_KERNEL_SYMBOL!r} for any supported batch; MoE symbols: "
        + ", ".join(repr(item) for item in MOE_GEMM1_SYMBOLS)
    )


def moe_cpp_target_isa() -> Path:
    return Path(__file__).resolve().parent.parent / MOE_CPP_ISA_BASENAME


def validate_cpp_backend_target(
    isa: Path,
    symbol: str,
    mode: str,
) -> None:
    """Restrict the C++ binding to its one audited source/symbol contract."""

    expected_isa = moe_cpp_target_isa().resolve()
    actual_isa = isa.resolve()
    same_path = os.path.normcase(str(actual_isa)) == os.path.normcase(
        str(expected_isa)
    )
    if (
        not same_path
        or actual_isa.name != MOE_CPP_ISA_BASENAME
        or symbol != MOE_GEMM1_WPT4_KERNEL_SYMBOL
        or mode != "moe-gemm1"
    ):
        raise single.GemmIsaRunnerError(
            "--cpp is restricted to the exact source "
            f"{expected_isa} and exact kernel symbol "
            f"{MOE_GEMM1_WPT4_KERNEL_SYMBOL!r}; got source={actual_isa}, "
            f"symbol={symbol!r}, mode={mode!r}"
        )


def validate_moe_v21_source_contract(source: str, symbol: str) -> None:
    if symbol != MOE_V21_LOADONLY_KERNEL_SYMBOL:
        return
    marker_lines = [
        re.sub(r"\s+", " ", single._strip_asm_comment(line)).strip()
        for line in source.splitlines()
        if MOE_V21_CONTRACT_MARKER in single._strip_asm_comment(line)
    ]
    expected = f".set {MOE_V21_CONTRACT_MARKER}, 1"
    if marker_lines != [expected]:
        raise single.GemmIsaRunnerError(
            f"{symbol} requires exactly one {expected!r} marker; "
            f"found {marker_lines}"
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
    store_detection: single.GlobalOutputStoreDetection,
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

            profiled_launch = (
                _make_profiler_visible_driver_launch(
                    torch,
                    launch_loadonly,
                    anchor=a,
                    output=None,
                )
                if args.timing_method == TIMING_METHOD_PROFILER
                else None
            )
            _, timing = _run_selected_target_timing(
                args,
                torch,
                launch_loadonly,
                profiler_launch=profiled_launch,
                target_output=a,
                target_tensors={"a": a, "b": b},
                target_shape=(m, n, k),
                stream=stream,
                symbol=symbol,
                mark_stream_synchronized=module.mark_stream_synchronized,
            )
        us = float(timing["device_time_avg"])
        traffic = single.make_logical_traffic(
            read_bytes=int(a.nbytes + b.nbytes),
            output_bytes_if_stored=batch * m * n * 2,
            store_detection=store_detection,
        )
        single.print_logical_traffic("[gemm_batch_isa_runner]", traffic)
        not_validated = "n/a (not validated)"
        if symbol == MAB_FULL_BATCH_LOADONLY_WV23_256K_KERNEL_SYMBOL:
            lds_note = _mab_wv23_lds_note(metadata_lds_bytes)
            print(
                f"[gemm_batch_isa_runner] MAB TDM load-only nominal logical "
                f"A+B bytes={traffic.read_bytes} = A ({a.nbytes}) + B ({b.nbytes}); "
                f"{lds_note}; actual HBM bytes also depend on multicast/cache "
                "behavior, and descriptor payload is not claimed as HBM bytes"
            )
        else:
            print(
                f"[gemm_batch_isa_runner] MAB TDM load-only logical unique "
                f"A+B bytes={traffic.read_bytes} = A ({a.nbytes}) + B ({b.nbytes}); "
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
            "gemm_a4w4 bytes": traffic.total_bytes,
            "gemm_a4w4 TB/s": round(
                single.bytes_per_microsecond_to_tbps(traffic.total_bytes, us), 3
            ),
            "gemm_a4w4 err": not_validated,
            "gemm_a4w4 out hash128": not_validated,
            "gemm_a4w4 max_err_info": not_validated,
            "gemm_a4w4 rel_l2": not_validated,
        }

        def validate_loadonly() -> tuple[None, bool]:
            return None, True

        return row, timing, validate_loadonly

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

        profiled_launch = (
            _make_profiler_visible_driver_launch(
                torch,
                launch,
                anchor=output_storage,
                output=output_storage,
            )
            if args.timing_method == TIMING_METHOD_PROFILER
            else None
        )
        timed_output_storage, timing = _run_selected_target_timing(
            args,
            torch,
            launch,
            profiler_launch=profiled_launch,
            target_output=output_storage,
            target_tensors={
                "a": a,
                "b": b,
                "c": c_storage,
                "out": output_storage,
            },
            target_shape=(m, n, k),
            stream=stream,
            symbol=symbol,
            mark_stream_synchronized=module.mark_stream_synchronized,
        )
    us = float(timing["device_time_avg"])
    traffic = single.make_logical_traffic(
        read_bytes=int(a.nbytes + b.nbytes),
        output_bytes_if_stored=int(output_storage.nbytes),
        store_detection=store_detection,
    )
    single.print_logical_traffic("[gemm_batch_isa_runner]", traffic)
    flops = 2 * batch * m * n * k
    print(
        f"[gemm_batch_isa_runner] MAB logical_bytes={traffic.total_bytes} = "
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
        "gemm_a4w4 bytes": traffic.total_bytes,
        "gemm_a4w4 TB/s": round(
            single.bytes_per_microsecond_to_tbps(traffic.total_bytes, us), 3
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

    return row, timing, validate



def _run_moe_gemm(
    args,
    *,
    symbol: str,
    profile,
    geometry,
    workload: MoeWorkload,
    code_object: Path,
    gfx: str,
    store_detection: single.GlobalOutputStoreDetection,
    cpp_extension: Any | None = None,
):
    """Launch the MoE stage-1 a8w4 TDM kernel and time it.

    The load-only variant writes nothing, so it reports bandwidth only; the
    full kernel additionally hashes its output so repeated runs are comparable.
    """
    import torch

    loadonly = symbol in MOE_LOADONLY_SYMBOLS
    m, n, k = workload.shape
    experts = workload.experts
    device = torch.device(f"cuda:{args.device}")
    stream = torch.cuda.current_stream(device)

    t = build_moe_inputs(
        torch,
        workload=workload,
        device=device,
        seed=args.seed,
    )
    contiguous_m = t["contiguous_m"]
    sa_rows = int(t["scale_a"].shape[-2])
    sa_cols = int(t["scale_a"].shape[-1])

    payload = pack_moe_kernargs(
        ptr_c=int(t["out"].data_ptr()),
        ptr_a=int(t["a"].data_ptr()),
        ptr_b=int(t["b"].data_ptr()),
        ptr_scale_a=int(t["scale_a"].data_ptr()),
        ptr_scale_b=int(t["scale_b"].data_ptr()),
        ptr_m_tile_map=int(t["m_tile_map"].data_ptr()),
        ptr_bias=int(t["bias"].data_ptr()),
        ptr_quant_scale=int(t["quant_scale"].data_ptr()),
        # stage1_act=1 gates gate/up down to N//2 output columns
        # (grouped_moe_gfx1250.py:632 allocates (1, contiguous_m, inter_dim)
        # while passing N=2*inter_dim), so the C descriptor is N//2 wide.
        c_shape=(1, contiguous_m, n // 2),
        c_strides=(contiguous_m * (n // 2), n // 2),
        sa_shape=(1, sa_rows, sa_cols),
        sa_strides=(sa_rows * sa_cols, sa_cols),
        sb_size0=int(t["scale_b"].shape[0]),
        qs_shape=(1, 1, 1),
        qs_strides=(1, 1),
        i32_m=contiguous_m,
        i32_n=n,
    )

    def measure_target(
        raw_target_launch: Callable[[], Any],
        *,
        profiler_target_launch: Callable[[], Any] | None = None,
        mark_stream_synchronized: Callable[[], Any] | None = None,
    ) -> dict[str, Any]:
        _, timing = _run_selected_target_timing(
            args,
            torch,
            raw_target_launch,
            profiler_launch=profiler_target_launch,
            target_output=t["out"],
            target_tensors={
                name: t[name]
                for name in (
                    "a",
                    "scale_a",
                    "b",
                    "scale_b",
                    "m_tile_map",
                    "out",
                    "bias",
                    "quant_scale",
                )
            },
            target_shape=(m, n, k),
            stream=stream,
            symbol=symbol,
            mark_stream_synchronized=mark_stream_synchronized,
        )
        if not args.inmoe:
            timing["timing_context"] = TIMING_CONTEXT_STANDALONE
        return timing

    if args.cpp:
        if cpp_extension is None:
            raise single.GemmIsaRunnerError(
                "the C++ launch backend was selected without a loaded extension"
            )

        def launch_cpp() -> Any:
            return cpp_extension.launch(
                t["out"],
                t["a"],
                t["b"],
                t["scale_a"],
                t["scale_b"],
                t["m_tile_map"],
                t["bias"],
                t["quant_scale"],
                payload,
                list(geometry.grid),
                list(geometry.block),
                list(geometry.cluster),
                m,
                n,
                k,
                experts,
                str(code_object),
            )

        # The compiled torch custom op already supplies profiler correlation
        # and is graph-capturable, so it serves as both the raw and
        # profiler-visible launch in the generic --inmoe adapter.
        timing = measure_target(
            launch_cpp,
            profiler_target_launch=launch_cpp,
        )
    else:
        with single._LoadedClusterKernel(
            code_object, symbol, args.device
        ) as module:
            module.configure(
                payload, geometry, int(stream.cuda_stream), MOE_KERNARG_SIZE
            )

            def launch() -> None:
                module.launch()

            profiled_launch = (
                _make_profiler_visible_driver_launch(
                    torch,
                    launch,
                    anchor=t["out"],
                    output=t["out"],
                )
                if args.timing_method == TIMING_METHOD_PROFILER
                else None
            )
            timing = measure_target(
                launch,
                profiler_target_launch=profiled_launch,
                mark_stream_synchronized=module.mark_stream_synchronized,
            )
    us = float(timing["device_time_avg"])

    a_bytes = int(t["a"].nbytes)
    b_bytes = int(t["b"].nbytes)
    sa_bytes = int(t["scale_a"].nbytes)
    sb_bytes = int(t["scale_b"].nbytes)

    # Match _calculate_effective_stage_metrics in the grouped-MoE benchmark:
    # unique valid A rows once and each active expert's complete B surface
    # once. Scheduled padding, tail capacity, descriptor replication, and
    # control metadata are excluded from this useful logical byte metric.
    m_tiles = t["m_tiles"]
    n_tiles = n // MOE_TILE_N
    useful = single.calculate_moe_a4w4_bf16_useful_bytes(
        valid_rows=int(t["valid_routed_rows"]),
        active_experts=int(t["active_experts"]),
        n=n,
        k=k,
        output_n=n // 2,
    )
    traffic = single.make_logical_traffic(
        read_bytes=useful.read_bytes,
        output_bytes_if_stored=useful.output_payload_bytes,
        store_detection=store_detection,
    )
    single.print_logical_traffic(
        "[gemm_batch_isa_runner] useful logical",
        traffic,
    )

    print(
        f"[gemm_batch_isa_runner] MoE stage-1 a4w4: experts={experts} "
        f"rows/expert={m} ({workload.valid_rows_per_expert} valid) routed_m="
        f"{t['routed_m']} contiguous_m={contiguous_m} N={n} K={k}; "
        f"m_tiles={m_tiles} carry data x n_tiles={n_tiles} = "
        f"{m_tiles * n_tiles} working WGs of grid={geometry.grid} "
        f"({geometry.grid[0] - m_tiles * n_tiles} exit at entry)"
    )
    print(
        "[gemm_batch_isa_runner] useful logical read components: "
        f"A MXFP4={useful.a_payload_bytes}; "
        f"ScaleA E8M0={useful.a_scale_bytes}; "
        f"B MXFP4={useful.b_payload_bytes}; "
        f"ScaleB E8M0={useful.b_scale_bytes}; "
        f"read total={useful.read_bytes}"
    )
    print(
        "[gemm_batch_isa_runner] useful logical output formula: "
        f"{t['valid_routed_rows']} valid routed rows x {n // 2} BF16 "
        f"columns x 2 bytes = {useful.output_payload_bytes}; "
        f"output allocation footprint="
        f"{t['out'].nbytes} ({contiguous_m} capacity rows); "
        f"logical_write_bytes={traffic.write_bytes}"
    )
    print(
        "[gemm_batch_isa_runner] scheduled/padding context (excluded from "
        f"useful bytes): rows/expert={m}; valid/expert="
        f"{workload.valid_rows_per_expert}; active m_tiles={m_tiles}; "
        f"working WGs={m_tiles * n_tiles}; tail capacity rows="
        f"{contiguous_m - t['routed_m']}; tensor footprints A={a_bytes} "
        f"B={b_bytes} sA={sa_bytes} sB={sb_bytes}"
    )

    not_validated = "n/a (not validated)"
    err_cell = ref_hash = max_err_cell = rel_l2_cell = not_validated
    if loadonly:
        out_hash = not_validated
        flops_cell = "n/a (no store)"
    else:
        flops = 2.0 * t["routed_m"] * n * k
        flops_cell = round(flops / (us * 1e-6) / 1e12, 1)

        # Only the rows below each expert's psum are stored; the tile-alignment
        # padding rows are left untouched, so comparing them would score the
        # kernel wrong for doing exactly what it should.  Check that they really
        # were skipped, then drop them from the comparison.
        valid = t["valid_rows"]
        padding = t["out_view"][~valid]
        print(
            f"[gemm_batch_isa_runner] MoE padding-row check: "
            f"{int(valid.sum())} of {contiguous_m} rows valid "
            f"({workload.valid_rows_per_expert}/{MOE_TILE_M} per expert); padding "
            f"rows have {int((padding != 0).sum())} nonzero of {padding.numel()} "
            f"(expect 0)"
        )
        ref = t["reference"][valid]
        got = t["out_view"][valid].float()
        out_hash = single.tensor_blake2b128(got)
        ref_hash = single.tensor_blake2b128(ref)
        # The same two helpers every other kernel in this runner uses, so that
        # err / max_err_info / rel_l2 mean the same thing in every row: err is
        # check_allclose's mismatch fraction, and max_err_info is the
        # (reference, result, absolute_error) triple at the worst element,
        # which _markdown_cell renders as a 3-tuple.
        err_cell = single._load_dependencies(args.device).check_allclose(
            ref,
            got,
            rtol=1e-1,
            atol=1.0,
            msg="a8w4 MoE stage-1 ISA runner",
        )
        max_err_cell, rel_l2_cell = single.float32_error_metrics(ref, got)

    row = {
        "intype": "a4w4",
        "batch": experts,
        "M": m,
        "N": n,
        "K": k,
        "apre": 1,
        "init": args.init,
        "seed": args.seed,
        "dtype": "bf16",
        "gfx": gfx,
        "launch backend": (
            LAUNCH_BACKEND_CPP if args.cpp else LAUNCH_BACKEND_PYTHON
        ),
        "timing context": (
            TIMING_CONTEXT_MOE_PIPELINE
            if args.inmoe
            else TIMING_CONTEXT_STANDALONE
        ),
        "logical cluster tasks/plane": m_tiles * n_tiles,
        "physical clusters/plane": m_tiles * n_tiles,
        "physical WGs/plane": geometry.grid[0],
        "encoded recurrence stride/plane": 0,
        "ref hash128": ref_hash,
        "gemm_a4w4 us": round(us, 3),
        "gemm_a4w4 TFLOPS": flops_cell,
        "gemm_a4w4 bytes": traffic.total_bytes,
        "gemm_a4w4 TB/s": round(
            single.bytes_per_microsecond_to_tbps(traffic.total_bytes, us), 3
        ),
        "gemm_a4w4 err": err_cell,
        "gemm_a4w4 out hash128": out_hash,
        "gemm_a4w4 max_err_info": max_err_cell,
        "gemm_a4w4 rel_l2": rel_l2_cell,
    }
    if args.inmoe:
        gemm2 = timing["pipeline_gemm2"]
        row.update(
            {
                "pipeline us": round(float(timing["pipeline_us"]), 3),
                "pipeline K3072 us": round(
                    float(gemm2["device_time_avg"]),
                    3,
                ),
                "pipeline K3072 count": int(gemm2["cnt"]),
                "pipeline random_y1 unchanged": bool(
                    timing["random_y1_unchanged"]
                ),
                "pipeline isolation pairs": int(
                    timing["isolation_pairs_checked"]
                ),
            }
        )

    def validate() -> tuple[None, bool]:
        return None, True

    return row, timing, validate


_select_target_profiler_timing = single._select_target_profiler_timing


def _make_profiler_visible_driver_launch(
    torch_module: Any,
    raw_launch: Callable[[], Any],
    *,
    anchor: Any,
    output: Any,
) -> Callable[[], Any]:
    return single._make_profiler_visible_driver_launch(
        torch_module,
        raw_launch,
        anchor=anchor,
        output=output,
    )


def _run_target_kernel_perftest(
    callable_: Callable[[], Any],
    *,
    symbol: str,
    warmup: int,
    iters: int,
    device: int,
    synchronize: Callable[[], Any],
    mark_stream_synchronized: Callable[[], Any] | None = None,
    reported_warmup: int | None = None,
    test_graph: bool = False,
    timing_source: str | None = None,
    run_perftest_impl: Callable[..., tuple[Any, float, Any]] | None = None,
) -> tuple[Any, float, dict[str, Any], Any]:
    return single._run_target_kernel_perftest(
        callable_,
        symbol=symbol,
        warmup=warmup,
        iters=iters,
        device=device,
        synchronize=synchronize,
        mark_stream_synchronized=mark_stream_synchronized,
        reported_warmup=reported_warmup,
        test_graph=test_graph,
        timing_source=timing_source,
        run_perftest_impl=run_perftest_impl,
    )


def _run_selected_target_timing(
    args: argparse.Namespace,
    torch_module: Any,
    raw_launch: Callable[[], Any],
    *,
    profiler_launch: Callable[[], Any] | None,
    target_output: Any,
    target_tensors: Mapping[str, Any],
    target_shape: tuple[int, int, int],
    stream: Any,
    symbol: str,
    mark_stream_synchronized: Callable[[], Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    if args.inmoe:
        if profiler_launch is None:
            raise single.GemmIsaRunnerError(
                "--inmoe requires profiler-visible target launch correlation"
            )
        from my_code.isa_runner import moe_pipeline_timing_context

        timing = moe_pipeline_timing_context.run_inmoe_context(
            torch_module=torch_module,
            device=args.device,
            symbol=symbol,
            target_shape=target_shape,
            target_tensors=target_tensors,
            raw_target_launch=raw_launch,
            profiler_target_launch=profiler_launch,
            target_warmup=args.warmup,
            iters=args.iters,
            backend_label=(
                LAUNCH_BACKEND_CPP if args.cpp else LAUNCH_BACKEND_PYTHON
            ),
            synchronize=torch_module.cuda.synchronize,
            mark_stream_synchronized=mark_stream_synchronized,
            run_target_perftest=_run_target_kernel_perftest,
            select_target_profiler_timing=_select_target_profiler_timing,
            test_graph=args.cudagh,
            timing_source=moe_pipeline_timing_source(args.cudagh),
            error_type=single.GemmIsaRunnerError,
        )
        return target_output, timing

    if args.timing_method == TIMING_METHOD_PROFILER:
        output, _, timing, _ = _run_target_kernel_perftest(
            raw_launch if profiler_launch is None else profiler_launch,
            symbol=symbol,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
            synchronize=torch_module.cuda.synchronize,
            mark_stream_synchronized=mark_stream_synchronized,
            test_graph=args.cudagh,
        )
        return output, timing

    output, avg_us = single._run_batched_cuda_event_timing(
        torch_module,
        raw_launch,
        stream,
        num_warmup=args.warmup,
        num_iters=args.iters,
    )
    if mark_stream_synchronized is not None:
        mark_stream_synchronized()
    timing = single.make_cuda_event_timing_row(
        symbol,
        args.iters,
        args.warmup,
        avg_us,
        args.device,
    )
    print(
        "[gemm_batch_isa_runner] timing source=cuda.Event batched; "
        f"configured symbol={symbol}; launch-loop count={args.iters}; "
        "no profiler kernel name/count is claimed"
    )
    return output, timing


def _run_moe_cpp_perftest(
    run_perftest: Callable[..., tuple[Any, float, Any]],
    launch: Callable[[], Any],
    *,
    symbol: str,
    warmup: int,
    iters: int,
    device: int,
    synchronize: Callable[[], Any] = lambda: None,
    test_graph: bool = False,
) -> tuple[Any, dict[str, Any]]:
    output, _, timing, _ = _run_target_kernel_perftest(
        launch,
        symbol=symbol,
        warmup=warmup,
        iters=iters,
        device=device,
        synchronize=synchronize,
        test_graph=test_graph,
        run_perftest_impl=run_perftest,
    )
    return output, timing


def _print_run_perftest_timing(row: Mapping[str, Any]) -> None:
    display = dict(row)
    display["device_time_sum"] = f"{float(row['device_time_sum']):.4f}"
    display["device_time_avg"] = f"{float(row['device_time_avg']):.4f}"
    print(
        "run_perftest kernel timing "
        f"(microseconds; source={row['source']}; "
        "summary uses the captured target-row device_time_avg):"
    )
    print(single._plain_markdown(display, single.EVENT_TIMING_COLUMNS))

def _run_batch_gemm(
    args: argparse.Namespace,
    code_object: Path,
    geometry: single.LaunchGeometry,
    store_detection: single.GlobalOutputStoreDetection,
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
    writes_output = store_detection.writes_output
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

        profiled_launch = (
            _make_profiler_visible_driver_launch(
                torch,
                launch,
                anchor=output,
                output=output,
            )
            if args.timing_method == TIMING_METHOD_PROFILER
            else None
        )
        timed_output, timing = _run_selected_target_timing(
            args,
            torch,
            launch,
            profiler_launch=profiled_launch,
            target_output=output,
            target_tensors={
                "A": inputs["A"],
                "B": inputs["B"],
                "sA": inputs["sA"],
                "sB": inputs["sB"],
                "out": output,
            },
            target_shape=(m, n, k),
            stream=stream,
            symbol=BATCH_KERNEL_SYMBOL,
            mark_stream_synchronized=module.mark_stream_synchronized,
        )
        us = float(timing["device_time_avg"])

    flops = 2 * batch * m * n * k
    # Distinct bytes each matrix contributes once; A/B/sA/sB are always read,
    # but D is only touched by kernels that actually store.
    input_bytes = (
        inputs["A"].nbytes
        + inputs["B"].nbytes
        + inputs["sA"].nbytes
        + inputs["sB"].nbytes
    )
    traffic = single.make_logical_traffic(
        read_bytes=int(input_bytes),
        output_bytes_if_stored=int(output.nbytes),
        store_detection=store_detection,
    )
    single.print_logical_traffic("[gemm_batch_isa_runner]", traffic)
    if writes_output:
        print(
            f"[gemm_batch_isa_runner] logical_bytes={traffic.total_bytes} = "
            f"A+B+sA+sB ({input_bytes}) + D ({output.nbytes})"
        )
    else:
        print(
            f"[gemm_batch_isa_runner] logical_bytes={traffic.total_bytes} = "
            f"A+B+sA+sB ({input_bytes}); excludes D ({output.nbytes}) "
            "(no Global output store found in ISA); TFLOPS is not reported"
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
        "gemm_a4w4 bytes": traffic.total_bytes,
        "gemm_a4w4 TB/s": round(
            single.bytes_per_microsecond_to_tbps(traffic.total_bytes, us), 2
        ),
    }

    if not writes_output:
        not_validated = "n/a (no global output store)"
        row.update(
            {
                "gemm_a4w4 err": not_validated,
                "gemm_a4w4 out hash128": not_validated,
                "gemm_a4w4 max_err_info": not_validated,
                "gemm_a4w4 rel_l2": not_validated,
            }
        )

        def validate_loadonly() -> tuple[BatchValidation, bool]:
            return BatchValidation(
                reference_hash=reference_hash,
                output_hash=not_validated,
                error=0.0,
                max_abs=0.0,
                max_err_info=(0.0, 0.0, 0.0),
                rel_l2=0.0,
            ), True

        return row, timing, validate_loadonly

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

    return row, timing, validate


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


def _self_test() -> None:
    """Run CPU-only profiler row-selection and call-contract checks."""

    class FakeTraceFrame:
        def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
            self.records = [dict(record) for record in records]

        def to_dict(self, orient: str) -> list[dict[str, Any]]:
            assert orient == "records"
            return [dict(record) for record in self.records]

        def to_string(self, index: bool = True) -> str:
            assert index is True
            return repr(self.records)

    symbol = "target_kernel_v17"

    def cuda_row(
        name: str,
        *,
        count: int = 7,
        total: float = 28.0,
        avg: float = 4.0,
        device: int = 0,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "cnt": count,
            "host_time_sum": 0.0,
            "device_time_sum": total,
            "device_time_avg": avg,
            "device_type": "CUDA",
            "device_index": str(device),
        }

    selected = _select_target_profiler_timing(
        FakeTraceFrame(
            [
                {
                    "name": symbol,
                    "cnt": 1,
                    "device_time_sum": 1.0,
                    "device_time_avg": 1.0,
                    "device_type": "CPU",
                    "device_index": "0",
                },
                cuda_row(f"void {symbol}(unsigned char*)", device=2),
            ]
        ),
        symbol=symbol,
        warmup=3,
        requested_device=0,
    )
    assert selected["name"] == f"void {symbol}(unsigned char*)"
    assert selected["cnt"] == 7
    assert selected["device_time_avg"] == 4.0
    assert selected["device_index"] == 2
    assert selected["source"] == RUN_PERFTEST_TIMING_SOURCE

    def expect_selection_error(
        records: Sequence[Mapping[str, Any]],
        message: str,
    ) -> None:
        try:
            _select_target_profiler_timing(
                FakeTraceFrame(records),
                symbol=symbol,
                warmup=0,
                requested_device=0,
            )
        except single.GemmIsaRunnerError as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError(f"expected profiler selection failure: {message}")

    expect_selection_error(
        [cuda_row(f"{symbol}_different")],
        "found 0",
    )
    expect_selection_error(
        [cuda_row(symbol), cuda_row(f"wrapper({symbol})")],
        "found 2",
    )

    call: dict[str, Any] = {}
    output = object()
    sync_calls: list[str] = []

    def launch() -> Any:
        call["launched"] = int(call.get("launched", 0)) + 1
        return output

    trace_df = FakeTraceFrame([cuda_row(symbol)])

    def fake_run_perftest(
        callable_: Callable[[], Any],
        **kwargs: Any,
    ) -> tuple[Any, float, Any]:
        call["callable"] = callable_
        call["kwargs"] = dict(kwargs)
        return callable_(), 999.0, trace_df

    actual_output, full_us, timing, actual_trace = _run_target_kernel_perftest(
        launch,
        symbol=symbol,
        warmup=3,
        iters=10,
        device=0,
        synchronize=lambda: sync_calls.append("sync"),
        mark_stream_synchronized=lambda: sync_calls.append("mark"),
        run_perftest_impl=fake_run_perftest,
    )
    assert actual_output is output
    assert actual_trace is trace_df
    assert full_us == 999.0
    assert call["callable"] is launch
    assert call["kwargs"] == {
        "num_warmup": 3,
        "num_iters": 10,
        "testGraph": False,
        "return_trace_df": True,
        "num_rotate_args": 1,
    }
    assert call["launched"] == 1
    assert sync_calls == ["sync", "mark"]
    assert timing["cnt"] == 7
    assert timing["device_time_sum"] == 28.0
    assert timing["device_time_avg"] == 4.0
    assert timing["source"] == RUN_PERFTEST_TIMING_SOURCE

    cuda_event_log: list[str] = []

    class FakeEvent:
        def __init__(self, label: str) -> None:
            self.label = label

        def record(self, actual_stream: Any) -> None:
            assert actual_stream == "stream"
            cuda_event_log.append(f"{self.label}.record")

        def synchronize(self) -> None:
            cuda_event_log.append(f"{self.label}.synchronize")

        def elapsed_time(self, other: Any) -> float:
            assert other.label == "end"
            cuda_event_log.append("start.elapsed_time")
            return 0.75

    class FakeCuda:
        def __init__(self) -> None:
            self.created = 0

        def Event(self, *, enable_timing: bool) -> FakeEvent:
            assert enable_timing is True
            label = ("start", "end")[self.created]
            self.created += 1
            cuda_event_log.append(f"{label}.create")
            return FakeEvent(label)

        def synchronize(self) -> None:
            cuda_event_log.append("cuda.synchronize")

    class FakeTorch:
        def __init__(self) -> None:
            self.cuda = FakeCuda()

    event_launches = 0

    def event_launch() -> str:
        nonlocal event_launches
        event_launches += 1
        cuda_event_log.append("launch")
        return "event-output"

    event_output, event_avg_us = single._run_batched_cuda_event_timing(
        FakeTorch(),
        event_launch,
        "stream",
        num_warmup=2,
        num_iters=3,
    )
    assert event_output == "event-output"
    assert event_avg_us == 250.0
    assert event_launches == 5
    assert cuda_event_log == [
        "launch",
        "launch",
        "cuda.synchronize",
        "start.create",
        "end.create",
        "start.record",
        "launch",
        "launch",
        "launch",
        "end.record",
        "end.synchronize",
        "start.elapsed_time",
    ]
    parser = _build_parser()
    timing_action = next(
        action for action in parser._actions if action.dest == "timing_method"
    )
    assert timing_action.default == TIMING_METHOD_PROFILER
    assert tuple(timing_action.choices) == TIMING_METHOD_CHOICES
    default_args = parser.parse_args(["--self-test"])
    assert default_args.inmoe is False
    assert default_args.cpp is False
    assert default_args.cudagh is False
    assert default_args.shape == (18432, 2048, 7168)
    assert default_args.batch == 1
    assert all(
        getattr(default_args, name) is None for name in MOE_CLI_FIELDS
    )
    moe_args = parser.parse_args(
        [
            "--self-test",
            "--experts",
            "96",
            "--tokens",
            "512",
            "--topk",
            "6",
            "--model-dim",
            "7168",
            "--inter-dim",
            "3072",
        ]
    )
    workload = resolve_moe_cli_workload(
        moe_args,
        shape_explicit=False,
        batch_explicit=False,
    )
    assert workload.valid_routes == 3072
    assert workload.valid_rows_per_expert == 32
    assert workload.raw_n == 6144
    assert workload.contiguous_m == 9216
    assert workload.grid == (3456, 1, 1)
    assert workload.working_wgs == 2304
    assert workload.tail_wgs == 1152
    geometry = make_moe_launch_geometry(workload)
    assert geometry.tiles == (24, 144)
    assert geometry.grid == (3456, 1, 1)
    assert geometry.block == (128, 1, 1)
    assert geometry.cluster == (1, 1, 1)
    noncluster_keepalive, noncluster_config = single._make_hip_launch_config(
        geometry,
        None,
    )
    assert noncluster_keepalive is None
    assert noncluster_config.numAttrs == 0
    assert not bool(noncluster_config.attrs)
    cluster_keepalive, cluster_config = single._make_hip_launch_config(
        replace(geometry, cluster=(4, 1, 1)),
        None,
    )
    assert cluster_keepalive is not None
    assert cluster_config.numAttrs == 1
    assert bool(cluster_config.attrs)
    assert (
        select_kernel_mode(MOE_V21_LOADONLY_KERNEL_SYMBOL, 1)
        == ("moe-v21-loadonly", MOE_V21_LOADONLY_PROFILE)
    )
    validate_moe_v21_source_contract(
        f".set {MOE_V21_CONTRACT_MARKER}, 1\n",
        MOE_V21_LOADONLY_KERNEL_SYMBOL,
    )
    try:
        validate_moe_v21_source_contract(
            "",
            MOE_V21_LOADONLY_KERNEL_SYMBOL,
        )
    except single.GemmIsaRunnerError:
        pass
    else:
        raise AssertionError("v21 contract marker must be mandatory")
    partial_moe_args = parser.parse_args(
        ["--self-test", "--experts", "96"]
    )
    try:
        resolve_moe_cli_workload(
            partial_moe_args,
            shape_explicit=False,
            batch_explicit=False,
        )
    except single.GemmIsaRunnerError as exc:
        assert "all-or-none" in str(exc)
    else:
        raise AssertionError("partial MoE user parameter group must be rejected")
    try:
        resolve_moe_cli_workload(
            moe_args,
            shape_explicit=True,
            batch_explicit=False,
        )
    except single.GemmIsaRunnerError as exc:
        assert "do not mix" in str(exc)
    else:
        raise AssertionError("mixed MoE/shape CLI forms must be rejected")
    assert parser.parse_args(["--self-test", "--cudagh"]).cudagh is True
    assert sum(action.dest == "cudagh" for action in parser._actions) == 1
    assert (
        parser.parse_args(["--self-test"]).timing_method
        == TIMING_METHOD_PROFILER
    )
    assert (
        parser.parse_args(
            ["--self-test", "--timing-method", TIMING_METHOD_CUDA_EVENT]
        ).timing_method
        == TIMING_METHOD_CUDA_EVENT
    )
    try:
        parser.parse_args(["--self-test", "--timing-method", "invalid"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("invalid --timing-method must be rejected")
    selected_flags = parser.parse_args(["--self-test", "--inmoe", "--cpp"])
    assert selected_flags.inmoe is True
    assert selected_flags.cpp is True
    for removed_args in (
        ["--self-test", "--timing-context", TIMING_CONTEXT_MOE_PIPELINE],
        ["--self-test", "--launch-backend", LAUNCH_BACKEND_CPP],
    ):
        try:
            parser.parse_args(removed_args)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"removed CLI must be rejected: {removed_args}")
    single.validate_timing_method_context(
        TIMING_METHOD_PROFILER,
        True,
    )
    try:
        single.validate_timing_method_context(
            TIMING_METHOD_CUDA_EVENT,
            True,
        )
    except single.GemmIsaRunnerError:
        pass
    else:
        raise AssertionError("pipeline+cuda-event must be rejected")
    try:
        single.validate_timing_method_context(
            TIMING_METHOD_CUDA_EVENT,
            False,
            True,
        )
    except single.GemmIsaRunnerError:
        pass
    else:
        raise AssertionError("cuda-event+--cudagh must be rejected")
    validate_cudagh_backend(False, False)
    validate_cudagh_backend(True, True)
    try:
        validate_cudagh_backend(True, False)
    except single.GemmIsaRunnerError:
        pass
    else:
        raise AssertionError("Python hipdrv+--cudagh must be rejected")
    full_detection = single.detect_global_output_stores(
        "; buffer_store_b128 in comment\n"
        "ds_store_b128 v0, v[0:3]\n"
        "tensor_store_from_lds s[0:3], s[4:11]\n"
    )
    loadonly_detection = single.detect_global_output_stores(
        "; tensor_store_from_lds is removed\n"
        "ds_store_b128 v0, v[0:3]\n"
    )
    useful = single.calculate_moe_a4w4_bf16_useful_bytes(
        valid_rows=3072,
        active_experts=96,
        n=6144,
        k=7168,
        output_n=3072,
    )
    assert useful.a_payload_bytes == 11_010_048
    assert useful.a_scale_bytes == 688_128
    assert useful.b_payload_bytes == 2_113_929_216
    assert useful.b_scale_bytes == 132_120_576
    assert useful.read_bytes == 2_257_747_968
    assert useful.output_payload_bytes == 18_874_368
    full_traffic = single.make_logical_traffic(
        read_bytes=useful.read_bytes,
        output_bytes_if_stored=useful.output_payload_bytes,
        store_detection=full_detection,
    )
    loadonly_traffic = single.make_logical_traffic(
        read_bytes=useful.read_bytes,
        output_bytes_if_stored=useful.output_payload_bytes,
        store_detection=loadonly_detection,
    )
    assert full_traffic.total_bytes == 2_276_622_336
    assert loadonly_traffic.total_bytes == 2_257_747_968
    assert (
        full_traffic.total_bytes - loadonly_traffic.total_bytes
        == 18_874_368
    )
    print(
        "[gemm_batch_isa_runner] SELF_TEST_OK: profiler row selection, "
        "run_perftest arguments, pure batched CUDA-event ordering, timing "
        "method CLI choices/default, MoE user-parameter derivation/conflict "
        "rejection, v21 geometry, pipeline combination rejection, and "
        "global-store-aware logical traffic accounting"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = single._build_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == "isa":
            action.required = False
            action.help = (
                "complete gfx1250 AMDGPU assembly source: the original MAB "
                "symbols require --batch 1; mab_tdm_gemm_full_batch and "
                "64x256_1x4_batch_ps support batched dispatch; omitted only "
                "with --self-test"
            )
        elif action.dest == "shape":
            action.help = (
                "per-matrix GEMM shape M,N,K; MAB modes require M%%16=0, "
                "N%%1024=0, and K%%512=0; omit for the preferred MoE "
                "user-level parameter form "
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
        elif action.dest == "warmup":
            action.help = (
                "untimed launches: run_perftest handles profiler mode; "
                "cuda-event mode queues them then synchronizes once "
                "(default: 101)"
            )
        elif action.dest == "iters":
            action.help = (
                "formal launches; profiler mode requires greater than 1, "
                "cuda-event mode requires at least 1 (default: 100)"
            )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help=(
            f"number of independent batch-major GEMMs in one dispatch "
            f"(default: 1; maximum: {MAX_BATCH}); omit for the preferred "
            "MoE user-level parameter form"
        ),
    )
    moe_group = parser.add_argument_group(
        "balanced MoE workload",
        "For MoE ISA symbols, supply all five options together and omit "
        "--shape/--batch. They do not alter dense-kernel defaults.",
    )
    moe_group.add_argument(
        "--experts",
        type=int,
        default=None,
        help=f"number of experts (audited specialization: {MOE_N_EXPERTS})",
    )
    moe_group.add_argument(
        "--tokens",
        type=int,
        default=None,
        help=(
            "input token count "
            f"(audited specialization: {MOE_REFERENCE_TOKENS})"
        ),
    )
    moe_group.add_argument(
        "--topk",
        type=int,
        default=None,
        help=(
            "routes per token "
            f"(audited specialization: {MOE_REFERENCE_TOPK})"
        ),
    )
    moe_group.add_argument(
        "--model-dim",
        type=int,
        default=None,
        help=(
            "raw GEMM K / model dimension "
            f"(audited specialization: {MOE_BAKED_K})"
        ),
    )
    moe_group.add_argument(
        "--inter-dim",
        type=int,
        default=None,
        help=(
            "stage-1 output dimension; raw GEMM N is twice this value "
            f"(audited specialization: {MOE_REFERENCE_INTER_DIM})"
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
        "--cpp",
        action="store_true",
        help=(
            "use the C++ hipModuleLaunchKernel backend; restricted to exact "
            f"{MOE_CPP_ISA_BASENAME}. Without this flag, use Python ctypes "
            "hipModuleLaunchKernel for ordinary 1x1x1 launches and "
            "hipDrvLaunchKernelEx for true workgroup clusters"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run CPU-only profiler timing helper tests and exit",
    )
    return parser


def validate_cudagh_backend(cudagh: bool, cpp: bool) -> None:
    if cudagh and not cpp:
        raise single.GemmIsaRunnerError(
            "--cudagh requires --cpp in gemm_batch_isa_runner.py: the Python "
            "hipDrvLaunchKernelEx launcher produced an empty CUDA Graph in "
            "runtime validation"
        )


def _argv_has_option(argv: Sequence[str], *options: str) -> bool:
    return any(
        token == option or token.startswith(f"{option}=")
        for token in argv
        for option in options
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    if args.self_test:
        _self_test()
        return 0
    if args.isa is None:
        parser.error("--isa is required unless --self-test is used")
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
        single.validate_timing_method_context(
            args.timing_method,
            args.inmoe,
            args.cudagh,
        )
    except single.GemmIsaRunnerError as exc:
        parser.error(str(exc))
    try:
        validate_cudagh_backend(args.cudagh, args.cpp)
    except single.GemmIsaRunnerError as exc:
        parser.error(str(exc))

    try:
        isa = single._resolve_isa(args.isa)
        source = single._read_isa_source(isa)
        symbol = single.resolve_kernel_symbol_from_text(source, args.symbol)
        validate_moe_v21_source_contract(source, symbol)
        mode, profile = select_kernel_mode(symbol, args.batch)
        workload: MoeWorkload | None = None
        moe_options_supplied = any(
            getattr(args, name) is not None for name in MOE_CLI_FIELDS
        )
        if mode in MOE_MODES:
            workload = resolve_moe_cli_workload(
                args,
                shape_explicit=_argv_has_option(
                    raw_argv,
                    "--shape",
                    "-mnk",
                ),
                batch_explicit=_argv_has_option(raw_argv, "--batch"),
            )
            args.shape = workload.shape
            args.batch = workload.experts
            print_moe_workload_contract(workload)
        elif moe_options_supplied:
            raise single.GemmIsaRunnerError(
                "--experts/--tokens/--topk/--model-dim/--inter-dim are "
                "accepted only by exact MoE ISA symbols"
            )
        if (
            args.timing_method == TIMING_METHOD_PROFILER
            and mode != "legacy"
            and args.iters <= 1
        ):
            raise single.GemmIsaRunnerError(
                "run_perftest profiler timing requires --iters greater than 1"
            )
        if args.cpp:
            validate_cpp_backend_target(isa, symbol, mode)
        store_detection = single.detect_global_output_stores(source)
        writes_output = store_detection.writes_output

        if mode in MOE_MODES:
            resources = single.parse_assembly_resources(source, symbol)
            if mode in ("moe-gemm1-loadonly", "moe-v21-loadonly") and writes_output:
                raise single.GemmIsaRunnerError(
                    f"{symbol} load-only contract requires stores to D=False"
                )
            if mode == "moe-gemm1" and not writes_output:
                raise single.GemmIsaRunnerError(
                    f"{symbol} full-kernel contract requires Global D stores"
                )
            assert workload is not None
            geometry = make_moe_launch_geometry(workload)
        elif mode in (
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
        if (
            args.cpp
            and args.clang is None
        ):
            # _resolve_clang canonicalizes the symlink to clang-23.  Keep the
            # configured DEFAULT_CLANG spelling in this backend's manifest and
            # build command so the requested toolchain path is auditable.
            clang = single.DEFAULT_CLANG.absolute()
        if single._clang_uses_default_runtime_libraries(clang):
            single._prepend_default_clang_runtime_libraries()
        print(f"[gemm_batch_isa_runner] selected clang: {clang}")
        cpp_extension = None
        build_context = (
            contextlib.nullcontext(None)
            if args.cpp
            else tempfile.TemporaryDirectory(prefix="gemm_batch_isa_runner_")
        )
        with build_context as temp:
            if args.cpp:
                try:
                    from . import moe_cpp_backend
                except ImportError:
                    import moe_cpp_backend

                dependencies = single._load_dependencies(args.device)
                artifacts = moe_cpp_backend.prepare_moe_cpp_backend(
                    isa=isa,
                    clang=clang,
                    symbol=symbol,
                    single_module=single,
                    torch_module=dependencies.torch,
                )
                code_object = artifacts.code.code_object
                cpp_extension = artifacts.extension.module
                print(
                    "[gemm_batch_isa_runner] C++ backend ready: "
                    f"co_key={artifacts.code.key}; "
                    f"extension_key={artifacts.extension.key}; "
                    f"co_cache_hit={artifacts.code.cache_hit}; "
                    f"extension_cache_hit={artifacts.extension.cache_hit}"
                )
            else:
                assert temp is not None
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
                code_object = result.code_object
            _keep_code_object_if_requested(
                args,
                isa,
                symbol,
                code_object,
            )

            if mode in MOE_MODES:
                assert workload is not None
                row, timing, validate = _run_moe_gemm(
                    args,
                    symbol=symbol,
                    profile=profile,
                    geometry=geometry,
                    workload=workload,
                    code_object=code_object,
                    gfx=single._load_dependencies(args.device).get_gfx(),
                    store_detection=store_detection,
                    cpp_extension=cpp_extension,
                )
            elif mode in (
                "mab",
                "mab-full",
                "mab-full-batch",
                "mab-full-batch-loadonly",
                "mab-full-batch-loadonly-wv23-256k",
            ):
                row, timing, validate = _run_mab_gemm(
                    args,
                    code_object,
                    geometry,
                    symbol,
                    resources.metadata_lds,
                    store_detection,
                )
            elif mode == "legacy":
                legacy_row, timing, legacy_passed = single._run_gemm(
                    args,
                    code_object,
                    symbol,
                    profile,
                    geometry,
                    store_detection,
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
                row, timing, validate = _run_batch_gemm(
                    args,
                    code_object,
                    geometry,
                    store_detection,
                )

            row["logical cluster tasks/plane"] = geometry.logical_cluster_tasks
            row["physical clusters/plane"] = (
                geometry.cluster_grid[0] * geometry.cluster_grid[1]
            )
            row["physical WGs/plane"] = geometry.grid[0] * geometry.grid[1]
            row["encoded recurrence stride/plane"] = geometry.persistent_stride
            if str(timing["source"]).startswith("run_perftest "):
                _print_run_perftest_timing(timing)
            else:
                single._print_cuda_event_timing(timing)
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
