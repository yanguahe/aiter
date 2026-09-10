# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Grouped MXFP4 GEMM launchers."""

from __future__ import annotations

import os

import torch

from .kernels.tensor_shim import ptr_arg

_SUPPORTED_CLUSTER_N = (4, 3, 2)
_USE_GEMM1_BASELINE_93665E = False


def use_gemm1_baseline_93665e() -> None:
    """Select the in-tree 93665e GEMM1 implementation for this process."""
    global _USE_GEMM1_BASELINE_93665E
    _USE_GEMM1_BASELINE_93665E = True


def _select_next_stage_prefetch(csv_next_stage_prefetch: int) -> int:
    """Selects the environment override or the CSV setting."""
    value = os.environ.get("AITER_TDM_NEXT_STAGE_PREFETCH")
    if value is None:
        return int(bool(csv_next_stage_prefetch))
    value = value.strip()
    if value not in ("0", "1"):
        raise ValueError("AITER_TDM_NEXT_STAGE_PREFETCH must be 0 or 1")
    return int(value)


def _select_cluster_n(n_tiles: int, csv_cluster_n: int) -> int:
    """Selects the environment override or CSV cluster degree."""
    env_cluster_n = os.environ.get("AITER_FLYDSL_MXFP4_CLUSTER_N")
    try:
        requested_cluster_n = (
            int(env_cluster_n) if env_cluster_n is not None else int(csv_cluster_n)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("AITER_FLYDSL_MXFP4_CLUSTER_N must be an integer") from exc
    if requested_cluster_n <= 1:
        return 1
    if requested_cluster_n not in _SUPPORTED_CLUSTER_N:
        return 1
    return requested_cluster_n if n_tiles % requested_cluster_n == 0 else 1


def _select_num_waves_per_tensor_tdm(csv_num_waves: int) -> int:
    """Selects the CSV value or falls back to the environment setting."""
    if csv_num_waves in (1, 2, 4):
        return csv_num_waves

    try:
        num_waves = int(os.environ.get("AITER_FLYDSL_NUM_WAVES_PER_TENSOR_TDM", "2"))
    except ValueError as exc:
        raise ValueError(
            "AITER_FLYDSL_NUM_WAVES_PER_TENSOR_TDM must be 1, 2, or 4"
        ) from exc
    if num_waves not in (1, 2, 4):
        raise ValueError(
            "AITER_FLYDSL_NUM_WAVES_PER_TENSOR_TDM must be 1, 2, or 4, got "
            f"{num_waves}"
        )
    return num_waves


def _select_epilogue_batch_wn(default: int) -> int:
    """Selects the target GEMM1 SiLU epilogue batch width."""
    try:
        batch_wn = int(
            os.environ.get("AITER_FLYDSL_GEMM1_EPILOGUE_BATCH_WN", str(default))
        )
    except ValueError as exc:
        raise ValueError(
            "AITER_FLYDSL_GEMM1_EPILOGUE_BATCH_WN must be 1, 2, 4, or 8"
        ) from exc
    if batch_wn not in (1, 2, 4, 8):
        raise ValueError("AITER_FLYDSL_GEMM1_EPILOGUE_BATCH_WN must be 1, 2, 4, or 8")
    return batch_wn


def _select_schedule_hints(default: int) -> int:
    value = os.environ.get("AITER_FLYDSL_GEMM1_SCHEDULE_HINTS", str(default)).strip()
    if value not in ("0", "1"):
        raise ValueError("AITER_FLYDSL_GEMM1_SCHEDULE_HINTS must be 0 or 1")
    return int(value)


def _select_relax_cluster_wrap_dscnt(default: int) -> int:
    value = os.environ.get(
        "AITER_FLYDSL_GEMM1_RELAX_CLUSTER_WRAP_DSCNT", str(default)
    ).strip()
    if value not in ("0", "1"):
        raise ValueError("AITER_FLYDSL_GEMM1_RELAX_CLUSTER_WRAP_DSCNT must be 0 or 1")
    return int(value)


def _select_binary_int(name: str, default: int) -> int:
    value = os.environ.get(name, str(default)).strip()
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return int(value)


def _select_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _select_tristate(name: str, default: int = -1) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be -1, 0, or 1") from exc
    if value not in (-1, 0, 1):
        raise ValueError(f"{name} must be -1, 0, or 1")
    return value


def _select_wmma_reuse(default: int = 0) -> int:
    try:
        value = int(os.environ.get("AITER_FLYDSL_GEMM1_WMMA_REUSE", str(default)))
    except ValueError as exc:
        raise ValueError("AITER_FLYDSL_GEMM1_WMMA_REUSE must be 0, 1, 2, or 3") from exc
    if value not in (0, 1, 2, 3):
        raise ValueError("AITER_FLYDSL_GEMM1_WMMA_REUSE must be 0, 1, 2, or 3")
    return value


def flydsl_grouped_gemm_a8w4_masked(
    out,
    a,
    w,
    a_scales,
    w_scales,
    m_tile_map,
    *,
    n_experts,
    contiguous_m,
    N,
    K,
    tile_m=64,
    tile_n=256,
    tile_k=256,
    m_warp=1,
    n_warp=4,
    num_buffers=3,
    out_is_f16=0,
    a_is_fp4=0,
    stage1_act=0,
    bias=None,
    swiglu_limit=7.0,
    stream=None,
    stage1_quant_out=0,
    quant_scale=None,
    quant_wmma_rep=1,
    cluster_n=-1,
    waves_per_tensor_tdm=-1,
    next_stage_prefetch=0,
    situ_beta=1.0,
    situ_linear_beta=1.0,
    a_preshuffle=0,
):
    """Launches a contiguous-M grouped a8w4 GEMM on the TDM kernel."""
    # Keep the historical kernel available in the current tree so performance
    # comparisons do not need to change Git state inside the ROCm container.
    if _USE_GEMM1_BASELINE_93665E:
        from .kernels.mxfp4_preshuffle_gfx1250_tdm_93665e import (
            launch_gemm_a8w4_tdm,
        )
    else:
        from .kernels.mxfp4_preshuffle_gfx1250_tdm import launch_gemm_a8w4_tdm

    if stream is None:
        stream = torch.cuda.current_stream()
    if stage1_act == 3:
        if float(situ_beta) <= 0.0:
            raise ValueError(f"situ_beta must be > 0, got {situ_beta!r}")
        if float(situ_linear_beta) <= 0.0:
            raise ValueError(f"situ_linear_beta must be > 0, got {situ_linear_beta!r}")
    num_buffers = min(num_buffers, max(1, K // tile_k))
    has_bias = 1 if bias is not None else 0
    bias_ptr = ptr_arg(bias) if bias is not None else ptr_arg(a)
    quant_scale_tensor = out if quant_scale is None else quant_scale.view(torch.uint8)
    n_tiles = (N + tile_n - 1) // tile_n
    cluster_n = _select_cluster_n(n_tiles, cluster_n)
    waves_per_tensor_tdm = _select_num_waves_per_tensor_tdm(waves_per_tensor_tdm)
    next_stage_prefetch = _select_next_stage_prefetch(next_stage_prefetch)
    if cluster_n > 1 and n_tiles % cluster_n:
        raise ValueError(
            f"[grouped-moe tdm] cluster_n={cluster_n} needs n_tiles={n_tiles} "
            f"(N={N}, tile_n={tile_n}) to be an exact multiple"
        )
    if _USE_GEMM1_BASELINE_93665E:
        launch_gemm_a8w4_tdm(
            out,
            ptr_arg(a),
            ptr_arg(w),
            a_scales.view(torch.int32),
            w_scales.view(torch.int32),
            contiguous_m,
            stream,
            N,
            K,
            tile_m,
            tile_n,
            tile_k,
            m_warp,
            n_warp,
            out_is_f16,
            num_buffers,
            a_is_fp4,
            ptr_arg(m_tile_map),
            n_experts,
            stage1_act,
            has_bias,
            bias_ptr,
            float(swiglu_limit),
            stage1_quant_out,
            quant_wmma_rep,
            quant_scale_tensor,
            cluster_n,
            next_stage_prefetch,
            waves_per_tensor_tdm,
            float(situ_beta),
            float(situ_linear_beta),
        )
        return out

    target_fp4_prefill_common = all(
        (
            a_is_fp4,
            K == 7168,
            tile_m in (128, 256),
            tile_n in (128, 256),
            (m_warp, n_warp) in ((2, 2), (4, 2), (4, 4), (8, 2)),
            stage1_act == 1,
            stage1_quant_out == 0,
            out_is_f16 == 0,
            has_bias == 0,
            cluster_n == 4,
            next_stage_prefetch == 1,
            n_experts > 0,
        )
    )
    target_fp4_prefill = target_fp4_prefill_common and (
        (tile_k, num_buffers, waves_per_tensor_tdm)
        in (
            (128, 4, 1),
            (128, 4, 2),
            (128, 4, 4),
            (256, 4, 1),
            (256, 4, 2),
            (256, 4, 4),
            (256, 3, 1),
            (256, 3, 2),
            (256, 3, 4),
            (256, 2, 1),
            (512, 2, 1),
        )
    )
    launch_gemm_a8w4_tdm(
        out,
        ptr_arg(a),
        ptr_arg(w),
        a_scales.view(torch.int32),
        w_scales.view(torch.int32),
        contiguous_m,
        stream,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        out_is_f16,
        num_buffers,
        a_is_fp4,
        ptr_arg(m_tile_map),
        n_experts,
        stage1_act,
        has_bias,
        bias_ptr,
        float(swiglu_limit),
        stage1_quant_out,
        quant_wmma_rep,
        quant_scale_tensor,
        cluster_n,
        next_stage_prefetch,
        waves_per_tensor_tdm,
        float(situ_beta),
        float(situ_linear_beta),
        _select_epilogue_batch_wn(8 if target_fp4_prefill else 1),
        int(bool(a_preshuffle)),
        _select_schedule_hints(1 if target_fp4_prefill else 0),
        _select_relax_cluster_wrap_dscnt(1 if target_fp4_prefill else 0),
        _select_binary_int("AITER_FLYDSL_GEMM1_DIRECT_SCALES", 0)
        if target_fp4_prefill
        else 0,
        _select_binary_int("AITER_FLYDSL_GEMM1_TRANSITIVE_CLUSTER_SYNC", 0)
        if target_fp4_prefill
        else 0,
        _select_binary_int("AITER_FLYDSL_GEMM1_TDM_EARLY_TIMEOUT", 1),
        _select_binary_int("AITER_FLYDSL_GEMM1_M_MAJOR_SWIZZLE", 0)
        if target_fp4_prefill
        else 0,
        _select_positive_int("AITER_FLYDSL_GEMM1_MMA_GROUP", 4),
        _select_positive_int("AITER_FLYDSL_GEMM1_FENCE_COVER_MMA", 8),
        _select_tristate("AITER_FLYDSL_GEMM1_DISABLE_XDL_ARB_STALL"),
        _select_binary_int("AITER_FLYDSL_GEMM1_SILU_POLY9", 0)
        if stage1_act == 1
        else 0,
        _select_binary_int("AITER_FLYDSL_GEMM1_SILU_HARD", 0)
        if stage1_act == 1
        else 0,
        _select_binary_int("AITER_FLYDSL_GEMM1_SILU_RELU", 0)
        if stage1_act == 1
        else 0,
        _select_wmma_reuse(),
        _select_binary_int("AITER_FLYDSL_GEMM1_DELAY_ACC_ZERO", 0),
    )
    return out
