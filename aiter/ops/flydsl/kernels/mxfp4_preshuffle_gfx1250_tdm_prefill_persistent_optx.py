# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""M128 window scheduling for gfx1250 fused-quant MoE GEMM1."""

from collections import namedtuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl, tdm_ops
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec

from aiter.utility.mx_types import MxDtypeInt as MxDtype

from .gemm_common_gfx1250 import (
    batched_silu_swiglu,
    fused_silu_swiglu_elem,
    make_lds_copy_ops,
    make_sgpr_opaque,
    pipeline_fence,
    workgroup_barrier,
)
from .mega_moe_gfx1250.gemm2_scatter import (
    SCATTER_ROWS,
    load_bf16_scatter_rows,
    make_bf16_scatter_view,
    store_bf16_scatter_group,
)
from .quant_utils import (
    emit_amax_e8m0_native_scale,
    emit_cvt_scalef32_pk8_fp4_bf16,
    emit_cvt_scalef32_pk8_fp8_f32,
)
from .tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE,
)

TDM_DESCRIPTOR_VERSION = 1


def _amax_inputs_from_bf16x16(all_vals_bf16):
    """Reduce sixteen rounded BF16 values to two FP32 absolute maxima.

    This is an IR builder, so the Python lists and loops are resolved while
    compiling. Packed maximumNumber preserves the numeric-over-NaN behavior
    of emit_amax_e8m0_native_scale; that helper still performs the FP32 clamp,
    peer reduction, and E8M0 rounding.
    """
    bf16_words = all_vals_bf16.bitcast(fx.Int32)
    abs_pairs = [
        Vec.from_elements(
            [bf16_words[i] & fx.Int32(0x7FFF7FFF)], fx.Int32
        ).bitcast(fx.BFloat16)
        for i in range(8)
    ]
    while len(abs_pairs) > 1:
        abs_pairs = [
            Vec(
                fx.arith.maxnumf(
                    abs_pairs[i].ir_value(), abs_pairs[i + 1].ir_value()
                )
            )
            for i in range(0, len(abs_pairs), 2)
        ]
    maxima = abs_pairs[0].to(fx.Float32)
    return [maxima[0], maxima[1]]


# Optimized A-preshuffle path for the production gfx1250 MoE tiles.

@flyc.jit
def launch_gemm_a8w4_tdm_fused_persistent_optx(
    arg_c: fx.Tensor,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Tensor,
    arg_scale_b: fx.Tensor,
    i32_m: fx.Int32,
    stream: fx.Stream,
    N: fx.Int32,
    K: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    out_is_f16: Constexpr[int],
    num_buffers: Constexpr[int],
    a_is_fp4: Constexpr[int],
    arg_m_tile_map: fx.Pointer,
    n_experts: fx.Int32,
    stage1_act: Constexpr[int],
    has_bias: Constexpr[int],
    arg_bias: fx.Pointer,
    f32_swiglu_limit: fx.Float32,
    stage1_quant_out: Constexpr[int] = 0,
    quant_wmma_rep: Constexpr[int] = 1,
    arg_quant_scale: fx.Tensor = None,
    cluster_n: Constexpr[int] = 1,
    cluster_m: Constexpr[int] = -1,
    next_stage_prefetch: Constexpr[int] = 0,
    num_waves_per_tensor_tdm: Constexpr[int] = 2,
    tdm_b_th: Constexpr[int] = 0,
    balanced_m192: Constexpr[int] = 0,
    partial_m_direct_scale: Constexpr[int] = 0,
    partial_m_drain_output: Constexpr[int] = 0,
    optx_kgroup: Constexpr[int] = 1,
    optx_skip: Constexpr[int] = 1,
    optx_stagger: Constexpr[int] = 0,
    optx_snake: Constexpr[int] = 0,
):
    """Launch the protected M256 parent / M128 window GEMM1 path.

    Expert count and tensor bounds are runtime arguments, so E64 and E65
    share one compiled GEMM1 object. Exact-partial-M semantics and the
    validated M256_FUSED_LAYOUT pipeline are retained.
    """
    assert (K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers) == (
        7168, 256, 256, 256, 2, 4, 4
    )
    assert (stage1_act, stage1_quant_out, quant_wmma_rep) == (1, 1, 8)
    assert (cluster_n, cluster_m, num_waves_per_tensor_tdm) == (4, 1, 2)
    assert a_is_fp4 and not out_is_f16 and not has_bias
    assert next_stage_prefetch and not balanced_m192
    assert not partial_m_direct_scale and not partial_m_drain_output
    assert optx_kgroup == 1, "merged READY requires G1/S4"
    assert optx_snake in (0, 1, 2)
    assert optx_skip in (0, 1) and not optx_stagger

    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    WAVE = 32
    PACK_TK = tile_k // 2
    KWS = tile_k // WMMA_K
    balanced_t256_m192 = bool(balanced_m192)
    partial_m_direct_scale = bool(partial_m_direct_scale)
    partial_m_drain_output = bool(partial_m_drain_output)
    compute_tile_m = 192 if balanced_t256_m192 else tile_m
    # A spare LDS buffer is required while the next tile's first k128 is carried.
    next_stage_on = 1 if (next_stage_prefetch and num_buffers >= 3) else 0
    # This launcher contains only the production A-preshuffle schedules. Other
    # tiles and EP variants use launch_gemm_a8w4_tdm instead.
    standard_geometry = all(
        (
            tile_m in (192, 256),
            tile_n == 256,
            tile_k == 256,
            m_warp == 2,
            n_warp == 2,
            num_buffers == 4,
        )
    )
    gemm2_eight_wave_geometry = all(
        (
            tile_m in (192, 256),
            tile_n == 256,
            tile_k == 256,
            m_warp == 2,
            n_warp == 4,
            num_buffers == 4,
        )
    )
    common_schedule = all(
        (
            a_is_fp4,
            standard_geometry or gemm2_eight_wave_geometry,
            next_stage_on == 1,
            num_waves_per_tensor_tdm == 2,
            stage1_quant_out in (0, 1),
            out_is_f16 == 0,
            has_bias == 0,
            cluster_n == 4,
        )
    )
    fp4_prefill_schedule = common_schedule and K == 7168 and stage1_act == 1
    gemm2_schedule = all(
        (
            common_schedule,
            K in (2048, 3072),
            stage1_act == 0,
        )
    )
    assert fp4_prefill_schedule or gemm2_schedule
    assert not partial_m_direct_scale or all(
        (
            fp4_prefill_schedule,
            stage1_quant_out == 1,
            tile_m in (192, 256),
        )
    )
    assert not partial_m_drain_output or all(
        (
            fp4_prefill_schedule,
            stage1_quant_out == 1,
            tile_m in (192, 256),
        )
    )
    epilogue_batch_wn = (
        (4 if gemm2_eight_wave_geometry else 8)
        if fp4_prefill_schedule
        else 1
    )
    relax_cluster_wrap_dscnt = fp4_prefill_schedule
    disable_xdl_arb_stall = (
        0 if fp4_prefill_schedule and not gemm2_eight_wave_geometry else -1
    )
    wmma_reuse = 3 if gemm2_eight_wave_geometry else fp4_prefill_schedule
    schedule_wmma_m_rep = compute_tile_m // m_warp // WMMA_M
    schedule_mma_n_rep = (tile_n // n_warp // WMMA_N) // 2
    schedule_mma_count = schedule_wmma_m_rep * schedule_mma_n_rep
    # Leave one four-MMA group before the closing group.  This gives t256 its
    # established 28 and gives t192 a valid 20 instead of the old literal 28.
    fence_cover_mma = schedule_mma_count - 4
    # Split the output rows evenly.  t256 naturally remains 4+4, while t192
    # becomes 3+3 instead of 4+2 so the first TDM store has more epilogue work
    # available to cover it.
    output_store_split_wm = (
        schedule_wmma_m_rep // 2
        if fp4_prefill_schedule
        else 3
    )
    output_store_wave_split = gemm2_schedule
    assert (tile_n // n_warp // WMMA_N) % epilogue_batch_wn == 0
    cluster_m = (4 if cluster_m < 0 else cluster_m) if fp4_prefill_schedule else 1
    assert cluster_m in (1, 4)
    cache_tag = (
        K,
        tile_m,
        compute_tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        out_is_f16,
        num_buffers,
        a_is_fp4,
        stage1_act,
        has_bias,
        TDM_DESCRIPTOR_VERSION,
        stage1_quant_out,
        quant_wmma_rep,
        cluster_n,
        cluster_m,
        next_stage_on,
        num_waves_per_tensor_tdm,
        tdm_b_th,
        epilogue_batch_wn,
        relax_cluster_wrap_dscnt,
        disable_xdl_arb_stall,
        wmma_reuse,
        fence_cover_mma,
        output_store_split_wm,
        output_store_wave_split,
        partial_m_direct_scale,
        partial_m_drain_output,
    )
    _ = cache_tag
    warp_tile_m = compute_tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_acc = wmma_m_rep * wmma_n_rep
    assert not a_is_fp4 or wmma_n_rep % 2 == 0
    # The dedicated FP4 instruction consumes two adjacent 16-column weight
    # fragments and produces a 32-column result.  Keep the epilogue's existing
    # 16-column view by splitting each vec16 accumulator after the K loop.
    mma_n_rep = wmma_n_rep // 2 if a_is_fp4 else wmma_n_rep
    mma_n_acc = wmma_m_rep * mma_n_rep
    num_waves = m_warp * n_warp
    block = num_waves * WAVE

    ACT_ELEM = fx.Float4E2M1FN
    ACT_NDW = 8

    A_LDS_OUTER = compute_tile_m // 16
    # Keep the established schedules for other geometries and partial-M
    # options. This branch changes internal LDS placement and scheduling;
    # the routed input/output layouts and all K contributions are retained.
    M256_FUSED_LAYOUT = bool(
        fp4_prefill_schedule
        and gemm2_eight_wave_geometry
        and tile_m == 256
        and compute_tile_m == 256
        and stage1_quant_out == 1
        and quant_wmma_rep == 8
        and cluster_m == 1
        and not partial_m_direct_scale
        and not partial_m_drain_output
    )
    SEGMENT_SPLIT_A = bool(
        fp4_prefill_schedule
        and gemm2_eight_wave_geometry
        and (compute_tile_m == 192 or M256_FUSED_LAYOUT)
        and stage1_quant_out == 1
    )
    LDS_SEGMENT_BYTES = 64 * 1024
    SPLIT_SCALE_ARENA = M256_FUSED_LAYOUT
    INPUT_LDS_PAD_INTERVAL = (
        1024
        if fp4_prefill_schedule and gemm2_eight_wave_geometry and not SPLIT_SCALE_ARENA
        else 0
    )
    INPUT_LDS_PAD_AMOUNT = 16 if INPUT_LDS_PAD_INTERVAL else 0
    INPUT_LDS_SUBTILE_STRIDE = 1024 + INPUT_LDS_PAD_AMOUNT
    INPUT_LDS_ROW_PAD = (
        (PACK_TK * 16 // INPUT_LDS_PAD_INTERVAL) * INPUT_LDS_PAD_AMOUNT
        if INPUT_LDS_PAD_INTERVAL
        else 0
    )
    A_LDS_ROW = PACK_TK * 16 + INPUT_LDS_ROW_PAD
    B_LDS_ROW = PACK_TK * 16 + INPUT_LDS_ROW_PAD
    STAGE_A = ((A_LDS_OUTER * A_LDS_ROW + 15) // 16) * 16
    STAGE_B = (((tile_n // 16) * B_LDS_ROW + 15) // 16) * 16
    A_OWNER_ROWS = A_LDS_OUTER // m_warp
    A_OWNER_BYTES = A_OWNER_ROWS * A_LDS_ROW

    SC_INNER = tile_k // 4
    SB_SUPERS = tile_n // 32
    AS_INNER = SC_INNER
    AS_SUPERS = compute_tile_m // 32
    # One outer row is one wave's M tile. Its inner (k128, wm, lane16)
    # layout gives each WMMA scale operand a contiguous 16-dword block.
    STAGE_SA = ((AS_SUPERS * AS_INNER * 4 + 15) // 16) * 16
    STAGE_SB = ((SB_SUPERS * SC_INNER * 4 + 15) // 16) * 16
    SA_OWNER_SUPERS = AS_SUPERS // m_warp
    SA_OWNER_BYTES = SA_OWNER_SUPERS * AS_INNER * 4
    # Cyclic ownership places A0 at the stage front and A1 at the following
    # stage tail. M192 keeps scales beside the data; M256 uses four exact
    # 64 KiB data segments and a separate scale arena below.
    B_OFF = A_OWNER_BYTES + SA_OWNER_BYTES if SEGMENT_SPLIT_A else STAGE_A
    SA_OFF = A_OWNER_BYTES if SEGMENT_SPLIT_A else B_OFF + STAGE_B
    SB_OFF = B_OFF + STAGE_B if SEGMENT_SPLIT_A else SA_OFF + STAGE_SA
    A_TAIL_OFF = SB_OFF + STAGE_SB if SEGMENT_SPLIT_A else 0
    SA_TAIL_OFF = A_TAIL_OFF + A_OWNER_BYTES if SEGMENT_SPLIT_A else 0
    SEGMENT_USED = (
        SA_TAIL_OFF + SA_OWNER_BYTES
        if SEGMENT_SPLIT_A
        else STAGE_A + STAGE_B + STAGE_SA + STAGE_SB
    )
    if SPLIT_SCALE_ARENA:
        # A0[16 KiB], B[32 KiB], prior-stage A1[16 KiB]. ScaleA/ScaleB
        # retain their canonical contents in a separate four-stage ring.
        B_OFF = A_OWNER_BYTES
        A_TAIL_OFF = B_OFF + STAGE_B
        SA_OFF = 0
        SA_TAIL_OFF = SA_OWNER_BYTES
        SB_OFF = STAGE_SA
        SEGMENT_USED = A_TAIL_OFF + A_OWNER_BYTES
    assert not SEGMENT_SPLIT_A or SEGMENT_USED <= LDS_SEGMENT_BYTES
    # The cyclic layout uses exactly four physical segments.  Other schedules
    # retain their compact 512-byte-aligned linear stage pitch.
    PITCH = (
        LDS_SEGMENT_BYTES
        if SEGMENT_SPLIT_A
        else ((SEGMENT_USED + 511) // 512) * 512
    )

    SCALE_ARENA_OFF = num_buffers * PITCH
    SCALE_STAGE_PITCH = STAGE_SA + STAGE_SB
    out_elem = T.f16 if out_is_f16 else T.bf16
    # +16 cols: the bf16 passthrough epilogue stages C with a padded row pitch
    # to break the ds_store bank conflict; reserve it so the padded tile fits.
    C_STORE_B = (
        (tile_m // 16) * (tile_n // 64) * 256 + m_warp * quant_wmma_rep * 64
        if stage1_quant_out == 1
        else ((compute_tile_m * (tile_n + 16) * 2 + 127) // 128) * 128
    )
    # Both physical expert counts use the same validated four-task pipeline.
    PERSISTENT_TASKS = 4
    T192_TASK_OVERLAP = bool(
        PERSISTENT_TASKS > 1
        and SEGMENT_SPLIT_A
        and tile_m == 192
        and not partial_m_direct_scale
        and not partial_m_drain_output
    )
    # The penultimate carry fence releases stage 2's front before the final
    # K tile. Exact t192 quant output reuses that front; next-task I1 only
    # writes stage 2's tail. Alternate schedules retain their placement.
    OUTPUT_IN_RELEASED_STAGE = False
    # Two independent window arenas, alternating across task generations.
    # Each wave owns four M16 payload slots; scales still go directly global.
    OUTPUT_LDS_OFF = num_buffers * PITCH + num_buffers * SCALE_STAGE_PITCH
    OPTX_WINDOW_BYTES = 8192 + 512
    A_CACHE_STAGES = 2 if gemm2_eight_wave_geometry and gemm2_schedule else 0
    A_CACHE_PITCH = ((STAGE_A + 511) // 512) * 512
    A_CACHE_OFF = num_buffers * PITCH
    ARENA_B = OUTPUT_LDS_OFF + 2 * OPTX_WINDOW_BYTES
    assert OUTPUT_LDS_OFF + C_STORE_B <= ARENA_B
    assert ARENA_B <= 320 * 1024

    # Quant epilogue compile-time constants.
    # Each wn subtile produces 8 output cols (4 per kgrp) after silu/swiglu;
    # 4 wn subtiles = 32 output cols = 1 MX block for per-32 scaling.
    WN_PER_MX_BLOCK = 4
    assert not T192_TASK_OVERLAP or WN_PER_MX_BLOCK * 4 == 16
    Q_MX_BLOCKS_PER_WG = tile_n // 64
    Q_MX_BLOCK_BYTES = 16 * 16
    Q_ROW_TILE_BYTES = Q_MX_BLOCKS_PER_WG * Q_MX_BLOCK_BYTES
    Q_M16_TILES = tile_m // 16
    Q_STORE_M16 = wmma_m_rep if balanced_t256_m192 else quant_wmma_rep
    Q_SCALE_ROW_BYTES = 16 * 4
    Q_SCALE_TILE_BYTES = quant_wmma_rep * Q_SCALE_ROW_BYTES
    Q_SCALE_LDS_OFF = Q_M16_TILES * Q_ROW_TILE_BYTES
    assert Q_SCALE_LDS_OFF + m_warp * Q_SCALE_TILE_BYTES <= C_STORE_B
    if OUTPUT_IN_RELEASED_STAGE and PERSISTENT_TASKS > 1:
        quant_lds_bytes = Q_SCALE_LDS_OFF + m_warp * Q_SCALE_TILE_BYTES
        # Exact t192 output fits before stage 2's tail, which holds I1's
        # wave_m=1 A/ScaleA. Final K input and I0 use other segments.
        assert quant_lds_bytes <= (A_TAIL_OFF if SEGMENT_SPLIT_A else PITCH)

    _kname = (
        "a8w4_tdm_fp4"
        f"_t{tile_m}x{tile_n}x{tile_k}_w{m_warp}x{n_warp}"
        f"_b{num_buffers}_K{K}_edyn"
        f"_act{stage1_act}_q{stage1_quant_out}r{quant_wmma_rep}"
        f"_cn{cluster_n}_cm{cluster_m}_prefetch_apre_persist"
        f"{'_vm192' if balanced_t256_m192 else ''}"
        f"{'_pmds' if partial_m_direct_scale else ''}"
        f"{'_pmdrain' if partial_m_drain_output else ''}"
        f"_optx_g{optx_kgroup}_s4_sk{optx_skip}_st{optx_stagger}_sn{optx_snake}_bcache_merge_wavewindow_wavepub190_edyn_earlybanks_splitS0_S1rows0_carryF_carryS"
    )

    @flyc.kernel(name=_kname, known_block_size=[block, 1, 1])
    def kernel(
        arg_c: fx.Tensor,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_m_tile_map: fx.Pointer,
        arg_bias: fx.Pointer,
        arg_quant_scale: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        n_experts: fx.Int32,
        f32_swiglu_limit: fx.Float32,
    ):
        xdl_arb_off = (
            disable_xdl_arb_stall
            if disable_xdl_arb_stall >= 0
            else int(tile_m == 256 and tile_n == 256 and num_waves == 4)
        )
        if const_expr(fp4_prefill_schedule and xdl_arb_off):
            from flydsl._mlir.dialects import llvm as llvm_dialect

            # gfx1250 SCHED_MODE bit 2; the helper writes bit 4. See CDNA5 ISA 5.7.2.1.
            # Restrict this to one wave per SIMD because disabling the XDL
            # arbitration stall can block co-execution opportunities.
            llvm_dialect.call_intrinsic(
                None,
                "llvm.amdgcn.s.setreg",
                [fx.Int32(26 | (2 << 6)).ir_value(), fx.Int32(1).ir_value()],
                [],
                [],
            )

        persistent_m_context = None
        if const_expr(PERSISTENT_TASKS > 1):
            persistent_bid_x = fx.block_idx.x
            persistent_local_n = persistent_bid_x % cluster_n
            persistent_local_m = 0
            persistent_m_tile = persistent_bid_x // cluster_n
            persistent_blk_m = persistent_m_tile * tile_m
            persistent_blk_m64 = fx.Int64(persistent_blk_m)
            persistent_i32_ptr = fx.PointerType.get(
                elem_ty=fx.Int32.ir_type,
                address_space=fx.AddressSpace.Global,
                alignment=4,
            )
            persistent_tile_map = fx.recast_iter(persistent_i32_ptr, arg_m_tile_map)
            persistent_lo = persistent_blk_m * 0
            persistent_hi = persistent_lo + n_experts
            # Upper-bound search is uniform within a WG and supports runtime E.
            while persistent_lo < persistent_hi:
                persistent_mid = (persistent_lo + persistent_hi) >> 1
                persistent_mid_clamped = (persistent_mid < n_experts - 1).select(
                    persistent_mid, n_experts - 1
                )
                persistent_go_right = (
                    persistent_tile_map[persistent_mid_clamped] <= persistent_blk_m
                )
                persistent_lo = persistent_go_right.select(
                    persistent_mid + 1, persistent_lo
                )
                persistent_hi = persistent_go_right.select(
                    persistent_hi, persistent_mid
                )
            persistent_expert = persistent_lo
            persistent_eb64 = fx.Int64(persistent_expert)
            persistent_expert_idx = (persistent_expert < n_experts).select(
                persistent_expert, n_experts - 1
            )
            persistent_mn_oob = (
                persistent_tile_map[persistent_expert_idx] - persistent_blk_m
            )
            persistent_m_context = (
                persistent_local_n,
                persistent_local_m,
                persistent_m_tile,
                persistent_blk_m,
                persistent_blk_m64,
                persistent_tile_map,
                persistent_expert,
                persistent_eb64,
                persistent_mn_oob,
            )

        def run_task(tile_idx):
            K_TILES = K // tile_k
            Kp16 = (K // 2) * 16
            K4 = K // 4
            first_half = ((tile_idx & 1) if optx_snake == 1
                          else fx.Int32(1 if optx_snake == 2 else 0))

            tid = fx.thread_idx.x
            bid_x = fx.block_idx.x
            wave = rocdl.readfirstlane(T.i32, tid // WAVE)
            lane = tid % WAVE
            lane16 = lane % 16
            kgrp = lane // 16
            wave_m = wave % m_warp if M256_FUSED_LAYOUT else wave // n_warp
            wave_n = wave // m_warp if M256_FUSED_LAYOUT else wave % n_warp

            total_n_tiles = (i32_n + (tile_n - 1)) // tile_n
            total_m_tiles = (i32_m + (tile_m - 1)) // tile_m
            if const_expr(PERSISTENT_TASKS > 1):
                assert cluster_m == 1 and cluster_n == 4
                (
                    local_n,
                    local_m,
                    m_tile,
                    blk_m,
                    blk_m64,
                    tile_map,
                    expert,
                    eb64,
                    mn_oob,
                ) = persistent_m_context
                n_unit = tile_idx
            else:
                # DeepGEMM contiguous-M swizzle, run at cluster granularity so peers land
                # on one m_tile. Ternaries, not `if`: the rewriter would trace a branch.
                TILES_PER_GROUP = 16
                assert TILES_PER_GROUP % cluster_m == 0
                total_n_tiles = (i32_n + (tile_n - 1)) // tile_n
                total_m_tiles = (i32_m + (tile_m - 1)) // tile_m
                swz_id = bid_x // cluster_n if cluster_n > 1 else bid_x
                local_n = bid_x - swz_id * cluster_n if cluster_n > 1 else None
                n_units = total_n_tiles // cluster_n if cluster_n > 1 else total_n_tiles
                local_m = fx.block_idx.y if cluster_m > 1 else 0
                m_units = (total_m_tiles + cluster_m - 1) // cluster_m
                group_m_units = TILES_PER_GROUP // cluster_m
                blocks_per_group = n_units * group_m_units
                group = swz_id // blocks_per_group
                group_first_tile = group * group_m_units
                in_group = swz_id - group * blocks_per_group
                rem_tiles = m_units - group_first_tile
                group_tiles = (rem_tiles < group_m_units).select(rem_tiles, group_m_units)
                m_unit = group_first_tile + (in_group - (in_group // group_tiles) * group_tiles)
                n_unit = in_group // group_tiles
                m_tile = m_unit * cluster_m + local_m
                blk_m = m_tile * tile_m
                blk_m64 = fx.Int64(blk_m)
                # In-kernel bisect: find expert owning this M-tile via psum.
                i32_ptr = fx.PointerType.get(
                    elem_ty=fx.Int32.ir_type,
                    address_space=fx.AddressSpace.Global,
                    alignment=4,
                )
                tile_map = fx.recast_iter(i32_ptr, arg_m_tile_map)
                lo, hi = blk_m * 0, blk_m * 0 + n_experts
                while lo < hi:
                    mid = (lo + hi) >> 1
                    mid_clamped = (mid < n_experts - 1).select(mid, n_experts - 1)
                    go_right = tile_map[mid_clamped] <= blk_m
                    lo = go_right.select(mid + 1, lo)
                    hi = go_right.select(hi, mid)
                expert = lo
                eb64 = fx.Int64(expert)
                # Per-expert A-data OOB: bound to the owning expert's valid-row.
                expert_idx = (expert < n_experts).select(expert, n_experts - 1)
                mn_oob = tile_map[expert_idx] - blk_m
            blk_n = (
                (n_unit * cluster_n + local_n) * tile_n
                if cluster_n > 1
                else n_unit * tile_n
            )
            # Each A multicast row keeps one m_tile and varies n_tile. A 2-D
            # cluster has a separate contiguous N mask for each local M row.
            a_mcast_mask = (
                ((1 << cluster_n) - 1) << (local_m * cluster_n) if cluster_n > 1 else None
            )
            blk_n64 = fx.Int64(blk_n)
            n64 = fx.Int64(i32_n)
            B_BATCH_ROWS = n64 // 16
            N_SUPERS = (n64 + 31) // 32
            AS_ROW = K // 4

            c_outer_off, c_inner_off, c_stride = blk_m64, blk_n64, i32_n
            SB_OUTER_STRIDE = K4
            sb_batch_off = eb64 * (N_SUPERS * K4)

            b_mcast_mask = None
            full_cluster = None
            if const_expr(cluster_m > 1):
                # M peers may belong to different experts. Each peer derives the
                # identical column mask for the intersection of its expert and
                # this cluster; weights are only shared inside that intersection.
                cluster_first_m = m_unit * cluster_m
                valid_m_tiles = (tile_map[n_experts - 1] + tile_m - 1) // tile_m
                full_cluster = cluster_first_m + cluster_m <= valid_m_tiles
                prev_expert = (expert > 0).select(expert - 1, 0)
                # The fixed-step bisect can return E+1 for capacity-tail tiles.
                # Those WGs skip compute, but this descriptor setup runs first.
                prev_expert = (prev_expert < n_experts).select(prev_expert, n_experts - 1)
                first_m = (expert > 0).select(
                    (tile_map[prev_expert] + tile_m - 1) // tile_m, 0
                )
                end_m = tile_map[(expert < n_experts).select(expert, n_experts - 1)]
                end_m = (end_m + tile_m - 1) // tile_m
                column_mask = fx.Int32(0)
                for mi in range_constexpr(cluster_m):
                    peer_m = cluster_first_m + mi
                    same_expert = (peer_m >= first_m) & (peer_m < end_m)
                    column_mask = column_mask | same_expert.select(1 << (mi * cluster_n), 0)
                # A cluster containing sentinel tiles cannot use a cluster barrier.
                # Its live rows use the existing independent 1-D A-only protocol.
                b_mcast_mask = full_cluster.select(column_mask << local_n, 0)

            def cluster_sync(drain_lds=True):
                if const_expr(cluster_m > 1):  # noqa: SIM102 - preserve DSL staging
                    if full_cluster:
                        if const_expr(drain_lds):
                            workgroup_barrier()
                        else:
                            # Ring wrap only needs every requester wave to issue its TDM.
                            # Keep carry DS reads outstanding across the cluster wait so
                            # their latency overlaps synchronization.
                            rocdl.s_barrier_signal(-1)
                            rocdl.s_barrier_wait(-1)
                        if wave == 0:
                            rocdl.s_barrier_signal(-3)
                        rocdl.s_barrier_wait(-3)

            # Keep the allocator's raw byte pointer. ``peek().ptr`` retypes this
            # arena and changes address lowering for the t192 specialization.
            base_ptr = fx.SharedAllocator().allocate(ARENA_B)._ptr

            def ptr_to_idx(p):
                return fx.index_cast(T.index, fx.ptrtoint(p))

            output_base_ptr = base_ptr + OUTPUT_LDS_OFF
            stC_idx = ptr_to_idx(output_base_ptr)

            def buf_ptr(s):
                return base_ptr + s * PITCH

            def buf_ptr_opaque(s):
                """``buf_ptr(s)`` as an index the backend cannot constant-fold.

                In the constexpr-unrolled drain tail ``s`` is a literal and
                ``base_ptr`` is LDS offset 0, so the stage base is a constant and the
                backend re-associates ``(base + lane) + off`` into ``(base + off) +
                lane``: one v_add and one depctr wait per ds_load instead of one
                shared base register plus 16-bit ``offset:`` immediates. Hiding the
                base restores the shape the rolled loop gets for free -- 63 address
                registers down to 12, matching the main loop's 8.

                Only ever apply this to the *carry target*. Making the buffer a tile
                computes from opaque returns wrong results (logits_diff 0.039 against
                0.001 with random activations); the cause is not understood, and
                neither a matching "=s,0" constraint nor a real s_mov_b32 body
                changes it. Note --const-init 0 hides this: zero activations make a
                wrong LDS address read zeros, so it must be checked on random data.
                """
                return fx.index_cast(
                    T.index,
                    fx.Int32(
                        make_sgpr_opaque(fx.index_cast(T.i32, ptr_to_idx(buf_ptr(s))))
                    ),
                )

            def global_view(base, off, shape, stride):
                return fx.Tensor(fx.make_view(base + off, fx.make_layout(shape, stride)))

            def lds_view(ptr, shape, stride):
                return fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))

            lds_load_b32, _lds_store_b32 = make_lds_copy_ops(32)
            _, lds_store_b64 = make_lds_copy_ops(64)
            lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)

            i16_lds_ptr_ty = fx.PointerType.get(
                elem_ty=fx.Int16.ir_type,
                address_space=fx.AddressSpace.Shared,
                alignment=2,
            )
            i8_lds_ptr_ty = fx.PointerType.get(
                elem_ty=fx.Int8.ir_type,
                address_space=fx.AddressSpace.Shared,
                alignment=1,
            )

            def lds_store_b16(lds_base_idx, byte_offset, data):
                addr = fx.Int32(lds_base_idx) + fx.Int32(byte_offset)
                fx.ptr_store(fx.Int16(data), fx.inttoptr(i16_lds_ptr_ty, addr))

            def lds_store_b8(lds_base_idx, byte_offset, data):
                addr = fx.Int32(lds_base_idx) + fx.Int32(byte_offset)
                fx.ptr_store(data, fx.inttoptr(i8_lds_ptr_ty, addr))

            gA_base = fx.recast_iter(fx.Int8, arg_a)
            gB_base = fx.recast_iter(fx.Int8, arg_b)
            gSA_base, gSB_base = fx.get_iter(arg_scale_a), fx.get_iter(arg_scale_b)
            b_outer_row = eb64 * B_BATCH_ROWS + blk_n64 // 16
            a_off0 = (blk_m64 // 16) * Kp16
            b_off0 = b_outer_row * Kp16
            sb_off0 = (blk_n64 // 32) * SB_OUTER_STRIDE + sb_batch_off

            assert num_waves_per_tensor_tdm in (
                1,
                2,
                4,
            ), "num_waves_per_tensor_tdm must be 1, 2, or 4"
            assert (
                num_waves_per_tensor_tdm <= num_waves
            ), "waves per tensor cannot exceed workgroup waves"
            assert (
                num_waves % num_waves_per_tensor_tdm == 0
            ), "workgroup waves must divide evenly into tensor-owner groups"
            shared = fx.AddressSpace.Shared
            p8_shared = fx.PointerType.get(
                elem_ty=fx.Int8.ir_type, address_space=shared, alignment=16
            )
            p32_shared = fx.PointerType.get(
                elem_ty=fx.Int32.ir_type, address_space=shared, alignment=16
            )
            wave_groups = [
                tuple(range(i, i + num_waves_per_tensor_tdm))
                for i in range(0, num_waves, num_waves_per_tensor_tdm)
            ]
            waves = [wave_groups[i % len(wave_groups)] for i in range(4)]
            data_waves = (waves[0], waves[1])
            # TDMs issued by an owner wave per k-tile. When the workgroup has more
            # waves than the four A/B/SA/SB owner groups cover, the remaining waves
            # are compute-only.
            TDM_PER = max(1, 4 * num_waves_per_tensor_tdm // num_waves)
            nw = 1
            base_i32 = fx.recast_iter(p32_shared, base_ptr)
            base_lds_idx = ptr_to_idx(base_ptr)

            # Keep descriptor construction inside the owning-wave branch. Building
            # all four descriptors here makes every wave execute the setup for A, B,
            # SA, and SB even though it issues only one of them when wpt=1.
            Job = namedtuple(
                "Job",
                (
                    "g_base g_off g_stride oob inner outer on_i32 lds_off lds_row "
                    "k_adv waves pad wg_mask split_inner cache_modifier a_cache_off "
                    "owner_lds_offsets"
                ),
            )
            jobs = []

            def add_tdm_loads(
                g_base,
                g_off,
                g_stride,
                oob,
                inner,
                outer,
                *,
                on_i32,
                lds_off,
                lds_row,
                k_adv,
                wv,
                pad=None,
                wg_mask=None,
                split_inner=False,
                cache_modifier=0,
                a_cache_off=None,
                owner_lds_offsets=None,
            ):
                jobs.append(
                    Job(
                        g_base,
                        g_off,
                        g_stride,
                        oob,
                        inner,
                        outer,
                        on_i32,
                        lds_off,
                        lds_row,
                        k_adv,
                        wv,
                        pad,
                        wg_mask,
                        split_inner,
                        cache_modifier,
                        a_cache_off,
                        owner_lds_offsets,
                    )
                )

            add_tdm_loads(
                gA_base,
                a_off0,
                Kp16,
                (mn_oob + 15) // 16,
                PACK_TK * 16,
                A_LDS_OUTER,
                on_i32=False,
                lds_off=0,
                lds_row=A_LDS_ROW,
                k_adv=PACK_TK * 16,
                wv=data_waves[0],
                pad=(INPUT_LDS_PAD_INTERVAL, INPUT_LDS_PAD_AMOUNT)
                if INPUT_LDS_PAD_INTERVAL
                else None,
                wg_mask=a_mcast_mask,
                a_cache_off=0,
                owner_lds_offsets=(0, A_TAIL_OFF) if SEGMENT_SPLIT_A else None,
            )
            add_tdm_loads(
                gB_base,
                b_off0,
                Kp16,
                None,
                PACK_TK * 16,
                tile_n // 16,
                on_i32=False,
                lds_off=B_OFF,
                lds_row=B_LDS_ROW,
                k_adv=PACK_TK * 16,
                wv=data_waves[1],
                pad=(INPUT_LDS_PAD_INTERVAL, INPUT_LDS_PAD_AMOUNT)
                if INPUT_LDS_PAD_INTERVAL
                else None,
                wg_mask=b_mcast_mask,
                cache_modifier=tdm_b_th,
            )
            add_tdm_loads(
                gSA_base,
                (blk_m64 // 32) * AS_ROW,
                AS_ROW,
                (mn_oob + 31) // 32,
                AS_INNER,
                AS_SUPERS,
                on_i32=True,
                lds_off=(
                    (SCALE_ARENA_OFF + SA_OFF) // 4
                    if SPLIT_SCALE_ARENA else SA_OFF // 4
                ),
                lds_row=AS_INNER,
                k_adv=AS_INNER * 4,
                wv=waves[2],
                split_inner=AS_SUPERS < len(waves[2]),
                wg_mask=a_mcast_mask if cluster_m > 1 else None,
                owner_lds_offsets=(
                    (
                        (SCALE_ARENA_OFF + SA_OFF) // 4,
                        (SCALE_ARENA_OFF + SA_TAIL_OFF) // 4,
                    )
                    if SPLIT_SCALE_ARENA
                    else (
                        (SA_OFF // 4, SA_TAIL_OFF // 4)
                        if SEGMENT_SPLIT_A else None
                    )
                ),
            )
            add_tdm_loads(
                gSB_base,
                sb_off0,
                SB_OUTER_STRIDE,
                None,
                SC_INNER,
                SB_SUPERS,
                on_i32=True,
                lds_off=(
                    (SCALE_ARENA_OFF + SB_OFF) // 4
                    if SPLIT_SCALE_ARENA else SB_OFF // 4
                ),
                lds_row=SC_INNER,
                k_adv=SC_INNER * 4,
                wv=waves[3],
                wg_mask=b_mcast_mask,
            )
            next_jobs = jobs
            cached_a_jobs = jobs
            if const_expr(PERSISTENT_TASKS > 1):
                next_blk_n = ((tile_idx + 1) * cluster_n + local_n) * tile_n
                next_blk_n64 = fx.Int64(next_blk_n)
                next_b_outer_row = eb64 * B_BATCH_ROWS + next_blk_n64 // 16
                next_b_off0 = next_b_outer_row * Kp16
                next_sb_off0 = (
                    (next_blk_n64 // 32) * SB_OUTER_STRIDE + sb_batch_off
                )
                if const_expr(A_CACHE_STAGES > 0):
                    # Cached A0/A1 remove owner-wave TDMs. Split B across the first
                    # resident waves and leave ScaleA/ScaleB on the second slots so
                    # every physical SIMD receives similar next-task prefetch bytes.
                    next_jobs = [
                        jobs[1]._replace(
                            g_off=next_b_off0, waves=tuple(range(4))
                        ),
                        jobs[2],
                        jobs[3]._replace(g_off=next_sb_off0),
                    ]
                    # When a persistent task reuses A4 from buffer 0, split B4
                    # across the first resident waves. One input TDM per wave keeps
                    # the four-buffer pipeline's tensorcnt distance unchanged.
                    cached_a_jobs = [
                        jobs[1]._replace(waves=tuple(range(4))),
                        jobs[2],
                        jobs[3],
                    ]
                else:
                    next_jobs = [
                        jobs[0],
                        jobs[1]._replace(g_off=next_b_off0),
                        jobs[2],
                        jobs[3]._replace(g_off=next_sb_off0),
                    ]

            # Wave ids are runtime, so one stream serves every wave and one test per
            # owner list is the floor. Dispatched bodies receive their jobs directly.
            job_waves = sorted({j.waves for j in jobs})

            def owns(wv):
                """Runtime predicate: is this wave one of ``wv``?"""
                pred = wave == wv[0]
                for w in wv[1:]:
                    pred = pred | (wave == w)
                return pred

            def issue(s, kt, my_jobs=None, cache_a=False, reuse_cached_a=False,
                      producer_first_half=None, source_jobs=None):
                producer_first_half = (first_half if producer_first_half is None
                                       else producer_first_half)
                pa = fx.recast_iter(p8_shared, buf_ptr(s))
                so4 = s * (SCALE_STAGE_PITCH // 4 if SPLIT_SCALE_ARENA else PITCH // 4)
                if const_expr(cache_a):
                    cache_ptr = base_ptr + A_CACHE_OFF + kt * A_CACHE_PITCH
                    cache_pa = fx.recast_iter(p8_shared, cache_ptr)
                    cache_i32 = fx.recast_iter(p32_shared, cache_ptr)

                def emit(j):
                    split_i = j.split_inner and len(j.waves) > 1
                    if const_expr(len(j.waves) > 1):
                        if const_expr(split_i):
                            assert (
                                j.inner % len(j.waves) == 0
                            ), "TDM inner extent must divide owners"
                        else:
                            assert (
                                j.outer % len(j.waves) == 0
                            ), "TDM outer extent must divide owners"
                    seg = j.outer if split_i else j.outer // len(j.waves)
                    inner_seg = j.inner // len(j.waves) if split_i else j.inner
                    wave_outer_off = (
                        0 if split_i or len(j.waves) == 1 else (wave - j.waves[0]) * seg
                    )
                    wave_inner_off = (wave - j.waves[0]) * inner_seg if split_i else 0
                    if const_expr(j.owner_lds_offsets is not None):
                        # Physical owner 0 is F and owner 1 is S. Only the global
                        # source half swaps; LDS regions and multicast peers agree.
                        wave_outer_off = ((wave - j.waves[0]) ^ producer_first_half) * seg
                    gt = global_view(
                        j.g_base,
                        j.g_off
                        + fx.Int64(wave_outer_off) * j.g_stride
                        + fx.Int64(wave_inner_off),
                        (seg, inner_seg),
                        (j.g_stride, 1),
                    )
                    ext = None if j.oob is None else j.oob - wave_outer_off
                    if const_expr(ext is not None):
                        ext = (ext > 0).select(ext, 0)
                    pad_kw = (
                        {"pad_interval": j.pad[0], "pad_amount": j.pad[1]} if j.pad else {}
                    )
                    atom = fx.rocdl.make_tdm_atom(
                        gt,
                        [ext, None],
                        strides=[j.g_stride, None],
                        num_warps=nw,
                        # Descriptor bit 21: release to the peers already present and
                        # re-broadcast later, so early arrivals are not held for a merge.
                        early_timeout=j.wg_mask is not None,
                        cache_modifier=j.cache_modifier,
                        **pad_kw,
                    )
                    if const_expr(j.wg_mask is not None):
                        # Non-zero mask switches the TDM from GLOBAL_LOAD_ASYNC to
                        # CLUSTER_LOAD_ASYNC, fanning one load out to every peer's LDS.
                        atom = fx.atom_set_value(
                            atom, "workgroup_mask", fx.Int32(j.wg_mask)
                        )
                    use_a_cache = cache_a and j.a_cache_off is not None
                    cyclic_owner = j.owner_lds_offsets is not None and not use_a_cache
                    if const_expr(cyclic_owner):
                        owner = wave - j.waves[0]
                        dst_stage = (fx.Int32(s) + owner) & fx.Int32(num_buffers - 1)
                        elem_pitch = (
                            SCALE_STAGE_PITCH // 4
                            if SPLIT_SCALE_ARENA and j.on_i32
                            else (PITCH // 4 if j.on_i32 else PITCH)
                        )
                        owner_off = j.owner_lds_offsets[0] + owner * (
                            j.owner_lds_offsets[1] - j.owner_lds_offsets[0]
                        )
                        base = (
                            base_i32
                            if j.on_i32
                            else fx.recast_iter(p8_shared, base_ptr)
                        )
                        lds_off = dst_stage * elem_pitch + owner_off
                        dst_wave_outer_off = 0
                        stage_offset = 0
                    else:
                        base = (
                            cache_i32 if j.on_i32 else cache_pa
                        ) if use_a_cache else (base_i32 if j.on_i32 else pa)
                        lds_off = (
                            j.a_cache_off // 4 if j.on_i32 else j.a_cache_off
                        ) if use_a_cache else j.lds_off
                        dst_wave_outer_off = wave_outer_off * j.lds_row
                        stage_offset = so4 if j.on_i32 and not use_a_cache else 0
                    dst = lds_view(
                        base
                        + lds_off
                        + dst_wave_outer_off
                        + wave_inner_off
                        + stage_offset,
                        (seg, inner_seg),
                        (j.lds_row, 1),
                    )
                    fx.copy(atom, gt, dst, imm_offset=fx.Int64(kt * j.k_adv))

                if const_expr(my_jobs is not None):
                    for j in my_jobs:
                        if const_expr(not (reuse_cached_a and j.a_cache_off is not None)):
                            emit(j)
                else:
                    issue_jobs = jobs if source_jobs is None else source_jobs
                    for g in range_constexpr(len(job_waves)):
                        if owns(job_waves[g]):
                            for j in issue_jobs:
                                if const_expr(
                                    j.waves == job_waves[g]
                                    and not (
                                        reuse_cached_a and j.a_cache_off is not None
                                    )
                                ):
                                    emit(j)

            def dispatch_wave_job(fn, source_jobs=None):
                """Run ``fn`` with the current wave's jobs."""
                source_jobs = jobs if source_jobs is None else source_jobs
                source_job_waves = sorted({j.waves for j in source_jobs})
                for g in range_constexpr(len(source_job_waves)):
                    if owns(source_job_waves[g]):
                        fn(
                            [
                                j
                                for j in source_jobs
                                if j.waves == source_job_waves[g]
                            ]
                        )
                if const_expr(  # noqa: SIM102 - preserve DSL staging
                    4 * num_waves_per_tensor_tdm < num_waves
                ):
                    if wave >= 4 * num_waves_per_tensor_tdm:
                        fn([])

            wnb = wave_n * 64
            lds_b_lane_off = B_OFF + (wnb // 16) * B_LDS_ROW + kgrp * 256 + lane16 * 16
            lds_sb_lane_off = SB_OFF + ((wnb // 32) * SC_INNER + lane) * 4
            LdsAddr = namedtuple("LdsAddr", "a b sa sb")

            def calc_window_addr(kt, window):
                # Temporal windows alternate the physical half across SIMD peers.
                # Producers still use the original wave-independent first_half.
                physical_half = fx.Int32(window) ^ wave_m
                owner_stage = (fx.Int32(kt) + physical_half) & 3
                stage = fx.Int32(kt) & 3
                a_off = (owner_stage * PITCH + physical_half * A_TAIL_OFF
                         + wave_m * 2 * A_LDS_ROW + kgrp * 256 + lane16 * 16)
                sa_off = (SCALE_ARENA_OFF + owner_stage * SCALE_STAGE_PITCH
                          + physical_half * SA_TAIL_OFF + wave_m * AS_INNER * 4 + lane * 4)
                return LdsAddr(
                    a=base_lds_idx + fx.index_cast(T.index, a_off),
                    b=base_lds_idx + fx.index_cast(T.index, stage * PITCH + lds_b_lane_off),
                    sa=base_lds_idx + fx.index_cast(T.index, sa_off),
                    sb=base_lds_idx + fx.index_cast(
                        T.index, SCALE_ARENA_OFF + stage * SCALE_STAGE_PITCH + lds_sb_lane_off
                    ),
                )

            def load_a(base, wm, ksl):
                # M32 pair 0/2 belong to wave_m=0 and pair 1/3 to wave_m=1.
                off = ((wm // 2) * 4 + wm % 2) * A_LDS_ROW + ksl * 1024
                return Vec(lds_load_b128(base, fx.Int32(off))).shuffle(
                    Vec(lds_load_b128(base, fx.Int32(off + 512))), list(range(8))
                )

            def load_b(base, wn, ksl):
                def half(h):
                    off = h * B_LDS_ROW + ksl * 1024
                    return Vec(lds_load_b128(base, fx.Int32(off))).shuffle(
                        Vec(lds_load_b128(base, fx.Int32(off + 512))), list(range(8))
                    )
                return half(wn * 2).shuffle(half(wn * 2 + 1), list(range(16)))

            RmemSlot = namedtuple("RmemSlot", "a b sa sb")
            CachedB = namedtuple("CachedB", "b sb")
            b_group = [CachedB(
                b=[fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(2)],
                sb=fx.make_rmem_tensor(2, fx.Int32),
            ) for _ in range_constexpr(2 * optx_kgroup)]
            rmem_slots = [
                RmemSlot(
                    a=[fx.make_rmem_tensor(8, fx.Int32) for _ in range_constexpr(4)],
                    b=b_group[0].b,
                    sa=fx.make_rmem_tensor(2, fx.Int32),
                    sb=b_group[0].sb,
                ) for _ in range_constexpr(3)
            ]

            def cache_slot(slot, position):
                return slot._replace(b=b_group[position].b, sb=b_group[position].sb)

            # Banks are execution order F/S, independent of the runtime row half.
            c_frags = [fx.make_rmem_tensor(16, fx.Float32) for _ in range_constexpr(16)]

            def zero_accumulators():
                for cf in c_frags:
                    cf.store(fx.constant_vector(0.0, T.vec(16, T.f32)))

            def load_lds_data(slot, addr, ksl, read_weight=True):
                slot.sa.store(Vec.from_elements([
                    lds_load_b32(addr.sa, fx.Int32((sm * 2 * AS_INNER + ksl * 32) * 4))[0]
                    for sm in range_constexpr(2)
                ]))
                if const_expr(read_weight):
                    slot.sb.store(Vec.from_elements([
                        lds_load_b32(addr.sb, fx.Int32((sn * SC_INNER + ksl * 32) * 4))[0]
                        for sn in range_constexpr(2)
                    ]))
                    for wn in range_constexpr(2):
                        slot.b[wn].store(load_b(addr.b, wn, ksl))
                for wm in range_constexpr(4):
                    slot.a[wm].store(load_a(addr.a, wm, ksl))

            def mma_rows(window, slot, row_begin=0, row_end=4):
                sa, sb = slot.sa.load(), slot.sb.load()
                for wm in range_constexpr(row_begin, row_end):
                    if const_expr(optx_skip):
                        row_start = ((first_half ^ window ^ wave_m) * 128 + wave_m * 32
                                     + (wm // 2) * 64 + (wm % 2) * 16)
                        live_row = rocdl.readfirstlane(T.i32, fx.Int32(row_start < mn_oob))
                    else:
                        live_row = fx.Int32(1)
                    if live_row != fx.Int32(0):
                        for wn_raw in range_constexpr(2):
                            wn = 1 - wn_raw if wm % 2 else wn_raw
                            idx = window * 8 + wm * 2 + wn
                            c_frags[idx].store(rocdl.wmma_scale_f32_32x16x128_f4(
                                T.vec(16, T.f32),
                                slot.b[wn].load().ir_value(), slot.a[wm].load().ir_value(),
                                c_frags[idx].load().ir_value(), sb[wn], sa[wm // 2],
                                scaleAType=0, scaleBType=wm % 2,
                                reuseA=False, reuseB=False,
                            ))

            WINDOW_DS = 4 * 2 + 2 * 4 + 2 + 2

            def window_k128(window, slot, load_next=None, issue_next=None, ds_next=WINDOW_DS):
                if const_expr(issue_next is not None):
                    issue_next()
                if const_expr(load_next is not None):
                    load_next()
                mma_rows(window, slot)
                if const_expr(issue_next is not None):
                    rocdl.sched_vmem(1)
                # Two groups leave a small closing compute region after reads.
                rocdl.sched_mfma(4)
                if const_expr(load_next is not None):
                    rocdl.sched_dsrd(ds_next // 2)
                rocdl.sched_mfma(4)
                if const_expr(load_next is not None):
                    rocdl.sched_dsrd(ds_next - ds_next // 2)
                rocdl.sched_barrier(0)

            def compute_window_group(group, window):
                assert window == 0 and optx_kgroup == 1
                kt = group
                addr = calc_window_addr(kt, 0)
                second_addr = calc_window_addr(kt, 1)
                first0 = cache_slot(rmem_slots[0], 0)
                first1 = cache_slot(rmem_slots[1], 1)
                second1 = cache_slot(rmem_slots[2], 1)
                window_k128(0, first0,
                    load_next=lambda: load_lds_data(first1, addr, 1))

                def read_after_first1():
                    load_lds_data(first0, second_addr, 0, read_weight=False)
                    load_lds_data(second1, second_addr, 1, read_weight=False)

                window_k128(0, first1, ds_next=20, load_next=read_after_first1)

            def issue_released(kt, my_jobs):
                if (kt + num_buffers < K_TILES) | (tile_idx + 1 < PERSISTENT_TASKS):
                    cross_task = kt + num_buffers >= K_TILES
                    source_tile = cross_task.select(kt + num_buffers - K_TILES, kt + num_buffers)
                    next_first = (((tile_idx + 1) & 1) if optx_snake == 1
                                  else fx.Int32(1 if optx_snake == 2 else 0))
                    producer_first = cross_task.select(next_first, first_half)
                    effective_jobs = []
                    for j in my_jobs:
                        if const_expr(j.waves == jobs[1].waves):
                            j = j._replace(g_off=cross_task.select(next_b_off0, b_off0))
                        elif const_expr(j.waves == jobs[3].waves):
                            j = j._replace(g_off=cross_task.select(next_sb_off0, sb_off0))
                        effective_jobs.append(j)
                    issue((kt + num_buffers) & 3, source_tile, effective_jobs,
                          producer_first_half=producer_first)

            def compute_second_group(group, my_jobs, carry_next):
                group_kt = group * optx_kgroup
                for step in range_constexpr(optx_kgroup):
                    kt = group_kt + step
                    second0 = cache_slot(rmem_slots[0], 0)
                    second1 = cache_slot(rmem_slots[2], 1)
                    next_first0 = cache_slot(rmem_slots[0], 0)
                    if (tile_idx > 0) & (kt < 3):
                        tdm_ops.tensor_wait(4)
                    elif (tile_idx + 1 == PERSISTENT_TASKS) & (kt >= K_TILES - 2):
                        tdm_ops.tensor_wait(0)
                    elif (tile_idx + 1 == PERSISTENT_TASKS) & (kt + 3 == K_TILES):
                        tdm_ops.tensor_wait(1)
                    else:
                        tdm_ops.tensor_wait(2)
                    # S0/S1 LDS reads were issued during F1. Complete all
                    # readers before publishing this input stage's RELEASE.
                    rocdl.s_wait_dscnt(0)
                    rocdl.s_barrier_signal(-1)
                    rocdl.sched_barrier(0)
                    mma_rows(1, second0)
                    rocdl.sched_mfma(8)
                    rocdl.sched_barrier(0)
                    rocdl.s_barrier_wait(-1)
                    issue_released(kt, my_jobs)
                    if const_expr(carry_next):
                        next_addr = calc_window_addr(kt + 1, 0)
                        load_lds_data(next_first0, next_addr, 0)
                    mma_rows(1, second1, 0, 4)
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(8)
                    if const_expr(carry_next):
                        rocdl.sched_dsrd(WINDOW_DS)
                    rocdl.sched_barrier(0)

            def issue_window_output(window, actual_half):
                payload_row_stride = (fx.Int64(i32_n) // 4) * 16
                global_off = ((blk_m64 // 16) + fx.Int64(actual_half * 8 + wave_m * 2)) * payload_row_stride
                global_off = global_off + (blk_n64 // 64 + fx.Int64(wave_n)) * 256
                c_iter = fx.recast_iter(fx.Int8, fx.get_iter(arg_c))
                shape = (2, 2, 256)
                gt = global_view(c_iter, global_off, shape,
                                 (4 * payload_row_stride, payload_row_stride, 1))
                atom = fx.rocdl.make_tdm_atom(gt, [None, None, None],
                    strides=[4 * payload_row_stride, payload_row_stride, None],
                    num_warps=1, cache_modifier=0)
                src = lds_view(fx.recast_iter(fx.Int8, output_base_ptr)
                    + window * OPTX_WINDOW_BYTES + fx.index_cast(T.index, wave * 1024),
                    shape, (512, 256, 1))
                fx.copy(atom, src, gt)

            def window_epilogue(window):
                actual_half = first_half ^ window ^ wave_m
                scale_ptr_ty = fx.PointerType.get(elem_ty=fx.Int8.ir_type,
                    address_space=fx.AddressSpace.Global, alignment=1)
                scale_ptr = fx.recast_iter(scale_ptr_ty, fx.get_iter(arg_quant_scale))
                q_dst_scale_dwpr = i32_n // 256
                neg_limit = fx.Float32(0.0) - f32_swiglu_limit
                for wm in range_constexpr(4):
                    row_group = actual_half * 128 + wave_m * 32 + (wm // 2) * 64 + (wm % 2) * 16
                    if row_group < mn_oob:
                        row_rel = row_group + lane16
                        pairs = []
                        for wn in range_constexpr(2):
                            acc = Vec(c_frags[window * 8 + wm * 2 + wn].load())
                            for half in range_constexpr(2):
                                for p in range_constexpr(4):
                                    pairs.append((acc[half * 8 + 2 * p], acc[half * 8 + 2 * p + 1]))
                        vals = batched_silu_swiglu(pairs, swiglu=False,
                            limit_f32=f32_swiglu_limit, neg_limit_f32=neg_limit,
                            range_constexpr=range_constexpr)
                        bf16 = Vec.from_elements(vals, fx.Float32).to(fx.BFloat16)
                        scale_f32, e8m0 = emit_amax_e8m0_native_scale(
                            _amax_inputs_from_bf16x16(bf16), wave_size=32,
                            dtype=MxDtype.FP4_E2M1)
                        payload_byte = window * OPTX_WINDOW_BYTES + (wave * 4 + wm) * 256 + lane16 * 16
                        for half in range_constexpr(2):
                            src_bf16 = Vec.from_elements(
                                [bf16[half * 8 + i] for i in range_constexpr(8)], fx.BFloat16)
                            packed = emit_cvt_scalef32_pk8_fp4_bf16(src_bf16.ir_value(), scale_f32, i32_ty=T.i32)
                            lds_store_b16(stC_idx, payload_byte + half * 8 + kgrp * 2, packed)
                            lds_store_b16(stC_idx, payload_byte + half * 8 + 4 + kgrp * 2, packed >> fx.Int32(16))
                        if (row_rel < mn_oob) & (kgrp == 0):
                            row_i32 = fx.Int32(blk_m + row_rel)
                            mx = (fx.Int32(blk_n) + wave_n * 64) >> 6
                            dst_byte = (((row_i32 >> 5) * q_dst_scale_dwpr * 32
                                + (mx >> 2) * 32 + (row_i32 & 31)) * 4 + (mx & 3))
                            fx.ptr_store(e8m0, scale_ptr + dst_byte)
                # This TDM reads only this wave's private payload slice.
                # Tensor operations do not order with LDS stores automatically.
                rocdl.s_wait_dscnt(0)
                issue_window_output(window, actual_half)

            zero_accumulators()
            if expert < n_experts:
                def run_groups(my_jobs):
                    if tile_idx == 0:
                        for q in range_constexpr(num_buffers):
                            issue(q, q, my_jobs)
                        pipeline_fence(outstanding=3)
                    initial_addr = calc_window_addr(fx.Int32(0), 0)
                    load_lds_data(cache_slot(rmem_slots[0], 0), initial_addr, 0)
                    rocdl.sched_dsrd(WINDOW_DS)
                    rocdl.sched_barrier(0)
                    for optx_group in range(fx.Int32(0), fx.Int32(K_TILES - 1), fx.Int32(1)):
                        compute_window_group(optx_group, 0)
                        compute_second_group(optx_group, my_jobs, True)
                    compute_window_group(fx.Int32(K_TILES - 1), 0)
                    compute_second_group(fx.Int32(K_TILES - 1), my_jobs, False)
                dispatch_wave_job(run_groups)
                window_epilogue(0)
                window_epilogue(1)
                rocdl.s_wait_storecnt(0)

        if const_expr(PERSISTENT_TASKS == 1):
            run_task(fx.Int32(0))
        else:
            for tile_idx in range(
                fx.Int32(0), fx.Int32(PERSISTENT_TASKS), fx.Int32(1)
            ):
                run_task(tile_idx)
        tdm_ops.tensor_wait(0)

    m_tiles = (i32_m + (tile_m - 1)) // tile_m
    n_tiles = (N + (tile_n - 1)) // tile_n
    kargs = (
        arg_c,
        arg_a,
        arg_b,
        arg_scale_a,
        arg_scale_b,
        arg_m_tile_map,
        arg_bias,
        arg_quant_scale,
        i32_m,
        N,
        n_experts,
        f32_swiglu_limit,
    )
    grid = (
        ((m_tiles + cluster_m - 1) // cluster_m) * n_tiles // PERSISTENT_TASKS,
        cluster_m,
        1,
    )
    if cluster_n > 1:
        # Geometry must reach BOTH the definition and the launch site, or the
        # cluster never forms and the TDM loads silently fall back to per-load.
        kernel(
            *kargs,
            value_attrs={"rocdl.cluster_dims": f"{cluster_n},{cluster_m},1"},
        ).launch(
            grid=grid,
            block=(block, 1, 1),
            stream=stream,
            cluster=(cluster_n, cluster_m, 1),
        )
    else:
        kernel(*kargs).launch(grid=grid, block=(block, 1, 1), stream=stream)


launch_gemm_a8w4_tdm_fused_persistent_optx.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": False,
    "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
    "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    "enable-post-misched": True,
    "lsr-drop-solution": False,
    "unroll-threshold": 0,
    "unroll-partial-threshold": 0,
    "unroll-max-count": 1,
}
