# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Fresh FlyDSL drop-in replacements for the mxfp4 a4w4 MoE GEMM kernels:
#   - mxfp4_moe_gemm1_a4w4  (gate/up + SiLU*mul + per-32 fp4 requant)   [TODO]
#   - mxfp4_moe_gemm2_a4w4  (down proj + atomic topk reduce)            [in progress]
#
# The FlyDSL gemm1/gemm2 here match the HIP kernels' function AND interface
# exactly (true drop-in); every *other* MoE kernel (sort / quant / sort_scales /
# scatter_reduce) is reused from the mxfp4 (mx_fn) HIP path unchanged.
#
# Reference contract (csrc/kernels/mxfp4_moe/gemm_a4w4/gemm2_a4w4.cuh, gfx950,
# Kimi-K2.5: NE=385, K=512, N_OUT=7168, TOPK=9; BM in {16,32} atomic):
#   A_q (inter)      [max_sorted, K/2=256]      fp4x2  (SORTED order; row=sorted_pos)
#   A_scale          make_preshuffle tile       e8m0
#   B_q (w2)         [E, N_OUT, K/2]            fp4x2  (a16w4 preshuffle, gate_up=False)
#   B_scale          [E*N_OUT, K/32]           e8m0   (a16w4 scale preshuffle)
#   sorted_token_ids [max_sorted]               i32    (token=id&0xFFFFFF)
#   sorted_weights   [max_sorted]               f32
#   out_bf16         [M, N_OUT]                  bf16   (atomic acc*weight add; pre-zeroed)
#   grid=(cumsum/BM)*(N_OUT/256), block=256 (4 waves), K_TILES=2.
#
# a16w4 B-load (gemm2_a4w4.cuh:316/230): base = (e*N_OUT + n_block*256 +
#   wave_n*64 + j*16) * (K/2);  v_voff = (lane/16)*256 + (lane%16)*16 + K_C*2048;
#   two b128 loads at imm {0, 1024}.
# Scale (kBS_*) == FlyDSL make_preshuffle_scale_layout (reused).

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import range_constexpr, const_expr
from flydsl.expr import arith, gpu, buffer_ops, vector, rocdl
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf, memref
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .mfma_preshuffle_pipeline import (
    make_preshuffle_scale_layout,
    swizzle_xor16,
    lds_store_16b_xor16,
    buffer_copy_gmem16_dwordx4,
    tile_chunk_coord_i32,
)
from .layout_utils import crd2idx, idx2crd, get as layout_get


def make_preshuffle_scale_bases(*, n_out, k, experts):
    """Per-expert / per-N-block scale stride constants (== HIP kBS_* / kAS_*,
    == FlyDSL make_preshuffle_scale_layout). All in dword units.
      kBS_c_n1 = N_OUT/16/2 ;  kBS_c_k1 = (K/32)/4/2
      stride_k0_dw = 64 ;  stride_n0_dw = c_k1*64
      per_expert_dw = c_n1 * stride_n0_dw
      as_per_chunk_dw = ((K/32)/4/2) * 64
    """
    c_n1 = n_out // 16 // 2
    c_k1 = (k // 32) // 4 // 2
    stride_k0_dw = 64
    stride_n0_dw = c_k1 * 64
    per_expert_dw = c_n1 * stride_n0_dw
    as_per_chunk_dw = ((k // 32) // 4 // 2) * 64
    return dict(
        c_n1=c_n1,
        c_k1=c_k1,
        stride_k0_dw=stride_k0_dw,
        stride_n0_dw=stride_n0_dw,
        per_expert_dw=per_expert_dw,
        as_per_chunk_dw=as_per_chunk_dw,
    )


def _ptr_buffer_resource(arg_ptr, num_records_bytes):
    """Build an AMD buffer resource (rsrc) from a raw pointer + byte size.

    Matches mixed_moe_gemm_2stage.py:_ptr_buffer_resource.
    """
    addr = fx.ptrtoint(arg_ptr)
    addr_i64 = arith.index_cast(T.i64, addr)
    return buffer_ops.create_buffer_resource_from_addr(
        addr_i64, num_records_bytes=num_records_bytes
    )


@functools.lru_cache(maxsize=None)
def compile_mxfp4_gemm2_a4w4(
    *,
    experts: int,
    model_dim: int,   # N_OUT (down-proj output) = 7168
    inter_dim: int,   # K (down-proj contraction) = 512
    topk: int,
    BM: int,          # 16/32 (atomic) or 128 (nonatomic bf16flat / mxfp4out)
    b_nt: int = 2,
    mxfp4out: bool = False,   # BM=128 per-row fp4 flat_out_q/scale + scatter_reduce_q
    bf16flat: bool = False,   # BM=128 per-row bf16 flat_out + scatter_reduce
):
    """Compile a fresh FlyDSL gemm2 that is a drop-in for mxfp4_moe_gemm2_a4w4
    (atomic epilogue, BM in {16,32}). Returns the @flyc.jit launcher.

    Simplest-correct first version: per (m_block, n_block) tile, load A (BM x K
    fp4, sorted-direct) into LDS, loop K (K/256 tiles), fp4 scaled-MFMA into
    accm, then cshuffle->LDS and atomic acc*weight into out[token].
    """
    assert not (mxfp4out and bf16flat), "pick one nonatomic mode"
    if mxfp4out or bf16flat:
        assert BM == 128, "nonatomic gemm2 (bf16flat/mxfp4out) is BM=128 only"
    else:
        assert BM in (16, 32), "atomic gemm2 supports BM in {16,32} here"
    # B(w2) global-load cache policy (== HIP issue_b_load_j kBQ_AUX =
    # kAtomic && kUseNT ? 2 : 0): the BM=128 nonatomic path runs a huge grid where
    # many m-blocks of the same expert reuse the same w2 tile, so cache it in L2
    # (cache_modifier=0); a non-temporal (nt=2) hint there evicts the reusable
    # weights and inflates per-request L2 latency (ATT: B-load 455-536 cyc/req nt
    # vs ~100 cyc cached). The BM=16/32 atomic _NT path keeps nt (b_nt).
    _b_cache = 0 if (mxfp4out or bf16flat) else b_nt
    assert inter_dim == 512, "gemm2 contract: K==512"
    assert model_dim % 256 == 0

    K = inter_dim
    N_OUT = model_dim
    K_HALF = K // 2
    BN = 256
    BK = 256
    K_TILES = K // BK            # 2
    num_n_blocks = N_OUT // BN   # 28
    num_waves = 4
    total_threads = num_waves * 64
    pack_M = BM // 16            # 2 (BM=32) or 1 (BM=16)
    pack_K = 2
    a_elem_vec_pack = 2          # fp4: 2 per byte
    elem_bytes = 1
    cbsz = 4                     # fp4 sub-format
    blgp = 4

    # scale layout (== HIP kBS_*); reused for A_scale and B_scale.
    _scale_pack_m = 2
    _scale_pack_n = 2

    gpu_arch = get_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_g2")

    # LDS sizing (Python ints; layouts/types built inside kernel where the MLIR
    # context is active).
    lds_stride = BK              # fp4 elems per LDS row (one K-tile, no pad)
    k_blocks16 = (BK * elem_bytes) // 16
    # All BMs stage A via direct-to-LDS (raw_ptr_buffer_load_lds, == HIP
    # issue_a_load_lds) into a 2-slot LDS double-buffer so both K-tiles can be
    # loaded up-front (HIP run_one: load all, then 2-stage drain). This removes
    # the serial single-slot reg->ds_write path that left the large-M (BM=128)
    # gemm2 WAITCNT-stall-bound (ATT: 53% of issue cycles vs HIP 8%). The DMA
    # writes each lane's 16B contiguously, so the LDS row is exactly BK/2=128
    # bytes (HIP s_Aq[BM][BK/2]) and the bank-conflict swizzle goes into the
    # per-lane global read (it cancels on the ds_read, see lds_load_packs_k64).
    _g2_a_directlds = BM in (16, 32, 128)
    _g2_a_slots = 2 if _g2_a_directlds else 1
    _a_lds_stride = (BK // 2) if _g2_a_directlds else lds_stride
    _a_slot_stride = BM * _a_lds_stride     # fp4-bytes for one A K-tile slot
    _a_lds_bytes = _g2_a_slots * _a_slot_stride
    _acc_lds_elems = BM * BN                # f32 cshuffle buffer
    _lds_a_offset = 0
    # Union the cshuffle accumulator (lds_acc) onto the A-load LDS (lds_a) at
    # offset 0 (== HIP's LDSLayout union). lds_a is GEMM-loop-only and lds_acc is
    # epilogue-only, separated by the pre-cshuffle barrier, so they never alias in
    # time. Without the union FlyDSL stacked them (BM=16 atomic: 20480 vs HIP
    # 16384), which capped occupancy at 3 blocks/CU instead of 4 (launch_bounds).
    _lds_acc_offset = 0
    allocator.ptr = max(_a_lds_bytes, _lds_acc_offset + _acc_lds_elems * 4)

    # A-load tiling (a_elem_bytes==1 fp4 convention, 16B dwordx4 loads).
    x_load_bytes = 16
    num_x_loads = (BM * BK * elem_bytes) // total_threads // x_load_bytes
    chunk_i32 = x_load_bytes // 4
    tile_k_dwords = (BK * elem_bytes) // (4 * a_elem_vec_pack)
    c_k_div4 = (K // a_elem_vec_pack) * elem_bytes // 4   # A_q row stride (dwords)

    # scale (make_preshuffle == HIP kBS_*/kAS_*) constants
    _sc = make_preshuffle_scale_bases(n_out=N_OUT, k=K, experts=experts)
    kBS_per_expert_dw = _sc["per_expert_dw"]
    kBS_stride_n0_dw = _sc["stride_n0_dw"]
    kAS_per_chunk_dw = _sc["as_per_chunk_dw"]
    kSubBlocks = 1 if BM < 32 else BM // 32   # A-scale 32-row sub-chunks (4 @ BM=128)
    _mni_n0 = BN // 16 // 2                    # 8
    _mni_wn = BN // 64 // 2                    # 2
    # inter A-scale make_preshuffle chunk granularity = 32 rows (BM>=32) or 16
    # (BM=16) -- matches what gemm1 wrote. NOT BM (BM=128 would read the wrong chunk).
    _as_chunk_div = min(BM, 32)

    _g2tag = "mxout" if mxfp4out else ("bf16flat" if bf16flat else "atomic")
    module_name = (
        f"mxfp4_g2_a4w4_E{experts}_K{K}_N{N_OUT}_TOPK{topk}_BM{BM}_{_g2tag}_v0"
    )

    @flyc.kernel
    def gemm2_kernel(
        arg_out: fx.Pointer,         # out_bf16 [M, N_OUT]  (atomic)
        arg_a: fx.Pointer,           # A_q (inter) [max_sorted, K/2] fp4
        arg_b: fx.Pointer,           # B_q (w2)    [E, N_OUT, K/2] fp4 a16w4
        arg_a_scale: fx.Pointer,     # A_scale (make_preshuffle)
        arg_b_scale: fx.Pointer,     # B_scale (make_preshuffle)
        arg_sorted_token_ids: fx.Pointer,   # (atomic)
        arg_sorted_expert_ids: fx.Pointer,
        arg_sorted_weights: fx.Pointer,      # (atomic)
        arg_num_valid_ids: fx.Pointer,   # cumsum_tensor [1]
        arg_flat_q: fx.Pointer,          # flat_out_q [max_sorted, N_OUT/2]  (mxfp4out)
        arg_flat_scale: fx.Pointer,      # flat_out_scale [max_sorted, N_OUT/32] (mxfp4out)
        i32_tokens: fx.Int32,            # M (logical tokens)
        i32_max_sorted: fx.Int32,
    ):
        # types/layouts (MLIR context active here)
        vec16_x = T.vec(16, T.i8)
        vec2_i64 = T.vec(2, T.i64)
        vec4_i64 = T.vec(4, T.i64)
        vec8_i32 = T.vec(8, T.i32)
        vec4_f32 = T.vec(4, T.f32)
        vec2_i32 = T.vec(2, T.i32)
        i32 = T.i32
        f32 = T.f32
        layout_lds = fx.make_layout((BM, lds_stride), (lds_stride, 1))

        tx = gpu.thread_id("x")
        # 1-D grid with HIP's m-block-major decode (== gemm2_a4w4.cuh:432/446
        # m_block=pid/num_n_blocks, n_block=pid%num_n_blocks). The previous 2-D
        # grid put m_block on the fast (x) dim, so each XCD's round-robin block
        # stream hit ALL experts for a given n -> w2 thrashed L2 (PMC: 28% hit).
        # m-major decode keeps each XCD on a few experts -> w2 stays L2-hot
        # (== HIP 52% hit). Pair (m,n) coverage is identical; only the schedule
        # order (hence L2 reuse) changes, so it is correctness-neutral.
        _pid = gpu.block_id("x")
        _cnnb = arith.constant(num_n_blocks, index=True)
        m_block = _pid / _cnnb
        n_block = _pid % _cnnb

        # wave / lane decomposition (4 waves x 64 lanes).
        coord_wl = idx2crd(tx, fx.make_layout((num_waves, 64), (64, 1)))
        wave = layout_get(coord_wl, 0)
        lane = layout_get(coord_wl, 1)
        coord_l16 = idx2crd(lane, fx.make_layout((4, 16), (16, 1)))
        lane_div_16 = layout_get(coord_l16, 0)
        lane_mod_16 = layout_get(coord_l16, 1)
        wave_n = wave

        c_tokens = arith.index_cast(ir.IndexType.get(), i32_tokens.ir_value())
        c_max_sorted = arith.index_cast(
            ir.IndexType.get(), i32_max_sorted.ir_value()
        )

        # --- buffer resources (byte sizes) ---
        out_nbytes = arith.index_cast(
            T.i32, c_tokens * arith.constant(N_OUT * 2, index=True)
        )
        out_rsrc = _ptr_buffer_resource(arg_out, out_nbytes)
        a_nbytes = arith.index_cast(
            T.i32, c_max_sorted * arith.constant(K_HALF, index=True)
        )
        a_rsrc = _ptr_buffer_resource(arg_a, a_nbytes)
        b_rsrc = _ptr_buffer_resource(
            arg_b, arith.constant(experts * N_OUT * K_HALF, type=T.i32)
        )
        eid_rsrc = _ptr_buffer_resource(
            arg_sorted_expert_ids,
            arith.index_cast(
                T.i32, (c_max_sorted / arith.constant(BM, index=True))
                * arith.constant(4, index=True),
            ),
        )
        numids_rsrc = _ptr_buffer_resource(
            arg_num_valid_ids, arith.constant(4, type=T.i32)
        )
        sti_rsrc = _ptr_buffer_resource(
            arg_sorted_token_ids,
            arith.index_cast(T.i32, c_max_sorted * arith.constant(4, index=True)),
        )
        sw_rsrc = _ptr_buffer_resource(
            arg_sorted_weights,
            arith.index_cast(T.i32, c_max_sorted * arith.constant(4, index=True)),
        )

        num_valid = buffer_ops.buffer_load(
            numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=T.i32
        )
        m_row = m_block * arith.constant(BM, index=True)
        m_row_i32 = arith.index_cast(T.i32, m_row)
        blk_valid = arith.cmpi(CmpIPredicate.ult, m_row_i32, num_valid)
        expert_i32 = buffer_ops.buffer_load(
            eid_rsrc, m_block, vec_width=1, dtype=T.i32
        )

        # LDS: A tile (fp4 bytes) + cshuffle acc (f32).
        lds_base = allocator.get_base()
        lds_a = SmemPtr(lds_base, _lds_a_offset, T.i8, shape=(_a_lds_bytes,)).get()
        lds_acc = SmemPtr(
            lds_base, _lds_acc_offset, T.f32, shape=(_acc_lds_elems,)
        ).get()

        # ---- A-load (build/test increment 2): A_q is sorted inter, read
        # directly by sorted_pos = m_row + row_local (NO token gather). One
        # K-tile (BK fp4) at a time into lds_a with xor16 swizzle. ----
        tx_i32_base = tx * arith.constant(chunk_i32, index=True)
        layout_x_tile = fx.make_layout((BM, tile_k_dwords), (tile_k_dwords, 1))
        x_row_local = []
        x_col_local = []
        x_row_base_div4 = []
        for i in range_constexpr(num_x_loads):
            _rl, _cl = tile_chunk_coord_i32(
                arith,
                tx_i32_base=tx_i32_base,
                i=i,
                total_threads=total_threads,
                layout_tile_div4=layout_x_tile,
                chunk_i32=chunk_i32,
            )
            x_row_local.append(_rl)
            x_col_local.append(_cl)
            _row = m_row + _rl
            x_row_base_div4.append(_row * arith.constant(c_k_div4, index=True))

        def load_a_tile(base_k):
            base_k_div4 = (
                (base_k / arith.constant(a_elem_vec_pack, index=True))
                * arith.constant(elem_bytes, index=True)
            ) / arith.index(4)
            parts = []
            for i in range_constexpr(num_x_loads):
                idx = x_row_base_div4[i] + base_k_div4 + x_col_local[i]
                v = buffer_copy_gmem16_dwordx4(
                    buffer_ops,
                    vector,
                    elem_type=T.i8,
                    idx_i32=idx,
                    rsrc=a_rsrc,
                    vec_elems=16,
                )
                parts.append(vector.bitcast(T.vec(4, i32), v))
            return parts

        def store_a_tile(parts, slot=0):
            _slot_base = arith.constant(slot * _a_slot_stride, index=True)
            for i in range_constexpr(num_x_loads):
                lds_store_16b_xor16(
                    arith,
                    vector,
                    lds_memref=lds_a,
                    vec16_ty=vec16_x,
                    layout_lds=layout_lds,
                    row_local=x_row_local[i],
                    col_local_i32=x_col_local[i],
                    tx_c4=arith.index(4),
                    k_blocks16=k_blocks16,
                    lds_base=_slot_base,
                    vec_part_i32x4=parts[i],
                    elem_bytes=elem_bytes,
                )

        _c8idx = arith.constant(8, index=True)
        _c16idx = arith.constant(16, index=True)
        _c2idx = arith.constant(2, index=True)
        _ptr3 = ir.Type.parse("!llvm.ptr<3>")

        # rows per wave for direct-LDS A staging: BM=16 uses only waves 0,1 (16
        # rows = 2 waves x 8); BM>=32 uses all 4 waves (BM/4 rows each, split into
        # kSubBlocks chunks of 8).
        _a_rows_per_wave = (BM // 2) if BM == 16 else (BM // 4)

        def _emit_a_directlds_chunk(slot, kt, sub):
            # one 8-row DMA chunk: lane L DMAs 16B from global to m0 + L*16
            # (contiguous), m0 = LDS row-group base. The bank-conflict swizzle
            # mask = lds_swizzle_mask<BK/2=128>(row) = (row & 14)<<3 goes into the
            # per-lane global read; lds_load_packs_k64 applies the same mask on
            # read so it cancels (data lands un-swizzled into the MFMA operand).
            row_off = lane / _c8idx                  # 0..7
            lib = lane % _c8idx                       # 0..7
            col_byte = lib * _c16idx                  # 0,16,...,112
            lds_row_base = (
                wave * arith.constant(_a_rows_per_wave, index=True)
                + arith.constant(sub * 8, index=True)
            )
            actual_row = lds_row_base + row_off       # 0..BM-1
            # (actual_row & 14)<<3 == ((actual_row%16)>>1)<<4; % keeps it < 128
            # for BM>=32 (HIP uses & 14, which only depends on bits 1-3).
            mask = ((actual_row % _c16idx) / _c2idx) * _c16idx
            car_row = m_row + actual_row
            voff = arith.index_cast(
                T.i32,
                car_row * arith.constant(K // 2, index=True)
                + (col_byte ^ mask))
            m0_idx = (
                memref.extract_aligned_pointer_as_index(lds_a)
                + arith.constant(slot * _a_slot_stride, index=True)
                + lds_row_base * arith.constant(_a_lds_stride, index=True))
            m0_i64 = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, m0_idx))
            lds_ptr = llvm.inttoptr(_ptr3, m0_i64)
            rocdl.raw_ptr_buffer_load_lds(
                a_rsrc, lds_ptr,
                arith.constant(16, type=T.i32),
                voff,
                arith.constant(kt * (BK // 2), type=T.i32),  # soffset = kt*128
                arith.constant(0, type=T.i32),
                arith.constant(0, type=T.i32),
            )

        def load_a_directlds(slot, kt):
            # HIP issue_a_load_lds: each active wave DMAs its rows straight to LDS,
            # no reg round-trip / ds_write. BM=16 -> waves 0,1 only; BM>=32 -> all
            # 4 waves, kSubBlocks (=BM/32) chunks of 8 rows.
            if const_expr(BM == 16):
                _w_lt2 = arith.cmpi(
                    CmpIPredicate.ult, arith.index_cast(T.i32, wave),
                    arith.constant(2, type=T.i32))
                _if = scf.IfOp(_w_lt2)
                with ir.InsertionPoint(_if.then_block):
                    _emit_a_directlds_chunk(slot, kt, 0)
                    scf.YieldOp([])
            else:
                for sub in range_constexpr(kSubBlocks):
                    _emit_a_directlds_chunk(slot, kt, sub)

        # (A/B/scale loads + MFMA are issued by the unified K-loop below, after
        #  all tile helpers are defined.)

        # ---- a16w4 B-load (build/test increment 3): exact HIP issue_b_load_j.
        #   base = (e*N_OUT + n_block*256 + wave_n*64 + j*16) * K_HALF   [bytes]
        #   v_voff = (lane/16)*256 + (lane%16)*16 + k_c*2048             [bytes]
        #   two b128 (16B) loads per j at +0 and +1024 bytes (= +0/+256 dwords)
        # -> b_tile[j] = [(b0,b1)_mfmaK0, (b0,b1)_mfmaK1]; each (b0,b1) is the
        #    lane's 32 fp4 for one MFMA-K=128.
        e_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
        _c256 = arith.constant(256, index=True)
        _c64 = arith.constant(64, index=True)
        _c16 = arith.constant(16, index=True)
        _c2048 = arith.constant(2048, index=True)
        _cKH = arith.constant(K_HALF, index=True)
        _cN = arith.constant(N_OUT, index=True)

        def load_b_tile(k_c):
            b_tile = []
            for j in range_constexpr(4):
                n_off = (
                    n_block * _c256
                    + wave_n * _c64
                    + arith.constant(j * 16, index=True)
                )
                base_bytes = (e_idx * _cN + n_off) * _cKH
                v_voff = (
                    lane_div_16 * _c256 + lane_mod_16 * _c16 + k_c * _c2048
                )
                base_dw = (base_bytes + v_voff) / arith.index(4)
                b_j = []
                for sub in range_constexpr(2):
                    off = base_dw + arith.constant(sub * 256, index=True)
                    v16 = buffer_ops.buffer_load(
                        b_rsrc, off, vec_width=4, dtype=i32, cache_modifier=_b_cache
                    )
                    i64x2 = vector.bitcast(vec2_i64, v16)
                    b0 = vector.extract(
                        i64x2, static_position=[0], dynamic_position=[]
                    )
                    b1 = vector.extract(
                        i64x2, static_position=[1], dynamic_position=[]
                    )
                    b_j.append((b0, b1))
                b_tile.append(b_j)
            return b_tile

        # ---- scale load (build/test increment 4): make_preshuffle A/B scale.
        #   a_scale dword off = (m_row/_as_chunk_div + sub)*kAS_per_chunk_dw
        #                     + (lane/16*16 + lane%16) + ku*64
        #   b_scale dword off = e*kBS_per_expert_dw
        #                     + (mni_base+mw)*kBS_stride_n0_dw
        #                     + (lane/16*16 + lane%16) + ku*64
        a_scale_nbytes = arith.index_cast(
            T.i32,
            (c_max_sorted / arith.constant(_as_chunk_div, index=True)
             + arith.constant(2, index=True))
            * arith.constant(kAS_per_chunk_dw * 4, index=True),
        )
        a_scale_rsrc = _ptr_buffer_resource(arg_a_scale, a_scale_nbytes)
        b_scale_rsrc = _ptr_buffer_resource(
            arg_b_scale,
            arith.constant(experts * kBS_per_expert_dw * 4, type=T.i32),
        )
        _scale_lane_dw = lane_div_16 * _c16 + lane_mod_16   # (lane/16*16+lane%16)
        _chunk_base = m_row / arith.constant(_as_chunk_div, index=True)
        _mni_base = (
            n_block * arith.constant(_mni_n0, index=True)
            + wave_n * arith.constant(_mni_wn, index=True)
        )

        def load_scales(ku):
            # returns ([a_scale_dword per 32-row sub], [b_scale_dword_mw0, mw1])
            ku_off = ku * arith.constant(64, index=True)
            a_scs = []
            for sub in range_constexpr(kSubBlocks):
                a_base = (
                    (_chunk_base + arith.constant(sub, index=True))
                    * arith.constant(kAS_per_chunk_dw, index=True)
                    + _scale_lane_dw + ku_off
                )
                a_scs.append(buffer_ops.buffer_load(
                    a_scale_rsrc, a_base, vec_width=1, dtype=T.i32, cache_modifier=0
                ))
            b_scs = []
            for mw in range_constexpr(2):
                b_base = (
                    e_idx * arith.constant(kBS_per_expert_dw, index=True)
                    + (_mni_base + arith.constant(mw, index=True))
                    * arith.constant(kBS_stride_n0_dw, index=True)
                    + _scale_lane_dw + ku_off
                )
                b_scs.append(
                    buffer_ops.buffer_load(
                        b_scale_rsrc, b_base, vec_width=1, dtype=T.i32,
                        cache_modifier=0,
                    )
                )
            return a_scs, b_scs

        # ---- fp4 scaled-MFMA (build/test increment 5) ----
        # MFMA 16x16x128 f8f6f4: a128/b128 = pack_i64x4_to_i32x8(b0,b1,0,0);
        # op_sel selects the e8m0 byte in the make_preshuffle scale dword.
        # (op_sel numeric correctness to be validated vs HIP cos.)
        row_a_lds = lane_mod_16
        col_offset_base = lane_div_16 * _c16
        m_repeat = BM // 16                  # 2 (BM=32)
        m_repeat_packed = m_repeat // pack_M  # 1
        num_acc_n = BN // num_waves // 16     # 4
        num_acc_n_packed = num_acc_n // 2     # 2 (pack_N=2)
        pack_N = 2

        def _pack_i64x4(x0, x1):
            c0 = arith.constant(0, type=T.i64)
            v4 = vector.from_elements(vec4_i64, [x0, x1, c0, c0])
            return vector.bitcast(vec8_i32, v4)

        def lds_load_packs_k64(curr_row, col_base, slot=0):
            if const_expr(_g2_a_directlds):
                # HIP issue_a_ds_read: 128-byte rows, mask =
                # lds_swizzle_mask<BK/2=128>(lane_row) = (lane_row & 14)<<3, which
                # depends ONLY on lane%16 (NOT curr_row's mi_idx*16 part), so it is
                # the same mask the direct-LDS write applied at this lane's row ->
                # the swizzle cancels and the lane's K-data lands at col_base.
                mask = (lane_mod_16 / arith.index(2)) * _c16
                col_swz = col_base ^ mask
                byte_idx = (
                    curr_row * arith.constant(_a_lds_stride, index=True)
                    + col_swz
                    + arith.constant(slot * _a_slot_stride, index=True)
                )
                loaded = vector.load_op(vec16_x, lds_a, [byte_idx])
            else:
                col_swz = swizzle_xor16(curr_row, col_base, k_blocks16)
                idx_a = crd2idx([curr_row, col_swz], layout_lds)
                if const_expr(slot != 0):
                    idx_a = idx_a + arith.constant(slot * _a_slot_stride, index=True)
                loaded = vector.load_op(vec16_x, lds_a, [idx_a])
            a_i64x2 = vector.bitcast(vec2_i64, loaded)
            a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
            a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
            return a0, a1

        acc_init = arith.constant_vector(0.0, vec4_f32)
        accm = [acc_init] * (m_repeat * num_acc_n)   # mi_idx*num_acc_n + ni_idx

        def mfma_ktile(b_tile, a_scs, b_scs, a_slot=0):
            for k_idx in range_constexpr(2):       # 2 MFMA-K (128) per K-tile
                ikxdl = k_idx
                col_base = col_offset_base + arith.constant(
                    (k_idx * 128) // a_elem_vec_pack, index=True
                )
                for imxdl in range_constexpr(pack_M):
                    mi_idx = imxdl                  # m_repeat_packed=1
                    curr_row = row_a_lds + arith.constant(mi_idx * 16, index=True)
                    a0, a1 = lds_load_packs_k64(curr_row, col_base, slot=a_slot)
                    a128 = _pack_i64x4(a0, a1)
                    # A-scale: one make_preshuffle dword per 32-row sub-chunk
                    # (a_scs[mi//2]); byte = ikxdl*2 + (mi%2). _scale_pack_m=2 is
                    # the fixed m-half stride in the dword.
                    a_scale_val = a_scs[mi_idx // 2]
                    op_sel_a = ikxdl * _scale_pack_m + (mi_idx % 2)
                    for ni in range_constexpr(num_acc_n_packed):
                        b_scale_val = b_scs[ni]
                        for inxdl in range_constexpr(pack_N):
                            ni_idx = ni * pack_N + inxdl
                            b0, b1 = b_tile[ni_idx][k_idx]
                            b128 = _pack_i64x4(b0, b1)
                            op_sel_b = ikxdl * pack_N + inxdl
                            acc_idx = mi_idx * num_acc_n + ni_idx
                            accm[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                vec4_f32,
                                [
                                    a128, b128, accm[acc_idx],
                                    cbsz, blgp,
                                    op_sel_a, a_scale_val,
                                    op_sel_b, b_scale_val,
                                ],
                            )

        # ---- K-loop (K=512 -> 2 K-tiles) ----
        if const_expr(_g2_a_directlds):
            # HIP run_one structure: stage BOTH K-tiles' A into the 2-slot LDS
            # double-buffer up-front via direct-to-LDS DMA (== HIP issue_a_load_lds,
            # no buffer_load->reg->ds_write round-trip), one barrier, then prefetch
            # both B/scale and drain the 2 MFMA stages. This replaced the serial
            # single-slot path that left BM=128 WAITCNT-stall-bound (ATT: 53% of
            # issue cycles in s_waitcnt vmcnt/lgkmcnt vs HIP 8%); now B VMEM
            # overlaps the A DMA and there is no mid-loop store/barrier.
            load_a_directlds(0, 0)
            load_a_directlds(1, 1)
            # sched fence (== HIP gemm2 issue path: sched_barrier(0) right after
            # the A direct-LDS DMA): pin the A loads ahead of the B/scale loads so
            # the scheduler cannot sink B (or MFMA) above the A DMA. mask 0 = no
            # instruction may cross. Keeps the load cluster -> MFMA cluster split.
            rocdl.sched_barrier(0)
            # issue B (down weights, the bulk of VMEM) + scales BEFORE the barrier
            # so their latency overlaps the A direct-LDS DMA (== HIP: all loads
            # issued, then the drain barrier only waits for the A DMA while B stays
            # in flight, gated per-MFMA by vmcnt).
            _b0 = load_b_tile(arith.index(0))
            _asc0, _bsc0 = load_scales(arith.index(0))
            _b1 = load_b_tile(arith.index(1))
            _asc1, _bsc1 = load_scales(arith.index(1))
            # One workgroup barrier: ds_read of every K-stage reads all BM rows
            # (cross-wave), and the backend flushes both A DMA slots (vmcnt) before
            # the barrier, so a single barrier makes both slots visible workgroup-
            # wide for both MFMA stages (same as the proven BM=16 path).
            rocdl.s_barrier()
            mfma_ktile(_b0, _asc0, _bsc0, a_slot=0)
            mfma_ktile(_b1, _asc1, _bsc1, a_slot=1)
        else:
            store_a_tile(load_a_tile(arith.index(0)))
            rocdl.s_barrier()
            _b_tile0 = load_b_tile(arith.index(0))
            _a_sc0, _b_scs0 = load_scales(arith.index(0))
            mfma_ktile(_b_tile0, _a_sc0, _b_scs0)
            rocdl.s_barrier()
            store_a_tile(load_a_tile(arith.constant(BK, index=True)))
            rocdl.s_barrier()
            _b_tile1 = load_b_tile(arith.index(1))
            _a_sc1, _b_scs1 = load_scales(arith.index(1))
            mfma_ktile(_b_tile1, _a_sc1, _b_scs1)

        # ---- BM=128 nonatomic bf16 flat epilogue (== apply_bf16_flat_epilog_bm128)
        # write accm straight to bf16 flat_out[max_sorted, N_OUT] per sorted row
        # (no LDS, no quant, no weight); mxfp4_moe_scatter_reduce does the topk
        # reduce. Done before the cshuffle so it isn't paid for this mode. ----
        if const_expr(bf16flat):
            _flat_base = arith.index_cast(ir.IndexType.get(), fx.ptrtoint(arg_out))
            _cN_out = arith.constant(N_OUT, index=True)
            for i in range_constexpr(m_repeat):
                for j in range_constexpr(num_acc_n):
                    gn = (n_block * _c256 + wave_n * _c64
                          + arith.constant(j * 16, index=True) + lane_mod_16)
                    acc_v = accm[i * num_acc_n + j]
                    for v in range_constexpr(4):
                        row = (m_row + arith.constant(i * 16, index=True)
                               + lane_div_16 * arith.index(4)
                               + arith.constant(v, index=True))
                        val = vector.extract(
                            acc_v, static_position=[v], dynamic_position=[])
                        bf = arith.trunc_f(T.bf16, val)
                        off = (row * _cN_out + gn) * arith.index(2)
                        _ptr = llvm.inttoptr(
                            ir.Type.parse("!llvm.ptr<1>"),
                            arith.index_cast(T.i64, _flat_base + off))
                        llvm.StoreOp(
                            bf._value if hasattr(bf, "_value") else bf,
                            _ptr, alignment=2)
            # match atomic/mxfp4out epilogues: a workgroup barrier before exit so
            # the GEMM's LDS (lds_a) reads are all retired before the workgroup
            # frees its 128KB LDS for the next (FlyDSL) kernel on the CU.
            rocdl.s_barrier()
            return

        # ---- cshuffle + atomic epilogue (build/test increment 6) ----
        # accm[mi*4+J][v]  ->  lds_acc[(mi*16 + lane/16*4 + v)*BN + col],
        #   col = wave_n*64 + J*16 + lane%16
        rocdl.s_barrier()
        c_BN = arith.constant(BN, index=True)
        for i in range_constexpr(m_repeat):
            for J in range_constexpr(num_acc_n):
                col = (
                    wave_n * _c64
                    + arith.constant(J * 16, index=True)
                    + lane_mod_16
                )
                row_base = arith.constant(i * 16, index=True) + lane_div_16 * arith.constant(
                    4, index=True
                )
                acc_v = accm[i * num_acc_n + J]
                for v in range_constexpr(4):
                    lds_idx = (
                        (row_base + arith.constant(v, index=True)) * c_BN + col
                    )
                    val = vector.extract(
                        acc_v, static_position=[v], dynamic_position=[]
                    )
                    v1 = vector.from_elements(T.vec(1, f32), [val])
                    vector.store(v1, lds_acc, [lds_idx], alignment=4)
        rocdl.s_barrier()

        # ---- nonatomic mxfp4out epilogue (== apply_mxfp4_flat_epilog_bm128) ----
        # per (sorted row, 32-col group): per-32 fp4 requant -> flat_out_q
        # [max_sorted, N_OUT/2] (row-major) + e8m0 -> flat_out_scale
        # [max_sorted, N_OUT/32]. No reduce here; scatter_reduce_q does topk.
        if const_expr(mxfp4out):
            _flat_q_base = arith.index_cast(
                ir.IndexType.get(), fx.ptrtoint(arg_flat_q))
            _flat_sc_base = arith.index_cast(
                ir.IndexType.get(), fx.ptrtoint(arg_flat_scale))
            _c32idx = arith.constant(32, index=True)
            _mlane = tx / _c16
            _nlane = tx % _c16
            _wgrp = _nlane / arith.index(4)
            _kk = _nlane % arith.index(4)
            _c7fff = arith.constant(0x7FFFFFFF, type=i32)
            _c64i = arith.constant(64, type=i32)
            _hi16 = arith.constant(-0x10000, type=i32)   # 0xFFFF0000
            NBLK = BN // 32                              # 8
            N_HALF = N_OUT // 2
            N_SC = N_OUT // 32
            for mr in range_constexpr(m_repeat):         # BM/16 (=8)
                row_local = arith.constant(mr * 16, index=True) + _mlane
                out_row = m_row + row_local
                for half in range_constexpr(NBLK // 4):  # 2
                    group = _wgrp + arith.constant(half * 4, index=True)
                    col0 = group * _c32idx + _kk * arith.index(8)
                    r = []
                    for e in range_constexpr(8):
                        rv = vector.load_op(
                            T.vec(1, f32), lds_acc,
                            [row_local * c_BN + col0 + arith.constant(e, index=True)],
                        )
                        r.append(vector.extract(
                            rv, static_position=[0], dynamic_position=[]))
                    local_max = (r[0].bitcast(i32) & _c7fff).bitcast(f32)
                    for e in range_constexpr(1, 8):
                        local_max = arith.maximumf(
                            local_max, (r[e].bitcast(i32) & _c7fff).bitcast(f32))
                    # quad amax over the 4 kk lanes (== HIP dpp 0xB1 + 0x4E).
                    local_max = arith.maximumf(
                        local_max, local_max.shuffle_xor(
                            arith.constant(1, type=i32), _c64i))
                    local_max = arith.maximumf(
                        local_max, local_max.shuffle_xor(
                            arith.constant(2, type=i32), _c64i))
                    # encode_e8m0(bf16(amax)): bexp=((amax&0xFFFF0000)+0x200000)>>23,
                    # e8m0=clamp(bexp-2,0,254); quant_scale=2^(e8m0-127).
                    amax_bf = local_max.bitcast(i32) & _hi16
                    bexp = (
                        (amax_bf + arith.constant(0x200000, type=i32))
                        >> arith.constant(23, type=i32)
                    ) & arith.constant(0xFF, type=i32)
                    e8m0 = arith.minsi(
                        arith.maxsi(bexp - arith.constant(2, type=i32),
                                    arith.constant(0, type=i32)),
                        arith.constant(254, type=i32))
                    qs = (e8m0 << arith.constant(23, type=i32)).bitcast(f32)
                    packed = arith.constant(0, type=i32)
                    for k in range_constexpr(4):
                        packed = llvm.call_intrinsic(
                            i32, "llvm.amdgcn.cvt.scalef32.pk.fp4.f32",
                            [packed, r[2 * k], r[2 * k + 1], qs,
                             arith.constant(k, type=i32)], [], [])
                    global_col = n_block * _c256 + col0
                    q_off = (out_row * arith.constant(N_HALF, index=True)
                             + global_col / arith.index(2))
                    _q_ptr = llvm.inttoptr(
                        ir.Type.parse("!llvm.ptr<1>"),
                        arith.index_cast(T.i64, _flat_q_base + q_off))
                    llvm.StoreOp(
                        packed._value if hasattr(packed, "_value") else packed,
                        _q_ptr, alignment=4)
                    _kk_i32 = arith.index_cast(i32, _kk)
                    _is0 = arith.cmpi(
                        CmpIPredicate.eq, _kk_i32, arith.constant(0, type=i32))
                    _ifsc = scf.IfOp(_is0)
                    with ir.InsertionPoint(_ifsc.then_block):
                        blk = n_block * arith.constant(NBLK, index=True) + group
                        sc_off = out_row * arith.constant(N_SC, index=True) + blk
                        _sc_ptr = llvm.inttoptr(
                            ir.Type.parse("!llvm.ptr<1>"),
                            arith.index_cast(T.i64, _flat_sc_base + sc_off))
                        _e8i8 = arith.TruncIOp(T.i8, e8m0)
                        llvm.StoreOp(
                            _e8i8._value if hasattr(_e8i8, "_value") else _e8i8,
                            _sc_ptr, alignment=1)
                        scf.YieldOp([])
            return

        # read by (m_lane=tid/32, n_lane=tid%32), atomic acc*weight -> out[token]
        _c32 = arith.constant(32, index=True)
        m_lane = tx / _c32
        n_lane = tx % _c32
        col_start = n_lane * arith.constant(2, index=True)
        out_base_idx = arith.index_cast(ir.IndexType.get(), fx.ptrtoint(arg_out))
        c_tokens_i32 = arith.index_cast(T.i32, c_tokens)
        _mask24 = arith.constant(0xFFFFFF, type=T.i32)
        M_REPS = BM // 8
        bf16x2 = T.vec(2, T.bf16)
        for mr in range_constexpr(M_REPS):
            row_in_block = arith.constant(mr * 8, index=True) + m_lane
            sorted_pos = m_row + row_in_block
            packed = buffer_ops.buffer_load(
                sti_rsrc, sorted_pos, vec_width=1, dtype=T.i32
            )
            token = arith.andi(packed, _mask24)
            # guard padding rows: sorted_pos < cumsum (HIP per-row bound) AND
            # token < M.  Launching max_sorted/BM blocks means trailing rows are
            # padding (uninitialized A_q); skipping them avoids garbage atomics.
            sorted_pos_i32 = arith.index_cast(T.i32, sorted_pos)
            pos_ok = arith.cmpi(CmpIPredicate.ult, sorted_pos_i32, num_valid)
            tok_lt_m = arith.cmpi(CmpIPredicate.ult, token, c_tokens_i32)
            tok_ok = arith.andi(pos_ok, tok_lt_m)
            _if_tok = scf.IfOp(tok_ok)
            with ir.InsertionPoint(_if_tok.then_block):
                token_idx = arith.index_cast(ir.IndexType.get(), token)
                weight = buffer_ops.buffer_load(
                    sw_rsrc, sorted_pos, vec_width=1, dtype=f32
                )
                row_out_base = (
                    token_idx * arith.constant(N_OUT, index=True)
                    + n_block * _c256
                    + col_start
                )
                _wv2 = vector.from_elements(T.vec(2, f32), [weight, weight])
                for s in range_constexpr(4):
                    lds_row_off = row_in_block * c_BN + col_start + arith.constant(
                        s * 64, index=True
                    )
                    # load the lane's 2 consecutive acc cols as a vec2 and scale by
                    # weight with a single packed v_pk_mul_f32 (== HIP), instead of
                    # 2 scalar v_mul_f32. Keeps the f32 pair packed end-to-end.
                    e = vector.load_op(T.vec(2, f32), lds_acc, [lds_row_off])
                    fw = arith.mulf(e, _wv2)
                    f0w = vector.extract(fw, static_position=[0], dynamic_position=[])
                    f1w = vector.extract(fw, static_position=[1], dynamic_position=[])
                    b0 = arith.trunc_f(T.bf16, f0w)
                    b1 = arith.trunc_f(T.bf16, f1w)
                    pk = vector.from_elements(bf16x2, [b0, b1])
                    out_addr_idx = (
                        out_base_idx
                        + (row_out_base + arith.constant(s * 64, index=True))
                        * arith.constant(2, index=True)
                    )
                    out_addr_i64 = arith.index_cast(T.i64, out_addr_idx)
                    out_ptr = llvm.inttoptr(
                        ir.Type.parse("!llvm.ptr<1>"), out_addr_i64
                    )
                    pk_raw = pk._value if hasattr(pk, "_value") else pk
                    llvm.AtomicRMWOp(
                        llvm.AtomicBinOp.fadd,
                        out_ptr,
                        pk_raw,
                        llvm.AtomicOrdering.monotonic,
                        syncscope="agent",
                        alignment=4,
                    )
                scf.YieldOp([])

    @flyc.jit
    def launch_gemm2(
        arg_out: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_a_scale: fx.Pointer,
        arg_b_scale: fx.Pointer,
        arg_sorted_token_ids: fx.Pointer,
        arg_sorted_expert_ids: fx.Pointer,
        arg_sorted_weights: fx.Pointer,
        arg_num_valid_ids: fx.Pointer,
        arg_flat_q: fx.Pointer,
        arg_flat_scale: fx.Pointer,
        i32_tokens: fx.Int32,
        i32_max_sorted: fx.Int32,
        i32_total_m_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        # 1-D grid (total_m_blocks * num_n_blocks); kernel decodes
        # m_block=pid/num_n_blocks, n_block=pid%num_n_blocks (== HIP wu decode) so
        # the block schedule is expert-local for L2 reuse of w2.
        gx = arith.index_cast(ir.IndexType.get(), i32_total_m_blocks.ir_value())
        gx = gx * arith.constant(num_n_blocks, index=True)
        gemm2_kernel(
            arg_out, arg_a, arg_b, arg_a_scale, arg_b_scale,
            arg_sorted_token_ids, arg_sorted_expert_ids, arg_sorted_weights,
            arg_num_valid_ids, arg_flat_q, arg_flat_scale,
            i32_tokens, i32_max_sorted,
        ).launch(
            grid=(gx, 1, 1),
            block=(total_threads, 1, 1),
            # LDS in allocator static global; smem= is additive (would double).
            smem=0,
            stream=stream,
        )

    return launch_gemm2


def _f32_to_e2m1(x_f32):
    """Software f32 -> fp4(e2m1) 4-bit int (matches HIP cvt + fp4_utils:
    saturate / denormal / round-to-nearest-even)."""
    T_i32 = T.i32
    qx = x_f32.bitcast(T_i32)
    s = qx & arith.constant(0x80000000, type=T_i32)
    qx_abs = qx & arith.constant(0x7FFFFFFF, type=T_i32)
    c_3F8 = arith.constant(0x3F800000, type=T_i32)
    c_40C = arith.constant(0x40C00000, type=T_i32)
    c_4A8 = arith.constant(0x4A800000, type=T_i32)
    dmask = arith.cmpi(CmpIPredicate.ult, qx_abs, c_3F8)
    nmask = arith.andi(
        arith.cmpi(CmpIPredicate.ult, qx_abs, c_40C),
        arith.cmpi(CmpIPredicate.uge, qx_abs, c_3F8),
    )
    dn = qx_abs.bitcast(T.f32) + c_4A8.bitcast(T.f32)
    dnx = dn.bitcast(T_i32) - c_4A8
    modd = (qx_abs >> arith.constant(22, type=T_i32)) & arith.constant(1, type=T_i32)
    nx = (qx_abs + arith.constant(0xC11FFFFF, type=T_i32) + modd) >> arith.constant(
        22, type=T_i32
    )
    e2 = arith.select(nmask, nx, arith.constant(0x7, type=T_i32))
    e2 = arith.select(dmask, dnx, e2)
    return (s >> arith.constant(28, type=T_i32)) | e2


@functools.lru_cache(maxsize=None)
def compile_mxfp4_gemm1_a4w4(
    *,
    experts: int,
    model_dim: int,   # D_HIDDEN (gemm1 contraction K) = 7168
    inter_dim: int,   # D_INTER  (gemm1 output inter)  = 512
    topk: int,
    BM: int,          # 16 (inline) or 32
    use_nt: bool = True,
    inline_quant: bool = False,
    debug_bf16: bool = False,   # write raw silu*mul (bf16) instead of fp4 requant
    debug_scaled: bool = False, # debug: write results*inv_scale (pre-e2m1) bf16
    a_load_variant: str = "direct_hip_2slot_kmajor",
):
    """Fresh FlyDSL gemm1 drop-in for mxfp4_moe_gemm1_a4w4 (gate/up a16w4 +
    SiLU*mul + per-32 fp4 requant).  K=D_HIDDEN, N_OUT=2*D_INTER (gate+up
    interleaved), output inter = D_INTER fp4 (sorted) + make_preshuffle scale.

    Simplest-correct first: NT path (pre-quant A_q), simple K-loop, software
    f32->fp4 requant (HW cvt.scalef32.pk_fp4 isn't exposed in rocdl).
    """
    assert BM in (16, 32, 128), "gemm1 here supports BM in {16,32,128}"
    assert not (inline_quant and BM == 128), "BM=128 is NT-only (no inline quant)"
    valid_a_load_variants = {
        "safe",
        "direct_current",
        "direct_hip_2slot_kmajor",
        "direct_hip_2slot_kmajor_wait",
        "direct_hip_2slot_kmajor_cached",
    }
    assert a_load_variant in valid_a_load_variants, (
        f"unknown a_load_variant={a_load_variant!r}, expected one of "
        f"{sorted(valid_a_load_variants)}"
    )
    _direct_a_load = a_load_variant != "safe"
    _hip_a_read = a_load_variant in {
        "direct_hip_2slot_kmajor",
        "direct_hip_2slot_kmajor_wait",
        "direct_hip_2slot_kmajor_cached",
    }
    _hip_a_kmajor = _hip_a_read
    _pre_read_wait = a_load_variant == "direct_hip_2slot_kmajor_wait"
    _cache_a_rows = a_load_variant == "direct_hip_2slot_kmajor_cached"
    # A-scale / output-scale make_preshuffle chunk = 32 rows (BM>=32) or 16 (BM=16).
    kSubBlocks = 1 if BM < 32 else (BM // 32)   # 32-row sub-chunks per m-block
    _out_chunk_div = 16 if BM == 16 else 32

    K = model_dim                # 7168 (contraction)
    N_OUT = 2 * inter_dim        # 1024 (gate+up)
    N_INTER = inter_dim          # 512  (output inter)
    K_HALF = K // 2
    BN = 256
    BK = 256
    K_TILES = K // BK            # 28
    num_n_blocks = N_OUT // BN   # 4
    num_waves = 4
    total_threads = num_waves * 64
    pack_M = BM // 16
    a_elem_vec_pack = 2
    elem_bytes = 1
    cbsz = 4
    blgp = 4
    _scale_pack_m = 2
    # B (expert-weight) global-load cache policy. cache_modifier=2 -> "nt"
    # (non-temporal, streaming, bypass/evict L2); 0 -> normal cached. In MoE the
    # same B tile is re-read by every m-block of the same expert, so non-temporal
    # evicts reusable weights from L2 -> extra HBM traffic -> the per-tile A-DMA
    # vmcnt + barrier stall inflates (main loop +14.7% vs HIP). HIP picks
    # kUseNT=false (cached) once estimated m/expert >= 64 (gemm1 dispatch); the
    # thread trace confirms HIP issues 0 nt loads while FlyDSL issued 224.
    # Respect the use_nt arg (was hardcoded to 2, ignoring it) so cached is the
    # default for the gemm1 BM>=32 path; cached is >= nt at every measured M.
    b_nt = 2 if use_nt else 0

    # B-scale (gate/up a16w4) — kBS_* over N_OUT=2*D_INTER.
    _bsc = make_preshuffle_scale_bases(n_out=N_OUT, k=K, experts=experts)
    kBS_per_expert_dw = _bsc["per_expert_dw"]
    kBS_stride_n0_dw = _bsc["stride_n0_dw"]
    kBS_stride_k0_dw = _bsc["stride_k0_dw"]   # 64
    # A-scale (activation, K=D_HIDDEN) chunk = 28 K-tiles.
    kAS_c_k1 = (K // 32) // 4 // 2
    kAS_per_chunk_dw = 1 * kAS_c_k1 * 64
    _as_chunk_div = 32                          # gemm1 A-scale: chunk = m_row/32
    _mni_n0 = BN // 16 // 2
    _mni_wn = BN // 64 // 2

    # output inter requant make_preshuffle (kAS over N_INTER).
    kOS_c_k1 = (N_INTER // 32) // 4 // 2
    kOS_per_chunk_dw = 1 * kOS_c_k1 * 64

    gpu_arch = get_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_g1")

    # A-LDS packed to HIP's 128-byte/row layout (== HIP s_Aq[slot][BM][BK/2]):
    # row stride = BK/2 (the fp4x2 byte width), swizzle = (row & 7)*16
    # (k_blocks16 = (BK/2)/16 = 8, vs the prior 256-stride / (row&15)*16). store
    # and read both key off lds_stride + k_blocks16, so the change is consistent.
    lds_stride = BK // 2
    k_blocks16 = (lds_stride * elem_bytes) // 16
    # Inline-quant BM=16: 3 A-LDS slots (== HIP kAStages=3) for the 2-stage
    # pipeline (read_slot=OFFSET%3, write_slot=(OFFSET+2)%3, inflight=(OFFSET+1)%3
    # all distinct -> no WAR). 3*4096 A + 3*128dw ascale = 13824B, still union'd
    # under the 16384B cshuffle acc, so occupancy is unchanged vs the 2-slot form.
    # ALL paths now run the HIP-faithful double-buffered K-loop (BM=16 inline-quant;
    # BM=32/128 NT), which needs a 3-slot A-LDS rotation (read_slot=OFFSET%3,
    # write_slot=(OFFSET+2)%3, inflight=(OFFSET+1)%3 all distinct -> no WAR). 3 A
    # slots still fit UNDER the cshuffle acc (BM*BN*4) in the LDS union for every BM
    # (BM16 12288<16384, BM32 24576<32768, BM128 98304<131072), so occupancy is
    # unchanged. (HIP uses kAStages=2 at BM=128 with direct-to-LDS DMA; FlyDSL
    # stages via buffer_load+ds_write, so 3 distinct slots avoid the WAR for free.)
    _a_slots = 2 if (BM == 128 and _hip_a_read) else 3
    _a_slot_stride = BM * lds_stride            # bytes per A LDS slot
    # A-scale LDS: BM=16 uses HIP's packed-byte layout (256B = 64 dwords / tile,
    # 2 e8m0 packed per quant-lane dword at byte0/byte2). BM>=32 keeps the
    # dword-per-e8m0 staging written by load_a_tile_inline (BM*8 dwords).
    _ascale_slot_dwords = 64 if (inline_quant and BM == 16) else (BM * 8)
    _a_lds_bytes = _a_slots * _a_slot_stride
    _acc_lds_elems = BM * BN
    _lds_a_offset = 0
    # LDS layout matches HIP's LDSPool union: the A-load tile (lds_a) and the
    # inline-quant A-scale staging (lds_ascale) are GEMM-loop-phase only; the
    # cshuffle accumulator (lds_acc) is epilogue-only and barrier-separated, so
    # union it onto offset 0 (overlapping lds_a + lds_ascale). Without the union
    # FlyDSL used lds_a + lds_acc + lds_ascale stacked (BM=16: 20992 vs HIP 16384),
    # lowering occupancy at large M.
    _lds_ascale_offset = ((_a_lds_bytes + 15) // 16) * 16     # after lds_a (loop)
    _lds_ascale_dwords = _a_slots * _ascale_slot_dwords
    _lds_ascale_bytes = _lds_ascale_dwords * 4
    # NT A-scale LDS staging (== HIP s_Ascale[kSubBlocks*K_TILES*256]): one-time
    # direct-DMA from the sorted A_scale VMEM, then ds_read per tile (replaces the
    # per-tile A-scale VMEM loads). Identity copy LDS[sub*kAS_per_chunk_dw + j] =
    # A_scale[(m_row/32+sub)*kAS_per_chunk_dw + j]. Shares _lds_ascale_offset with
    # the inline staging (mutually exclusive) and stays under the cshuffle acc.
    _nt_ascale_bytes = (kSubBlocks * K_TILES * 256) if not inline_quant else 0
    _loop_lds_end = _lds_ascale_offset + (
        _lds_ascale_bytes if inline_quant else _nt_ascale_bytes)
    _lds_acc_offset = 0                                       # union onto loop LDS
    allocator.ptr = max(_loop_lds_end, _lds_acc_offset + _acc_lds_elems * 4)

    # A-load tiling (gather rows via m_indices; per-K-tile, 16B dwordx4 loads).
    x_load_bytes = 16
    num_x_loads = (BM * BK * elem_bytes) // total_threads // x_load_bytes
    chunk_i32 = x_load_bytes // 4
    tile_k_dwords = (BK * elem_bytes) // (4 * a_elem_vec_pack)
    c_k_div4 = (K // a_elem_vec_pack) * elem_bytes // 4   # A_q row stride (dwords)

    module_name = (
        f"mxfp4_g1_a4w4_E{experts}_K{K}_N{N_OUT}_TOPK{topk}_BM{BM}"
        f"_{'IQ' if inline_quant else 'NT'}_{'dbg' if debug_bf16 else 'q'}"
        f"_aload_{a_load_variant}_v0"
    )

    # =====================================================================
    # HIP reference: csrc/kernels/mxfp4_moe/gemm_a4w4/gemm1_a4w4.cuh
    #   template<int MAX_M, int NUM_EXPERTS, int K, int N_OUT, int BM,
    #            bool kUseNT=false, bool kInlineQuant=false, int kXcdSwizzle=0>
    #   __launch_bounds__(256, (BM==128)?1:((BM==16)?3:2)) kernel(...)
    # This FlyDSL kernel = the BM=16, kInlineQuant=true, kUseNT=false
    # instantiation (M<=16 inline-quant gate/up GEMM). The HIP body is
    # `run_one(m_block_idx, n_block_idx, e)`; the dispatcher at the bottom of
    # the HIP kernel maps blockIdx.x -> (m_block_idx, n_block_idx). FlyDSL
    # instead launches a 2-D grid (block_id x = m, y = n), so run_one's body
    # maps directly to this kernel body. Line-by-line HIP correspondence is
    # given in the comments below ("HIP:" tags).
    #
    # Signature  (FlyDSL arg            <-> HIP kernel param):
    #   arg_a               <-> const __hip_fp4x2_storage_t* A_q
    #   arg_a_scale         <-> const __amd_scale_t*         A_scale
    #   arg_b               <-> const __hip_fp4x2_storage_t* B_ps_q   (w12)
    #   arg_b_scale         <-> const __amd_scale_t*         B_ps_scale
    #   arg_sorted_expert_ids <-> const int* sorted_expert_ids
    #   arg_cumsum          <-> const int* cumsum_tensor
    #   arg_m_indices       <-> const int* m_indices
    #   arg_aq_out          <-> uint8_t* A_q_out      (inter_sorted_quant)
    #   arg_ascale_out      <-> uint8_t* A_scale_out  (inter_sorted_shuffled_scale)
    #   arg_hidden          <-> const __hip_bfloat16* hidden_states
    #   i32_tokens          <-> int n_tokens
    #   i32_max_sorted      <-> MAX_M (host-side buffer bound; HIP is constexpr)
    # =====================================================================
    @flyc.kernel
    def gemm1_kernel(
        arg_a: fx.Pointer,           # A_q [max_sorted, K/2] fp4 (pre-quant)
        arg_a_scale: fx.Pointer,     # A_scale (make_preshuffle, sorted)
        arg_b: fx.Pointer,           # B_ps_q (w12) [E, N_OUT, K/2] fp4 a16w4
        arg_b_scale: fx.Pointer,     # B_ps_scale (make_preshuffle)
        arg_sorted_expert_ids: fx.Pointer,
        arg_cumsum: fx.Pointer,
        arg_m_indices: fx.Pointer,
        arg_aq_out: fx.Pointer,      # inter_sorted_quant [max_sorted, N_INTER/2]
        arg_ascale_out: fx.Pointer,  # inter_sorted_shuffled_scale
        arg_hidden: fx.Pointer,      # hidden_states [M, K] bf16 (inline quant)
        i32_tokens: fx.Int32,
        i32_max_sorted: fx.Int32,
    ):
        i32 = T.i32
        f32 = T.f32
        vec16_x = T.vec(16, T.i8)
        vec2_i64 = T.vec(2, T.i64)
        vec4_i64 = T.vec(4, T.i64)
        vec8_i32 = T.vec(8, T.i32)
        vec4_f32 = T.vec(4, T.f32)
        layout_lds = fx.make_layout((BM, lds_stride), (lds_stride, 1))

        # HIP: const int tid = threadIdx.x; const int wave = tid/64; lane = tid%64;
        #      const int wave_n = wave; m_block_idx = pid/num_n, n_block_idx = pid%num_n.
        # GRID: HIP-style 1-D pid -> (m,n) decode (NOT a 2-D grid). With a 2-D grid
        # the hardware interleaved valid + padding blocks per n-column and clustered
        # 2 valid blocks onto some CUs -> straggler CUs (occupancy.json: CU5/CU6 ran
        # 2 concurrent valid blocks, ~117k vs ~73k) -> slower kernel at small M.
        # A 1-D pid keeps the valid blocks CONTIGUOUS (pid < valid_m*num_n) so they
        # spread 1-per-CU like HIP.
        tx = gpu.thread_id("x")
        _pid = gpu.block_id("x")
        m_block = _pid / arith.constant(num_n_blocks, index=True)
        n_block = _pid % arith.constant(num_n_blocks, index=True)
        coord_wl = idx2crd(tx, fx.make_layout((num_waves, 64), (64, 1)))
        wave = layout_get(coord_wl, 0)
        lane = layout_get(coord_wl, 1)
        coord_l16 = idx2crd(lane, fx.make_layout((4, 16), (16, 1)))
        lane_div_16 = layout_get(coord_l16, 0)   # HIP: lane/16 (lds_col group)
        lane_mod_16 = layout_get(coord_l16, 1)   # HIP: lane%16 (lds_row)
        wave_n = wave

        c_tokens = arith.index_cast(ir.IndexType.get(), i32_tokens.ir_value())
        c_max_sorted = arith.index_cast(
            ir.IndexType.get(), i32_max_sorted.ir_value()
        )

        # HIP make_buffer_rsrc(...) block (gemm1_a4w4.cuh:81-97):
        #   A_q_rsrc       = make_buffer_rsrc(A_q,  n_tokens*K_HALF*sizeof(fp4x2))
        #   B_ps_q_rsrc    = make_buffer_rsrc(B_ps_q, E*N_OUT*K_HALF*sizeof(fp4x2))
        #   A_scale_rsrc   = make_buffer_rsrc(A_scale, (MAX_M/32)*kAS_per_chunk_dw*4)
        #   B_ps_scale_rsrc= make_buffer_rsrc(B_ps_scale, E*kBS_per_expert_dw*4)
        #   hidden_rsrc    = make_buffer_rsrc(hidden_states, n_tokens*K*sizeof(bf16))
        # (inline-quant reads hidden via hidden_rsrc; the pre-quant A_q path is
        #  unused here. The descriptor size bounds OOB rows -> hardware returns 0.)
        # A_q is per-token a_quant [M, K/2] (NOT sorted): bound by M so that
        # padding rows whose gathered token >= M clamp to 0 (OOB buffer load).
        a_nbytes = arith.index_cast(
            T.i32, c_tokens * arith.constant(K_HALF, index=True)
        )
        a_rsrc = _ptr_buffer_resource(arg_a, a_nbytes)
        b_rsrc = _ptr_buffer_resource(   # HIP B_ps_q_rsrc
            arg_b, arith.constant(experts * N_OUT * K_HALF, type=T.i32)
        )
        aqout_nbytes = arith.index_cast(
            T.i32,
            c_max_sorted * arith.constant(
                (N_INTER * 2) if debug_bf16 else (N_INTER // 2), index=True
            ),
        )
        aqout_rsrc = _ptr_buffer_resource(arg_aq_out, aqout_nbytes)
        eid_rsrc = _ptr_buffer_resource(
            arg_sorted_expert_ids,
            arith.index_cast(
                T.i32, (c_max_sorted / arith.constant(BM, index=True))
                * arith.constant(4, index=True),
            ),
        )

        # HIP: const int e = __ldg(&sorted_expert_ids[m_block_idx]);
        #      const int m_row = m_block_idx * BM;
        expert_i32 = buffer_ops.buffer_load(
            eid_rsrc, m_block, vec_width=1, dtype=T.i32
        )
        m_row = m_block * arith.constant(BM, index=True)

        # HIP LDS union (gemm1_a4w4.cuh:99-109): a single __shared__ pool reused
        # by (1) the A direct-to-LDS staging buffer (lds_a) and (2) the cshuffle
        # accumulator scratch (lds_acc). FlyDSL mirrors this with one lds_base and
        # two views at _lds_a_offset / _lds_acc_offset (sizes unioned, not summed).
        lds_base = allocator.get_base()
        lds_a = SmemPtr(lds_base, _lds_a_offset, T.i8, shape=(_a_lds_bytes,)).get()
        lds_acc = SmemPtr(
            lds_base, _lds_acc_offset, T.f32, shape=(_acc_lds_elems,)
        ).get()

        # ---- gemm1 GEMM (increment 2): A-load(gather) + a16w4 gate/up B-load
        # + A/B scale + K=K_TILES fp4 MFMA -> accm[kMChunks][4]. ----
        # HIP: m_indices is read directly (m_indices[m_row + ...]); A_scale_rsrc /
        # B_ps_scale_rsrc are the make_buffer_rsrc(...) from gemm1_a4w4.cuh:86-90.
        m_idx_rsrc = _ptr_buffer_resource(
            arg_m_indices,
            arith.index_cast(T.i32, c_max_sorted * arith.constant(4, index=True)),
        )
        a_scale_nbytes = arith.index_cast(
            T.i32,
            (c_max_sorted / arith.constant(32, index=True)
             + arith.constant(2, index=True))
            * arith.constant(kAS_per_chunk_dw * 4, index=True),
        )
        a_scale_rsrc = _ptr_buffer_resource(arg_a_scale, a_scale_nbytes)
        b_scale_rsrc = _ptr_buffer_resource(
            arg_b_scale,
            arith.constant(experts * kBS_per_expert_dw * 4, type=T.i32),
        )
        e_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
        _c256 = arith.constant(256, index=True)
        _c64 = arith.constant(64, index=True)
        _c16 = arith.constant(16, index=True)
        _c2048 = arith.constant(2048, index=True)
        _cKH = arith.constant(K_HALF, index=True)
        _cN = arith.constant(N_OUT, index=True)

        # A-load tiling + per-row token gather (m_indices[m_row + row_local]).
        # HIP non-inline equivalent: cached_actual_row[] in run_one (gemm1_a4w4.cuh
        # :383-400) caches m_indices once; issue_a_load_lds then uses it as the
        # gather base. (Inline-quant uses iq_token_hrow() below instead.)
        tx_i32_base = tx * arith.constant(chunk_i32, index=True)
        layout_x_tile = fx.make_layout((BM, tile_k_dwords), (tile_k_dwords, 1))
        x_row_local = []
        x_col_local = []
        x_row_base_div4 = []
        x_row_token = []
        for i in range_constexpr(num_x_loads):
            _rl, _cl = tile_chunk_coord_i32(
                arith, tx_i32_base=tx_i32_base, i=i,
                total_threads=total_threads, layout_tile_div4=layout_x_tile,
                chunk_i32=chunk_i32,
            )
            x_row_local.append(_rl)
            x_col_local.append(_cl)
            _tok = buffer_ops.buffer_load(
                m_idx_rsrc, m_row + _rl, vec_width=1, dtype=T.i32
            )
            _tok_idx = arith.index_cast(ir.IndexType.get(), _tok)
            x_row_token.append(_tok_idx)
            x_row_base_div4.append(_tok_idx * arith.constant(c_k_div4, index=True))

        # inline-quant resources (hidden + A-scale staging LDS)
        if const_expr(inline_quant):
            hidden_rsrc = _ptr_buffer_resource(
                arg_hidden,
                arith.index_cast(
                    T.i32, c_tokens * arith.constant(K * 2, index=True)
                ),
            )
            lds_ascale_i32 = SmemPtr(
                lds_base, _lds_ascale_offset, T.i32, shape=(_lds_ascale_dwords,)
            ).get()
        if const_expr(not inline_quant):
            # NT A-scale LDS staging region (== HIP s_Ascale): i8 view for the
            # direct-DMA dest pointer, i32 view for the per-tile ds_read.
            lds_ascale_nt = SmemPtr(
                lds_base, _lds_ascale_offset, T.i8, shape=(_nt_ascale_bytes,)
            ).get()
            lds_ascale_nt_i32 = SmemPtr(
                lds_base, _lds_ascale_offset, T.i32, shape=(_nt_ascale_bytes // 4,)
            ).get()

        # HIP issue_a_load_lds (gemm1_a4w4.cuh:121-146): the pre-quant A path that
        # buffer_load_lds's the sorted A_q fp4 directly into s_Aq. FlyDSL splits it
        # into load_a_tile (VMEM gather to regs) + store_a_tile (xor16 -> lds_a).
        # Both are UNUSED in the inline-quant BM=16 path (iq_load/iq_finish write
        # lds_a directly); kept for the non-inline / debug paths.
        def load_a_tile(kt):
            base_k_div4 = arith.constant(kt * tile_k_dwords, index=True)
            parts = []
            for i in range_constexpr(num_x_loads):
                idx = x_row_base_div4[i] + base_k_div4 + x_col_local[i]
                v = buffer_copy_gmem16_dwordx4(
                    buffer_ops, vector, elem_type=T.i8, idx_i32=idx,
                    rsrc=a_rsrc, vec_elems=16,
                )
                parts.append(vector.bitcast(T.vec(4, i32), v))
            return parts

        def store_a_tile(parts, slot=0):
            # slot-aware: write into A-LDS slot `slot` (== HIP issue_a_load_lds's
            # write_slot). lds_base is added to crd2idx in lds_store_16b_xor16, the
            # exact mirror of lds_load_packs_k64's slot offset.
            _sl_off = (arith.index(0) if slot == 0
                       else arith.constant(slot * _a_slot_stride, index=True))
            for i in range_constexpr(num_x_loads):
                lds_store_16b_xor16(
                    arith, vector, lds_memref=lds_a, vec16_ty=vec16_x,
                    layout_lds=layout_lds, row_local=x_row_local[i],
                    col_local_i32=x_col_local[i], tx_c4=arith.index(4),
                    k_blocks16=k_blocks16, lds_base=_sl_off,
                    vec_part_i32x4=parts[i], elem_bytes=elem_bytes,
                )

        # HIP issue_a_load_lds (gemm1_a4w4.cuh:135-145, BM>=32): direct VMEM->LDS
        # DMA of the token-gathered A_q, with the (row&7)*16 xor16 swizzle baked
        # into the voffset so the linear per-lane DMA (lane L -> m0+L*16) lands
        # exactly where read_a_tile_all (== HIP issue_a_ds_read) reads it. Replaces
        # load_a_tile (buffer_load->reg) + store_a_tile (ds_write) -> zero ds_write,
        # no register round-trip. Each (wave,sub) DMAs 8 rows (row_off=lane/8); the
        # 8 lanes/row (lane%8) write the 128B row at 16B each.
        _ptr3_as = ir.Type.parse("!llvm.ptr<3>")
        _c0i32_as = arith.constant(0, type=T.i32)
        _c8idx_a = arith.constant(8, index=True)
        _c14idx_a = arith.constant(14, index=True)
        _a_row_off = lane / _c8idx_a
        # ADDRESS SCALARIZATION (== HIP cached_actual_row + hoisted base). The A
        # DMA voffset = (swizzle ^ col) + token*(K/2) does NOT depend on kt (the
        # per-tile K offset goes into the scalar soffset = kt*(BK/2)), and the LDS
        # dest base lds_row*lds_stride does NOT depend on the rotating slot. HIP
        # gathers the token ONCE and computes token*(K/2) once. FlyDSL used to
        # re-gather + redo the v_mul_lo_u32 EVERY tile (kSubBlocks x K_TILES =
        # 112 v_mul + 112 m_indices loads) interleaved with the per-tile vmcnt, so
        # the A-DMAs could not burst-issue -> the loads' latency landed on the
        # vmcnt(3) stall (~2200 cyc/tile) instead of overlapping MFMA. Precompute
        # per-sub voff (token base + xor16 swizzle) and the m0 row-base ONCE;
        # load_a_directlds then only adds the compile-time slot/kt offsets, so the
        # 4 sub DMAs issue back-to-back with no address ALU on the critical path.
        if const_expr(BM == 128 and _direct_a_load):
            _a_col_byte = arith.index_cast(T.i32, (lane % _c8idx_a) * _c16)
            _a_dl_base = memref.extract_aligned_pointer_as_index(lds_a)
            _a_dl_voff = []     # per-sub voffset (token row base + swizzle), kt-independent
            _a_dl_m0row = []    # per-sub LDS dest base (lds_row*lds_stride), slot-independent
            for sub in range_constexpr(kSubBlocks):
                _lds_row = (wave * arith.constant(BM // 4, index=True)
                            + arith.constant(sub * 8, index=True))
                # HIP lds_swizzle_mask<BK/2=128>(lds_row+row_off) = ((row)&14)<<3
                _mask = arith.index_cast(
                    T.i32, ((_lds_row + _a_row_off) & _c14idx_a) * arith.index(8))
                _colsw = _a_col_byte ^ _mask
                _tok = buffer_ops.buffer_load(
                    m_idx_rsrc, m_row + _lds_row + _a_row_off,
                    vec_width=1, dtype=T.i32)
                _a_dl_voff.append(
                    _colsw + (_tok * arith.constant(K // 2, type=T.i32)))
                _a_dl_m0row.append(rocdl.readfirstlane(
                    T.i64, arith.index_cast(
                        T.i64,
                        _a_dl_base + _lds_row * arith.constant(lds_stride, index=True))))

        def load_a_directlds(slot, kt):
            _ksoff = arith.constant(kt * (BK // 2), type=T.i32)
            _slot_off = arith.constant(slot * _a_slot_stride, type=T.i64)
            for sub in range_constexpr(kSubBlocks):
                _m0 = _a_dl_m0row[sub] + _slot_off          # scalar i64 base + const slot offset
                rocdl.raw_ptr_buffer_load_lds(
                    a_rsrc, llvm.inttoptr(_ptr3_as, _m0),
                    arith.constant(16, type=T.i32), _a_dl_voff[sub], _ksoff,
                    _c0i32_as, _c0i32_as)

        def issue_a_load_nt(slot, kt):
            if const_expr(BM == 128 and _direct_a_load):
                load_a_directlds(slot, kt)
            else:
                store_a_tile(load_a_tile(kt), slot=slot)

        # HIP inline_quant_kt (gemm1_a4w4.cuh:209-259), NON-split prologue form:
        # gather hidden[token], compute the MX-block amax, e8m0 + 4x HW cvt -> fp4,
        # write lds_a + lds_ascale. This per-thread variant covers a full 32-K
        # block alone (16 cvt + 4 b128); the K-loop uses the split iq_load/iq_finish
        # (8 cvt + 2 b128/thread, == HIP) instead. Kept only as a reference / for
        # the non-pipelined else-branch.
        # inline bf16->fp4 quant: gather hidden[token], per-lane amax over the
        # lane's 32-K-group, e8m0 + software fp4 -> lds_a; e8m0 -> lds_ascale.
        _c0x7f = arith.constant(0x7FFFFFFF, type=i32)
        _khalf_dw = K // 2          # hidden row stride (dwords; K bf16 = K/2 dw)

        def load_a_tile_inline(kt):
            parts = []
            for i in range_constexpr(num_x_loads):
                token = x_row_token[i]
                col_dw = x_col_local[i]
                hbase = (
                    token * arith.constant(_khalf_dw, index=True)
                    + arith.constant(kt * 128, index=True)
                    + col_dw * arith.index(4)
                )
                d_list = []
                f32v = []
                for j in range_constexpr(4):
                    vv = buffer_ops.buffer_load(
                        hidden_rsrc, hbase + arith.constant(j * 4, index=True),
                        vec_width=4, dtype=i32,
                    )
                    for kk in range_constexpr(4):
                        d = vector.extract(
                            vv, static_position=[kk], dynamic_position=[]
                        )
                        d_list.append(d)
                        f32v.append((d << arith.constant(16, type=i32)).bitcast(f32))
                        f32v.append(
                            (d & arith.constant(-0x10000, type=i32)).bitcast(f32)
                        )
                amax = (f32v[0].bitcast(i32) & _c0x7f).bitcast(f32)
                for kk in range_constexpr(1, 32):
                    amax = arith.maximumf(
                        amax, (f32v[kk].bitcast(i32) & _c0x7f).bitcast(f32)
                    )
                max_i = amax.bitcast(i32)
                exp_field = (
                    (max_i + arith.constant(0x200000, type=i32))
                    >> arith.constant(23, type=i32)
                ) & arith.constant(0xFF, type=i32)
                e8m0 = arith.minsi(
                    arith.maxsi(exp_field - arith.constant(2, type=i32),
                                arith.constant(0, type=i32)),
                    arith.constant(254, type=i32),
                )
                # qs = 2^(e8m0-127); HW cvt divides bf16 by qs -> fp4 (16 instrs)
                qs = (e8m0 << arith.constant(23, type=i32)).bitcast(f32)
                vec2bf16 = T.vec(2, T.bf16)
                dwords = []
                for dd in range_constexpr(4):
                    pk = arith.constant(0, type=i32)
                    for idx in range_constexpr(4):
                        src = vector.bitcast(
                            vec2bf16,
                            vector.from_elements(T.vec(1, i32), [d_list[dd * 4 + idx]]),
                        )
                        pk = llvm.call_intrinsic(
                            i32, "llvm.amdgcn.cvt.scalef32.pk.fp4.bf16",
                            [pk, src, qs, arith.constant(idx, type=i32)], [], [],
                        )
                    dwords.append(pk)
                parts.append(vector.from_elements(T.vec(4, i32), dwords))
                # e8m0 -> its OWN dword lds_ascale_i32[row*8 + group] (no race)
                group = col_dw / arith.index(4)
                asc_idx = x_row_local[i] * arith.index(8) + group
                vector.store(
                    vector.from_elements(T.vec(1, i32), [e8m0]),
                    lds_ascale_i32, [asc_idx], alignment=4,
                )
            return parts

        # HIP-faithful inline-quant for BM=16 (M<=16): each lane handles 2 b128
        # (B128_IDX 0/1), computes a local amax over its own 8 bf16, then a
        # cross-lane quad reduction (shuffle_xor 1,2 == HIP dpp 0xB1+0x4E) to get
        # the 32-element MX-block scale shared by the 4 lanes of the quad. Each
        # lane then does 4 cvt -> 1 fp4 dword and writes it directly to lds_a.
        # This removes the 2x redundant per-thread coverage of load_a_tile_inline
        # (which had each thread cover a full 32-block alone -> 16 cvt + 4 b128,
        # 256 threads overlapping a 512-dword tile 2x). Result: 8 cvt + 2 b128 per
        # thread, matching HIP. The LDS byte content (lds_a fp4 + lds_ascale_i32
        # e8m0) is byte-identical to load_a_tile_inline, so the read side
        # (mfma_ktile / load_a_scale) is unchanged.
        _vec2bf16 = T.vec(2, T.bf16)
        _vec1_i32 = T.vec(1, i32)
        _vec4_i8 = T.vec(4, T.i8)
        _vec2_i16 = T.vec(2, T.i16)
        _c4idx = arith.constant(4, index=True)
        _i32ty = ir.IntegerType.get_signless(32)

        def _v2i16(x):
            return vector.bitcast(_vec2_i16, vector.from_elements(_vec1_i32, [x]))

        def _dpp_quad_amax(amax):
            # HIP inline_quant_dpp_quad_amax: quad reduction over the 4 lanes of an
            # MX block. HIP IR is umax.i32(x, update.dpp.i32(x)), which LLVM ISel
            # FUSES into a single v_max_u32_dpp. We MUST do the max in the u32
            # INTEGER domain to get that fusion: amax holds |value| (>=0), and for
            # non-negative floats the u32 bit pattern is monotonic, so maxui on the
            # bits == maxf (numerically identical). The previous f32 maximumf form
            # did NOT fuse -> emitted v_mov_b32_dpp + v_maximum3_f32 (2x the VALU);
            # this matches HIP's 112 v_max_u32_dpp exactly.
            AV = type(amax)
            ai = amax.bitcast(i32)
            p = rocdl.update_dpp(_i32ty, ai, ai, 0xB1, 0xF, 0xF, True)
            ai = arith.maxui(ai, AV(p))                       # -> v_max_u32_dpp
            p = rocdl.update_dpp(_i32ty, ai, ai, 0x4E, 0xF, 0xF, True)
            ai = arith.maxui(ai, AV(p))                       # -> v_max_u32_dpp
            return ai.bitcast(f32)

        # HIP cached_row_inline (gemm1_a4w4.cuh:402-410):
        #   const int rcls = wave*4 + (lane/16);
        #   cached_row_inline[s] = m_indices[m_row + s*16 + rcls];  // s=0 for BM=16
        # token-row gather is loop-invariant (same A-row across all K-tiles), so
        # hoist it out of the K-loop: HIP caches it once (cached_row_inline). The
        # per-tile gather had each K-tile re-load m_indices, and its vmcnt(0) wait
        # drained the prefetched B (defeating the B double-buffer).
        def iq_token_hrow():
            row = wave * _c4idx + lane_div_16              # 0..15  (HIP r/rcls)
            tok = buffer_ops.buffer_load(
                m_idx_rsrc, m_row + row, vec_width=1, dtype=T.i32
            )
            return row, (
                arith.index_cast(ir.IndexType.get(), tok)
                * arith.constant(_khalf_dw, index=True)
            )

        # HIP inline_quant_load_kt (gemm1_a4w4.cuh:197-208) -- the LOAD half:
        #   v_voff = row_token*(K*2) + ((lane>>2)&3)*64 + (lane&3)*16;  // bytes
        #   s_soff = kt*(BK*2) + B128_IDX*256;
        #   return raw_buffer_load_b128(hidden_rsrc, v_voff, s_soff, 0);
        # FlyDSL byte offset = hrow*4 + kt*512 + b128*256 + (lane%16)*16, and
        # (lane%16)*16 == ((lane>>2)&3)*64 + (lane&3)*16 (the same lane[3:0]*16),
        # so the per-lane hidden address is BIT-IDENTICAL to HIP; both lower to
        # buffer_load_dwordx4. b128 loop == HIP's two B128_IDX 0/1 calls.
        # Split quant into LOAD (issue the 2 b128 hidden VMEM loads) and FINISH
        # (amax + DPP + cvt + LDS store). The pipelined K-loop issues iq_load for
        # the next tile, runs the current tile's MFMA cluster (hiding the hidden
        # load latency), then runs iq_finish -- so the amax's vmcnt wait lands
        # after MFMA instead of stalling it (HIP load_kt / finish_kt split).
        def iq_load(kt, hrow):
            # HIP inline_quant_load_kt: v_voff = row_token*(K*2) + lane part (per
            # lane), s_soff = kt*(BK*2) + B128*256 (scalar). The per-lane voffset
            # (hrow + lane%16*4 dwords) is kt/b128-independent (CSE'd once); the
            # kt*512+b128*256-byte offset goes to the scalar soffset, so no per-load
            # vector address add.
            off = hrow + lane_mod_16 * _c4idx          # dwords (buffer x4 -> bytes)
            vvs = []
            for b128 in range_constexpr(2):
                vvs.append(buffer_ops.buffer_load(
                    hidden_rsrc, off, vec_width=4, dtype=i32,
                    soffset_bytes=(kt * 512 + b128 * 256)))
            return vvs

        # HIP inline_quant_finish_kt (gemm1_a4w4.cuh:268-310) -- the FINISH half:
        #   hm[j]=h_dw[j]&0x7FFF7FFF; m01=pkmax_u16(hm0,hm1); m23=pkmax_u16(hm2,hm3);
        #   m0123=pkmax_u16(m01,m23); local_amax=max(lo,hi);
        #   amax=dpp_quad_amax(local_amax); e8m0=encode_e8m0(amax); qs=2^(e8m0<<23);
        #   pk = 4x cvt_scalef32_pk_fp4_bf16(h_dw, qs); s_Aq[slot][r][swz]=pk;
        #   s_Ascale[...]=e8m0.
        # FlyDSL matches op-for-op: arith.maxui on vec2(i16) == v_pk_max_u16,
        # _dpp_quad_amax == inline_quant_dpp_quad_amax (fused v_max_u32_dpp 0xB1/0x4E),
        # the e8m0 exp-extract is bit-exact vs encode_e8m0, and the 4 cvt intrinsics
        # are identical. (FlyDSL does amax in f32 bits via m<<16; the bf16 exponent
        # lands in the f32 exponent field, so the e8m0 result is unchanged.)
        def iq_finish(row, vvs, slot=0):
            _a_slot_off = arith.constant(slot * _a_slot_stride, index=True)
            _sc_slot_off = arith.constant(slot * _ascale_slot_dwords, index=True)
            # HIP inline_quant pack path: accumulate the 2 b128 e8m0 into one u32
            # (byte0 = b128 0, byte2 = b128 1 == HIP pack_byte = B128_IDX*2+SUB,
            # SUB=0) and write it once after the loop (inline_quant_pack_write).
            scale_accum = arith.constant(0, type=i32)
            for b128 in range_constexpr(2):
                # col_dw (fp4 dword col 0..31) = B128*16 + lane_mod_16
                col_dw = arith.constant(b128 * 16, index=True) + lane_mod_16
                vv = vvs[b128]
                d_list = [
                    vector.extract(vv, static_position=[kk], dynamic_position=[])
                    for kk in range_constexpr(4)
                ]
                # local amax over the lane's 8 bf16 via packed u16 max
                # (v_pk_max_u16, == HIP inline_quant_pkmax_u16): |bf16| ordering
                # equals u16 ordering once the sign bits are cleared. Halves the
                # amax VALU vs the scalar-f32 maximumf tree and drops VGPR
                # 158 -> 140 (= HIP 139), recovering occupancy headroom.
                _m7f = arith.constant(0x7FFF7FFF, type=i32)
                p01 = arith.maxui(_v2i16(d_list[0] & _m7f), _v2i16(d_list[1] & _m7f))
                p23 = arith.maxui(_v2i16(d_list[2] & _m7f), _v2i16(d_list[3] & _m7f))
                p = arith.maxui(p01, p23)              # vec2(i16): [max lo, max hi]
                # lo/hi 16-bit combine as a u16 max (== HIP
                # max((uint16_t)(m&0xFFFF), (uint16_t)(m>>16))): max the two i16
                # lanes of p directly. This emits v_max_u16(_sdwa) like HIP rather
                # than v_max_u32_sdwa (the prior i32 mask+shift form). Both are
                # bit-equal here (values <= 0x7FFF after the sign-clear).
                p_lo = vector.extract(p, static_position=[0], dynamic_position=[])
                p_hi = vector.extract(p, static_position=[1], dynamic_position=[])
                m = arith.extui(i32, arith.maxui(p_lo, p_hi))
                # bf16(|amax|) -> f32 bits, then quad reduce over the 4 lanes.
                amax = (m << arith.constant(16, type=i32)).bitcast(f32)
                amax = _dpp_quad_amax(amax)
                max_i = amax.bitcast(i32)
                exp_field = (
                    (max_i + arith.constant(0x200000, type=i32))
                    >> arith.constant(23, type=i32)
                ) & arith.constant(0xFF, type=i32)
                e8m0 = arith.minsi(
                    arith.maxsi(exp_field - arith.constant(2, type=i32),
                                arith.constant(0, type=i32)),
                    arith.constant(254, type=i32),
                )
                qs = (e8m0 << arith.constant(23, type=i32)).bitcast(f32)
                pk = arith.constant(0, type=i32)
                for idx in range_constexpr(4):
                    src = vector.bitcast(
                        _vec2bf16,
                        vector.from_elements(_vec1_i32, [d_list[idx]]),
                    )
                    pk = llvm.call_intrinsic(
                        i32, "llvm.amdgcn.cvt.scalef32.pk.fp4.bf16",
                        [pk, src, qs, arith.constant(idx, type=i32)], [], [],
                    )
                # store fp4 dword at the SAME byte position load_a_tile_inline's
                # 16B chunk store would place dword col_dw: swizzle_xor16 keeps the
                # low 4 bits, so a per-dword 4B store lands inside the swizzled
                # 16B line exactly where the chunk store's dword col_dw goes.
                col_swz = swizzle_xor16(row, col_dw * _c4idx, k_blocks16)
                byte_idx = (row * arith.constant(lds_stride, index=True)
                            + col_swz + _a_slot_off)
                v4i8 = vector.bitcast(
                    _vec4_i8, vector.from_elements(_vec1_i32, [pk]))
                vector.store(v4i8, lds_a, [byte_idx], alignment=4)
                # accumulate e8m0 into scale_accum byte (b128*2): == HIP
                # scale_accum |= e8m0 << (pack_byte*8), pack_byte = B128_IDX*2.
                scale_accum = scale_accum | (
                    e8m0 << arith.constant(b128 * 16, type=i32))
            # HIP inline_quant_pack_write: one packed dword per quant-lane at
            #   lane_tgt = ((lane>>2)&3)*16 + r_in_chunk   (r_in_chunk == row);
            # the 4 quad lanes share scale_accum so the write is idempotent. This
            # lowers to ds_write_b32 x1/tile (vs the prior 2x dword-per-e8m0) and
            # is read back as one dword by load_a_scale (ds_read_b32, op_sel byte).
            lane_hi = lane_mod_16 / _c4idx              # (lane>>2)&3
            asc_idx = lane_hi * _c16 + row + _sc_slot_off
            vector.store(
                vector.from_elements(_vec1_i32, [scale_accum]),
                lds_ascale_i32, [asc_idx], alignment=4,
            )

        # HIP issue_b_load_j<K_C>(b_sub, j) (gemm1_a4w4.cuh:312-320):
        #   v_voff = (lane/16)*256 + (lane%16)*16 + K_C*2048;     // bytes
        #   buffer_load_b128_imm<   0>(b_sub[j][0], B_ps_q_rsrc, v_voff, b_load_s_base[j]);
        #   buffer_load_b128_imm<1024>(b_sub[j][1], B_ps_q_rsrc, v_voff, b_load_s_base[j]);
        # with b_load_s_base[j] = (e*N_OUT + n_block*BN + wave_n*(BN/4) + j*16)*(K/2).
        # FlyDSL is identical: base_bytes==b_load_s_base[j], v_voff matches, and the
        # sub=0/1 loads (+256 dwords = +1024 bytes) == the imm 0/1024. Each is a
        # buffer_load_dwordx4 (kUseNT aux via b_nt). 4 j x 2 sub = 8 b128 / tile.
        # HIP issue_b_load_j<K_C>(b_sub, j): one n-output J = 2 b128 (sub 0/1).
        # ADDRESSING (== HIP): the per-j base is a wave-uniform SCALAR soffset
        # b_load_s_base[j] (readfirstlane), the lane part is one shared voffset
        # vreg, and the tile/sub offsets fold into the buffer imm. This removes the
        # per-load vector address arithmetic (v_or/v_add/v_lshl) FlyDSL used when it
        # materialized the full voffset per (j,kt,sub) into a vreg.
        #   b_load_s_base[j] = (e*N_OUT + n_block*256 + wave_n*64 + j*16) * K_HALF  [bytes]
        _b_base_j = [
            rocdl.readfirstlane(
                T.i32,
                arith.index_cast(
                    T.i32,
                    (e_idx * _cN + n_block * _c256 + wave_n * _c64
                     + arith.constant(j * 16, index=True)) * _cKH,
                ),
            )
            for j in range_constexpr(4)
        ]
        # lane voffset in ELEMENTS (buffer_load multiplies by 4):
        #   (lane/16*256 + lane%16*16) / 4 = lane/16*64 + lane%16*4
        _b_voff_lane = arith.index_cast(
            T.i32, lane_div_16 * _c64 + lane_mod_16 * _c4idx)

        # Split out so the 2-stage K-loop can interleave one issue_b_load_j after
        # each per-J MFMA cluster (== HIP run_one cuh:529-543).
        def load_b_tile_j(kt, j):
            sb = _b_base_j[j]                                   # scalar soffset (bytes)
            # voff_kt shared across the 4 j of a tile (CSE -> 1 vreg/tile); the
            # kt*2048-byte part lives in the vreg (exceeds the 4KB imm), the
            # sub*1024-byte part folds into the buffer imm offset.
            voff_kt = _b_voff_lane + arith.constant(kt * 512, type=T.i32)
            b_j = []
            for sub in range_constexpr(2):
                off = voff_kt + arith.constant(sub * 256, type=T.i32)
                v16 = buffer_ops.buffer_load(
                    b_rsrc, off, vec_width=4, dtype=i32, cache_modifier=b_nt,
                    soffset_bytes=sb,
                )
                i64x2 = vector.bitcast(vec2_i64, v16)
                b0 = vector.extract(i64x2, static_position=[0], dynamic_position=[])
                b1 = vector.extract(i64x2, static_position=[1], dynamic_position=[])
                b_j.append((b0, b1))
            return b_j

        def load_b_tile(kt):
            return [load_b_tile_j(kt, j) for j in range_constexpr(4)]

        _scale_lane_dw = lane_div_16 * _c16 + lane_mod_16
        _chunk_base = m_row / arith.constant(32, index=True)
        _mni_base = (
            n_block * arith.constant(_mni_n0, index=True)
            + wave_n * arith.constant(_mni_wn, index=True)
        )

        # HIP issue_a_scale_ds_read (gemm1_a4w4.cuh:188-195) [inline+NT both read
        # the A-scale from LDS here]:
        #   lds_dw = sub*kAS_per_chunk_dw + kt*64 + (lane/16)*16 + (lane%16);
        #   a_scale_aiter[sub] = *(int*)&s_Ascale[lds_dw*4];     // ds_read_b32
        # DIVERGENCE (intentional): for inline-quant FlyDSL reads from a SEPARATE
        # lds_ascale_i32 buffer with a dword-per-e8m0 layout (asc_idx = row*8 +
        # block), written idempotently by iq_finish, instead of HIP's packed-byte
        # s_Ascale. This avoids the sub-dword store races that HIP resolves with
        # the inline_quant_pack_write path. The e8m0 value handed to the MFMA `sa`
        # operand is identical; both lower to ds_read_b32. The NT branch below is
        # the buffer-load form == HIP issue_a_scale_load (165-186).
        def load_a_scale(kt, slot=0):
            # HIP issue_a_scale_ds_read (cuh:188-195), BM=16 packed layout: read
            # ONE dword (4 e8m0); the MFMA op_sel_a selects the byte (k0 -> 0,
            # k1 -> 2). lds_dw_inner = (lane/16)*16 + (lane%16). ds_read_b32 x1.
            if const_expr(inline_quant and BM == 16):
                _sc_slot_off = arith.constant(
                    slot * _ascale_slot_dwords, index=True)
                asc_idx = lane_div_16 * _c16 + lane_mod_16 + _sc_slot_off
                v = vector.load_op(T.vec(1, i32), lds_ascale_i32, [asc_idx])
                return vector.extract(v, static_position=[0], dynamic_position=[])
            # BM>=32 inline: dword-per-e8m0 staging (load_a_tile_inline writer),
            # returns [a_sc_k0, a_sc_k1] (one A-scale dword per MFMA-K).
            if const_expr(inline_quant):
                _sc_slot_off = arith.constant(
                    slot * _ascale_slot_dwords, index=True)
                out = []
                for k_idx in range_constexpr(2):
                    group = arith.constant(k_idx * 4, index=True) + lane_div_16
                    asc_idx = lane_mod_16 * arith.index(8) + group + _sc_slot_off
                    v = vector.load_op(T.vec(1, i32), lds_ascale_i32, [asc_idx])
                    out.append(
                        vector.extract(v, static_position=[0], dynamic_position=[])
                    )
                return out
            # NT: one A-scale dword per 32-row sub-chunk (kSubBlocks). Each dword
            # holds 4 e8m0 (2 MFMA-K x 2 m-chunks-in-sub); mfma_ktile picks
            # a_scs[imxdl//2] with op_sel = ikxdl*2 + imxdl%2.
            a_scs = []
            for sub in range_constexpr(kSubBlocks):
                a_base = (
                    (_chunk_base + arith.constant(sub, index=True))
                    * arith.constant(kAS_per_chunk_dw, index=True)
                    + arith.constant(kt * 64, index=True) + _scale_lane_dw
                )
                a_scs.append(buffer_ops.buffer_load(
                    a_scale_rsrc, a_base, vec_width=1, dtype=T.i32, cache_modifier=0
                ))
            return a_scs

        # HIP issue_a_scale_load (gemm1_a4w4.cuh:165-186): ONE-TIME direct-DMA of
        # this m-block's kSubBlocks 32-row A-scale chunks (all K_TILES) straight to
        # LDS via raw_ptr_buffer_load_lds. Identity copy: LDS[sub*kAS_per_chunk_dw +
        # j] = A_scale[(m_row/32+sub)*kAS_per_chunk_dw + j], j=0..1791. Per sub: one
        # 16B DMA (lane L wave w -> bytes w*1024+L*16, dwords 0..1023) + three 4B
        # DMA (byte_off 4096/5120/6144, dwords 1024..1791). voff/m0 use the same
        # (wave*64+lane) ordering so LDS offset == VMEM offset (identity).
        _ptr3_as = ir.Type.parse("!llvm.ptr<3>")
        _c0i32_as = arith.constant(0, type=T.i32)
        def issue_a_scale_stage():
            _as_base_idx = memref.extract_aligned_pointer_as_index(lds_ascale_nt)
            wl = wave * _c64 + lane                             # wave*64 + lane
            v_voff_dx4 = arith.index_cast(T.i32, wl * _c16)     # (wave*64+lane)*16
            v_voff_dw = arith.index_cast(T.i32, wl * arith.index(4))
            for sub in range_constexpr(kSubBlocks):
                s_chunk_base = rocdl.readfirstlane(
                    T.i32, arith.index_cast(
                        T.i32,
                        (_chunk_base + arith.constant(sub, index=True))
                        * arith.constant(kAS_per_chunk_dw * 4, index=True)))
                _sub_off = arith.constant(sub * kAS_per_chunk_dw * 4, index=True)
                m0_16 = rocdl.readfirstlane(
                    T.i64, arith.index_cast(
                        T.i64,
                        _as_base_idx + _sub_off
                        + wave * arith.constant(1024, index=True)))
                rocdl.raw_ptr_buffer_load_lds(
                    a_scale_rsrc, llvm.inttoptr(_ptr3_as, m0_16),
                    arith.constant(16, type=T.i32), v_voff_dx4, s_chunk_base,
                    _c0i32_as, _c0i32_as)
                for d in range_constexpr(3):
                    _bo = 4096 + d * 1024
                    s_off = rocdl.readfirstlane(
                        T.i32, s_chunk_base + arith.constant(_bo, type=T.i32))
                    m0_4 = rocdl.readfirstlane(
                        T.i64, arith.index_cast(
                            T.i64,
                            _as_base_idx + _sub_off
                            + arith.constant(_bo, index=True)
                            + wave * arith.constant(256, index=True)))
                    rocdl.raw_ptr_buffer_load_lds(
                        a_scale_rsrc, llvm.inttoptr(_ptr3_as, m0_4),
                        arith.constant(4, type=T.i32), v_voff_dw, s_off,
                        _c0i32_as, _c0i32_as)

        # HIP issue_a_scale_ds_read (gemm1_a4w4.cuh:188-195): per-tile LDS read,
        # lds_dw = sub*kAS_per_chunk_dw + kt*64 + (lane/16)*16 + (lane%16). Returns
        # kSubBlocks dwords (== load_a_scale NT shape) for mfma_cluster_J_all.
        def issue_a_scale_lds_read(kt):
            a_scs = []
            for sub in range_constexpr(kSubBlocks):
                lds_dw = (
                    arith.constant(sub * kAS_per_chunk_dw + kt * 64, index=True)
                    + lane_div_16 * _c16 + lane_mod_16)
                v = vector.load_op(T.vec(1, i32), lds_ascale_nt_i32, [lds_dw])
                a_scs.append(
                    vector.extract(v, static_position=[0], dynamic_position=[]))
            return a_scs

        # HIP issue_b_scale_load<K_C>(bs_sub) (gemm1_a4w4.cuh:322-332): scalar
        # per-mw base soffset + shared lane voffset + kt offset (== B-load style).
        #   b_scale_s_base[mw] = (e*kBS_per_expert_dw + (mni_base+mw)*kBS_stride_n0_dw)*4
        _bs_base_mw = [
            rocdl.readfirstlane(
                T.i32,
                arith.index_cast(
                    T.i32,
                    (e_idx * arith.constant(kBS_per_expert_dw, index=True)
                     + (_mni_base + arith.constant(mw, index=True))
                     * arith.constant(kBS_stride_n0_dw, index=True))
                    * arith.index(4),
                ),
            )
            for mw in range_constexpr(2)
        ]
        _bs_voff_lane = arith.index_cast(T.i32, _scale_lane_dw)   # elements (x4 -> bytes)

        def load_b_scale(kt):
            b_scs = []
            voff_kt = _bs_voff_lane + arith.constant(kt * 64, type=T.i32)  # CSE/tile
            for mw in range_constexpr(2):
                b_scs.append(buffer_ops.buffer_load(
                    b_scale_rsrc, voff_kt, vec_width=1, dtype=T.i32, cache_modifier=0,
                    soffset_bytes=_bs_base_mw[mw],
                ))
            return b_scs

        # MFMA setup
        row_a_lds = lane_mod_16
        col_offset_base = lane_div_16 * _c16
        m_repeat = BM // 16
        num_acc_n = BN // num_waves // 16     # 4
        num_acc_n_packed = num_acc_n // 2     # 2
        pack_N = 2

        def _pack_i64x4(x0, x1):
            c0 = arith.constant(0, type=T.i64)
            v4 = vector.from_elements(vec4_i64, [x0, x1, c0, c0])
            return vector.bitcast(vec8_i32, v4)

        # HIP issue_a_ds_read (gemm1_a4w4.cuh:148-163), inner load:
        #   lane_row=lane%16; lane_col=(lane/16)*16; mask=lds_swizzle_mask<BK/2>(lane_row);
        #   lds_col = (lane_col + k*64) ^ mask;
        #   a[i][k] = *(i32x4*)&s_Aq[slot][lane_row + i*16][lds_col];  // ds_read_b128
        # FlyDSL col_base = lane_div_16*16 + (k_idx*128)//a_elem_vec_pack (==k*64) and
        # swizzle_xor16(curr_row, col_base) == (lane_col + k*64) ^ mask, so the per-lane
        # LDS address is identical; vector.load_op(vec16) lowers to ds_read_b128.
        def lds_load_packs_k64(curr_row, col_base, slot=0):
            if const_expr(BM == 128 and _hip_a_read):
                # HIP issue_a_ds_read uses lds_swizzle_mask<BK/2>(row):
                # ROW_BYTES=128 -> mask=((row)&14)<<3.
                col_swz = col_base ^ ((curr_row & _c14idx_a) * arith.index(8))
            else:
                col_swz = swizzle_xor16(curr_row, col_base, k_blocks16)
            idx_a = crd2idx([curr_row, col_swz], layout_lds)
            if const_expr(slot != 0):
                idx_a = idx_a + arith.constant(slot * _a_slot_stride, index=True)
            loaded = vector.load_op(vec16_x, lds_a, [idx_a])
            a_i64x2 = vector.bitcast(vec2_i64, loaded)
            a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
            a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
            return a0, a1

        acc_init = arith.constant_vector(0.0, vec4_f32)
        accm = [acc_init] * (m_repeat * num_acc_n)

        # HIP issue_mfma_cluster<J,kInit>(slot) (gemm1_a4w4.cuh:334-375), BM==16:
        #   sa = a_scale_aiter[0]; sb = b_scale_v[slot][J/2];
        #   mfma_f4f4_vgpr<0, 0+J%2>(accm[0][J], a[0][0], b[slot][J][0], sa, sb);  // k0
        #   mfma_f4f4_vgpr<2, 2+J%2>(accm[0][J], a[0][1], b[slot][J][1], sa, sb);  // k1
        # mfma_ktile runs ALL 8 MFMA of the tile (k_idx x ni x inxdl). The per-MFMA
        # operands/op_sel/accumulator are bit-identical to HIP (op_sel_b = k_idx*2 +
        # J%2, acc_idx = J, sa/sb same). TWO ISA-PATTERN DIVERGENCES vs HIP (the
        # main remaining gemm1 alignment gap, see isa_compare):
        #   (1) ISSUE ORDER: FlyDSL is K-major (k0:{J0..3}, k1:{J0..3}); HIP is
        #       N-major (per J: k0,k1). Same 8 MFMA, different interleave.
        #   (2) B-LOAD INTERLEAVE: HIP issues issue_b_load_j(J) BETWEEN the 4 per-J
        #       MFMA clusters (sched_barrier-fenced) so B VMEM overlaps MFMA at a
        #       fine grain; FlyDSL prefetches the whole next B-tile (load_b_tile
        #       (kt+1)) before mfma_ktile, relying on the scheduler to overlap.
        def mfma_ktile(b_tile, a_sc_per_k, b_scs, a_slot=0):
            for k_idx in range_constexpr(2):
                ikxdl = k_idx
                col_base = col_offset_base + arith.constant(
                    (k_idx * 128) // a_elem_vec_pack, index=True
                )
                for imxdl in range_constexpr(pack_M):
                    mi_idx = imxdl
                    curr_row = row_a_lds + arith.constant(mi_idx * 16, index=True)
                    a0, a1 = lds_load_packs_k64(curr_row, col_base, slot=a_slot)
                    a128 = _pack_i64x4(a0, a1)
                    # inline: per-k e8m0 in byte 0 (op_sel 0). NT: one dword per
                    # 32-row sub (a_scs[imxdl//2]); byte = ikxdl*2 + imxdl%2.
                    if const_expr(inline_quant):
                        a_sc = a_sc_per_k[k_idx]
                        op_sel_a = 0
                    else:
                        a_sc = a_sc_per_k[imxdl // 2]
                        op_sel_a = ikxdl * _scale_pack_m + (imxdl % 2)
                    for ni in range_constexpr(num_acc_n_packed):
                        b_scale_val = b_scs[ni]
                        for inxdl in range_constexpr(pack_N):
                            ni_idx = ni * pack_N + inxdl
                            b0, b1 = b_tile[ni_idx][k_idx]
                            b128 = _pack_i64x4(b0, b1)
                            op_sel_b = ikxdl * pack_N + inxdl
                            acc_idx = mi_idx * num_acc_n + ni_idx
                            accm[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                vec4_f32,
                                [a128, b128, accm[acc_idx], cbsz, blgp,
                                 op_sel_a, a_sc, op_sel_b, b_scale_val],
                            )

        # HIP issue_a_ds_read (cuh:148-163): read the tile's A once (BM=16 -> 1
        # m-chunk, 2 k128) and reuse across the 4 per-J MFMA clusters. Returns
        # [a_k0, a_k1] (each a packed i32x8 A operand). Used by the HIP-faithful
        # 2-stage loop; mfma_ktile above (non-BM16 / else path) reads inline.
        def read_a_tile(a_slot=0):
            a_k = []
            for k_idx in range_constexpr(2):
                col_base = col_offset_base + arith.constant(
                    (k_idx * 128) // a_elem_vec_pack, index=True
                )
                a0, a1 = lds_load_packs_k64(row_a_lds, col_base, slot=a_slot)
                a_k.append(_pack_i64x4(a0, a1))
            return a_k

        # HIP issue_mfma_cluster<J,kInit>(slot) (cuh:334-345), BM=16: ONE n-output
        # J = 2 MFMA (k0: op_sel_a=0; k1: op_sel_a=0 on FlyDSL's per-k a_sc dword),
        # op_sel_b = k_idx*2 + J%2, accumulating into accm[J]. Issued N-major (one
        # J at a time) so the K-loop can fence it with sched_barrier/s_setprio and
        # interleave issue_b_load_j(J) right after -- byte-for-byte HIP's cluster.
        def mfma_cluster_J(a_k, b_J, a_sc, b_scs, J):
            ni = J // pack_N
            inxdl = J % pack_N
            b_scale_val = b_scs[ni]
            for k_idx in range_constexpr(2):
                a128 = a_k[k_idx]
                # HIP: k0 uses op_sel_a=0 (scale byte0), k1 op_sel_a=2 (byte2) of
                # the SAME packed A-scale dword a_sc.
                op_sel_a = k_idx * 2
                b0, b1 = b_J[k_idx]
                b128 = _pack_i64x4(b0, b1)
                op_sel_b = k_idx * pack_N + inxdl
                accm[J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    vec4_f32,
                    [a128, b128, accm[J], cbsz, blgp,
                     op_sel_a, a_sc, op_sel_b, b_scale_val],
                )

        # HIP issue_a_ds_read (cuh:148-163), BM>=32 form: read ALL m_repeat
        # (=BM/16) m-chunks of the tile into a[mi][k] (each a packed i32x8 A
        # operand), reused across the 4 per-J MFMA clusters. (read_a_tile above is
        # the BM=16 single-m-chunk specialization.)
        def read_a_tile_all(a_slot=0):
            # returns a_all[mi][k] = (a0, a1) raw i64 pair (128-bit A operand). The
            # AGPR inline-asm path uses the v4i32 form directly; the VGPR rocdl path
            # repacks via _pack_i64x4.
            if const_expr(BM == 128 and _hip_a_kmajor):
                # HIP issue_a_ds_read is k-major:
                #   for k in 0..1:
                #     for i in 0..kMChunks:
                #       a[i][k] = ds_read_b128(...)
                # Preserve that issue order for waitcnt/ds_read pattern alignment,
                # while keeping the consumer-facing a_all[mi][k] structure.
                a_all = [[None, None] for _ in range(m_repeat)]
                for k_idx in range_constexpr(2):
                    col_base = col_offset_base + arith.constant(
                        (k_idx * 128) // a_elem_vec_pack, index=True
                    )
                    for mi in range_constexpr(m_repeat):
                        curr_row = row_a_lds + arith.constant(mi * 16, index=True)
                        a0, a1 = lds_load_packs_k64(curr_row, col_base, slot=a_slot)
                        a_all[mi][k_idx] = (a0, a1)
                return a_all

            a_all = []
            for mi in range_constexpr(m_repeat):
                curr_row = row_a_lds + arith.constant(mi * 16, index=True)
                a_k = []
                for k_idx in range_constexpr(2):
                    col_base = col_offset_base + arith.constant(
                        (k_idx * 128) // a_elem_vec_pack, index=True
                    )
                    a0, a1 = lds_load_packs_k64(curr_row, col_base, slot=a_slot)
                    a_k.append((a0, a1))
                a_all.append(a_k)
            return a_all

        # HIP mfma_f4f4_agpr (mfma_f4f4.hpp:45-61): force the accumulator into AGPR
        # via inline asm ("+a"), and use the v4i32 (fp4-width) A/B operands -- HIP's
        # .v4i32.v4i32 intrinsic. (FlyDSL's rocdl.mfma op otherwise pads A/B to
        # v8i32, doubling their VGPR; the inline-asm v4i32 form halves A/B VGPR AND
        # moves accm out of VGPR into AGPR.) op_sel selects the scale byte:
        # alo/ahi = scA lo/hi (op_sel_a bit0/bit1), blo/bhi = scB (op_sel_b).
        _vec4_i32 = T.vec(4, i32)
        def _mfma_agpr(acc, a0, a1, b0, b1, sa, sb, op_sel_a, op_sel_b,
                       kinit=False):
            a4 = vector.bitcast(
                _vec4_i32, vector.from_elements(vec2_i64, [a0, a1]))
            b4 = vector.bitcast(
                _vec4_i32, vector.from_elements(vec2_i64, [b0, b1]))
            # op_sel/op_sel_hi select the scale byte: byte = op_sel_hi*2 + op_sel.
            alo = op_sel_a & 1
            ahi = (op_sel_a >> 1) & 1
            blo = op_sel_b & 1
            bhi = (op_sel_b >> 1) & 1
            osel = (f"op_sel:[{alo},{blo},0] op_sel_hi:[{ahi},{bhi},0] "
                    f"cbsz:4 blgp:4")
            if kinit:
                # == HIP mfma_f4f4_agpr_init_zero (hpp:54-61): C-src is literal 0,
                # so the AGPR accumulator is initialized WITHOUT tying the shared
                # VGPR acc_init constant into 32 distinct AGPRs (that tie corrupted
                # accm -> cos 0.9965). Output-only "=a".
                asm = (
                    "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 "
                    + osel
                )
                return llvm.inline_asm(
                    vec4_f32, [a4, b4, sa, sb], asm, "=a,v,v,v,v",
                    has_side_effects=True,
                )
            # == HIP mfma_f4f4_agpr (hpp:45-52): "+a" accumulate (C-src = output).
            asm = (
                "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 "
                + osel
            )
            return llvm.inline_asm(
                vec4_f32, [acc, a4, b4, sa, sb], asm, "=a,0,v,v,v,v",
                has_side_effects=True,
            )

        # HIP issue_mfma_cluster<J,kInit>(slot) (cuh:334-375), BM>=32 form: ONE
        # n-output J, all m_repeat m-chunks x 2 k = 2*m_repeat MFMA into
        # accm[mi*num_acc_n + J]. Per-MFMA operands are bit-identical to mfma_ktile
        # (NT: a_sc=a_sc_per_k[mi//2], op_sel_a=k*_scale_pack_m + mi%2; op_sel_b=
        # k*pack_N + inxdl; sb=b_scs[J//pack_N]). Issued N-major (one J at a time)
        # so the K-loop fences it and interleaves issue_b_load_j(J) right after.
        def mfma_cluster_J_all_direct(a_all, b_J, a_sc_per_k, b_scs, J, kinit=False):
            ni = J // pack_N
            inxdl = J % pack_N
            b_scale_val = b_scs[ni]
            # ISSUE ORDER = k-outer, mi-inner (== HIP's ACTUAL ISA in the thread
            # trace: all m_repeat m-chunks' k0 back-to-back, then all k1). This
            # keeps CONSECUTIVE MFMA on DISTINCT accm and leaves an m_repeat-long
            # gap between accm[mi].k0 and accm[mi].k1, so the MFMA unit issues
            # back-to-back with NO hazard s_nop. The previous per-sub order
            # (i0.k0,i1.k0,i0.k1,i1.k1) recurred the SAME accm every 2 MFMA -> the
            # hazard recognizer inserted ~1 s_nop per 2 MFMA (428 extra s_nop,
            # +1448 cyc). HIP's 16 MFMA run with 0 interleaved s_nop.
            # accm starts = acc_init 0 (pre-zeroed); k0 accumulates onto it (NOT the
            # output-only init-zero form "=a,v,v,v,v", which LLVM left as a dead
            # side-effecting MFMA on a REAL accm reg -> +32 dead MFMA + accm
            # corruption, cos 0.976). (kinit unused.)
            for k_idx in range_constexpr(2):
                b0, b1 = b_J[k_idx]
                for mi in range_constexpr(m_repeat):
                    a0, a1 = a_all[mi][k_idx]
                    a_sc = a_sc_per_k[mi // 2]
                    op_sel_a = (k_idx << 1) | (mi % 2)
                    op_sel_b = k_idx * pack_N + inxdl
                    acc_idx = mi * num_acc_n + J
                    accm[acc_idx] = _mfma_agpr(
                        accm[acc_idx], a0, a1, b0, b1, a_sc, b_scale_val,
                        op_sel_a, op_sel_b, kinit=False)

        def mfma_cluster_J_all(a_all, b_J, a_sc_per_k, b_scs, J, kinit=False):
            ni = J // pack_N
            inxdl = J % pack_N
            b_scale_val = b_scs[ni]

            def _one(mi, k_idx):
                a0, a1 = a_all[mi][k_idx]
                b0, b1 = b_J[k_idx]
                if const_expr(inline_quant):
                    a_sc = a_sc_per_k[k_idx]
                    op_sel_a = 0
                else:
                    a_sc = a_sc_per_k[mi // 2]
                    op_sel_a = k_idx * _scale_pack_m + (mi % 2)
                op_sel_b = k_idx * pack_N + inxdl
                acc_idx = mi * num_acc_n + J
                # BM=128: AGPR accumulator + v4i32 A/B via inline asm (== HIP
                # mfma_f4f4_agpr). init-zero ONLY on k0 (== HIP: agpr_init_zero for
                # k0, agpr-accumulate for k1 onto the same accm; cuh:354-361).
                if const_expr(BM == 128):
                    accm[acc_idx] = _mfma_agpr(
                        accm[acc_idx], a0, a1, b0, b1, a_sc, b_scale_val,
                        op_sel_a, op_sel_b, kinit=(kinit and k_idx == 0))
                else:
                    a128 = _pack_i64x4(a0, a1)
                    b128 = _pack_i64x4(b0, b1)
                    accm[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        vec4_f32,
                        [a128, b128, accm[acc_idx], cbsz, blgp,
                         op_sel_a, a_sc, op_sel_b, b_scale_val],
                    )

            # HIP issue_mfma_cluster order (cuh:347-372): per sub-pair (i0,i1) do
            # i0.k0, i1.k0, i0.k1, i1.k1 -- interleave the 2 m-chunks of the pair.
            for sub in range_constexpr(kSubBlocks):
                i0 = sub * 2
                i1 = sub * 2 + 1
                _one(i0, 0)
                _one(i1, 0)
                _one(i0, 1)
                _one(i1, 1)

        # HIP grid dispatcher (gemm1_a4w4.cuh:573-592): pid=blockIdx.x is decoded to
        # (m_block_idx, n_block_idx); blocks whose m_row >= cumsum_tensor[0] are
        # padding and return early. FlyDSL keeps the 2-D grid and replicates the
        # padding skip with an on-device cumsum guard (scf.if) so no cumsum value is
        # copied back to the host.
        # device-side grid guard: kernel is launched with a fixed max grid
        # (max_sorted/BM); read cumsum on-device and skip padding blocks whose
        # m_row >= cumsum so we neither read cumsum back to the host (no per-iter
        # .item() DtoH + sync) nor waste the GEMM/epilogue on padding blocks.
        # (debug_bf16 keeps the un-guarded body: its const_expr early-return at
        #  the scale store would leave the scf.if then-block unterminated.)
        # const_expr -> compile-time Python if (NOT scf_if_dispatch), so _guard_ip
        # stays in the kernel-builder scope and is visible to the close below.
        if const_expr(not debug_bf16):
            _cumsum_rsrc = _ptr_buffer_resource(
                arg_cumsum, arith.constant(4, type=i32)
            )
            _num_valid = buffer_ops.buffer_load(
                _cumsum_rsrc, arith.constant(0, index=True), vec_width=1, dtype=i32
            )
            _m_row_i32 = arith.index_cast(i32, m_row)
            _blk_valid = arith.cmpi(CmpIPredicate.ult, _m_row_i32, _num_valid)
            _guard_if = scf.IfOp(_blk_valid)
            _guard_ip = ir.InsertionPoint(_guard_if.then_block)
            _guard_ip.__enter__()

        if const_expr(inline_quant and BM == 16):
            # ===== HIP-faithful run_one K-loop (gemm1_a4w4.cuh:434-565) =====
            # Byte-for-byte port of HIP's 2-STAGE software pipeline:
            #   kStages=2 (B + A-quant 2 tiles ahead), kAStages=3 (_a_slots),
            #   kUnroll=K_TILES-2=26, then a separate 2-iter drain.
            #   read_slot = OFFSET % 3, write_slot = (OFFSET+2) % 3,
            #   slot_b     = OFFSET % 2   (all distinct -> no WAR).
            # Each tile's 4 n-output MFMA clusters are issued N-major (one J at a
            # time), fenced with sched_barrier + s_setprio(1)/(0), and a single
            # issue_b_load_j(K_C, J) is interleaved right after each cluster so B
            # VMEM overlaps MFMA at HIP's exact granularity.
            _AS = _a_slots                                     # 3 == kAStages
            _iq_row, _iq_hrow = iq_token_hrow()                # HIP cached_row_inline
            # ---- PROLOGUE: stages 0,1 (== cuh:434-463) ----
            # COUPLE load+process per stage (== HIP inline_quant_kt: load hidden,
            # then amax+cvt+store right after). HIP only SPLITS load/finish in the
            # main loop (to hide hidden-load latency behind MFMA); the prologue has
            # no MFMA to hide behind, so it couples. Coupling keeps the hidden
            # load's use-distance short so the scheduler does NOT defer the
            # token-dependent hidden loads behind the (dependency-free) B loads --
            # i.e. hidden stays among the oldest VMEM, so the prologue amax's vmcnt
            # drains only the hidden (not the 16 B). (Batching iq_load for all
            # stages then iq_finish made hidden's use-distance long -> scheduler
            # issued B first -> amax over-waited vmcnt down to 4.)
            b_slot = [None, None]                              # HIP b[kStages][4][2]
            bsc_slot = [None, None]                            # HIP b_scale_v[kStages][2]
            for K_C in range_constexpr(2):                     # static_for<0,kStages>
                iq_finish(_iq_row, iq_load(K_C, _iq_hrow), slot=K_C)  # inline_quant_kt
                b_slot[K_C] = load_b_tile(K_C)                 # issue_b_load_j(K_C, 0..3)
                bsc_slot[K_C] = load_b_scale(K_C)              # issue_b_scale_load(K_C)
            # ---- MAIN LOOP: OFFSET 0..K_TILES-3 (== cuh:465-552) ----
            for off in range_constexpr(K_TILES - 2):          # static_for<0,kUnroll>
                K_C = off + 2
                read_slot = off % _AS
                write_slot = K_C % _AS
                slot_b = off % 2
                rocdl.s_barrier()                                 # __syncthreads() (472)
                a_k = read_a_tile(a_slot=read_slot)           # issue_a_ds_read(read_slot)
                asc_cur = load_a_scale(off, slot=read_slot)   # issue_a_scale_ds_read(OFFSET)
                hv = iq_load(K_C, _iq_hrow)                    # inline_quant_load_kt(K_C) [2 ahead]
                rocdl.sched_barrier(0)                         # cuh:527
                b_cur = b_slot[slot_b]
                bsc_cur = bsc_slot[slot_b]
                for J in range_constexpr(4):                  # cuh:529-543 for-J
                    rocdl.sched_barrier(0)
                    rocdl.s_setprio(1)
                    mfma_cluster_J(a_k, b_cur[J], asc_cur, bsc_cur, J)  # issue_mfma_cluster<J>
                    rocdl.s_setprio(0)
                    rocdl.sched_barrier(0)
                    b_slot[slot_b][J] = load_b_tile_j(K_C, J)  # issue_b_load_j(K_C, J) -> slot_b
                    rocdl.sched_barrier(0)
                bsc_slot[slot_b] = load_b_scale(K_C)           # issue_b_scale_load(K_C)
                iq_finish(_iq_row, hv, slot=write_slot)        # inline_quant_finish_kt(write_slot,K_C)
            # ---- DRAIN: last kStages tiles (== cuh:554-565) ----
            for s in range_constexpr(2):                       # static_for<0,kStages>
                kt = K_TILES - 2 + s
                read_slot = kt % _AS
                slot_b = kt % 2
                rocdl.s_barrier()                                 # __syncthreads() (559)
                a_k = read_a_tile(a_slot=read_slot)           # issue_a_ds_read(kt%3)
                asc_cur = load_a_scale(kt, slot=read_slot)    # issue_a_scale_ds_read(kt)
                b_cur = b_slot[slot_b]
                bsc_cur = bsc_slot[slot_b]
                for J in range_constexpr(4):                  # 4x issue_mfma_cluster<J>
                    mfma_cluster_J(a_k, b_cur[J], asc_cur, bsc_cur, J)
        if const_expr(not inline_quant):
            # ===== HIP-faithful NT (BM=128) run_one K-loop (gemm1_a4w4.cuh:434-565,
            # kInlineQuant=false) =====
            # 2-stage double-buffer (kStages=2): B + A staged 2 tiles ahead; A goes
            # through a 3-slot LDS rotation (read=OFFSET%3, write=(OFFSET+2)%3 -> no
            # WAR). Each tile's 4 n-output MFMA clusters are issued N-major (one J at
            # a time), sched_barrier-fenced, with issue_b_load_j(K_C,J) interleaved
            # right after each cluster so B VMEM overlaps MFMA. No s_setprio at
            # BM=128 (HIP guards `if BM != 128`). Replaces the old naive 2-barrier/
            # tile non-pipelined loop (57 s_barrier / 721 s_waitcnt -> HIP-like).
            _AS = _a_slots                                  # 3
            b_slot = [None, None]                           # HIP b[kStages][4][2]
            bsc_slot = [None, None]                         # HIP b_scale_v[kStages][2]
            issue_a_scale_stage()                           # HIP issue_a_scale_load (cuh:431-433): A-scale -> LDS once
            # A-load: BM=128 now uses HIP's direct VMEM->LDS path; smaller NT
            # configs keep the verified reg->LDS path via issue_a_load_nt().
            # ---- PROLOGUE: stages 0,1 (== cuh:456-463 NT else + 462) ----
            for K_C in range_constexpr(2):                  # static_for<0,kStages>
                issue_a_load_nt(K_C, K_C)                   # issue_a_load_lds(K_C)
                b_slot[K_C] = load_b_tile(K_C)              # 4x issue_b_load_j(K_C)
                bsc_slot[K_C] = load_b_scale(K_C)           # issue_b_scale_load(K_C)
            # ---- MAIN LOOP: OFFSET 0..K_TILES-3 (== cuh:465-552, NT else branch) --
            for off in range_constexpr(K_TILES - 2):        # static_for<0,kUnroll>
                K_C = off + 2
                read_slot = off % _AS
                write_slot = K_C % _AS
                slot_b = off % 2
                rocdl.s_barrier()                           # __syncthreads() (472)
                if const_expr(BM == 128 and _pre_read_wait):
                    rocdl.s_waitcnt(0)                       # diagnostic: drain direct-LDS before ds_read
                # Issue the A-scale ds_read BEFORE the 16 A ds_read so the FIRST
                # MFMA's scale operand (asc) is ready early. Previously the scales
                # were read AFTER all 16 A reads, so the first MFMA had to wait for
                # the 17th/18th lgkm op -> SIInsertWaitcnts emitted lgkmcnt(1)
                # (drain ~all) before the cluster. HIP interleaves the scale read
                # among the A reads (early) -> incremental lgkmcnt(14->..->0).
                asc_cur = issue_a_scale_lds_read(off)       # issue_a_scale_ds_read(OFFSET)
                a_all = read_a_tile_all(a_slot=read_slot)   # issue_a_ds_read(read_slot)
                issue_a_load_nt(write_slot, K_C)             # issue_a_load_lds(write_slot)
                b_cur = b_slot[slot_b]
                bsc_cur = bsc_slot[slot_b]
                for J in range_constexpr(4):                # cuh:529-543 (BM=128: no setprio)
                    mfma_cluster_J_all_direct(
                        a_all, b_cur[J], asc_cur, bsc_cur, J,
                        kinit=(off == 0))  # issue_mfma_cluster<J,kInit=(OFFSET==0)>
                    rocdl.sched_barrier(0)
                    b_slot[slot_b][J] = load_b_tile_j(K_C, J)  # issue_b_load_j(K_C,J) -> slot_b
                    rocdl.sched_barrier(0)
                bsc_slot[slot_b] = load_b_scale(K_C)        # issue_b_scale_load(K_C)
            # ---- DRAIN: last kStages tiles (== cuh:554-565) ----
            for s in range_constexpr(2):                    # static_for<0,kStages>
                kt = K_TILES - 2 + s
                read_slot = kt % _AS
                slot_b = kt % 2
                rocdl.s_barrier()                           # __syncthreads() (559)
                if const_expr(BM == 128 and _pre_read_wait):
                    rocdl.s_waitcnt(0)                       # diagnostic: drain direct-LDS before ds_read
                asc_cur = issue_a_scale_lds_read(kt)        # issue_a_scale_ds_read(kt)
                a_all = read_a_tile_all(a_slot=read_slot)   # issue_a_ds_read(kt%3)
                b_cur = b_slot[slot_b]
                bsc_cur = bsc_slot[slot_b]
                for J in range_constexpr(4):                # 4x issue_mfma_cluster<J>
                    mfma_cluster_J_all_direct(a_all, b_cur[J], asc_cur, bsc_cur, J)
        if const_expr(inline_quant and BM != 16):
            # BM=32 inline-quant: naive non-pipelined loop (HIP pipeline port pending).
            for kt in range_constexpr(K_TILES):
                if const_expr(kt > 0):
                    rocdl.s_barrier()
                store_a_tile(load_a_tile_inline(kt))
                rocdl.s_barrier()
                mfma_ktile(load_b_tile(kt), load_a_scale(kt), load_b_scale(kt))

        # ===== HIP apply_cshuffle_quant_epilog<N_OUT,BM> (mxfp4_epilogs.hpp:21-145)
        # called at gemm1_a4w4.cuh:567-570 after the final __syncthreads(). FlyDSL
        # inlines the same three phases:
        #   (1) cshuffle accm -> lds_acc          (hpp:39-53)
        #   (2) SiLU*mul + amax(DPP) + e8m0 + 4x cvt_pk_fp4_f32 + nt store (hpp:57-122)
        #   (3) kk==0 A-scale store               (hpp:124-144)
        # =====
        # ---- SiLU*mul + fp4 requant epilogue (increment 3) ----
        # output inter-scale chunk granularity = 16 rows (BM=16) or 32 rows
        # (BM>=32, kSubBlocks 32-row sub-chunks). The rsrc num_records bound must
        # cover max_sorted/_out_chunk_div (+slack); /BM would under-bound BM=128
        # (4 sub-chunks per block) and drop 3/4 of the stores.
        ascale_out_rsrc = _ptr_buffer_resource(
            arg_ascale_out,
            arith.index_cast(
                T.i32,
                (c_max_sorted / arith.constant(_out_chunk_div, index=True)
                 + arith.constant(2, index=True))
                * arith.constant(kOS_per_chunk_dw * 4, index=True),
            ),
        )
        rocdl.s_barrier()   # pre-epilogue __syncthreads() (cuh:567)
        c_BN = arith.constant(BN, index=True)
        _c128 = arith.constant(128, index=True)
        _c32idx = arith.constant(32, index=True)
        # Phase (1) cshuffle == HIP hpp:39-53:
        #   row_base = i*16 + (lane/16)*4; col_local = wave_n*32 + (J/2)*16 + lane%16;
        #   lds_col = (J%2)? 128+col_local : col_local;
        #   lds_acc[(row_base+v)*BN + lds_col] = accm[i][J][v];   (v=0..3)
        # cshuffle: accm[i][J] -> lds_acc; J even=gate (col), J odd=up (128+col)
        for i in range_constexpr(m_repeat):
            row_base = arith.constant(i * 16, index=True) + lane_div_16 * arith.constant(
                4, index=True
            )
            for J in range_constexpr(num_acc_n):
                is_up = (J % 2) == 1
                J_local = J // 2
                col_local = (
                    wave_n * _c32idx
                    + arith.constant(J_local * 16, index=True)
                    + lane_mod_16
                )
                lds_col = (col_local + _c128) if is_up else col_local
                acc_v = accm[i * num_acc_n + J]
                for v in range_constexpr(4):
                    lds_idx = (row_base + arith.constant(v, index=True)) * c_BN + lds_col
                    val = vector.extract(acc_v, static_position=[v], dynamic_position=[])
                    vv = vector.from_elements(T.vec(1, f32), [val])
                    vector.store(vv, lds_acc, [lds_idx], alignment=4)
        rocdl.s_barrier()

        # Phase (2) == HIP hpp:57-122: read gate/up, SiLU*mul (silu_mul_fast =
        # g*rcp(1+exp2(-log2e*g))*u), amax over 8 + DPP quad (0xB1,0x4E), e8m0 via
        # (amax+0x200000)*0.25 >> 23, 4x cvt_scalef32_pk_fp4_f32, nontemporal store.
        # m_lane=tid/16, n_lane=tid%16, wave_grp=n_lane/4, kk=n_lane%4 == HIP (62-65).
        # read gate/up, SiLU*mul, per-32 amax(DPP) -> fp4 + e8m0
        m_lane = tx / _c16
        n_lane = tx % _c16
        wave_grp = n_lane / arith.index(4)
        kk = n_lane % arith.index(4)
        M_REPS = BM // 16
        _neg_log2e = arith.constant(-1.4426950408889634, type=f32)
        _one_f = arith.constant(1.0, type=f32)
        _c7fff = arith.constant(0x7FFFFFFF, type=i32)
        _c2e21 = arith.constant(0x200000, type=i32)
        _quarter = arith.constant(0.25, type=f32)
        _c254i = arith.constant(254, type=i32)
        K_G2_HALF = N_INTER // 2
        scales_per_mr = []
        for mr in range_constexpr(M_REPS):
            row_local = arith.constant(mr * 16, index=True) + m_lane
            gate_col0 = wave_grp * _c32idx + kk * arith.index(8)
            results = []
            for e in range_constexpr(8):
                gcol = gate_col0 + arith.constant(e, index=True)
                g = vector.extract(
                    vector.load_op(T.vec(1, f32), lds_acc, [row_local * c_BN + gcol]),
                    static_position=[0], dynamic_position=[],
                )
                u = vector.extract(
                    vector.load_op(
                        T.vec(1, f32), lds_acc, [row_local * c_BN + gcol + _c128]
                    ),
                    static_position=[0], dynamic_position=[],
                )
                t = g * _neg_log2e
                emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                sig = llvm.call_intrinsic(
                    f32, "llvm.amdgcn.rcp.f32", [_one_f + emu], [], []
                )
                results.append((g * sig) * u)
            # amax over 8 + quad (kk) cross-lane reduction
            local_max = (results[0].bitcast(i32) & _c7fff).bitcast(f32)
            for e in range_constexpr(1, 8):
                local_max = arith.maximumf(
                    local_max, (results[e].bitcast(i32) & _c7fff).bitcast(f32)
                )
            # quad cross-lane amax via DPP mov (== HIP epilogue mov_dpp 0xB1/0x4E +
            # fmaxf, mxfp4_epilogs.hpp:96-101), not shuffle_xor which lowers to
            # ds_swizzle_b32. Keeps the cross-lane reduce on the VALU/DPP path.
            _e_i32ty = ir.IntegerType.get_signless(32)
            _LMV = type(local_max)
            _lm_i = local_max.bitcast(i32)
            _p = rocdl.update_dpp(_e_i32ty, _lm_i, _lm_i, 0xB1, 0xF, 0xF, True)
            local_max = arith.maximumf(local_max, _LMV(_p).bitcast(f32))
            _lm_i = local_max.bitcast(i32)
            _p = rocdl.update_dpp(_e_i32ty, _lm_i, _lm_i, 0x4E, 0xF, 0xF, True)
            local_max = arith.maximumf(local_max, _LMV(_p).bitcast(f32))
            # HIP quant_scale (non-pow2) + HW cvt -> bit-exact with HIP:
            #   quant_scale = uint(amax + 0x200000) * 0.25
            #   e8m0 = min(uint(quant_scale) >> 23, 254)
            #   fp4 = cvt_scalef32_pk_fp4_f32(results, quant_scale)
            amax_i = local_max.bitcast(i32)
            quant_scale = ((amax_i + _c2e21).bitcast(f32)) * _quarter
            e8m0_biased = arith.minsi(
                quant_scale.bitcast(i32) >> arith.constant(23, type=i32), _c254i
            )
            scales_per_mr.append(e8m0_biased)
            if const_expr(debug_bf16):
                col0 = n_block * _c128 + wave_grp * _c32idx + kk * arith.index(8)
                dbg_base = (m_row + row_local) * arith.constant(
                    N_INTER, index=True
                ) + col0
                if const_expr(debug_scaled):
                    bf16_vals = [
                        arith.trunc_f(T.bf16, arith.divf(results[e], quant_scale))
                        for e in range_constexpr(8)
                    ]
                else:
                    bf16_vals = [arith.trunc_f(T.bf16, results[e])
                                 for e in range_constexpr(8)]
                v8 = vector.from_elements(T.vec(8, T.bf16), bf16_vals)
                buffer_ops.buffer_store(
                    v8, aqout_rsrc, dbg_base * arith.index(2), offset_is_bytes=True
                )
                continue
            packed = arith.constant(0, type=i32)
            for k in range_constexpr(4):
                packed = llvm.call_intrinsic(
                    i32, "llvm.amdgcn.cvt.scalef32.pk.fp4.f32",
                    [packed, results[2 * k], results[2 * k + 1], quant_scale,
                     arith.constant(k, type=i32)], [], [],
                )
            byte_pos = (
                n_block * arith.constant(64, index=True)
                + wave_grp * _c16 + kk * arith.index(4)
            )
            out_row = m_row + row_local
            aq_off = out_row * arith.constant(K_G2_HALF, index=True) + byte_pos
            _aqout_base = arith.index_cast(
                ir.IndexType.get(), fx.ptrtoint(arg_aq_out)
            )
            _aq_addr = arith.index_cast(T.i64, _aqout_base + aq_off)
            _aq_ptr = llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _aq_addr)
            llvm.StoreOp(
                packed._value if hasattr(packed, "_value") else packed,
                _aq_ptr, alignment=4,
            )

        # Phase (3) == HIP hpp:124-144: scale store (kk==0): make_preshuffle inter
        # scale. ku = n_block>>1, ikxdl = n_block&1; BM=16 writes the LOW byte only
        # (dword_off*4 + ikxdl*2), BM>=32 packs the e8m0 pair of rows {2sub,2sub+1}.
        if const_expr(debug_bf16):
            return
        kk_i32 = arith.index_cast(i32, kk)
        is_w = arith.cmpi(CmpIPredicate.eq, kk_i32, arith.constant(0, type=i32))
        _ifw = scf.IfOp(is_w)
        with ir.InsertionPoint(_ifw.then_block):
            ku = n_block / arith.index(2)
            ikxdl = n_block % arith.index(2)
            ikxdl_b = arith.index_cast(i32, ikxdl)
            _ikx_byte = arith.index_cast(
                ir.IndexType.get(), ikxdl_b * arith.constant(2, type=i32)
            )
            if const_expr(BM == 16):
                dword_off = (
                    m_block * arith.constant(kOS_per_chunk_dw, index=True)
                    + ku * _c64 + wave_grp * _c16 + m_lane
                )
                sc_byte = dword_off * arith.index(4) + _ikx_byte
                s0 = arith.TruncIOp(T.i8, scales_per_mr[0])
                buffer_ops.buffer_store(
                    s0, ascale_out_rsrc, sc_byte, offset_is_bytes=True
                )
            else:
                # BM>=32: kSubBlocks 32-row sub-chunks. chunk = m_block*kSubBlocks
                # + sub; each sub packs e8m0 of rows {2*sub, 2*sub+1} (u16).
                for sub in range_constexpr(kSubBlocks):
                    chunk = (
                        m_block * arith.constant(kSubBlocks, index=True)
                        + arith.constant(sub, index=True)
                    )
                    dword_off = (
                        chunk * arith.constant(kOS_per_chunk_dw, index=True)
                        + ku * _c64 + wave_grp * _c16 + m_lane
                    )
                    sc_byte = dword_off * arith.index(4) + _ikx_byte
                    pair = arith.TruncIOp(
                        T.i16,
                        scales_per_mr[sub * 2]
                        | (scales_per_mr[sub * 2 + 1]
                           << arith.constant(8, type=i32)),
                    )
                    buffer_ops.buffer_store(
                        pair, ascale_out_rsrc, sc_byte, offset_is_bytes=True
                    )
            scf.YieldOp([])

        # close the device-side grid guard opened before the GEMM loop.
        if const_expr(not debug_bf16):
            scf.YieldOp([])
            _guard_ip.__exit__(None, None, None)

    @flyc.jit
    def launch_gemm1(
        arg_a: fx.Pointer,
        arg_a_scale: fx.Pointer,
        arg_b: fx.Pointer,
        arg_b_scale: fx.Pointer,
        arg_sorted_expert_ids: fx.Pointer,
        arg_cumsum: fx.Pointer,
        arg_m_indices: fx.Pointer,
        arg_aq_out: fx.Pointer,
        arg_ascale_out: fx.Pointer,
        arg_hidden: fx.Pointer,
        i32_tokens: fx.Int32,
        i32_max_sorted: fx.Int32,
        i32_total_m_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        gx = arith.index_cast(ir.IndexType.get(), i32_total_m_blocks.ir_value())
        # 1-D grid of total tiles (= total_m_blocks * num_n_blocks), decoded to
        # (m,n) on-device as m=pid/num_n, n=pid%num_n (== HIP). Keeps valid blocks
        # contiguous in pid for even 1-per-CU distribution (vs the 2-D grid that
        # clustered valid blocks onto some CUs -> straggler CUs at small M).
        g_total = gx * arith.constant(num_n_blocks, index=True)
        gemm1_kernel(
            arg_a, arg_a_scale, arg_b, arg_b_scale,
            arg_sorted_expert_ids, arg_cumsum, arg_m_indices,
            arg_aq_out, arg_ascale_out, arg_hidden,
            i32_tokens, i32_max_sorted,
            # Pin occupancy to match HIP __launch_bounds__(256, 3): emit the
            # LLVM "amdgpu-waves-per-eu"="3" function attribute (via the
            # rocdl.waves_per_eu gpu.func attr -> rocdl-attach-target). Without
            # it FlyDSL left waves_per_eu unset (compiler default ~4 from VGPR).
            # value_attrs={"rocdl.waves_per_eu": 3},
        ).launch(
            grid=(g_total, 1, 1),
            block=(total_threads, 1, 1),
            # LDS lives in the allocator's static group-segment global (== allocator.ptr,
            # 128KB for BM=128 via the union); no extra dynamic LDS. smem= is additive
            # to the static fixed size, so passing the static size again would double it
            # (262KB > 160KB cap -> INVALID_ALLOCATION).
            smem=0,
            stream=stream,
        )

    return launch_gemm1


def get_arch():
    from flydsl.runtime.device import get_rocm_arch
    return get_rocm_arch()
