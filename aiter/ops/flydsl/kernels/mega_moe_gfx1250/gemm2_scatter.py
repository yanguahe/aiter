# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""BF16 EP scatter builders for the A-preshuffled GEMM2 launchers."""

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T

from .tdm_gather_shim import make_tensor_gather_descriptor, tensor_store_gather

SCATTER_ROWS = 8


def make_bf16_scatter_view(
    arena_handle, combine_offset, slot_stride, world_size, tile_n
):
    """Resolve the symmetric window once, before the GEMM input pipeline."""
    import mori.cco.device.flydsl as cco

    window = cco.Window(fx.Int64(arena_handle))
    base = fx.Int64(window.lsa_ptr(fx.Int32(0), 0))
    peer = fx.Int64(window.lsa_ptr(fx.Int32(1), 0))
    slots_per_rank = fx.Int32((peer - base) // fx.Int64(slot_stride))
    oob = slots_per_rank * fx.Int32(world_size)
    pointer_type = fx.PointerType.get(
        T.i16, address_space=fx.AddressSpace.Global, alignment=16
    )
    pointer = fx.inttoptr(pointer_type, base + fx.Int64(combine_offset))
    view = fx.Tensor(
        fx.make_view(pointer, fx.make_layout((oob, tile_n), (slot_stride // 2, 1)))
    )
    return view, slots_per_rank, oob


def load_bf16_scatter_rows(
    row_map,
    blk_m,
    mn_oob,
    wave,
    lane,
    wmb,
    lane16,
    *,
    wmma_m_rep,
    split_wm,
    m_warp,
    num_waves,
    destination_stride,
    slots_per_rank,
    oob,
    second_slice_wave_shift=0,
):
    """Prefetch route weights and both output slices' destination row indices.

    The destination vectors use logical GEMM rows. Their LDS source rows are
    supplied separately, so the same indices serve ordinary and packed slices.
    All Python loops here have compile-time bounds; validity uses IR selects.
    """
    row_map = fx.recast_iter(fx.Int32, fx.get_iter(row_map))
    weights = []
    for wm in range(wmma_m_rep):
        row = wmb + wm * 16 + lane16
        valid = row < mn_oob
        safe_row = valid.select(row, fx.Int32(0))
        bits = fx.Int32(
            fx.ptr_load(row_map + fx.Int64(blk_m) * 2 + fx.Int64(safe_row) * 2 + 1)
        )
        weights.append(valid.select(bits.bitcast(fx.Float32), fx.Float32(0.0)))

    destinations = []
    warp_rows = wmma_m_rep * 16
    for wm_base, wm_count in ((0, split_wm), (split_wm, wmma_m_rep - split_wm)):
        slice_rows = wm_count * 16
        groups = m_warp * slice_rows // SCATTER_ROWS
        passes = (groups + num_waves - 1) // num_waves
        passes_per_vector = 32 // SCATTER_ROWS
        vectors = (passes + passes_per_vector - 1) // passes_per_vector
        slice_indices = []
        slice_wave = (
            wave if wm_base == 0 else (wave + second_slice_wave_shift) % num_waves
        )
        for vector in range(vectors):
            scatter_pass = lane // SCATTER_ROWS + vector * passes_per_vector
            group = slice_wave + scatter_pass * num_waves
            slice_row = group * SCATTER_ROWS + lane % SCATTER_ROWS
            logical_row = (
                (slice_row // slice_rows) * warp_rows
                + wm_base * 16
                + slice_row % slice_rows
            )
            valid = (scatter_pass < passes) & (group < groups) & (logical_row < mn_oob)
            safe_row = valid.select(logical_row, fx.Int32(0))
            destination = fx.Int32(
                fx.ptr_load(row_map + fx.Int64(blk_m) * 2 + fx.Int64(safe_row) * 2)
            )
            peer = destination // fx.Int32(destination_stride)
            slot = destination % fx.Int32(destination_stride)
            keep = valid & (destination >= fx.Int32(0))
            slice_indices.append(keep.select(peer * slots_per_rank + slot, oob))
        destinations.append(slice_indices)
    return weights, destinations


def store_bf16_scatter_group(
    global_view,
    lds_view,
    destinations,
    group,
    vector_pass,
    *,
    oob,
    tile_n,
    store_pitch,
    slot_stride,
    lds_row,
    column,
):
    """Issue one eight-row TDM store against the peer/slot indexed window."""
    row_indices = [
        fx.Int32(
            rocdl.readlane(
                T.i32,
                _raw(destinations),
                _raw(fx.Int32(vector_pass * SCATTER_ROWS + i)),
            )
        )
        for i in range(SCATTER_ROWS)
    ]
    descriptor = make_tensor_gather_descriptor(
        global_view,
        lds_view,
        row_indices,
        row_width=store_pitch,
        tensor_dim0=tile_n,
        tensor_dim1=oob.ir_value(),
        stride=slot_stride // 2,
        elem_bytes=2,
        index_size=32,
        lds_byte_offset=fx.index_cast(T.index, lds_row * fx.Int32(store_pitch * 2)),
        global_byte_offset=fx.Int64(column) * 2,
    )
    tensor_store_gather(descriptor)
