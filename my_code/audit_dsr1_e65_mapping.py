#!/usr/bin/env python3
"""Audit task coverage and expert bounds for the proposed G1 schedules on CPU."""

import bisect
from collections import Counter
import json
import random


def task_coordinates(m_tiles, tasks, group_m, cluster_n=4):
    n_tiles = 16
    assert tasks == 1 or cluster_n == 4
    for block in range(m_tiles * n_tiles // tasks):
        local_n = block % cluster_n
        if tasks > 1:
            groups = 4 // tasks
            m_tile = block // (cluster_n * groups)
            n_group = (block // cluster_n) % groups
            for task in range(tasks):
                yield m_tile, (n_group * tasks + task) * cluster_n + local_n
        else:
            swizzle = block // cluster_n
            n_units = n_tiles // cluster_n
            group = swizzle // (n_units * group_m)
            first_m = group * group_m
            in_group = swizzle - group * (n_units * group_m)
            active_m = min(m_tiles - first_m, group_m)
            m_tile = first_m + in_group % active_m
            n_unit = in_group // active_m
            yield m_tile, n_unit * cluster_n + local_n


def audit_quant_addresses():
    """Check every produced M16/M32 byte against the external t256 layout."""
    checks = 0
    for tile_m, m_warp in ((64, 2), (128, 2), (256, 2), (128, 1)):
        rows_per_wave = tile_m // m_warp
        m16_per_wave = rows_per_wave // 16
        q_scale_base = (tile_m // 16) * 4 * 256
        # Include a tile crossing an expert's final partial M16/M32 groups.
        for block_m in (0, 256, 1536):
            for block_n in range(0, 4096, 256):
                for wave_m in range(m_warp):
                    for wave_n in range(4):
                        wave = wave_m * 4 + wave_n
                        for wm in range(m16_per_wave):
                            for lane in range(16):
                                row = block_m + wave_m * rows_per_wave + wm * 16 + lane
                                mx = block_n // 64 + wave_n
                                for byte in range(16):
                                    logical = (row // 16) * 16384 + mx * 256 + (row % 16) * 16 + byte
                                    tensor = ((block_m // 16 + wave_m * m16_per_wave + wm) * 16384
                                              + (block_n // 64 + wave_n) * 256 + lane * 16 + byte)
                                    staged = (wave * m16_per_wave + wm) * 256 + lane * 16 + byte
                                    assert logical == tensor
                                    assert 0 <= staged < q_scale_base
                                    checks += 1
                                logical_scale = (row // 32) * 2048 + (mx // 4) * 128 + (row % 32) * 4 + mx % 4
                                if tile_m == 256:
                                    direct = ((row // 32) * 16 * 32 + (mx // 4) * 32 + row % 32) * 4 + mx % 4
                                    assert logical_scale == direct
                                else:
                                    tensor = ((block_m // 32 + wave_m * (m16_per_wave // 2) + wm // 2) * 2048
                                              + (block_n // 256) * 128 + (wm % 2 * 16 + lane) * 4 + wave_n)
                                    assert logical_scale == tensor
                                    staged = q_scale_base + (wave_m * m16_per_wave + wm) * 64 + lane * 4 + wave_n
                                    assert q_scale_base <= staged < q_scale_base + m_warp * m16_per_wave * 64
                                checks += 1
    return checks


def audit_input_rings():
    """Prove the candidate issue schedule never overwrites an unread K tile.

    This checks logical ordering only. The source keeps the actual tensor and
    workgroup fences, whose runtime behavior needs GPU replay validation.
    """
    checks = 0
    for tile_k in (128, 256):
        k_tiles = 7168 // tile_k
        halves = tile_k // 128
        for buffers in (3, 4):
            ring = [None] * buffers
            loaded = set()
            for kt in range(buffers):
                ring[kt] = kt
            loaded.add((0, 0))
            for kt in range(k_tiles):
                assert (kt, 0) in loaded
                for half in range(halves):
                    assert (kt, half) in loaded
                    if half + 1 < halves:
                        assert ring[kt % buffers] == kt
                        loaded.add((kt, half + 1))
                    else:
                        if kt < k_tiles - buffers:
                            future = kt + buffers
                            slot = future % buffers
                            old = ring[slot]
                            assert old == kt
                            assert all((old, h) in loaded for h in range(halves))
                            ring[slot] = future
                        if kt + 1 < k_tiles:
                            assert ring[(kt + 1) % buffers] == kt + 1
                            loaded.add((kt + 1, 0))
                    checks += 1
            assert len(loaded) == 56
    return checks


def audit_input_addresses():
    """Decode candidate LDS reads using the public FP4/M32 permutations.

    The reference is the inverse of shuffle_weight_f4 / shuffle_scale_f4,
    independent of the consumer's fragment coordinates. Check every byte of
    each unique A/B/scale fragment, all K tiles and both register subtiles.
    This does not validate device visibility or the backend's pointer lowering.
    """
    model_k, weight_n = 7168, 4096
    payload_checks = scale_checks = 0

    def payload_coordinate(global_byte):
        row16, remainder = divmod(global_byte, model_k // 2 * 16)
        k16, inner = divmod(remainder, 256)
        row, packed_k = divmod(inner, 16)
        return row16 * 16 + row, k16 * 16 + packed_k

    def scale_coordinate(global_byte):
        row32, remainder = divmod(global_byte, model_k)
        k4, inner = divmod(remainder, 128)
        row, scale_k = divmod(inner, 4)
        return row32 * 32 + row, k4 * 4 + scale_k

    def payload_source(lds_byte, row_pitch, source_row16, k_tile, tile_k):
        row16, padded_inner = divmod(lds_byte, row_pitch)
        chunk, inner = divmod(padded_inner, 1040)
        assert inner < 1024, "operand touches a TDM padding byte"
        logical_inner = chunk * 1024 + inner
        assert logical_inner < tile_k // 2 * 16
        global_byte = ((source_row16 + row16) * (model_k // 2 * 16)
                       + k_tile * (tile_k // 2 * 16) + logical_inner)
        return payload_coordinate(global_byte)

    for tile_m, m_warp in ((64, 2), (128, 2), (256, 2), (128, 1)):
        warp_m = tile_m // m_warp
        m_reps = warp_m // 16
        for tile_k in (128, 256):
            row_pitch = tile_k // 2 * 16 // 1024 * 1040
            # Use distinct expert and M/N origins to expose offset mixing.
            block_m = 1536 + (128 if tile_m == 128 else 64 if tile_m == 64 else 0)
            block_n, expert = 768, 2
            for kt in range(model_k // tile_k):
                for ksl in range(tile_k // 128):
                    for wave_m in range(m_warp):
                        for wm in range(m_reps):
                            row16 = (wave_m * warp_m // 16) + wm
                            for lane16 in range(16):
                                for kgrp in range(2):
                                    for half in range(2):
                                        for byte in range(16):
                                            lds_byte = (row16 * row_pitch + ksl * 1040
                                                        + half * 512 + kgrp * 256
                                                        + lane16 * 16 + byte)
                                            actual = payload_source(lds_byte, row_pitch,
                                                                    block_m // 16, kt, tile_k)
                                            expected = (block_m + row16 * 16 + lane16,
                                                        kt * tile_k // 2 + ksl * 64
                                                        + half * 32 + kgrp * 16 + byte)
                                            assert actual == expected, (tile_m, tile_k, actual, expected)
                                            payload_checks += 1
                        for sm in range((m_reps + 1) // 2):
                            row32 = wave_m * warp_m // 32 + sm
                            for lane in range(32):
                                for byte in range(4):
                                    global_byte = ((block_m // 32 + row32) * model_k
                                                   + kt * tile_k + (ksl * 32 + lane) * 4 + byte)
                                    actual = scale_coordinate(global_byte)
                                    expected = (block_m + row32 * 32 + lane,
                                                kt * tile_k // 32 + ksl * 4 + byte)
                                    assert actual == expected, (tile_m, tile_k, actual, expected)
                                    scale_checks += 1
                    for wave_n in range(4):
                        for wn in range(2):
                            for row_half in range(2):
                                row16 = wave_n * 4 + wn * 2 + row_half
                                for lane16 in range(16):
                                    for kgrp in range(2):
                                        for half in range(2):
                                            for byte in range(16):
                                                lds_byte = (row16 * row_pitch + ksl * 1040
                                                            + half * 512 + kgrp * 256
                                                            + lane16 * 16 + byte)
                                                actual = payload_source(
                                                    lds_byte, row_pitch,
                                                    (expert * weight_n + block_n) // 16, kt, tile_k)
                                                expected = (expert * weight_n + block_n
                                                            + row16 * 16 + lane16,
                                                            kt * tile_k // 2 + ksl * 64
                                                            + half * 32 + kgrp * 16 + byte)
                                                assert actual == expected, (tile_m, tile_k, actual, expected)
                                                payload_checks += 1
                        for sn in range(2):
                            row32 = wave_n * 2 + sn
                            for lane in range(32):
                                for byte in range(4):
                                    global_byte = (((expert * weight_n + block_n) // 32 + row32)
                                                   * model_k + kt * tile_k
                                                   + (ksl * 32 + lane) * 4 + byte)
                                    actual = scale_coordinate(global_byte)
                                    expected = (expert * weight_n + block_n + row32 * 32 + lane,
                                                kt * tile_k // 32 + ksl * 4 + byte)
                                    assert actual == expected, (tile_m, tile_k, actual, expected)
                                    scale_checks += 1
    return {"payload_bytes": payload_checks, "scale_bytes": scale_checks}


def audit_row_tail():
    """Every live M16 group executes the complete original K128 update list."""
    checks = 0
    saved_groups = 0
    range_checks = 0
    pair_checks = 0
    for tile_m, m_warp in ((256, 2), (256, 4)):
        rows_per_wave = tile_m // m_warp
        for live_rows in range(tile_m + 1):
            groups = []
            for wave_m in range(m_warp):
                reps = rows_per_wave // 16
                split = (reps + 1) // 2
                for wms in (list(range(split)), list(range(split, reps))):
                    starts = [wave_m * rows_per_wave + wm * 16 for wm in wms]
                    expected = [wm for wm, first in zip(wms, starts) if first < live_rows]
                    actual_range = (wms if starts[-1] < live_rows else
                                    [wm for wm, first in zip(wms, starts) if first < live_rows]
                                    if starts[0] < live_rows else [])
                    assert actual_range == expected
                    range_checks += 1
                    actual_pairs = []
                    for i in range(0, len(wms), 2):
                        if starts[i] < live_rows:
                            actual_pairs.extend(wms[i:i + 2])
                    assert all(wm in actual_pairs for wm in expected)
                    assert all(wm in wms for wm in actual_pairs)
                    assert len(actual_pairs) - len(expected) <= 1
                    pair_checks += 1
                for wm in range(rows_per_wave // 16):
                    first = wave_m * rows_per_wave + wm * 16
                    executes = first < live_rows
                    actual = range(first, min(first + 16, live_rows)) if executes else ()
                    groups.extend(actual)
                    if not executes:
                        assert not any(row < live_rows for row in range(first, first + 16))
                        saved_groups += 1
                    else:
                        # No valid accumulator loses a K128 contribution.
                        assert tuple(range(0, 7168, 128)) == tuple(k * 128 for k in range(56))
                    checks += 1
            assert groups == list(range(live_rows))
    return {"group_checks": checks, "range_checks": range_checks,
            "pair_checks": pair_checks,
            "wholly_padded_groups": saved_groups}


def audit_compact_n():
    """N128/w2x2 keeps complete N4096 coverage and canonical output bytes."""
    tile_m, tile_n, n_warp = 256, 128, 2
    coverage = Counter()
    payload_checks = scale_checks = 0
    for block_m in (0, 256, 1536):
        for block_n in range(0, 4096, tile_n):
            for wave_m in range(2):
                for wave_n in range(n_warp):
                    wave = wave_m * n_warp + wave_n
                    for wm in range(8):
                        for lane in range(16):
                            row = block_m + wave_m * 128 + wm * 16 + lane
                            mx = block_n // 64 + wave_n
                            coverage[row, mx] += 1
                            for byte in range(16):
                                logical = row // 16 * 16384 + mx * 256 + row % 16 * 16 + byte
                                tensor = ((block_m // 16 + wave_m * 8 + wm) * 16384
                                          + (block_n // 64 + wave_n) * 256 + lane * 16 + byte)
                                staged = (wave * 8 + wm) * 256 + lane * 16 + byte
                                assert logical == tensor
                                assert 0 <= staged < (tile_m // 16) * (tile_n // 64) * 256
                                payload_checks += 1
                            logical_scale = row // 32 * 2048 + mx // 4 * 128 + row % 32 * 4 + mx % 4
                            direct_scale = ((row // 32) * 16 * 32 + mx // 4 * 32 + row % 32) * 4 + mx % 4
                            assert logical_scale == direct_scale
                            scale_checks += 1
    expected = Counter((row, mx) for base in (0, 256, 1536)
                       for row in range(base, base + 256) for mx in range(64))
    assert coverage == expected
    return {"row_mx_pairs": len(coverage), "payload_bytes": payload_checks,
            "scale_bytes": scale_checks}


def audit_interleaved_m():
    """Alternate M16 groups, preserving original A/ScaleA and output bytes."""
    coverage = Counter()
    payload_checks = scale_checks = input_checks = row_checks = 0
    for block_m in (0, 256, 1536):
        for block_n in range(0, 4096, 256):
            for wave_m in range(2):
                for wave_n in range(4):
                    wave = wave_m * 4 + wave_n
                    for wm in range(8):
                        m16 = wm * 2 + wave_m
                        for lane16 in range(16):
                            row = block_m + m16 * 16 + lane16
                            mx = block_n // 64 + wave_n
                            coverage[row, mx] += 1
                            for byte in range(16):
                                logical = row // 16 * 16384 + mx * 256 + row % 16 * 16 + byte
                                tdm = ((block_m // 16 + wave_m + wm * 2) * 16384
                                       + mx * 256 + lane16 * 16 + byte)
                                staged = (wave * 8 + wm) * 256 + lane16 * 16 + byte
                                assert logical == tdm and 0 <= staged < 16384
                                payload_checks += 1
                            logical = row // 32 * 2048 + mx // 4 * 128 + row % 32 * 4 + mx % 4
                            direct = ((row // 32) * 16 * 32 + mx // 4 * 32 + row % 32) * 4 + mx % 4
                            staged = 16384 + (m16 * 16 + lane16) * 4 + wave_n
                            scale_tdm = (16384 + (row % 256 // 128) * 512
                                         + (row % 128 // 32) * 128
                                         + (row % 32) * 4 + wave_n)
                            assert direct == logical and staged == scale_tdm
                            scale_checks += 1
    expected = Counter((row, mx) for base in (0, 256, 1536)
                       for row in range(base, base + 256) for mx in range(64))
    assert coverage == expected
    # Both halves of each lane-pair load the selected M16 group's scales into
    # opsel0. The operand encoding uses that lower half with scaleBType=0.
    for kt in range(28):
        for ksl in range(2):
            for wave_m in range(2):
                for wm in range(8):
                    for lane in range(32):
                        m16 = wm * 2 + wave_m
                        lane16, kgrp = lane % 16, lane // 16
                        for half in range(2):
                            for byte in range(16):
                                lds = (m16 * 2080 + ksl * 1040 + half * 512
                                       + kgrp * 256 + lane16 * 16 + byte)
                                row16, rem = divmod(lds, 2080)
                                chunk, inner = divmod(rem, 1040)
                                assert inner < 1024
                                global_byte = row16 * 57344 + kt * 2048 + chunk * 1024 + inner
                                row16, rem = divmod(global_byte, 57344)
                                mx, inner = divmod(rem, 256)
                                row, packed_k = divmod(inner, 16)
                                assert (row16 * 16 + row, mx * 16 + packed_k) == (
                                    m16 * 16 + lane16, kt * 128 + ksl * 64
                                    + half * 32 + kgrp * 16 + byte)
                                input_checks += 1
                        for byte in range(4):
                            g = wm * 7168 + kt * 256 + (ksl * 32 + lane16 + wave_m * 16) * 4 + byte
                            row32, rem = divmod(g, 7168)
                            k4, inner = divmod(rem, 128)
                            row, scale_k = divmod(inner, 4)
                            assert (row32 * 32 + row, k4 * 4 + scale_k) == (
                                m16 * 16 + lane16, kt * 8 + ksl * 4 + byte)
                            input_checks += 1
    for live_rows in range(257):
        actual = Counter()
        group_counts = []
        for wave_m in range(2):
            groups = 0
            for wm in range(8):
                first = (wm * 2 + wave_m) * 16
                if first < live_rows:
                    actual.update(range(first, min(first + 16, live_rows)))
                    assert len(range(0, 7168, 128)) == 56
                    groups += 1
                row_checks += 1
            group_counts.append(groups)
        assert actual == Counter(range(live_rows))
        assert abs(group_counts[0] - group_counts[1]) <= 1
    return {"row_mx_pairs": len(coverage), "payload_bytes": payload_checks,
            "scale_bytes": scale_checks, "input_bytes": input_checks,
            "row_groups": row_checks}


def audit_interleaved_m32():
    """Paired M16 groups retain ScaleA's complete M32/opsel organization."""
    payload_checks = scale_checks = input_checks = row_checks = 0
    coverage = Counter()
    for block_m in (0, 256, 1536):
        for block_n in range(0, 4096, 256):
            for wave_m in range(2):
                for wave_n in range(4):
                    wave = wave_m * 4 + wave_n
                    for wm in range(8):
                        m16 = (wm // 2) * 4 + wave_m * 2 + wm % 2
                        for lane16 in range(16):
                            row = block_m + m16 * 16 + lane16
                            mx = block_n // 64 + wave_n
                            coverage[row, mx] += 1
                            for byte in range(16):
                                logical = row // 16 * 16384 + mx * 256 + lane16 * 16 + byte
                                tensor = ((block_m // 16 + wave_m * 2) * 16384
                                          + (wm // 2) * 65536 + (wm % 2) * 16384
                                          + mx * 256 + lane16 * 16 + byte)
                                out_wave = (m16 // 8) * 4 + wave_n
                                out_wm = m16 % 8
                                staged = (out_wave * 8 + out_wm) * 256 + lane16 * 16 + byte
                                canonical_stage = ((m16 // 8 * 4 + wave_n) * 8
                                                   + m16 % 8) * 256 + lane16 * 16 + byte
                                assert staged == canonical_stage
                                assert logical == tensor and 0 <= staged < 16384
                                payload_checks += 1
                            logical = row // 32 * 2048 + mx // 4 * 128 + row % 32 * 4 + mx % 4
                            staged = 16384 + (m16 * 16 + lane16) * 4 + wave_n
                            scale_tdm = (16384 + (row % 256 // 128) * 512
                                         + (row % 128 // 32) * 128 + row % 32 * 4 + wave_n)
                            assert staged == scale_tdm
                            assert logical == (row // 32 * 16 * 32 + mx // 4 * 32 + row % 32) * 4 + mx % 4
                            scale_checks += 1
    expected = Counter((row, mx) for base in (0, 256, 1536)
                       for row in range(base, base + 256) for mx in range(64))
    assert coverage == expected
    for kt in range(28):
        for ksl in range(2):
            for wave_m in range(2):
                for wm in range(8):
                    m16 = (wm // 2) * 4 + wave_m * 2 + wm % 2
                    for lane in range(32):
                        lane16, kgrp = lane % 16, lane // 16
                        for half in range(2):
                            for byte in range(16):
                                lds = m16 * 2080 + ksl * 1040 + half * 512 + kgrp * 256 + lane16 * 16 + byte
                                row16, rem = divmod(lds, 2080)
                                chunk, inner = divmod(rem, 1040)
                                assert inner < 1024
                                global_byte = row16 * 57344 + kt * 2048 + chunk * 1024 + inner
                                row16, rem = divmod(global_byte, 57344)
                                mx, inner = divmod(rem, 256)
                                row, packed_k = divmod(inner, 16)
                                assert (row16 * 16 + row, mx * 16 + packed_k) == (
                                    m16 * 16 + lane16, kt * 128 + ksl * 64 + half * 32 + kgrp * 16 + byte)
                                input_checks += 1
                        # One load covers both M16 halves; the unchanged opsel
                        # selects the lower/upper 16 lanes for this wm pair.
                        for byte in range(4):
                            g = ((wave_m + 2 * (wm // 2)) * 7168 + kt * 256
                                 + (ksl * 32 + lane) * 4 + byte)
                            row32, rem = divmod(g, 7168)
                            k4, inner = divmod(rem, 128)
                            row, scale_k = divmod(inner, 4)
                            assert (row32 * 32 + row, k4 * 4 + scale_k) == (
                                m16 // 2 * 32 + lane, kt * 8 + ksl * 4 + byte)
                            if lane // 16 == wm % 2:
                                assert row32 * 32 + row == m16 * 16 + lane16
                            input_checks += 1
    for live_rows in range(257):
        actual = Counter()
        counts = []
        for wave_m in range(2):
            groups = 0
            for wm in range(8):
                first = ((wm // 2) * 4 + wave_m * 2 + wm % 2) * 16
                if first < live_rows:
                    actual.update(range(first, min(first + 16, live_rows)))
                    groups += 1
                row_checks += 1
            counts.append(groups)
        assert actual == Counter(range(live_rows)) and abs(counts[0] - counts[1]) <= 2
    return {"row_mx_pairs": len(coverage), "payload_bytes": payload_checks,
            "scale_bytes": scale_checks, "input_bytes": input_checks,
            "row_groups": row_checks}


def main():
    coverage_checks = expert_checks = search_checks = subtile_owner_checks = 0
    # Include all final swizzle-group remainders and the actual capacity.
    for m_tiles in (*range(1, 65), 119, 238, 476):
        expected = Counter((m, n) for m in range(m_tiles) for n in range(16))
        for tasks in (1, 2, 4):
            for group_m in (1, 4, 8, 16, 32):
                actual = Counter(task_coordinates(m_tiles, tasks, group_m))
                assert actual == expected, (m_tiles, tasks, group_m)
                coverage_checks += 1
        for cluster_n in (1, 2):
            for group_m in (1, 4, 8, 16, 32):
                actual = Counter(task_coordinates(m_tiles, 1, group_m, cluster_n))
                assert actual == expected, (m_tiles, cluster_n, group_m)
                coverage_checks += 1
    rng = random.Random(17)
    cases = [[0] * 65, [1] * 65, [0] * 64 + [1536],
             [257] * 64 + [1536]]
    cases.extend([[rng.randrange(0, 513) for _ in range(64)] + [1536]
                  for _ in range(100)])
    for counts in cases:
        ends = []
        start = 0
        for count in counts:
            end = start + count
            ends.append(end)
            start = ((end + 255) // 256) * 256
        for tile_m in (64, 128, 256):
            for block_m in range(0, start + 1024, tile_m):
                upper = bisect.bisect_right(ends, block_m)
                lo, hi = 0, len(ends)
                for _ in range(len(ends).bit_length()):
                    mid = (lo + hi) >> 1
                    right = ends[min(mid, len(ends) - 1)] <= block_m
                    lo = mid + 1 if right else lo
                    hi = hi if right else mid
                assert (lo == upper if upper < len(ends) else lo >= len(ends))
                search_checks += 1
                parallel = sum(end <= block_m for end in ends)
                assert upper == parallel
                if upper < len(ends):
                    expected_end = ends[upper]
                    selected_end = min(end for end in ends if end > block_m)
                else:
                    expected_end = selected_end = max(ends)
                assert selected_end == expected_end
                # Every live row of an internal M64/M128 tile belongs to one
                # original 256-aligned expert interval.
                for row in range(block_m, min(block_m + tile_m, expected_end)):
                    assert bisect.bisect_right(ends, row) == upper
                expert_checks += 1
                parent_m = block_m // 256 * 256
                parent_expert = bisect.bisect_right(ends, parent_m)
                live_rows = []
                for expert, count in enumerate(counts):
                    begin = 0 if expert == 0 else (ends[expert - 1] + 255) // 256 * 256
                    live_rows.extend((row, expert) for row in range(
                        max(begin, block_m), min(ends[expert], block_m + tile_m)))
                if parent_expert < len(ends) and ends[parent_expert] > block_m:
                    selected_rows = [(row, parent_expert) for row in range(
                        block_m, min(ends[parent_expert], block_m + tile_m))]
                else:
                    selected_rows = []
                assert selected_rows == live_rows, (counts, tile_m, block_m)
                subtile_owner_checks += 1
    print(json.dumps({"coverage_checks": coverage_checks,
                      "expert_bound_checks": expert_checks,
                      "minimal_search_checks": search_checks,
                      "parent_aligned_subtile_checks": subtile_owner_checks,
                      "quant_output_byte_checks": audit_quant_addresses(),
                      "input_ring_step_checks": audit_input_rings(),
                      "input_fragment_address_checks": audit_input_addresses(),
                      "row_tail_coverage": audit_row_tail(),
                      "compact_n_coverage": audit_compact_n(),
                      "interleaved_m_coverage": audit_interleaved_m(),
                      "interleaved_m32_coverage": audit_interleaved_m32(),
                      "gpu_dispatch": False,
                      "note": "CPU mapping proof only; GPU byte checks are still required."}, indent=2))


if __name__ == "__main__":
    main()
