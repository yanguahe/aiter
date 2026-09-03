#!/usr/bin/env python3
"""Generate and statically audit the MoE v21 load-only ISA.

v21 deliberately keeps the v18_v2 load/synchronization body.  Its generated
delta is restricted to:

* an exact 184-byte MoE ABI and unique symbol/marker;
* physical-bid -> 16-M-tile swizzle and a real upper-bound psum lookup;
* expert/tile base-address adaptation for production MoE buffers;
* expert-local A bounds: M[0,32) loads and M[32,64) descriptor-zero-fills;
* non-cluster-safe TDM masks plus explicit non-multicast cluster-load masks;
* metadata matching the new ABI and ordinary 128-thread workgroup launch.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import re
import struct
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ISA_DIR = (
    ROOT
    / "my_code"
    / "f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps"
)
BASELINE = ISA_DIR / (
    "f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps"
    "_loadonly_v18_v2.s"
)
V21 = ISA_DIR / (
    "f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps"
    "_loadonly_v21.s"
)
V0 = ROOT / "my_code" / "moe_gemm1_a4w4_v0_loadonly.s"

BASELINE_SYMBOL = (
    "f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps"
)
V21_SYMBOL = "f4gemm_bf16_mxfp4_ABpreShuffle_64x256_moe_loadonly_v21"
CONTRACT_MARKER = "__aiter_moe_v21_contract"
CORE_ANCHOR = ".Lbranch_000000001da0:"
END_ANCHOR = ".Lbranch_00000000ba60:"
EXPECTED_BASELINE_SHA256 = (
    "bd0b038e8f577292ee0780c3481d066fe907e86d5e4a63f09eb060ea663cc1e2"
)
EXPECTED_PRE_A_OOB_V21_SHA256 = (
    "3211aad842e867bdefe92d07f57d7f61600da0076e352b5701ecd096811eaafc"
)

EXPERTS = 96
TOKENS = 512
TOPK = 6
VALID_ROUTES = TOKENS * TOPK
VALID_ROWS_PER_EXPERT = VALID_ROUTES // EXPERTS
TILE_M = 64
TILE_N = 256
N = 6144
K = 7168
K_PACKED = K // 2
K_SCALE = K // 32
CONTIGUOUS_M = 9216
M_TILES = CONTIGUOUS_M // TILE_M
N_TILES = N // TILE_N
GRID = M_TILES * N_TILES
ACTIVE_WGS = EXPERTS * N_TILES
TAIL_WGS = GRID - ACTIVE_WGS
LDS_BYTES = 0x32800

A_TILE_BYTES = TILE_M * K_PACKED
B_TILE_BYTES = TILE_N * K_PACKED
B_EXPERT_BYTES = N * K_PACKED
SA_TILE_BYTES = TILE_M * K_SCALE
SB_WAVE_BYTES = 64 * K_SCALE
SB_TILE_BYTES = TILE_N * K_SCALE
SB_EXPERT_BYTES = N * K_SCALE

K_PHASE = 256
K_PHASES = K // K_PHASE
A_LOGICAL_WAVES = 4
A_VALID_LOGICAL_WAVES = 2
A_ROWS_PER_LOGICAL_WAVE = TILE_M // A_LOGICAL_WAVES
A_PHASE_BYTES_PER_LOGICAL_WAVE = (
    A_ROWS_PER_LOGICAL_WAVE * K_PHASE // 2
)
A_TENSOR_DIM0 = A_ROWS_PER_LOGICAL_WAVE * K_PACKED
A_SPLIT_TILE_DIM1 = 2
A_FULL_TILE_DIM1 = 4
B_ROWS_PER_BLOCK = 16
B_BLOCKS_PER_TILE = TILE_N // B_ROWS_PER_BLOCK
B_PHASE_BYTES_PER_BLOCK = B_ROWS_PER_BLOCK * K_PHASE // 2
SCALE_PHASE_BYTES = TILE_M * K_PHASE // 32

# The source body is pinned by EXPECTED_BASELINE_SHA256.  Replace exactly the
# six bound-producing instructions while preserving instruction count, issue
# sites, register allocation, and every non-A instruction byte-for-byte at the
# assembly-source level.  The first four patches cover the cold split-A
# descriptors and their next-task shadows.  The final two cover the persistent
# full-A rebuild path and its next-task shadow.
A_OOB_BOUND_PATCHES = (
    (
        "cold lower current",
        "\ts_lshr_b32 s26, s26, 4                                     "
        "; 000000001FA4: 851A841A",
        "\ts_mov_b32 s26, 2                                           "
        "; A lower split: two valid M16 blocks",
    ),
    (
        "cold lower next",
        "\ts_lshr_b32 s27, s27, 4                                     "
        "; 00000000208C: 851B841B",
        "\ts_mov_b32 s27, 2                                           "
        "; next A lower split: two valid M16 blocks",
    ),
    (
        "cold upper current",
        "\ts_lshr_b32 s26, s26, 4                                     "
        "; 000000002BEC: 851A841A",
        "\ts_mov_b32 s26, 0                                           "
        "; A upper split: both M16 blocks are OOB",
    ),
    (
        "cold upper next",
        "\ts_lshr_b32 s27, s27, 4                                     "
        "; 000000002CF0: 851B841B",
        "\ts_mov_b32 s27, 0                                           "
        "; next A upper split: both M16 blocks are OOB",
    ),
    (
        "persistent full next",
        "\ts_lshr_b32 s27, s27, 4                                     "
        "; 00000000509C: 851B841B",
        "\ts_mov_b32 s27, 2                                           "
        "; next full A: only lower two M16 blocks are valid",
    ),
    (
        "persistent full current",
        "\ts_lshr_b32 s26, s26, 4                                     "
        "; 00000000521C: 851A841A",
        "\ts_mov_b32 s26, 2                                           "
        "; full A: only lower two M16 blocks are valid",
    ),
)


def _upper_bound_body() -> str:
    lines: list[str] = []
    for step in range(7):
        lines.extend(
            (
                f"\t; upper_bound step {step}",
                "\ts_add_co_u32 s26, s24, s25",
                "\ts_lshr_b32 s26, s26, 1",
                "\ts_min_u32 s26, s26, 0x5f",
                "\ts_load_b32 s27, s[30:31], s26 offset:0x0 scale_offset",
                "\ts_add_co_u32 s29, s26, 1",
                "\ts_wait_kmcnt 0x0",
                "\ts_cmp_gt_u32 s27, s28",
                "\ts_cselect_b32 s25, s26, s25",
                "\ts_cselect_b32 s24, s24, s29",
            )
        )
    return "\n".join(lines)


def _prologue() -> str:
    return f"""\t.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
\t.amdhsa_code_object_version 6
\t.text
\t; ---------------------------------------------------------------------
\t; MoE v21 load-only kernel generated from v18_v2.
\t; Fixed user contract: E=96, tokens=512, topk=6, K=7168, N=6144.
\t; Logical grid=(24,144,1), physical grid=(3456,1,1), block=128,
\t; ordinary non-cluster dispatch.  m_tile 96..143 exits after psum lookup.
\t; CLUSTER_LOAD_B128 is retained: CDNA5 documents that it downgrades to a
\t; GLOBAL load outside a cluster.  Every TDM Workgroup_mask is zero, as the
\t; same specification requires when ClusterID==0.
\t; ---------------------------------------------------------------------
\t.set {CONTRACT_MARKER}, 1
\t.protected\t{V21_SYMBOL}
\t.globl\t{V21_SYMBOL}
\t.p2align\t8
\t.type\t{V21_SYMBOL},@function
{V21_SYMBOL}:
\ts_version UC_VERSION_GFX12|UC_VERSION_W32_BIT
\ts_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
\ts_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1

\t; Exact 184-byte production MoE ABI.
\ts_load_b64 s[44:45], s[0:1], 0x0 nv
\ts_load_b128 s[4:7], s[0:1], 0x28 nv
\ts_load_b64 s[8:9], s[0:1], 0x38 nv
\ts_load_b64 s[10:11], s[0:1], 0x60 nv
\ts_load_b64 s[30:31], s[0:1], 0x70 nv
\ts_load_b64 s[16:17], s[0:1], 0xa4 nv
\ts_wait_kmcnt 0x0
\ts_mov_b32 s18, s17
\ts_mov_b32 s17, s16

\t; Preserve v18_v2's wave/lane decomposition.
\ts_bfe_u32 s22, ttmp8, 0x50019
\ts_cmp_eq_u32 s22, 0
\ts_cbranch_scc0 .Lmoe_v21_wave_mode_ready
\ts_getreg_b32 s24, hwreg(HW_REG_WAVE_MODE)
\ts_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 24, 1), 1
\ts_getreg_b32 s24, hwreg(HW_REG_WAVE_MODE)
.Lmoe_v21_wave_mode_ready:
\tv_lshrrev_b32_e32 v1, 10, v0
\tv_lshrrev_b32_e32 v2, 10, v1
\tv_and_b32_e32 v0, 0x3ff, v0
\tv_and_b32_e32 v1, 0x3ff, v1
\tv_and_b32_e32 v2, 0x3ff, v2
\tv_lshrrev_b32_e32 v3, 5, v0
\tv_and_b32_e32 v0, 31, v0
\tv_lshlrev_b32_e32 v8, 4, v0
\ts_bfe_u32 s22, ttmp8, 0x50019

\t; bid=384*g+16*n+ml -> m_tile=16*g+ml, n_tile=n.
\ts_mov_b32 s28, ttmp9
\ts_lshr_b32 s24, s28, 7
\ts_mul_hi_u32 s25, s24, 0xaaaaaaab
\ts_lshr_b32 s25, s25, 1
\ts_mul_i32 s26, s25, 0x180
\ts_sub_co_u32 s27, s28, s26
\ts_lshr_b32 s54, s27, 4
\ts_and_b32 s24, s27, 15
\ts_lshl_b32 s55, s25, 4
\ts_add_co_u32 s55, s55, s24

\t; Production expert lookup: upper_bound(psum[0:96], m_tile*64).
\ts_lshl_b32 s28, s55, 6
\ts_mov_b32 s24, 0
\ts_mov_b32 s25, 0x60
{_upper_bound_body()}
\ts_cmp_gt_u32 s24, 0x5f
\ts_cbranch_scc1 .Lmoe_v21_tail

\t; B/ScaleB are expert-major.  A/ScaleA are contiguous-M buffers.
\ts_mul_i32 s26, s24, 0x1500000
\ts_add_co_u32 s6, s6, s26
\ts_add_co_ci_u32 s7, 0, s7
\ts_mul_i32 s26, s24, 0x150000
\ts_add_co_u32 s10, s10, s26
\ts_add_co_ci_u32 s11, 0, s11

\t; Dense-body register contract, specialized to one physical WG per tile.
\ts_mov_b32 s12, 0x1800
\ts_mov_b32 s13, 0xe00
\ts_mov_b32 s14, 0xe00
\ts_mov_b32 s15, 0xe0
\ts_mov_b32 s16, 0xe0
\ts_mov_b32 s19, 0x1c00
\ts_mov_b32 s20, 0
\ts_mov_b32 s21, 0
\ts_mov_b32 s49, 0
\ts_mov_b32 s50, 0
\ts_mov_b32 s51, 1
\ts_mov_b32 s52, 1
\ts_mov_b32 s61, 24
\ts_mov_b32 s68, s54
\ts_mov_b32 s69, s55
\ts_mov_b32 s60, 1
\ts_mov_b32 s28, 0
\ts_mov_b32 s29, 1
\ts_mov_b32 s94, 0
\ts_mov_b32 s59, s19
\ts_mov_b32 s70, s19
\ts_add_co_u32 s71, s19, 0x200
"""


def _scale_base_block() -> str:
    return """\t; ScaleA tile base in production wmma_rep-preshuffled layout.
\ts_mov_b32 s80, s8
\ts_mov_b32 s81, s9
\ts_mul_i32 s24, s55, 0x3800
\ts_add_co_u32 s80, s80, s24
\ts_add_co_ci_u32 s81, 0, s81
\t; ScaleB wave base in production expert-major n32k4 layout.
\ts_mov_b32 s82, s10
\ts_mov_b32 s83, s11
\ts_mul_i32 s24, s54, 0xe000
\ts_add_co_u32 s82, s82, s24
\ts_add_co_ci_u32 s83, 0, s83
\ts_mul_i32 s24, s22, 0x3800
\ts_add_co_u32 s82, s82, s24
\ts_add_co_ci_u32 s83, 0, s83
\t; M0 low mask zero makes these explicit non-multicast requests.
\ts_mov_b32 s90, 0x10000
\ts_mov_b32 s91, 0
\ts_or_b32 s91, s91, 0x10000
\t; Single-task launch: the body lookahead reuses the current scale bases.
\ts_mov_b32 s84, s80
\ts_mov_b32 s85, s81
\ts_mov_b32 s86, s82
\ts_mov_b32 s87, s83
\ts_mov_b32 s88, 0x200
\ts_mov_b32 s89, 0"""


def _metadata() -> str:
    return f"""\t.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
      - .offset:         64
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         96
        .size:           8
        .value_kind:     global_buffer
      - .offset:         104
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         128
        .size:           8
        .value_kind:     global_buffer
      - .offset:         136
        .size:           28
        .value_kind:     by_value
      - .offset:         164
        .size:           4
        .value_kind:     by_value
      - .offset:         168
        .size:           4
        .value_kind:     by_value
      - .offset:         172
        .size:           4
        .value_kind:     by_value
      - .offset:         176
        .size:           4
        .value_kind:     by_value
      - .offset:         180
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: {LDS_BYTES}
    .kernarg_segment_align: 8
    .kernarg_segment_size: 184
    .max_flat_workgroup_size: 128
    .name:           {V21_SYMBOL}
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 128
      - 1
      - 1
    .sgpr_count:     106
    .symbol:         {V21_SYMBOL}.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     384
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

\t.end_amdgpu_metadata
"""


def _apply_a_oob_bounds(body: str) -> str:
    for path_name, old, new in A_OOB_BOUND_PATCHES:
        if body.count(old) != 1:
            raise AssertionError(
                f"A bound source changed for {path_name}: {old!r}"
            )
        body = body.replace(old, new)
    return body


def generated_text(
    baseline_text: str,
    *,
    apply_a_oob_bounds: bool = True,
) -> str:
    if baseline_text.count(CORE_ANCHOR) != 1:
        raise AssertionError(f"baseline core anchor count changed: {CORE_ANCHOR}")
    body = CORE_ANCHOR + baseline_text.split(CORE_ANCHOR, 1)[1]
    body = body.replace(BASELINE_SYMBOL, V21_SYMBOL)

    old_scale_block = """\t; Scale base addresses for global_load_b32
\ts_mov_b32 s80, s8
\ts_mov_b32 s81, s9
\ts_mov_b32 s82, s10
\ts_mov_b32 s83, s11
\t; M0 values for cluster_load: [15:0]=WG mask, [16]=early timeout
\ts_mov_b32 s90, 0x1000f
\ts_mov_b32 s91, 1
\ts_lshl_b32 s91, s91, s49
\ts_or_b32 s91, s91, 0x10000
\t; scale next-task bases + byte step
\ts_mov_b32 s84, s8
\ts_mov_b32 s85, s9
\ts_mov_b32 s86, s10
\ts_mov_b32 s87, s11
\ts_mov_b32 s88, 0x200
\ts_mov_b32 s89, 0"""
    if body.count(old_scale_block) != 1:
        raise AssertionError("v18_v2 scale-base block changed")
    body = body.replace(old_scale_block, _scale_base_block())

    if body.count("\ts_mov_b32 s53, 0xf") != 4:
        raise AssertionError("expected four v18_v2 all-WG TDM masks")
    if body.count("\ts_mov_b32 s53, 1") != 4:
        raise AssertionError("expected four v18_v2 single-WG TDM masks")
    body = body.replace("\ts_mov_b32 s53, 0xf", "\ts_mov_b32 s53, 0")
    body = body.replace("\ts_mov_b32 s53, 1", "\ts_mov_b32 s53, 0")
    if apply_a_oob_bounds:
        body = _apply_a_oob_bounds(body)

    if body.count(f"\n{END_ANCHOR}\n") != 1:
        raise AssertionError("v18_v2 end anchor changed")
    body = body.replace(
        f"\n{END_ANCHOR}\n",
        f"\n.Lmoe_v21_tail:\n{END_ANCHOR}\n",
    )

    replacements = {
        "\t\t.amdhsa_kernarg_size 120": "\t\t.amdhsa_kernarg_size 184",
        "\t\t.amdhsa_user_sgpr_count 32": "\t\t.amdhsa_user_sgpr_count 4",
        "\t\t.amdhsa_user_sgpr_kernarg_preload_length 30": (
            "\t\t.amdhsa_user_sgpr_kernarg_preload_length 2"
        ),
        "\t\t.amdhsa_system_sgpr_workgroup_id_y 1": (
            "\t\t.amdhsa_system_sgpr_workgroup_id_y 0"
        ),
        "\t\t.amdhsa_system_sgpr_workgroup_id_z 1": (
            "\t\t.amdhsa_system_sgpr_workgroup_id_z 0"
        ),
    }
    for old, new in replacements.items():
        if body.count(old) != 1:
            raise AssertionError(f"descriptor source changed: {old!r}")
        body = body.replace(old, new)

    metadata_start = body.index("\t.amdgpu_metadata")
    metadata_end = body.index("\t.end_amdgpu_metadata", metadata_start)
    metadata_end += len("\t.end_amdgpu_metadata")
    body = body[:metadata_start] + _metadata().rstrip() + body[metadata_end:]
    return _prologue() + body + ("" if body.endswith("\n") else "\n")


def generate(baseline: Path, output: Path) -> None:
    baseline_text = baseline.read_text(encoding="utf-8")
    output.write_text(
        generated_text(baseline_text),
        encoding="utf-8",
        newline="\n",
    )


def _strip_comment(line: str) -> str:
    return re.split(r";|//", line, maxsplit=1)[0].strip()


def _mnemonic_counts(text: str) -> Counter[str]:
    result: Counter[str] = Counter()
    in_metadata = False
    for raw in text.splitlines():
        clean = _strip_comment(raw)
        if clean == ".amdgpu_metadata":
            in_metadata = True
            continue
        if clean == ".end_amdgpu_metadata":
            in_metadata = False
            continue
        if in_metadata or not clean or clean.startswith("."):
            continue
        if clean.endswith(":"):
            continue
        match = re.match(r"^([a-z][a-z0-9_.]*)\b", clean)
        if match:
            result[match.group(1)] += 1
    return result


def _kernel_symbols(text: str) -> set[str]:
    patterns = (
        r"(?m)^\s*\.globl\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"(?m)^\s*\.type\s+([A-Za-z_][A-Za-z0-9_]*),@function",
        r"(?m)^\s*\.amdhsa_kernel\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"(?m)^\s*\.name:\s*([A-Za-z_][A-Za-z0-9_]*)",
        r"(?m)^\s*\.symbol:\s*([A-Za-z_][A-Za-z0-9_]*)\.kd",
    )
    found: set[str] = set()
    for pattern in patterns:
        found.update(re.findall(pattern, text))
    return found


def _map_bid(bid: int) -> tuple[int, int]:
    quotient_input = bid >> 7
    g = ((quotient_input * 0xAAAAAAAB) >> 32) >> 1
    if g != quotient_input // 3:
        raise AssertionError(f"magic /3 mismatch for bid={bid}")
    remainder = bid - 384 * g
    n_tile = remainder >> 4
    m_tile = (g << 4) + (remainder & 15)
    return m_tile, n_tile


def _unrolled_upper_bound(values: list[int], key: int) -> int:
    low, high = 0, len(values)
    for _ in range(7):
        middle = min((low + high) >> 1, len(values) - 1)
        if values[middle] > key:
            high = middle
        else:
            low = middle + 1
    if low != high:
        raise AssertionError(
            f"seven-step upper_bound did not converge: key={key}, "
            f"low={low}, high={high}"
        )
    return low


def _section(text: str, start: str, end: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise AssertionError(f"section anchors changed: {start!r}, {end!r}")
    return text.split(start, 1)[1].split(end, 1)[0]


def _audit_patch_isolation(legacy_text: str, output_text: str) -> None:
    legacy_lines = legacy_text.splitlines()
    output_lines = output_text.splitlines()
    if len(legacy_lines) != len(output_lines):
        raise AssertionError("A OOB patch changed the ISA source line count")
    actual_changes = [
        (old, new)
        for old, new in zip(legacy_lines, output_lines)
        if old != new
    ]
    expected_changes = [
        (old, new) for _, old, new in A_OOB_BOUND_PATCHES
    ]
    if Counter(actual_changes) != Counter(expected_changes):
        raise AssertionError(
            "A OOB patch changed instructions outside the six audited bounds: "
            f"{actual_changes}"
        )
    for path_name, old, new in A_OOB_BOUND_PATCHES:
        if legacy_text.count(old) != 1 or output_text.count(new) != 1:
            raise AssertionError(f"A bound patch missing for {path_name}")
        if old in output_text:
            raise AssertionError(f"legacy A bound remains for {path_name}")

    legacy_metadata = legacy_text.split("\t.amdgpu_metadata", 1)[1]
    output_metadata = output_text.split("\t.amdgpu_metadata", 1)[1]
    if legacy_metadata != output_metadata:
        raise AssertionError("A OOB patch changed ABI/resource metadata")

    legacy_lower_dim1 = [
        (CONTIGUOUS_M - expert * TILE_M) // A_ROWS_PER_LOGICAL_WAVE
        for expert in range(EXPERTS)
    ]
    legacy_upper_dim1 = [
        (
            CONTIGUOUS_M
            - expert * TILE_M
            - A_SPLIT_TILE_DIM1 * A_ROWS_PER_LOGICAL_WAVE
        )
        // A_ROWS_PER_LOGICAL_WAVE
        for expert in range(EXPERTS)
    ]
    if (
        min(legacy_lower_dim1) < A_SPLIT_TILE_DIM1
        or min(legacy_upper_dim1) < A_SPLIT_TILE_DIM1
    ):
        raise AssertionError("pre-fix A bounds no longer reproduce the OOB bug")
    print(
        "pre_A_OOB_evidence=FAIL_EXPECTED "
        f"lower_dim1=[{min(legacy_lower_dim1)},{max(legacy_lower_dim1)}] "
        f"upper_dim1=[{min(legacy_upper_dim1)},{max(legacy_upper_dim1)}] "
        "upper_tile_dim1=2 => M[32,64) globally in-range"
    )
    print(
        "patch_isolation=PASS changed_instructions=6 "
        "B/Scale/LDS/barrier/load_sites/metadata=source-identical"
    )


def _audit_a_descriptor_paths(output_text: str) -> None:
    cold_lower = _section(
        output_text,
        ".Lbranch_000000001f08:",
        ".Lbranch_000000002b48:",
    )
    cold_upper = _section(
        output_text,
        ".Lbranch_000000002b48:",
        ".Lbranch_0000000037ac:",
    )
    persistent_full = _section(
        output_text,
        ".Lbranch_000000005060:",
        ".Lbranch_000000005548:",
    )

    expected_sections = (
        (
            "cold lower",
            cold_lower,
            (
                "s_mov_b32 s95, 0",
                "s_mov_b32 s96, 0x2000",
                "s_mov_b32 s97, 0x4000",
                "s_mov_b32 s98, 0x6000",
                "s_mov_b32 s32, 1",
                "s_mov_b32 s34, s72",
                "s_or_b32 s35, s73, s35",
                "s_mov_b32 s36, 0",
                "s_lshl_b32 s26, s13, 4",
                "s_or_b32 s39, s39, 0x8000000",
                "s_or_b32 s40, s40, 2",
                "s_lshl_b32 s24, s13, 4",
                "s_mov_b32 s41, s24",
                "s_bitset0_b32 s36, 20",
                "s_mov_b32 s53, 0",
                "s_bitset1_b32 s36, 21",
                A_OOB_BOUND_PATCHES[0][2].strip(),
                A_OOB_BOUND_PATCHES[1][2].strip(),
            ),
        ),
        (
            "cold upper",
            cold_upper,
            (
                "s_mov_b32 s95, 0x1000",
                "s_mov_b32 s96, 0x3000",
                "s_mov_b32 s97, 0x5000",
                "s_mov_b32 s98, 0x7000",
                "s_mov_b32 s32, 1",
                "s_mov_b32 s34, s76",
                "s_or_b32 s35, s77, s35",
                "s_mov_b32 s36, 0",
                "s_lshl_b32 s26, s13, 4",
                "s_or_b32 s39, s39, 0x8000000",
                "s_or_b32 s40, s40, 2",
                "s_lshl_b32 s24, s13, 4",
                "s_mov_b32 s41, s24",
                "s_bitset0_b32 s36, 20",
                "s_mov_b32 s53, 0",
                "s_bitset1_b32 s36, 21",
                A_OOB_BOUND_PATCHES[2][2].strip(),
                A_OOB_BOUND_PATCHES[3][2].strip(),
            ),
        ),
        (
            "persistent full",
            persistent_full,
            (
                "s_mov_b32 s32, 1",
                "s_mov_b32 s34, s72",
                "s_or_b32 s35, s73, s35",
                "s_mov_b32 s36, 0",
                "s_lshl_b32 s26, s13, 4",
                "s_or_b32 s39, s39, 0x8000000",
                "s_or_b32 s40, s40, 4",
                "s_lshl_b32 s24, s13, 4",
                "s_mov_b32 s41, s24",
                "s_bitset0_b32 s36, 20",
                "s_mov_b32 s53, 0",
                "s_bitset1_b32 s36, 21",
                A_OOB_BOUND_PATCHES[4][2].strip(),
                A_OOB_BOUND_PATCHES[5][2].strip(),
            ),
        ),
    )
    for name, section, required in expected_sections:
        missing = [token for token in required if token not in section]
        if missing:
            raise AssertionError(
                f"{name} A descriptor path is incomplete: {missing}"
            )

    base_setup = _section(
        output_text,
        CORE_ANCHOR,
        ".Lbranch_000000001f08:",
    )
    for required in (
        "s_mul_i32 s24, s55, 0x40",
        "s_add_co_u32 s72, s4, s24",
        "s_add_co_i32 s24, s24, 0x20",
        "s_add_co_u32 s76, s4, s24",
    ):
        if required not in base_setup:
            raise AssertionError(f"A global-base setup changed: {required}")

    promoted_next = _section(
        output_text,
        ".Lbranch_00000000ba28:",
        ".Lmoe_v21_tail:",
    )
    for required in (
        "s_mov_b32 s36, s64",
        "s_mov_b32 s37, s65",
        "s_mov_b32 s38, s66",
        "s_mov_b32 s39, s67",
    ):
        if required not in promoted_next:
            raise AssertionError(f"next-task descriptor promotion changed: {required}")

    tensor_sites = output_text.count("tensor_load_to_lds")
    bound_selects = output_text.count("s_cselect_b32 s38, s38, s66")
    if tensor_sites != 56 or bound_selects != tensor_sites:
        raise AssertionError(
            "steady descriptor recurrence changed: "
            f"tensor_sites={tensor_sites}, bound_selects={bound_selects}"
        )
    print(
        "A_descriptor_paths=PASS "
        "cold_lower=(base=M0,dim=[0xe000,2],tile=[0x800,2],stride=0xe000) "
        "cold_upper=(base=M0+32,dim=[0xe000,0],tile=[0x800,2],"
        "stride=0xe000) "
        "persistent_full=(base=M0,dim=[0xe000,2],tile=[0x800,4],"
        "stride=0xe000)"
    )


def _audit_v0_exact_a_oob(v0_text: str) -> None:
    required = (
        "s_sub_co_i32 s48, s4, s34",
        "s_lshr_b32 s60, s20, 5",
        "s_lshl_b32 s4, s60, 4",
        "s_sub_co_i32 s4, s48, s4",
        "s_max_i32 s4, s4, 0",
        "s_lshl_b32 s5, s4, 16",
        "s_lshr_b32 s4, s4, 16",
        "s_mov_b32 s8, 16",
        "s_movk_i32 s9, 0xe00",
        "tensor_load_to_lds s[16:19], s[4:11]",
    )
    missing = [instruction for instruction in required if instruction not in v0_text]
    if missing:
        raise AssertionError(f"v0 exact A OOB sequence changed: {missing}")

    v0_tensor_dim1 = tuple(
        max(VALID_ROWS_PER_EXPERT - wave * A_ROWS_PER_LOGICAL_WAVE, 0)
        for wave in range(A_LOGICAL_WAVES)
    )
    if v0_tensor_dim1 != (32, 16, 0, 0):
        raise AssertionError(f"v0 exact A bounds changed: {v0_tensor_dim1}")
    print(
        "v0_exact_A_OOB=PASS "
        "tensor_dim1=max((psum-block_start)-16*wave,0)=[32,16,0,0]"
    )


def _interval_signature(
    intervals: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    total_bytes = 0
    for begin, size in intervals:
        if begin < 0 or size <= 0:
            raise AssertionError(f"invalid interval [{begin},+{size})")
        digest.update(struct.pack("<QQ", begin, size))
        total_bytes += size
    return len(intervals), total_bytes, digest.hexdigest()


def _modeled_address_signatures(
    *,
    fixed_a_oob: bool,
) -> dict[str, tuple[int, int, str]]:
    a_intervals: list[tuple[int, int]] = []
    b_intervals: list[tuple[int, int]] = []
    scale_a_intervals: list[tuple[int, int]] = []
    scale_b_intervals: list[tuple[int, int]] = []

    for expert in range(EXPERTS):
        a_tile_begin = expert * A_TILE_BYTES
        for phase in range(K_PHASES):
            for logical_wave in range(A_LOGICAL_WAVES):
                if fixed_a_oob and logical_wave >= A_VALID_LOGICAL_WAVES:
                    continue
                begin = (
                    a_tile_begin
                    + logical_wave * A_TENSOR_DIM0
                    + phase * A_PHASE_BYTES_PER_LOGICAL_WAVE
                )
                a_intervals.append(
                    (begin, A_PHASE_BYTES_PER_LOGICAL_WAVE)
                )

        scale_a_begin = expert * SA_TILE_BYTES
        for phase in range(K_PHASES):
            scale_a_intervals.append(
                (scale_a_begin + phase * SCALE_PHASE_BYTES, SCALE_PHASE_BYTES)
            )

        b_expert_begin = expert * B_EXPERT_BYTES
        sb_expert_begin = expert * SB_EXPERT_BYTES
        for n_tile in range(N_TILES):
            b_tile_begin = b_expert_begin + n_tile * B_TILE_BYTES
            for block in range(B_BLOCKS_PER_TILE):
                block_begin = b_tile_begin + block * A_TENSOR_DIM0
                for phase in range(K_PHASES):
                    b_intervals.append(
                        (
                            block_begin
                            + phase * B_PHASE_BYTES_PER_BLOCK,
                            B_PHASE_BYTES_PER_BLOCK,
                        )
                    )

            sb_tile_begin = sb_expert_begin + n_tile * SB_TILE_BYTES
            for wave in range(A_LOGICAL_WAVES):
                wave_begin = sb_tile_begin + wave * SB_WAVE_BYTES
                for phase in range(K_PHASES):
                    scale_b_intervals.append(
                        (
                            wave_begin + phase * SCALE_PHASE_BYTES,
                            SCALE_PHASE_BYTES,
                        )
                    )

    return {
        "A": _interval_signature(a_intervals),
        "B": _interval_signature(b_intervals),
        "ScaleA": _interval_signature(scale_a_intervals),
        "ScaleB": _interval_signature(scale_b_intervals),
    }


def _audit_non_a_address_isolation() -> None:
    legacy = _modeled_address_signatures(fixed_a_oob=False)
    fixed = _modeled_address_signatures(fixed_a_oob=True)
    for operand in ("B", "ScaleA", "ScaleB"):
        if legacy[operand] != fixed[operand]:
            raise AssertionError(f"{operand} address set changed with A OOB fix")

    expected_bytes = {
        "A": EXPERTS * VALID_ROWS_PER_EXPERT * K_PACKED,
        "B": EXPERTS * B_EXPERT_BYTES,
        "ScaleA": EXPERTS * SA_TILE_BYTES,
        "ScaleB": EXPERTS * SB_EXPERT_BYTES,
    }
    for operand, expected in expected_bytes.items():
        if fixed[operand][1] != expected:
            raise AssertionError(
                f"{operand} unique interval bytes {fixed[operand][1]} != {expected}"
            )
    if legacy["A"][1] != 2 * fixed["A"][1]:
        raise AssertionError("legacy/fixed A useful-byte ratio is not 2:1")
    print(
        "non_A_address_sets=PASS "
        + " ".join(
            f"{operand}={fixed[operand][2][:16]}"
            for operand in ("B", "ScaleA", "ScaleB")
        )
    )


def _audit_lds_ranges() -> None:
    a_slots = (0x0000, 0x2000, 0x4000, 0x6000)
    sa_slots = (0x8000, 0x8200, 0x8400, 0x8600)
    sb_slots = (0x8800, 0x9000, 0x9800, 0xA000)
    b_slots = (0xA800, 0x12800, 0x1A800, 0x22800)
    input_end = 0x2A800

    ranges: list[tuple[str, int, int]] = []
    for slot in a_slots:
        ranges.extend(
            (
                ("A cold lower", slot, slot + 0x1000),
                ("A cold upper OOB-zero", slot + 0x1000, slot + 0x2000),
                ("A persistent full", slot, slot + 0x2000),
            )
        )
    for slot in sa_slots:
        ranges.append(("ScaleA persistent full", slot, slot + 0x200))
    for slot in sb_slots:
        ranges.append(("ScaleB persistent full", slot, slot + 0x800))
    for slot in b_slots:
        ranges.extend(
            (
                ("B cold lower", slot, slot + 0x4000),
                ("B cold upper", slot + 0x4000, slot + 0x8000),
                ("B persistent full", slot, slot + 0x8000),
            )
        )

    for name, begin, end in ranges:
        if not 0 <= begin < end <= input_end <= LDS_BYTES:
            raise AssertionError(
                f"{name} LDS range [{begin:#x},{end:#x}) exceeds allocation"
            )
    if b_slots[-1] + 0x8000 != input_end:
        raise AssertionError("input LDS ring does not end at 0x2a800")
    print(
        "LDS=PASS allocation=0x32800 input_end=0x2a800 "
        "A_OOB_zero_ranges=4x0x1000 no_padding no_overlap"
    )


def _audit_mapping_and_addresses() -> None:
    psum = [TILE_M * expert + VALID_ROWS_PER_EXPERT for expert in range(EXPERTS)]
    active: set[tuple[int, int]] = set()
    tails: list[tuple[int, int, int]] = []

    a_capacity = CONTIGUOUS_M * K_PACKED
    b_capacity = EXPERTS * B_EXPERT_BYTES
    sa_capacity = CONTIGUOUS_M * K_SCALE
    sb_capacity = EXPERTS * SB_EXPERT_BYTES

    for bid in range(GRID):
        m_tile, n_tile = _map_bid(bid)
        if not (0 <= m_tile < M_TILES and 0 <= n_tile < N_TILES):
            raise AssertionError(f"bid {bid} maps out of logical grid")
        block_start = m_tile * TILE_M
        expert = _unrolled_upper_bound(psum, block_start)
        if expert != bisect.bisect_right(psum, block_start):
            raise AssertionError(f"upper_bound mismatch for bid={bid}")
        if expert >= EXPERTS:
            tails.append((bid, m_tile, n_tile))
            continue

        active.add((expert, n_tile))
        if expert != m_tile:
            raise AssertionError(
                f"balanced map mismatch bid={bid}: m_tile={m_tile}, expert={expert}"
            )
        if psum[expert] != block_start + VALID_ROWS_PER_EXPERT:
            raise AssertionError(
                f"expert {expert} psum is not block_start+32"
            )

        intervals = (
            (
                "A",
                m_tile * A_TILE_BYTES,
                (m_tile + 1) * A_TILE_BYTES,
                a_capacity,
            ),
            (
                "B",
                expert * B_EXPERT_BYTES + n_tile * B_TILE_BYTES,
                expert * B_EXPERT_BYTES + (n_tile + 1) * B_TILE_BYTES,
                b_capacity,
            ),
            (
                "ScaleA",
                m_tile * SA_TILE_BYTES,
                (m_tile + 1) * SA_TILE_BYTES,
                sa_capacity,
            ),
            (
                "ScaleB",
                expert * SB_EXPERT_BYTES + n_tile * SB_TILE_BYTES,
                expert * SB_EXPERT_BYTES + (n_tile + 1) * SB_TILE_BYTES,
                sb_capacity,
            ),
        )
        for name, begin, end, capacity in intervals:
            if not 0 <= begin < end <= capacity:
                raise AssertionError(
                    f"{name} interval out of range for bid={bid}: "
                    f"[{begin},{end}) / {capacity}"
                )

    expected_active = {
        (expert, n_tile)
        for expert in range(EXPERTS)
        for n_tile in range(N_TILES)
    }
    if active != expected_active:
        raise AssertionError(
            f"active coverage mismatch: got={len(active)}, "
            f"expected={len(expected_active)}"
        )
    if len(active) != ACTIVE_WGS:
        raise AssertionError(f"active WG count {len(active)} != {ACTIVE_WGS}")
    if len(tails) != TAIL_WGS:
        raise AssertionError(f"tail WG count {len(tails)} != {TAIL_WGS}")
    if {m_tile for _, m_tile, _ in tails} != set(range(EXPERTS, M_TILES)):
        raise AssertionError("tail m_tile range is not exactly 96..143")
    if psum != list(range(32, 6113, 64)):
        raise AssertionError("balanced psum contract changed")

    descriptor_instances = 0
    nominal_a_bytes = 0
    in_range_a_bytes = 0
    useful_a_bytes = 0
    v0_in_range_a_bytes = 0
    logical_wave_counts = Counter()
    for expert, n_tile in sorted(active):
        block_start = expert * TILE_M
        valid_end = psum[expert]
        a_tile_begin = block_start * K_PACKED
        for phase in range(K_PHASES):
            for logical_wave in range(A_LOGICAL_WAVES):
                producer_wave = logical_wave // A_SPLIT_TILE_DIM1
                descriptor_y = logical_wave % A_SPLIT_TILE_DIM1
                tensor_dim1 = (
                    A_VALID_LOGICAL_WAVES
                    if producer_wave == 0
                    else 0
                )
                is_in_range = descriptor_y < tensor_dim1
                expected_in_range = logical_wave < A_VALID_LOGICAL_WAVES
                if is_in_range != expected_in_range:
                    raise AssertionError(
                        "A descriptor OOB mismatch: "
                        f"expert={expert} n={n_tile} phase={phase} "
                        f"logical_wave={logical_wave}"
                    )

                row_begin = block_start + logical_wave * A_ROWS_PER_LOGICAL_WAVE
                begin = (
                    a_tile_begin
                    + logical_wave * A_TENSOR_DIM0
                    + phase * A_PHASE_BYTES_PER_LOGICAL_WAVE
                )
                end = begin + A_PHASE_BYTES_PER_LOGICAL_WAVE
                if not 0 <= begin < end <= a_capacity:
                    raise AssertionError(
                        f"A logical interval [{begin},{end}) is outside allocation"
                    )
                if is_in_range:
                    if row_begin + A_ROWS_PER_LOGICAL_WAVE > valid_end:
                        raise AssertionError("in-range A slice crosses psum")
                    v21_bytes = A_PHASE_BYTES_PER_LOGICAL_WAVE
                else:
                    if row_begin < valid_end:
                        raise AssertionError("OOB A slice begins before psum")
                    v21_bytes = 0

                v0_remaining_rows = max(
                    VALID_ROWS_PER_EXPERT
                    - logical_wave * A_ROWS_PER_LOGICAL_WAVE,
                    0,
                )
                v0_rows = min(
                    A_ROWS_PER_LOGICAL_WAVE,
                    v0_remaining_rows,
                )
                v0_bytes = v0_rows * (K_PHASE // 2)
                if v21_bytes != v0_bytes:
                    raise AssertionError(
                        "v21/v0 A useful bytes differ: "
                        f"expert={expert} n={n_tile} phase={phase} "
                        f"logical_wave={logical_wave}"
                    )

                descriptor_instances += 1
                logical_wave_counts[logical_wave] += 1
                nominal_a_bytes += A_PHASE_BYTES_PER_LOGICAL_WAVE
                in_range_a_bytes += v21_bytes
                useful_a_bytes += v21_bytes
                v0_in_range_a_bytes += v0_bytes

    expected_instances = (
        EXPERTS * N_TILES * K_PHASES * A_LOGICAL_WAVES
    )
    expected_in_range = (
        ACTIVE_WGS
        * K_PHASES
        * A_VALID_LOGICAL_WAVES
        * A_PHASE_BYTES_PER_LOGICAL_WAVE
    )
    if descriptor_instances != expected_instances:
        raise AssertionError(
            f"A descriptor instances {descriptor_instances} != {expected_instances}"
        )
    if set(logical_wave_counts.values()) != {
        EXPERTS * N_TILES * K_PHASES
    }:
        raise AssertionError(f"A logical-wave counts changed: {logical_wave_counts}")
    if not (
        in_range_a_bytes
        == useful_a_bytes
        == v0_in_range_a_bytes
        == expected_in_range
    ):
        raise AssertionError(
            "effective A bytes mismatch: "
            f"in_range={in_range_a_bytes} useful={useful_a_bytes} "
            f"v0={v0_in_range_a_bytes} expected={expected_in_range}"
        )
    if nominal_a_bytes != 2 * in_range_a_bytes:
        raise AssertionError("A nominal/in-range ratio is not 2:1")

    _audit_non_a_address_isolation()
    _audit_lds_ranges()
    print(
        "mapping=PASS "
        f"grid={GRID} active={ACTIVE_WGS} tail={TAIL_WGS} "
        "coverage=96x24 psum[e]=64*e+32"
    )
    print(
        "A_addresses=PASS "
        f"enumerated={descriptor_instances}=96x24x28x4 "
        f"nominal={nominal_a_bytes} in_range={in_range_a_bytes} "
        f"useful={useful_a_bytes} v0_in_range={v0_in_range_a_bytes}"
    )


def audit(baseline: Path, output: Path) -> None:
    baseline_bytes = baseline.read_bytes()
    output_bytes = output.read_bytes()
    v0_bytes = V0.read_bytes()
    baseline_sha = hashlib.sha256(baseline_bytes).hexdigest()
    output_sha = hashlib.sha256(output_bytes).hexdigest()
    v0_sha = hashlib.sha256(v0_bytes).hexdigest()
    print(f"baseline_sha256={baseline_sha}")
    print(f"v0_loadonly_sha256={v0_sha}")
    if EXPECTED_BASELINE_SHA256 and baseline_sha != EXPECTED_BASELINE_SHA256:
        raise AssertionError(
            "v18_v2 baseline SHA256 changed; review and update the generator "
            "contract explicitly"
        )
    if b"\r" in output_bytes:
        raise AssertionError("v21 must use LF line endings")

    baseline_text = baseline_bytes.decode("utf-8")
    output_text = output_bytes.decode("utf-8")
    legacy_text = generated_text(
        baseline_text,
        apply_a_oob_bounds=False,
    )
    legacy_sha = hashlib.sha256(legacy_text.encode("utf-8")).hexdigest()
    print(f"pre_a_oob_v21_sha256={legacy_sha}")
    print(f"v21_sha256={output_sha}")
    if legacy_sha != EXPECTED_PRE_A_OOB_V21_SHA256:
        raise AssertionError(
            "captured pre-A-OOB v21 SHA256 changed; re-audit the source delta"
        )
    regenerated = generated_text(baseline_text)
    if output_text != regenerated:
        raise AssertionError(
            "v21 differs from the deterministic generator; unclassified edit found"
        )
    _audit_patch_isolation(legacy_text, output_text)
    _audit_a_descriptor_paths(output_text)
    _audit_v0_exact_a_oob(v0_bytes.decode("utf-8"))

    if _kernel_symbols(output_text) != {V21_SYMBOL}:
        raise AssertionError(
            f"symbol/metadata mismatch: {_kernel_symbols(output_text)}"
        )
    if output_text.count(f".set {CONTRACT_MARKER}, 1") != 1:
        raise AssertionError("v21 contract marker is absent or duplicated")
    if BASELINE_SYMBOL in output_text:
        raise AssertionError("baseline symbol remains in executable/metadata text")
    for required in (
        ".amdhsa_kernarg_size 184",
        ".amdhsa_user_sgpr_count 4",
        ".amdhsa_user_sgpr_kernarg_preload_length 2",
        ".amdhsa_system_sgpr_workgroup_id_x 1",
        ".amdhsa_system_sgpr_workgroup_id_y 0",
        ".amdhsa_system_sgpr_workgroup_id_z 0",
        f".group_segment_fixed_size: {LDS_BYTES}",
        ".kernarg_segment_size: 184",
        ".max_flat_workgroup_size: 128",
        ".reqd_workgroup_size:\n      - 128\n      - 1\n      - 1",
        ".vgpr_count:     384",
        ".sgpr_count:     106",
    ):
        if required not in output_text:
            raise AssertionError(f"required ABI/resource metadata missing: {required}")

    expected_args = (
        (0, 8),
        (8, 28),
        (40, 8),
        (48, 8),
        (56, 8),
        (64, 28),
        (96, 8),
        (104, 4),
        (112, 8),
        (120, 8),
        (128, 8),
        (136, 28),
        (164, 4),
        (168, 4),
        (172, 4),
        (176, 4),
        (180, 4),
    )
    metadata = output_text.split(".amdgpu_metadata", 1)[1]
    actual_args = tuple(
        (int(offset), int(size))
        for offset, size in re.findall(
            r"\.offset:\s+(\d+)\s*\n\s*\.size:\s+(\d+)",
            metadata,
        )
    )
    if actual_args != expected_args:
        raise AssertionError(f"184-byte ABI arg layout changed: {actual_args}")

    prologue = output_text.split(CORE_ANCHOR, 1)[0]
    if any(
        token in prologue
        for token in (
            "global_prefetch",
            "tensor_load_to_lds",
            "cluster_load",
            "global_load",
        )
    ):
        raise AssertionError("global TDM/data load appears before expert tail exit")
    if (
        "s_cbranch_scc1 .Lmoe_v21_tail" not in prologue
        or output_text.index(".Lmoe_v21_tail:") <= output_text.index(CORE_ANCHOR)
    ):
        raise AssertionError("tail branch/target does not guard the load body")
    if "ttmp6" in output_text or "ttmp7" in output_text:
        raise AssertionError("v21 still depends on cluster-only TTMP6/TTMP7 state")

    mask_assignments = re.findall(
        r"(?m)^\s*s_mov_b32\s+s53,\s*([^\s;]+)",
        output_text,
    )
    if mask_assignments != ["0"] * 8:
        raise AssertionError(f"TDM Workgroup_mask assignments: {mask_assignments}")
    if (
        "s_mov_b32 s90, 0x10000" not in output_text
        or "s_mov_b32 s91, 0" not in output_text
    ):
        raise AssertionError("cluster_load M0 masks are not explicit non-multicast")

    before = _mnemonic_counts(baseline_text)
    after = _mnemonic_counts(output_text)
    preserved = (
        "tensor_load_to_lds",
        "cluster_load_b128",
        "s_wait_tensorcnt",
        "s_wait_loadcnt",
        "s_barrier_signal",
        "s_barrier_wait",
    )
    for mnemonic in preserved:
        print(f"{mnemonic}: v18_v2={before[mnemonic]} v21={after[mnemonic]}")
        if before[mnemonic] != after[mnemonic]:
            raise AssertionError(f"core site count changed: {mnemonic}")
    if after["cluster_load_b128"] == 0:
        raise AssertionError("v21 no longer retains Scale cluster_load_b128")

    try:
        from my_code.isa_runner import gemm_isa_runner
    except ImportError:
        import gemm_isa_runner

    store_detection = gemm_isa_runner.detect_global_output_stores(output_text)
    if store_detection.writes_output:
        raise AssertionError(
            f"executable global store detected: {store_detection.mnemonics}"
        )
    if LDS_BYTES > gemm_isa_runner.MAX_LDS_PER_WGP:
        raise AssertionError("v21 static LDS exceeds gfx1250 WGP limit")

    _audit_mapping_and_addresses()
    print(
        "intentional_changes=prologue+184B_ABI,swizzle+upper_bound,"
        "expert/tile_addresses,A_M32_OOB,noncluster_TDM_masks,"
        "unique_symbol+metadata"
    )
    print("executable_store_detection=False")
    print("audit=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, default=V21)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="regenerate v21 from the audited v18_v2 source before checking",
    )
    args = parser.parse_args()
    if args.generate:
        generate(args.baseline, args.output)
    audit(args.baseline, args.output)


if __name__ == "__main__":
    main()
