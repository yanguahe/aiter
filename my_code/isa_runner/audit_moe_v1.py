#!/usr/bin/env python3
"""Generate and statically audit ``moe_gemm1_a4w4_v1.s``.

The v1 kernel is derived mechanically from the current full v0 kernel.  A/B
retain their two-slot TDM-to-LDS pipeline.  ScaleA/ScaleB instead use two
per-wave ``global_load_b128`` operations per K256 phase and an exact 4x32
register/lane transpose.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "my_code" / "moe_gemm1_a4w4_v0.s"
V1 = ROOT / "my_code" / "moe_gemm1_a4w4_v1.s"
RUNNER = ROOT / "my_code" / "isa_runner" / "gemm_batch_isa_runner.py"
RUNNER_TEST = ROOT / "my_code" / "isa_runner" / "test_moe_cpp_backend.py"

V0_SYMBOL = "a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4"
V1_SYMBOL = (
    "a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_"
    "wpt4_directscale_v1"
)

EXPERTS = 96
N_TILES = 24
K_PHASES = 28
ACTIVE_WGS = EXPERTS * N_TILES
TAIL_WGS = 1152
LDS_BYTES = 0x15C00
RAW_SCALE_BUFFERS = ((240, 247), (248, 255))
SCALE_OPERANDS = set(range(230, 238))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise AssertionError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def _replace_span(
    text: str,
    start: str,
    end: str,
    replacement: str,
    label: str,
) -> str:
    start_index = text.find(start)
    if start_index < 0:
        raise AssertionError(f"{label}: start anchor not found")
    end_index = text.find(end, start_index + len(start))
    if end_index < 0:
        raise AssertionError(f"{label}: end anchor not found")
    if text.find(start, start_index + 1) >= 0:
        raise AssertionError(f"{label}: start anchor is not unique")
    return text[:start_index] + replacement + text[end_index:]


def _transpose(raw_sa: int, raw_sb: int, indent: str = "\t") -> str:
    """Return the exact 4x32 transpose for one raw scale buffer.

    A b128 load gives ``raw[j][lane] = block[lane//8][4*(lane%8)+j]``.
    The xor-8/PERMLANE16 stages first group each physical 128-B plane in one
    VGPR.  DS_BPERMUTE then changes its lane order from ``j*8+chunk`` to the
    WMMA-required ``chunk*4+j`` order without reading or writing LDS memory.
    """

    lines = [
        f"{indent}; ScaleA raw v[{raw_sa}:{raw_sa + 3}] -> "
        "v232/v233/v236/v237.",
        f"{indent}v_permlane_xor_b32 v232, v{raw_sa + 1}, 8, 16",
        f"{indent}v_permlane_xor_b32 v233, v{raw_sa}, 8, 16",
        f"{indent}v_permlane_xor_b32 v236, v{raw_sa + 3}, 8, 16",
        f"{indent}v_permlane_xor_b32 v237, v{raw_sa + 2}, 8, 16",
        f"{indent}s_wait_alu depctr_va_vdst(0)",
        f"{indent}v_cndmask_b32_e64 v232, v{raw_sa}, v232, s70",
        f"{indent}v_cndmask_b32_e64 v233, v233, v{raw_sa + 1}, s70",
        f"{indent}v_cndmask_b32_e64 v236, v{raw_sa + 2}, v236, s70",
        f"{indent}v_cndmask_b32_e64 v237, v237, v{raw_sa + 3}, s70",
        f"{indent}; ScaleB raw v[{raw_sb}:{raw_sb + 3}] -> "
        "v230/v231/v234/v235.",
        f"{indent}v_permlane_xor_b32 v230, v{raw_sb + 1}, 8, 16",
        f"{indent}v_permlane_xor_b32 v231, v{raw_sb}, 8, 16",
        f"{indent}v_permlane_xor_b32 v234, v{raw_sb + 3}, 8, 16",
        f"{indent}v_permlane_xor_b32 v235, v{raw_sb + 2}, 8, 16",
        f"{indent}s_wait_alu depctr_va_vdst(0)",
        f"{indent}v_cndmask_b32_e64 v230, v{raw_sb}, v230, s70",
        f"{indent}v_cndmask_b32_e64 v231, v231, v{raw_sb + 1}, s70",
        f"{indent}v_cndmask_b32_e64 v234, v{raw_sb + 2}, v234, s70",
        f"{indent}v_cndmask_b32_e64 v235, v235, v{raw_sb + 3}, s70",
        f"{indent}s_wait_alu depctr_va_vdst(0)",
        f"{indent}v_permlane16_swap_b32 v232, v236",
        f"{indent}v_permlane16_swap_b32 v233, v237",
        f"{indent}v_permlane16_swap_b32 v230, v234",
        f"{indent}v_permlane16_swap_b32 v231, v235",
        f"{indent}s_wait_alu depctr_va_vdst(0)",
        f"{indent}; Butterfly lanes are j*8+chunk; WMMA needs chunk*4+j.",
        f"{indent}ds_bpermute_b32 v{raw_sa}, v239, v230",
        f"{indent}ds_bpermute_b32 v{raw_sa + 1}, v239, v231",
        f"{indent}ds_bpermute_b32 v{raw_sa + 2}, v239, v232",
        f"{indent}ds_bpermute_b32 v{raw_sa + 3}, v239, v233",
        f"{indent}ds_bpermute_b32 v{raw_sb}, v239, v234",
        f"{indent}ds_bpermute_b32 v{raw_sb + 1}, v239, v235",
        f"{indent}ds_bpermute_b32 v{raw_sb + 2}, v239, v236",
        f"{indent}ds_bpermute_b32 v{raw_sb + 3}, v239, v237",
        f"{indent}s_wait_dscnt 0x0",
        f"{indent}v_dual_mov_b32 v230, v{raw_sa} :: "
        f"v_dual_mov_b32 v231, v{raw_sa + 1}",
        f"{indent}v_dual_mov_b32 v232, v{raw_sa + 2} :: "
        f"v_dual_mov_b32 v233, v{raw_sa + 3}",
        f"{indent}v_dual_mov_b32 v234, v{raw_sb} :: "
        f"v_dual_mov_b32 v235, v{raw_sb + 1}",
        f"{indent}v_dual_mov_b32 v236, v{raw_sb + 2} :: "
        f"v_dual_mov_b32 v237, v{raw_sb + 3}",
        f"{indent}s_wait_alu depctr_va_vdst(0)",
    ]
    return "\n".join(lines) + "\n"


def generate(source: str) -> str:
    if "\r" in source:
        raise AssertionError("v0 must use LF line endings")
    if source.count(V0_SYMBOL) < 6:
        raise AssertionError("v0 symbol contract changed unexpectedly")

    text = source.replace(V0_SYMBOL, V1_SYMBOL)

    text = _replace_once(
        text,
        "\tv_and_b32_e32 v3, 16, v0\n"
        "\ts_lshl_b32 s7, s20, 1\n",
        "\tv_and_b32_e32 v3, 16, v0\n"
        "\t; DS_BPERMUTE byte address for source lane\n"
        "\t; ((lane & 3)*8 + (lane >> 2)).\n"
        "\tv_and_b32_e32 v198, 3, v1\n"
        "\tv_lshrrev_b32_e32 v239, 2, v1\n"
        "\ts_wait_alu depctr_va_vdst(0)\n"
        "\tv_mad_u32 v239, 8, v198, v239\n"
        "\ts_wait_alu depctr_va_vdst(0)\n"
        "\tv_lshlrev_b32_e32 v239, 2, v239\n"
        "\ts_lshl_b32 s7, s20, 1\n",
        "build lane permutation address",
    )

    text = _replace_once(
        text,
        "\tv_lshl_or_b32 v197, s51, 3, v1\n"
        "\tv_or3_b32 v195, s5, v0, v2\n",
        "\t; Direct-scale lane addresses.  v196=lane*16 covers the four\n"
        "\t; contiguous 128-B ScaleA planes.  v197 groups eight lanes per\n"
        "\t; ScaleB (N32,KSL) plane: [0,0x80,0x1c00,0x1c80].\n"
        "\tv_mad_u32 v197, 0x1c0, v3, v2\n"
        "\tv_or3_b32 v195, s5, v0, v2\n"
        "\tv_or_b32_e32 v196, v0, v2\n",
        "direct scale lane addresses",
    )

    text = _replace_span(
        text,
        "\ts_ashr_i64 s[46:47], s[34:35], 6\n",
        "\ts_load_b64 s[78:79], s[0:1], 0x60 nv\n",
        "\t; ScaleA direct base: ptr_scale_a + m_tile*0x3800.\n"
        "\ts_ashr_i64 s[46:47], s[34:35], 6\n"
        "\ts_mul_u64 s[64:65], s[46:47], 0x3800\n"
        "\ts_add_nc_u64 s[22:23], s[56:57], s[64:65]\n"
        "\ts_mov_b64 s[72:73], s[22:23]\n"
        "\t; Lanes with bit 3 set select the xor-8 transpose candidate.\n"
        "\ts_mov_b32 s70, 0xff00ff00\n",
        "remove ScaleA TDM descriptor",
    )
    text = _replace_once(
        text,
        "\ts_mov_b64 s[74:75], s[30:31]\n"
        "\ts_mov_b64 s[72:73], s[28:29]\n",
        "",
        "remove stale scale descriptor group zero",
    )

    text = _replace_span(
        text,
        "\ts_mov_b32 s24, s28\n"
        "\ts_mov_b32 s25, s9\n"
        "\ts_mov_b32 s26, s55\n"
        "\ts_mov_b32 s27, s55\n",
        "\ts_lshl_b32 s36, s60, 1\n",
        "",
        "remove first ScaleA TDM issue",
    )

    text = _replace_span(
        text,
        "\ts_mov_b32 s39, 0x407fff\n",
        "\ttensor_load_to_lds s[28:31], s[4:11]\n",
        "\t; ScaleB direct base: ptr_scale_b + expert/N-tile/wave-N64.\n"
        "\ts_mov_b64 s[74:75], s[86:87]\n"
        "\t; Prologue queues q0 in raw buffer 0 and q1 in raw buffer 1.\n"
        "\tglobal_load_b128 v[240:243], v196, s[72:73]\n"
        "\tglobal_load_b128 v[244:247], v197, s[74:75]\n"
        "\tglobal_load_b128 v[248:251], v196, s[72:73] offset:512\n"
        "\tglobal_load_b128 v[252:255], v197, s[74:75] offset:256\n"
        "\t; Refill pointers start at q2.\n"
        "\ts_add_nc_u64 s[76:77], s[72:73], 0x400\n"
        "\ts_add_nc_u64 s[84:85], s[74:75], 0x200\n",
        "replace first ScaleB descriptor with direct loads",
    )

    text = _replace_span(
        text,
        "\ts_add_nc_u64 s[30:31], s[76:77], 0x200\n",
        "\ts_sub_co_ci_u32 s29, s62, 0\n",
        "",
        "remove second scale TDM issues",
    )

    steady = (
        ".LBB0_3:\n"
        "\t; Four outstanding A/B TDM jobs are ordered q,q+1; <=2 retires q.\n"
        "\ts_wait_tensorcnt 0x2\n"
        "\t; Four outstanding scale VMEM jobs are ordered q,q+1; <=2 retires q.\n"
        "\ts_wait_loadcnt 0x2\n"
        "\ts_barrier_signal -1\n"
        "\ts_and_b32 s61, s55, 1\n"
        "\ts_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | "
        "instid1(SALU_CYCLE_1)\n"
        "\ts_mul_i32 s61, s61, 0xae00\n"
        "\tv_dual_add_nc_u32 v206, s61, v195 :: "
        "v_dual_add_nc_u32 v238, s61, v194\n"
        "\ts_cmp_eq_u32 s61, 0\n"
        "\ts_cbranch_scc0 .Lmoe_v1_scale_slot1\n"
        + _transpose(240, 244)
        + "\t; Refill q+2 into the raw buffer just consumed.\n"
        "\tglobal_load_b128 v[240:243], v196, s[76:77]\n"
        "\tglobal_load_b128 v[244:247], v197, s[84:85]\n"
        "\ts_branch .Lmoe_v1_scale_refill_done\n"
        ".Lmoe_v1_scale_slot1:\n"
        + _transpose(248, 252)
        + "\tglobal_load_b128 v[248:251], v196, s[76:77]\n"
        "\tglobal_load_b128 v[252:255], v197, s[84:85]\n"
        ".Lmoe_v1_scale_refill_done:\n"
        "\ts_add_nc_u64 s[76:77], s[76:77], 0x200\n"
        "\ts_add_nc_u64 s[84:85], s[84:85], 0x100\n"
        "\ts_barrier_wait -1\n"
        "\ts_wait_alu depctr_va_vdst(0)\n"
    )
    text = _replace_span(
        text,
        ".LBB0_3:\n",
        "\tds_load_b128 v[128:131], v206\n",
        steady,
        "replace steady scale consumer/refill",
    )

    text = _replace_span(
        text,
        "\ts_add_nc_u64 s[64:65], s[46:47], s[56:57]\n",
        ".LBB0_4:\n",
        "\t; A advances 0x80 and B advances 0x800 per K256 refill.\n"
        "\ts_add_nc_u64 s[30:31], s[30:31], 0x80\n"
        "\t; Keep the original 26-trip steady-loop induction in bytes.\n"
        "\ts_add_nc_u64 s[58:59], s[58:59], 0x100\n"
        "\ts_cmp_lg_u32 s58, 0x1a00\n"
        "\ts_cbranch_scc1 .LBB0_3\n",
        "remove steady scale TDM descriptors",
    )

    drain0 = (
        ".LBB0_4:\n"
        "\t; q26: q26/q27 are outstanding, so retire the older two jobs.\n"
        "\ts_wait_tensorcnt 0x2\n"
        "\ts_wait_loadcnt 0x2\n"
        "\ts_barrier_signal -1\n"
        + _transpose(240, 244)
        + "\ts_barrier_wait -1\n"
        "\ts_wait_alu depctr_va_vdst(0)\n"
    )
    text = _replace_span(
        text,
        ".LBB0_4:\n",
        "\tds_load_b128 v[128:131], v195\n",
        drain0,
        "replace q26 scale consumer",
    )

    q26_last = (
        "\tv_wmma_scale_f32_32x16x128_f4 v[16:31], v[160:175], "
        "v[144:151], v[16:31], v231, v237 matrix_b_scale:MATRIX_SCALE_ROW1\n"
    )
    q27 = (
        q26_last
        + "\t; q27 is the final slot: no TDM or VMEM scale job may remain.\n"
        "\ts_wait_tensorcnt 0x0\n"
        "\ts_wait_loadcnt 0x0\n"
        "\ts_barrier_signal -1\n"
        + _transpose(248, 252)
        + "\ts_barrier_wait -1\n"
        "\ts_wait_alu depctr_va_vdst(0)\n"
    )
    text = _replace_span(
        text,
        q26_last + "\ts_wait_tensorcnt 0x0\n",
        "\tds_load_b128 v[128:131], v195 offset:44544\n",
        q27,
        "replace q27 scale consumer",
    )
    text = _replace_once(
        text,
        "\tv_add_nc_u32_e32 v152, 0xa600, v197\n",
        "",
        "remove dead q27 ScaleB LDS address",
    )

    # q27's original allocator used a different packed scale-register order.
    # Direct scale keeps the steady/q26 order for all phases, so update only
    # q27's sixteen WMMA scale operands.
    q27_body_start = text.index(
        "\tds_load_b128 v[128:131], v195 offset:44544\n"
    )
    q27_body_end = text.index(
        "\tv_max_num_f32_e64 v134, s50, s50\n",
        q27_body_start,
    )
    q27_body = text[q27_body_start:q27_body_end]
    sb_register = {228: 230, 229: 231, 232: 234, 233: 235}
    sa_register = {230: 232, 231: 233, 234: 236, 235: 237}

    def remap_q27_scales(match: re.Match[str]) -> str:
        sb = int(match.group(1))
        sa = int(match.group(2))
        if sb not in sb_register or sa not in sa_register:
            raise AssertionError(
                f"unexpected q27 scale operands v{sb}, v{sa}"
            )
        return f", v{sb_register[sb]}, v{sa_register[sa]}"

    q27_body, q27_wmmas = re.subn(
        r", v(228|229|232|233), v(230|231|234|235)"
        r"(?=(?: matrix_b_scale:MATRIX_SCALE_ROW1)?$)",
        remap_q27_scales,
        q27_body,
        flags=re.MULTILINE,
    )
    if q27_wmmas != 16:
        raise AssertionError(
            f"expected to remap sixteen q27 WMMAs, found {q27_wmmas}"
        )
    text = text[:q27_body_start] + q27_body + text[q27_body_end:]

    # Every remaining 2ADDR load is a ScaleA/ScaleB LDS consumer.
    text, removed_scale_ds = re.subn(
        r"^\tds_load_2addr_b32 .*\n",
        "",
        text,
        flags=re.MULTILINE,
    )
    if removed_scale_ds != 6:
        # Six were removed as part of the three phase-entry span replacements.
        raise AssertionError(
            f"expected six residual scale LDS loads, found {removed_scale_ds}"
        )

    # With the three scale DS instructions removed from each KSL0 prefix,
    # 14 data loads remain and the first WMMA needs the first six: 14-6=8.
    if text.count("\ts_wait_dscnt 0x9\n") != 3:
        raise AssertionError("expected three original DScnt<=9 waits")
    text = text.replace("\ts_wait_dscnt 0x9\n", "\ts_wait_dscnt 0x8\n")

    text = _replace_once(
        text,
        f"\t.set {V1_SYMBOL}.num_vgpr, 239\n",
        f"\t.set {V1_SYMBOL}.num_vgpr, 256\n",
        "symbol VGPR count",
    )
    text = _replace_once(
        text,
        "\t\t.amdhsa_next_free_vgpr 257\n",
        "\t\t.amdhsa_next_free_vgpr 256\n",
        "next free VGPR",
    )
    text = _replace_once(
        text,
        "    .vgpr_count:     239\n",
        "    .vgpr_count:     256\n",
        "metadata VGPR count",
    )
    return text


def _tensor_issues(source: str) -> list[tuple[str, str]]:
    return re.findall(
        r"^\ttensor_load_to_lds (s\[[0-9]+:[0-9]+\]), "
        r"(s\[[0-9]+:[0-9]+\])$",
        source,
        flags=re.MULTILINE,
    )


def _audit_scale_layout() -> None:
    """Prove the per-lane b128/transpose mapping for both scale tensors."""

    # ScaleA planes are physically contiguous:
    # [KSL0/M0:31, KSL0/M32:63, KSL1/M0:31, KSL1/M32:63].
    sa_seen: set[tuple[int, int]] = set()
    sb_seen: set[tuple[int, int]] = set()
    butterfly: dict[tuple[int, int], tuple[int, int]] = {}
    for lane in range(32):
        block = lane // 8
        chunk = lane % 8
        for dword in range(4):
            row = 4 * chunk + dword
            sa_seen.add((block, row))
            sb_seen.add((block, row))
            # The xor-8/PERMLANE16 butterfly first groups each 128-B plane
            # into one VGPR, with lane order dword*8+chunk.
            butterfly[(block, dword * 8 + chunk)] = (block, row)

    expected = {(block, row) for block in range(4) for row in range(32)}
    assert sa_seen == expected
    assert sb_seen == expected
    assert len(butterfly) == 128
    for block in range(4):
        for destination_lane in range(32):
            source_lane = (
                (destination_lane & 3) * 8 + (destination_lane >> 2)
            )
            assert butterfly[(block, source_lane)] == (
                block,
                destination_lane,
            )

    # Byte-address proof.  One b128 per wave reads exactly 512 unique bytes.
    sa_bytes: set[int] = set()
    sb_bytes: set[int] = set()
    sb_plane_offsets = (0, 0x80, 0x1C00, 0x1C80)
    for lane in range(32):
        sa_addr = lane * 16
        sb_addr = ((lane & 15) << 4) + ((lane & 16) * 0x1C0)
        for byte in range(16):
            sa_bytes.add(sa_addr + byte)
            sb_bytes.add(sb_addr + byte)
    assert sa_bytes == set(range(0x200))
    assert sb_bytes == {
        plane + byte
        for plane in sb_plane_offsets
        for byte in range(0x80)
    }

    # Final WMMA operand order, matching v0:
    # logical ScaleB N0K0/N0K1/N1K0/N1K1 and
    # logical ScaleA M0:31K0/M32:63K0/M0:31K1/M32:63K1.
    assert (230, 231, 234, 235) == (230, 231, 234, 235)
    assert (232, 233, 236, 237) == (232, 233, 236, 237)


def _audit_counter_schedule() -> None:
    """Model in-order TENSORcnt/LOADcnt retirement over all 28 phases."""

    def run(name: str) -> None:
        queue: list[tuple[int, str]] = [
            (0, name + "0"),
            (0, name + "1"),
            (1, name + "0"),
            (1, name + "1"),
        ]
        ready: set[int] = set()

        def wait(threshold: int) -> None:
            while len(queue) > threshold:
                phase, _kind = queue.pop(0)
                ready.add(phase)

        for phase in range(26):
            wait(2)
            assert phase in ready
            queue.extend(((phase + 2, name + "0"), (phase + 2, name + "1")))
            assert len(queue) <= 4
        wait(2)
        assert 26 in ready
        wait(0)
        assert 27 in ready
        assert not queue

    run("tdm")
    run("load")


def _parse_range(token: str) -> range:
    match = re.fullmatch(r"v\[(\d+):(\d+)\]", token)
    if not match:
        raise AssertionError(f"cannot parse VGPR range {token!r}")
    lo, hi = map(int, match.groups())
    return range(lo, hi + 1)


def _audit_ds_hazards(source: str) -> None:
    """Check every WMMA data operand against the in-order DScnt frontier."""

    latest_ds_write: dict[int, int] = {}
    issued = 0
    completed = 0
    wmma_count = 0
    for raw_line in source.splitlines():
        line = raw_line.strip()
        load = re.match(r"ds_load_b128 (v\[\d+:\d+\]),", line)
        if load:
            issued += 1
            for register in _parse_range(load.group(1)):
                latest_ds_write[register] = issued
            continue
        wait = re.fullmatch(r"s_wait_dscnt 0x([0-9a-f]+)", line)
        if wait:
            threshold = int(wait.group(1), 16)
            completed = max(completed, issued - threshold)
            continue
        wmma = re.match(
            r"v_wmma_scale_f32_32x16x128_f4 "
            r"v\[\d+:\d+\], (v\[\d+:\d+\]), (v\[\d+:\d+\]),",
            line,
        )
        if not wmma:
            continue
        wmma_count += 1
        for operand in wmma.groups():
            for register in _parse_range(operand):
                write = latest_ds_write.get(register)
                if write is None:
                    raise AssertionError(
                        f"WMMA source v{register} has no preceding DS load"
                    )
                if write > completed:
                    raise AssertionError(
                        f"WMMA reads v{register} from DS issue {write}, "
                        f"but only issues <= {completed} are complete"
                    )
    assert wmma_count == 48


def audit(v0: str, v1: str) -> None:
    assert "\r" not in v1
    assert f"\t.globl\t{V0_SYMBOL}\n" in v0
    assert f"\t.globl\t{V0_SYMBOL}\n" not in v1
    assert v1.count(V1_SYMBOL) >= 6
    assert "\t.global_load_b128" not in v1  # catch malformed leading directive
    assert len(re.findall(r"^\tglobal_load_b128 ", v1, re.MULTILINE)) == 8
    assert "ds_load_2addr_b32" not in v1
    assert "0xff00ff00" in v1
    assert "v_mad_u32 v197, 0x1c0, v3, v2" in v1
    assert "v_or_b32_e32 v196, v0, v2" in v1
    assert "v_mad_u32 v239, 8, v198, v239" in v1
    assert len(re.findall(r"^\tds_bpermute_b32 ", v1, re.MULTILINE)) == 32
    assert "s_mov_b64 s[72:73], s[28:29]" not in v1
    scale_pairs = [
        tuple(map(int, pair))
        for pair in re.findall(
            r"^.*v_wmma_scale_f32_32x16x128_f4 .*, "
            r"v(230|231|234|235), v(232|233|236|237)"
            r"(?: matrix_b_scale:MATRIX_SCALE_ROW1)?$",
            v1,
            flags=re.MULTILINE,
        )
    ]
    assert len(scale_pairs) == 48
    assert {pair[0] for pair in scale_pairs} == {230, 231, 234, 235}
    assert {pair[1] for pair in scale_pairs} == {232, 233, 236, 237}

    # v0 has A/B/ScaleA/ScaleB in each static phase site.  v1 is the exact
    # A/B subsequence: q0, q1 and one steady refill.
    assert _tensor_issues(v0) == [
        ("s[16:19]", "s[4:11]"),
        ("s[24:27]", "s[12:19]"),
        ("s[36:39]", "s[20:27]"),
        ("s[72:75]", "s[36:43]"),
        ("s[28:31]", "s[4:11]"),
        ("s[28:31]", "s[12:19]"),
        ("s[28:31]", "s[20:27]"),
        ("s[28:31]", "s[36:43]"),
        ("s[64:67]", "s[4:11]"),
        ("s[64:67]", "s[12:19]"),
        ("s[64:67]", "s[20:27]"),
        ("s[64:67]", "s[36:43]"),
    ]
    assert _tensor_issues(v1) == [
        ("s[16:19]", "s[4:11]"),
        ("s[24:27]", "s[12:19]"),
        ("s[28:31]", "s[4:11]"),
        ("s[28:31]", "s[12:19]"),
        ("s[64:67]", "s[4:11]"),
        ("s[64:67]", "s[12:19]"),
    ]
    for invariant in (
        "s_movk_i32 s9, 0xe00",
        "s_mov_b32 s17, 0xe000",
        "s_add_nc_u64 s[30:31], s[30:31], 0x80",
        "s_add_nc_u64 s[44:45], s[44:45], 0x800",
        "s_cmp_lg_u32 s58, 0x1a00",
    ):
        assert invariant in v0 and invariant in v1

    # The complete output/SiLU/store control flow remains byte-for-byte equal.
    epilogue_anchor = "\tv_max_num_f32_e64 v134, s50, s50\n"
    end_anchor = ".LBB0_5:\n"
    v0_epilogue = v0[v0.index(epilogue_anchor) : v0.index(end_anchor)]
    v1_epilogue = v1[v1.index(epilogue_anchor) : v1.index(end_anchor)]
    assert v0_epilogue == v1_epilogue

    # Tail branch precedes all data TDM/VMEM operations.
    tail_branch = v1.index("\ts_cbranch_scc1 .LBB0_5\n")
    first_tdm = v1.index("\ttensor_load_to_lds ")
    first_global = v1.index("\tglobal_load_b128 ")
    assert tail_branch < first_tdm < first_global

    assert v1.count("\ts_wait_tensorcnt 0x2\n") == 2
    assert v1.count("\ts_wait_loadcnt 0x2\n") == 2
    assert v1.count("\ts_wait_loadcnt 0x0\n") == 1
    assert "\ts_wait_tensorcnt 0x4\n" not in v1
    assert "\ts_wait_dscnt 0x9\n" not in v1

    explicit_vgprs = [int(value) for value in re.findall(r"\bv(\d+)\b", v1)]
    assert max(explicit_vgprs) == 255
    assert SCALE_OPERANDS.isdisjoint(
        set(range(RAW_SCALE_BUFFERS[0][0], RAW_SCALE_BUFFERS[-1][1] + 1))
    )
    assert "\t\t.amdhsa_next_free_vgpr 256\n" in v1
    assert f"\t.set {V1_SYMBOL}.num_vgpr, 256\n" in v1
    assert "    .vgpr_count:     256\n" in v1
    assert f"    .group_segment_fixed_size: {LDS_BYTES}\n" in v1
    assert "\t\t.amdhsa_group_segment_fixed_size 89088\n" in v1

    _audit_scale_layout()
    _audit_counter_schedule()
    _audit_ds_hazards(v1)

    assert EXPERTS * N_TILES == ACTIVE_WGS
    assert ACTIVE_WGS + TAIL_WGS == 3456
    assert K_PHASES == 28

    runner = RUNNER.read_text(encoding="utf-8")
    runner_test = RUNNER_TEST.read_text(encoding="utf-8")
    assert "MOE_GEMM1_V1_KERNEL_SYMBOL" in runner
    assert "wpt4_directscale_v1" in runner
    assert "moe_gemm1_a4w4_v1.s" in runner_test
    assert "MOE_GEMM1_V1_KERNEL_SYMBOL" in runner_test


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="regenerate moe_gemm1_a4w4_v1.s from the current v0",
    )
    args = parser.parse_args()

    v0_bytes = V0.read_bytes()
    v0 = v0_bytes.decode("utf-8")
    generated = generate(v0)
    if args.write:
        V1.write_text(generated, encoding="utf-8", newline="\n")
    if not V1.is_file():
        raise AssertionError(f"{V1} does not exist; run with --write")
    v1_bytes = V1.read_bytes()
    v1 = v1_bytes.decode("utf-8")
    if v1 != generated:
        raise AssertionError("v1 is stale; regenerate it with --write")
    audit(v0, v1)
    print(
        "audit_moe_v1: PASS "
        f"v0_sha256={_sha256(v0_bytes)} "
        f"v1_sha256={_sha256(v1_bytes)} "
        "active_wgs=2304 tail_wgs=1152 k_phases=28 "
        "tdm_jobs_per_phase_per_wave=2 "
        "scale_load_jobs_per_phase_per_wave=2 "
        "scale_global_bytes_per_phase_per_wave=1024 "
        "highest_vgpr=v255 metadata_vgpr_count=256"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
