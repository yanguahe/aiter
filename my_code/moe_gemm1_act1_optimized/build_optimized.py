from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline_act1_independent.s"
OUTPUT = (
    HERE
    / "moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_opt.s"
)

MULTICAST_DISABLED = (
    "\ts_and_b32 s36, s36, 0xffff0000\n"
    "\ts_and_b32 s36, s36, 0xffdfffff"
)
MULTICAST_ENABLED = "\ts_bitset1_b32 s36, 21"


def regs(out_i: int) -> tuple[int, int, int, int]:
    gate0 = 100 + out_i * 4
    up0 = gate0 + 1
    gate1 = gate0 + 2
    up1 = gate0 + 3
    return gate0, up0, gate1, up1


def temp_pair(batch_index: int, lane: int) -> tuple[int, int]:
    # Double-buffer the transcendental temporaries so RCP for the current batch
    # can overlap preparation/EXP for the next batch. v91 remains the LDS base.
    base = 64 if batch_index % 2 == 0 else 228
    return base + lane * 2, base + lane * 2 + 1


def prepare_gate_pairs(batch_start: int, batch_index: int) -> list[str]:
    lines: list[str] = []
    for lane, out_i in enumerate(range(batch_start, batch_start + 8)):
        gate0, up0, gate1, _up1 = regs(out_i)
        t0, t1 = temp_pair(batch_index, lane)
        lines.extend(
            [
                # Reorder G0,U0,G1,U1 -> G0,G1,U0,U1. Gate and up values are
                # now adjacent pairs and can use packed FP32 operations.
                f"\tv_swap_b32 v{up0}, v{gate1}",
                (
                    f"\tv_dual_min_num_f32 v{gate0}, v{gate0}, v98 :: "
                    f"v_dual_min_num_f32 v{up0}, v{up0}, v99"
                ),
                f"\tv_pk_mul_f32 v[{t0}:{t1}], v[{gate0}:{up0}], v[96:97]",
            ]
        )
    return lines


def issue_exp_and_clamp_up(batch_start: int, batch_index: int) -> list[str]:
    lines: list[str] = []
    for lane, out_i in enumerate(range(batch_start, batch_start + 8)):
        _gate0, _gate1_slot, up0, up1 = regs(out_i)
        t0, t1 = temp_pair(batch_index, lane)
        lines.extend(
            [
                f"\tv_exp_f32_e32 v{t0}, v{t0}",
                f"\tv_med3_num_f32 v{up0}, v{up0}, s103, s102",
                f"\tv_exp_f32_e32 v{t1}, v{t1}",
                f"\tv_med3_num_f32 v{up1}, v{up1}, s103, s102",
            ]
        )
    return lines


def add_denominators(batch_index: int) -> list[str]:
    lines: list[str] = []
    for lane in range(8):
        t0, t1 = temp_pair(batch_index, lane)
        lines.append(
            f"\tv_dual_add_f32 v{t0}, 1.0, v{t0} :: "
            f"v_dual_add_f32 v{t1}, 1.0, v{t1}"
        )
    return lines


def finish_batch(batch_start: int, batch_index: int) -> list[str]:
    lines: list[str] = []
    for lane, out_i in enumerate(range(batch_start, batch_start + 8)):
        gate0, gate1_slot, up0, up1 = regs(out_i)
        t0, t1 = temp_pair(batch_index, lane)
        lines.extend(
            [
                f"\tv_pk_mul_f32 v[{gate0}:{gate1_slot}], "
                f"v[{gate0}:{gate1_slot}], v[{t0}:{t1}]",
                f"\tv_pk_mul_f32 v[{gate0}:{gate1_slot}], "
                f"v[{gate0}:{gate1_slot}], v[{up0}:{up1}]",
                f"\tv_cvt_pk_bf16_f32 v{100 + out_i}, v{gate0}, v{gate1_slot}",
            ]
        )
    return lines


def optimized_activation(bank: int) -> str:
    lines = [
        f"\t; Optimized accumulator bank {bank}: packed gate/up pairs,",
        "\t; dual-issue clamps/adds, and cross-batch EXP/RCP software pipeline.",
        f"\ts_set_vgpr_msb 0x{bank * 0x55:x}",
        "\tv_mov_b32_e32 v96, 0xbfb8aa3b",
        "\tv_mov_b32_e32 v97, v96",
        "\tv_mov_b32_e32 v98, s102",
        "\tv_mov_b32_e32 v99, s102",
    ]

    batches = tuple(range(0, 32, 8))
    lines.extend(prepare_gate_pairs(batches[0], 0))
    lines.extend(issue_exp_and_clamp_up(batches[0], 0))
    lines.extend(add_denominators(0))

    for batch_index, batch_start in enumerate(batches):
        has_next = batch_index + 1 < len(batches)
        if has_next:
            next_index = batch_index + 1
            next_start = batches[next_index]

            # Swaps and gate clamps are issued first. By the time the packed
            # pre-EXP multiply is interleaved with current RCPs, the clamp
            # result has enough independent VALU distance to avoid a stall.
            for out_i in range(next_start, next_start + 8):
                gate0, up0, gate1, _up1 = regs(out_i)
                lines.extend(
                    [
                        f"\tv_swap_b32 v{up0}, v{gate1}",
                        (
                            f"\tv_dual_min_num_f32 v{gate0}, v{gate0}, v98 :: "
                            f"v_dual_min_num_f32 v{up0}, v{up0}, v99"
                        ),
                    ]
                )

            for lane, next_out in enumerate(range(next_start, next_start + 8)):
                curr_t0, curr_t1 = temp_pair(batch_index, lane)
                next_t0, next_t1 = temp_pair(next_index, lane)
                next_gate0, next_gate1_slot, next_up0, _next_up1 = regs(next_out)
                lines.extend(
                    [
                        f"\tv_rcp_f32_e32 v{curr_t0}, v{curr_t0}",
                        (
                            f"\tv_pk_mul_f32 v[{next_t0}:{next_t1}], "
                            f"v[{next_gate0}:{next_gate1_slot}], v[96:97]"
                        ),
                        f"\tv_rcp_f32_e32 v{curr_t1}, v{curr_t1}",
                        (
                            f"\tv_med3_num_f32 v{next_up0}, v{next_up0}, "
                            "s103, s102"
                        ),
                    ]
                )

            # EXP for the next batch co-executes with packed multiplies and
            # BF16 conversion from the current batch.
            for lane, (curr_out, next_out) in enumerate(
                zip(
                    range(batch_start, batch_start + 8),
                    range(next_start, next_start + 8),
                )
            ):
                curr_gate0, curr_gate1_slot, curr_up0, curr_up1 = regs(curr_out)
                next_t0, next_t1 = temp_pair(next_index, lane)
                _ng0, _ng1, _nu0, next_up1 = regs(next_out)
                curr_t0, curr_t1 = temp_pair(batch_index, lane)
                lines.extend(
                    [
                        f"\tv_exp_f32_e32 v{next_t0}, v{next_t0}",
                        (
                            f"\tv_pk_mul_f32 v[{curr_gate0}:{curr_gate1_slot}], "
                            f"v[{curr_gate0}:{curr_gate1_slot}], "
                            f"v[{curr_t0}:{curr_t1}]"
                        ),
                        f"\tv_exp_f32_e32 v{next_t1}, v{next_t1}",
                        (
                            f"\tv_pk_mul_f32 v[{curr_gate0}:{curr_gate1_slot}], "
                            f"v[{curr_gate0}:{curr_gate1_slot}], "
                            f"v[{curr_up0}:{curr_up1}]"
                        ),
                        f"\tv_med3_num_f32 v{next_up1}, v{next_up1}, s103, s102",
                        (
                            f"\tv_cvt_pk_bf16_f32 v{100 + curr_out}, "
                            f"v{curr_gate0}, v{curr_gate1_slot}"
                        ),
                    ]
                )

            lines.extend(add_denominators(next_index))
        else:
            # The final batch has no following work to fill Trans32 slots.
            # Issue all reciprocals first, then consume them after sufficient
            # independent distance.
            for lane in range(8):
                t0, t1 = temp_pair(batch_index, lane)
                lines.extend(
                    [
                        f"\tv_rcp_f32_e32 v{t0}, v{t0}",
                        f"\tv_rcp_f32_e32 v{t1}, v{t1}",
                    ]
                )
            lines.extend(finish_batch(batch_start, batch_index))

    return "\n".join(lines)


text = BASELINE.read_text(encoding="utf-8")
if text.count(MULTICAST_DISABLED) != 8:
    raise RuntimeError(
        f"expected 8 independent TDM sites, got {text.count(MULTICAST_DISABLED)}"
    )
text = text.replace(MULTICAST_DISABLED, MULTICAST_ENABLED)
if text.count(MULTICAST_ENABLED) != 8:
    raise RuntimeError("the optimized source must retain eight multicast descriptors")
if text.count("s_barrier_signal -3") != 3:
    raise RuntimeError("the inherited 4x4 cluster barrier protocol changed")

for bank in range(4):
    start_marker = f"\t; Accumulator bank {bank}: eight 16-column fragments become eight"
    start = text.index(start_marker)
    end = text.index("\ts_wait_alu depctr_va_vdst(0)", start)
    text = text[:start] + optimized_activation(bank) + "\n" + text[end:]

first_store = text.index("\ttensor_store_from_lds s[80:83], s[84:91]")
output_wait = text.index("\ts_wait_tensorcnt 0x0", first_store)
bank2 = text.index("\t; Optimized accumulator bank 2:", output_wait)
if output_wait > bank2:
    raise RuntimeError("output TensorCnt wait must precede bank-2 LDS reuse")

text = text.replace(
    "\t; MoE GEMM1 specialization for E=96, tokens=16384, topk=6, N=6144, K=7168.",
    "\t; Optimized MoE GEMM1: production multicast plus packed/pipelined SiLU.",
    1,
)
OUTPUT.write_text(text, encoding="utf-8", newline="\n")

print(f"wrote {OUTPUT}")
for mnemonic in (
    "s_bitset1_b32 s36, 21",
    "v_swap_b32",
    "v_dual_min_num_f32",
    "v_dual_add_f32",
    "v_pk_mul_f32",
    "v_exp_f32",
    "v_rcp_f32",
    "s_wait_tensorcnt 0x0",
):
    print(f"{mnemonic}: {text.count(mnemonic)}")
