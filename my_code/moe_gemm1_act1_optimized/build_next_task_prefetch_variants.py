#!/usr/bin/env python3
"""Build next-persistent-task input-prefetch variants from output-pad8."""

from __future__ import annotations

import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "persistent_overlap_output_pad8.s"
OUTPUT_STAGE0 = HERE / "persistent_overlap_pad8_prefetch_stage0.s"
OUTPUT_STAGE01 = HERE / "persistent_overlap_pad8_prefetch_stage01.s"
EXPECTED_SOURCE_SHA256 = (
    "039ed787b1f136ee402b76e3bd0b7c9bf439148a0156d0fcca856ed6ab25ccad"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_exact(text: str, old: str, new: str, *, count: int = 1) -> str:
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(f"expected {count} matches for {old!r}, got {actual}")
    return text.replace(old, new, count)


def make_early_prefetch(stage_count: int, source: str) -> str:
    if stage_count not in (1, 2):
        raise ValueError(stage_count)

    def original_descriptor_block(start_label: str, role: str) -> str:
        role_begin = source.index(start_label) + len(start_label)
        begin = source.index("\ts_mov_b32 s32, 1", role_begin)
        finish = source.index("\ts_mov_b32 s56,", begin)
        block = source[begin:finish].strip("\n")
        if role == "b":
            block = block.replace(
                ".Lbranch_000000002c70",
                ".Lmoe_next_prefetch_column_mask_loop_b",
            )
        elif role == "scale_b":
            block = block.replace(
                ".Lbranch_000000004520",
                ".Lmoe_next_prefetch_column_mask_loop_scale_b",
            )
        return block

    role_specs = {
        "a": dict(
            descriptor=original_descriptor_block(
                ".Lbranch_000000001f08:", "a"
            ),
            lds_stage1=0x08000,
            hint=" th:TH_LOAD_NT",
            base_lo=4,
            pointer_lo=72,
            expert_stride=0x380000,
            tile_register="s55",
            stride_register="s13",
            stage_step=0x800,
        ),
        "b": dict(
            descriptor=original_descriptor_block(
                ".Lbranch_000000002b48:", "b"
            ),
            lds_stage1=0x38000,
            hint="",
            base_lo=6,
            pointer_lo=74,
            expert_stride=0x1500000,
            tile_register="s54",
            stride_register="s14",
            stage_step=0x800,
        ),
        "scale_a": dict(
            descriptor=original_descriptor_block(
                ".Lbranch_0000000037ac:", "scale_a"
            ),
            lds_stage1=0x10800,
            hint="",
            base_lo=8,
            pointer_lo=76,
            expert_stride=0x38000,
            tile_register="s55",
            stride_register="s15",
            stage_step=0x100,
        ),
        "scale_b": dict(
            descriptor=original_descriptor_block(
                ".Lbranch_0000000043f8:", "scale_b"
            ),
            lds_stage1=0x22800,
            hint="",
            base_lo=10,
            pointer_lo=78,
            expert_stride=0x150000,
            tile_register="s54",
            stride_register="s16",
            stage_step=0x100,
        ),
    }

    blocks: list[str] = []
    for role, spec in role_specs.items():
        issue = [
            spec["descriptor"],
            f"\ttensor_load_to_lds s[32:35], s[36:43]{spec['hint']}",
        ]
        if stage_count == 2:
            issue.extend(
                [
                    f"\ts_add_co_u32 s34, s34, 0x{spec['stage_step']:x}",
                    "\ts_add_co_ci_u32 s35, s35, 0",
                    f"\ts_mov_b32 s33, 0x{spec['lds_stage1']:x}",
                    f"\ttensor_load_to_lds s[32:35], s[36:43]{spec['hint']}",
                    f"\ts_sub_co_u32 s34, s34, 0x{spec['stage_step']:x}",
                    "\ts_sub_co_ci_u32 s35, s35, 0",
                    f"\ts_mov_b32 s33, 0x{spec['lds_stage1'] - (0x8000 if role in ('a', 'b') else 0x800):x}",
                ]
            )
        issue.append("\ts_branch .Lmoe_next_prefetch_done")
        blocks.append(f".Lmoe_next_prefetch_{role}:\n" + "\n".join(issue))

    return f"""
\t; Prefetch stage {'0/1' if stage_count == 2 else '0'} of the next persistent
\t; task after the current task's final cluster synchronization and output
\t; descriptor materialization.  Advance the complete persistent scalar state
\t; here so the following task can reuse it instead of repeating its prologue.
\t; The original descriptor construction is retained exactly.
\ts_add_co_u32 s29, s28, 16
\ts_cmp_lt_u32 s29, 0x240
\ts_cbranch_scc0 .Lmoe_next_prefetch_done
+
\t; For a +16 persistent stride, local M is unchanged.  If the current
\t; task's remainder modulo 24 is below 8, N advances by 16 in the same
\t; expert; otherwise N wraps by -8 and the expert advances by four.
\ts_mul_hi_u32 s30, s28, 0xaaaaaaab
\ts_lshr_b32 s30, s30, 4
\ts_mul_i32 s31, s30, 24
\ts_sub_co_u32 s31, s28, s31
\ts_add_co_u32 s46, s54, 16
\ts_sub_co_u32 s47, s54, 8
\ts_cmp_lt_u32 s31, 8
\ts_cselect_b32 s54, s46, s47
\ts_cselect_b32 s29, 0, 4
+
\t; Advance every expert-local base. s55 is unchanged by the +16 stride.
\ts_mul_i32 s24, s29, 0x600000
\ts_mov_b32 s25, 0
\ts_add_nc_u64 s[2:3], s[2:3], s[24:25]
\ts_mul_i32 s24, s29, 0x380000
\ts_add_nc_u64 s[4:5], s[4:5], s[24:25]
\ts_mul_i32 s24, s29, 0x1500000
\ts_add_nc_u64 s[6:7], s[6:7], s[24:25]
\ts_mul_i32 s24, s29, 0x38000
\ts_add_nc_u64 s[8:9], s[8:9], s[24:25]
\ts_mul_i32 s24, s29, 0x150000
\ts_add_nc_u64 s[10:11], s[10:11], s[24:25]
+
\t; Materialize all four tensor bases for the next task.
\ts_mul_i32 s24, s55, 0x100
\ts_mul_hi_u32 s73, s24, s13
\ts_mul_i32 s24, s24, s13
\ts_add_co_u32 s72, s4, s24
\ts_add_co_ci_u32 s73, s73, s5
\ts_mul_i32 s24, s55, 0x100
\ts_mul_hi_u32 s77, s24, s15
\ts_mul_i32 s24, s24, s15
\ts_add_co_u32 s76, s8, s24
\ts_add_co_ci_u32 s77, s77, s9
\ts_mul_i32 s24, s54, 0x100
\ts_mul_hi_u32 s75, s24, s14
\ts_mul_i32 s24, s24, s14
\ts_add_co_u32 s74, s6, s24
\ts_add_co_ci_u32 s75, s75, s7
\ts_mul_i32 s24, s54, 0x100
\ts_mul_hi_u32 s79, s24, s16
\ts_mul_i32 s24, s24, s16
\ts_add_co_u32 s78, s10, s24
\ts_add_co_ci_u32 s79, s79, s11
+
\ts_cmp_eq_u32 s22, 0
\ts_cbranch_scc1 .Lmoe_next_prefetch_a
\ts_cmp_eq_u32 s22, 1
\ts_cbranch_scc1 .Lmoe_next_prefetch_b
\ts_cmp_eq_u32 s22, 2
\ts_cbranch_scc1 .Lmoe_next_prefetch_scale_a
\ts_branch .Lmoe_next_prefetch_scale_b
+
+{"\n\n".join(blocks)}
+.Lmoe_next_prefetch_done:
+""".replace("\n+", "\n")

def conditional_initial_wait(role: str, count: int, barrier_line: str) -> str:
    return f"""\ts_cmp_eq_u32 s101, 0
\ts_cbranch_scc0 .Lmoe_{role}_prefetched_input_wait
\ts_wait_tensorcnt 0x0
\ts_branch .Lmoe_{role}_input_wait_done
.Lmoe_{role}_prefetched_input_wait:
\ts_wait_tensorcnt 0x{count:x}
.Lmoe_{role}_input_wait_done:
{barrier_line}"""


def guarded_issue(role: str, stage: int, instruction: str) -> str:
    return f"""\ts_cmp_eq_u32 s101, 0
\ts_cbranch_scc0 .Lmoe_{role}_stage{stage}_already_prefetched
{instruction}
.Lmoe_{role}_stage{stage}_already_prefetched:"""


def conditional_old_output_wait(role: str, count: int, instruction: str) -> str:
    return f"""\ts_cmp_eq_u32 s101, 0
\ts_cbranch_scc1 .Lmoe_{role}_old_output_already_safe
\ts_wait_tensorcnt 0x{count:x}
.Lmoe_{role}_old_output_already_safe:
{instruction}"""


def build_variant(source: str, *, stage_count: int) -> str:
    wait_count = 2 if stage_count == 1 else 3
    old_output_wait = 1 if stage_count == 1 else 0
    name = "stage-0" if stage_count == 1 else "stage-0/1"

    text = replace_exact(
        source,
        "\t; Persistent MoE GEMM1 output-pad8 candidate.",
        f"\t; Persistent MoE GEMM1 output-pad8 with next-task {name} prefetch.",
    )
    text = replace_exact(
        text,
        ".Lmoe_persistent_task:\n"
        "\ts_cmp_lt_u32 s28, 0x240",
        ".Lmoe_persistent_task:\n"
        "\ts_mov_b32 s101, 0\n"
        "\ts_branch .Lmoe_persistent_task_full_setup\n"
        ".Lmoe_persistent_task_prefetched:\n"
        "\ts_mov_b32 s101, 1\n"
        "\ts_branch .Lmoe_persistent_state_ready\n"
        ".Lmoe_persistent_task_full_setup:\n"
        "\ts_cmp_lt_u32 s28, 0x240",
    )
    text = replace_exact(
        text,
        "\n\t; Legacy core scalar contract for one expert-local 1024x6144x7168 GEMM.",
        "\n.Lmoe_persistent_state_ready:\n"
        "\t; Legacy core scalar contract for one expert-local 1024x6144x7168 GEMM.",
    )

    wait_sites = {
        "a": "\ts_barrier_signal -1                                        ; 0000000020AC: BE804EC1",
        "b": "\ts_barrier_signal -1                                        ; 000000002D10: BE804EC1",
        "scale_a": "\ts_barrier_signal -1                                        ; 000000003958: BE804EC1",
        "scale_b": "\ts_barrier_signal -1                                        ; 0000000045C0: BE804EC1",
    }
    for role, barrier in wait_sites.items():
        descriptor_entry = {
            "a": ".Lbranch_000000001f08:",
            "b": ".Lbranch_000000002b48:",
            "scale_a": ".Lbranch_0000000037ac:",
            "scale_b": ".Lbranch_0000000043f8:",
        }[role]
        text = replace_exact(
            text,
            descriptor_entry,
            descriptor_entry
            + f"\n\ts_cmp_eq_u32 s101, 0\n"
            + f"\ts_cbranch_scc0 .Lmoe_{role}_prefetched_descriptor_ready",
        )
        text = replace_exact(
            text,
            "\ts_wait_tensorcnt 0x0\n" + barrier,
            f".Lmoe_{role}_prefetched_descriptor_ready:\n"
            + conditional_initial_wait(role, wait_count, barrier),
        )

    # Scale waves have a second historical full drain after their initial
    # readiness barriers.  The prefetched path has already established I0
    # readiness and must retain O0/O1 overlap, while the first task keeps the
    # original drain unchanged.
    for role, wait_line in (
        ("scale_a", "\ts_wait_tensorcnt 0x0                                       ; 000000003964: BFCB0000"),
        ("scale_b", "\ts_wait_tensorcnt 0x0                                       ; 0000000045CC: BFCB0000"),
    ):
        text = replace_exact(
            text,
            wait_line,
            f"\ts_cmp_eq_u32 s101, 0\n"
            f"\ts_cbranch_scc0 .Lmoe_{role}_skip_redundant_full_drain\n"
            f"{wait_line}\n"
            f".Lmoe_{role}_skip_redundant_full_drain:",
        )

    stage0_sites = {
        "a": "\ttensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 0000000020C0: D0310000 00100000 7C7C2420",
        "b": "\ttensor_load_to_lds s[32:35], s[36:43]                      ; 000000003530: D0310000 00000000 7C7C2420",
        "scale_a": "\ttensor_load_to_lds s[32:35], s[36:43]                      ; 00000000417C: D0310000 00000000 7C7C2420",
        "scale_b": "\ttensor_load_to_lds s[32:35], s[36:43]                      ; 000000004DE4: D0310000 00000000 7C7C2420",
    }
    stage1_sites = {
        "a": "\ttensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 00000000210C: D0310000 00100000 7C7C2420",
        "b": "\ttensor_load_to_lds s[32:35], s[36:43]                      ; 00000000357C: D0310000 00000000 7C7C2420",
        "scale_a": "\ttensor_load_to_lds s[32:35], s[36:43]                      ; 0000000041C8: D0310000 00000000 7C7C2420",
        "scale_b": "\ttensor_load_to_lds s[32:35], s[36:43]                      ; 000000004E30: D0310000 00000000 7C7C2420",
    }
    stage2_sites = {
        "a": "\ttensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 000000002158: D0310000 00100000 7C7C2420",
        "b": "\ttensor_load_to_lds s[32:35], s[36:43]                      ; 0000000035C8: D0310000 00000000 7C7C2420",
        "scale_a": "\ttensor_load_to_lds s[32:35], s[36:43]                      ; 000000004214: D0310000 00000000 7C7C2420",
        "scale_b": "\ttensor_load_to_lds s[32:35], s[36:43]                      ; 000000004E7C: D0310000 00000000 7C7C2420",
    }
    for role, instruction in stage0_sites.items():
        text = replace_exact(text, instruction, guarded_issue(role, 0, instruction))
    if stage_count == 2:
        for role, instruction in stage1_sites.items():
            text = replace_exact(text, instruction, guarded_issue(role, 1, instruction))
    for role, instruction in stage2_sites.items():
        text = replace_exact(
            text,
            instruction,
            conditional_old_output_wait(role, old_output_wait, instruction),
        )

    insertion_marker = (
        ".Lbranch_00000000ad68:\n"
        "\t; Activated output uses a 144-byte LDS row pitch: 128-byte payload + 16-byte skew."
    )
    text = replace_exact(
        text,
        insertion_marker,
        ".Lbranch_00000000ad68:\n"
        + make_early_prefetch(stage_count, source)
        + "\t; Activated output uses a 144-byte LDS row pitch: 128-byte payload + 16-byte skew.",
    )
    text = replace_exact(
        text,
        "\ts_branch .Lmoe_persistent_task\n"
        ".Lmoe_persistent_final_drain:",
        "\ts_branch .Lmoe_persistent_task_prefetched\n"
        ".Lmoe_persistent_final_drain:",
    )
    return text


def main() -> None:
    actual = sha256(SOURCE)
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "output-pad8 source checksum changed: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {actual}"
        )
    source = SOURCE.read_text(encoding="utf-8")
    for stage_count, output in ((1, OUTPUT_STAGE0), (2, OUTPUT_STAGE01)):
        output.write_text(
            build_variant(source, stage_count=stage_count),
            encoding="utf-8",
            newline="\n",
        )
        print(f"wrote {output}")
        print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
