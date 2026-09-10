#!/usr/bin/env python3
"""Build persistent-task variants of the validated double-output-LDS kernel."""

from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_double_lds.s"
PERSISTENT = HERE / "moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent.s"
PERSISTENT_OVERLAP = HERE / "moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent_overlap.s"


def replace_exact(text: str, old: str, new: str, *, count: int = 1) -> str:
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(f"expected {count} matches for {old!r}, got {actual}")
    return text.replace(old, new, count)


def build_persistent() -> str:
    text = SOURCE.read_text(encoding="utf-8")
    text = replace_exact(
        text,
        "\t; Optimized MoE GEMM1 with double-buffered output LDS/TDM overlap.\n"
        "\t; It keeps the optimized ABpreShuffle compute core, consumes the production\n"
        "\t; 184-byte FlyDSL ABI, applies SiLU(gate)*up, and writes BF16 [M,3072].\n"
        "\t; Physical grid=(2880,4,1), block=(128,1,1), cluster=(4,4,1).",
        "\t; Persistent MoE GEMM1 with double-buffered output LDS/TDM overlap.\n"
        "\t; It keeps the optimized ABpreShuffle compute core, consumes the production\n"
        "\t; 184-byte FlyDSL ABI, applies SiLU(gate)*up, and writes BF16 [M,3072].\n"
        "\t; Physical grid=(16,16,1), block=(128,1,1), cluster=(4,4,1).\n"
        "\t; Sixteen resident clusters stride over 576 useful logical cluster tasks.",
    )

    start = text.index("\ts_load_b64 s[4:5], s[0:1], 0x28 nv")
    end = text.index("\n\t; Legacy core scalar contract", start)
    persistent_setup = """
\t; The 16 physical 4x4 clusters form a 4x4 cluster grid.  Each cluster
\t; starts at its flattened physical ID and advances by 16 through the 576
\t; useful DeepGEMM-swizzled logical cluster tasks.
\ts_and_b32 s28, ttmp7, 0xffff
\ts_lshl_b32 s28, s28, 2
\ts_add_co_u32 s28, s28, ttmp9

.Lmoe_persistent_task:
\ts_cmp_lt_u32 s28, 0x240
\ts_cbranch_scc0 .Lmoe_persistent_done

\t; Reload unmodified tensor bases and the SiLU limit for this task. gfx1250
\t; exposes only s0..s105 to this kernel, so all five bases cannot remain in
\t; extra SGPRs across iterations.
\ts_load_b64 s[2:3], s[0:1], 0x0 nv
\ts_load_b64 s[4:5], s[0:1], 0x28 nv
\ts_load_b64 s[6:7], s[0:1], 0x30 nv
\ts_load_b64 s[8:9], s[0:1], 0x38 nv
\ts_load_b64 s[10:11], s[0:1], 0x60 nv
\ts_load_b32 s102, s[0:1], 0xac nv
\ts_wait_kmcnt 0x0
\ts_xor_b32 s103, s102, 0x80000000

\t; Reconstruct the original local workitem ID after the prior task has
\t; reused low VGPRs for operands and epilogue temporaries.
\ts_bfe_u32 s22, ttmp8, 0x50019
\ts_mov_b32 exec_lo, -1
\ts_set_vgpr_msb 0
\tv_mbcnt_lo_u32_b32 v0, -1, 0
\tv_mov_b32_e32 v1, s22
\tv_lshlrev_b32_e32 v1, 5, v1
\tv_or_b32_e32 v0, v0, v1

\t; Preserve the production cluster-granular DeepGEMM 16-M-tile swizzle.
\ts_and_b32 s49, ttmp6, 15
\ts_bfe_u32 s48, ttmp6, 0x40004
\ts_mul_hi_u32 s35, s28, 0xaaaaaaab
\ts_lshr_b32 s35, s35, 4
\ts_mul_i32 s46, s35, 24
\ts_sub_co_u32 s47, s28, s46
\ts_lshr_b32 s54, s47, 2
\ts_lshl_b32 s54, s54, 2
\ts_add_co_u32 s54, s54, s49
\ts_and_b32 s34, s47, 3
\ts_lshl_b32 s34, s34, 2
\ts_lshl_b32 s55, s35, 4
\ts_add_co_u32 s55, s55, s34
\ts_add_co_u32 s55, s55, s48

\t; Balanced E96/T16384 routing gives four M256 tiles per expert.
\ts_lshr_b32 s32, s55, 2
\ts_cmp_ge_u32 s32, 0x60
\ts_cbranch_scc1 .Lmoe_persistent_done
\ts_and_b32 s55, s55, 3

\t; Rebase C/A/B/ScaleA/ScaleB to the owning expert. A and ScaleA use the
\t; ABpreShuffle layouts emitted by the paired quant kernel.
\ts_mul_i32 s34, s32, 0x600000
\ts_mov_b32 s35, 0
\ts_add_nc_u64 s[2:3], s[2:3], s[34:35]
\ts_mul_i32 s34, s32, 0x380000
\ts_add_nc_u64 s[4:5], s[4:5], s[34:35]
\ts_mul_i32 s34, s32, 0x1500000
\ts_add_nc_u64 s[6:7], s[6:7], s[34:35]
\ts_mul_i32 s34, s32, 0x38000
\ts_add_nc_u64 s[8:9], s[8:9], s[34:35]
\ts_mul_i32 s34, s32, 0x150000
\ts_add_nc_u64 s[10:11], s[10:11], s[34:35]
"""
    text = text[:start] + persistent_setup + text[end:]
    text = replace_exact(
        text,
        "\ts_mov_b32 s28, 0\n\ts_mov_b32 s29, 1",
        "\t; s28 retains the persistent logical-cluster task ID.\n"
        "\ts_mov_b32 s29, 1",
    )

    old_tail = """\ts_add_co_u32 s81, s81, 0x2000
\ttensor_store_from_lds s[80:83], s[84:91]                   ; second output half
.Lbranch_00000000ba60:
.Lmoe_256_tail:
\ts_wait_idle
\ts_endpgm
"""
    new_tail = """\ts_add_co_u32 s81, s81, 0x2000
\ttensor_store_from_lds s[80:83], s[84:91]                   ; second output half
\t; Safe persistent boundary: drain this task before any following task can
\t; reuse its LDS, then keep the complete 4x4 cluster in lockstep.
\ts_wait_idle
\ts_add_co_u32 s28, s28, 16
\ts_cmp_lt_u32 s28, 0x240
\ts_cbranch_scc0 .Lmoe_persistent_done
\ts_cmp_eq_u32 s22, 0
\ts_cbranch_scc0 .Lmoe_persistent_boundary_wait
\ts_barrier_signal -3
.Lmoe_persistent_boundary_wait:
\ts_barrier_wait 0xfffd
\ts_branch .Lmoe_persistent_task
.Lbranch_00000000ba60:
.Lmoe_256_tail:
\ts_wait_idle
.Lmoe_persistent_done:
\ts_endpgm
"""
    text = replace_exact(text, old_tail, new_tail)
    return text


def build_overlap(persistent: str) -> str:
    text = persistent.replace(
        "\t; Persistent MoE GEMM1 with double-buffered output LDS/TDM overlap.",
        "\t; Persistent MoE GEMM1 with cross-task output-drain overlap.",
        1,
    )
    # Delay the previous task's output-TDM wait until immediately before the
    # next task enters its first workgroup/cluster barrier and input TDM load.
    # All address/descriptor setup and the cache prefetch above these sites can
    # overlap the prior output store.  TENSORcnt is sufficient here: it protects
    # LDS bytes consumed by tensor_store_from_lds without unnecessarily draining
    # unrelated ALU state.
    first_task_barriers = (
        "\ts_barrier_signal -1                                        ; 0000000020AC: BE804EC1",
        "\ts_barrier_signal -1                                        ; 000000002D10: BE804EC1",
        "\ts_barrier_signal -1                                        ; 000000003958: BE804EC1",
        "\ts_barrier_signal -1                                        ; 0000000045C0: BE804EC1",
    )
    for barrier in first_task_barriers:
        text = replace_exact(
            text,
            barrier,
            "\ts_wait_tensorcnt 0x0\n" + barrier,
        )
    safe_boundary = """\t; Safe persistent boundary: drain this task before any following task can
\t; reuse its LDS, then keep the complete 4x4 cluster in lockstep.
\ts_wait_idle
\ts_add_co_u32 s28, s28, 16
\ts_cmp_lt_u32 s28, 0x240
\ts_cbranch_scc0 .Lmoe_persistent_done
\ts_cmp_eq_u32 s22, 0
\ts_cbranch_scc0 .Lmoe_persistent_boundary_wait
\ts_barrier_signal -3
.Lmoe_persistent_boundary_wait:
\ts_barrier_wait 0xfffd
\ts_branch .Lmoe_persistent_task
"""
    overlap_boundary = """\t; Keep the complete 4x4 cluster in lockstep after every output store.
\t; For a following task, leave the output TDM in flight while independent
\t; task/address setup executes.  The task prologue waits before issuing its
\t; first input TDM operation, so the reused 320 KiB LDS remains protected.
\ts_add_co_u32 s28, s28, 16
\ts_cmp_lt_u32 s28, 0x240
\ts_cbranch_scc0 .Lmoe_persistent_final_drain
\ts_cmp_eq_u32 s22, 0
\ts_cbranch_scc0 .Lmoe_persistent_overlap_boundary_wait
\ts_barrier_signal -3
.Lmoe_persistent_overlap_boundary_wait:
\ts_barrier_wait 0xfffd
\ts_branch .Lmoe_persistent_task
.Lmoe_persistent_final_drain:
\ts_wait_idle
\ts_branch .Lmoe_persistent_done
"""
    text = replace_exact(text, safe_boundary, overlap_boundary)
    return text


def main() -> None:
    persistent = build_persistent()
    PERSISTENT.write_text(persistent, encoding="utf-8", newline="\n")
    PERSISTENT_OVERLAP.write_text(
        build_overlap(persistent), encoding="utf-8", newline="\n"
    )
    print(f"wrote {PERSISTENT}")
    print(f"wrote {PERSISTENT_OVERLAP}")


if __name__ == "__main__":
    main()
