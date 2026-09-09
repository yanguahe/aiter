#!/usr/bin/env python3
"""Run the gfx1250 grouped-MoE test with an isolated GEMM1 ISA candidate."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_ISA = (
    HERE
    / "moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_opt.s"
)
TEST = REPO / "op_tests" / "test_flydsl_grouped_gemm_gfx1250.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Patch the existing C++ GEMM1 injection in-process so the normal "
            "grouped-MoE e2e test loads an ISA from this experiment directory."
        )
    )
    parser.add_argument("--isa", type=Path, default=DEFAULT_ISA)
    args, test_args = parser.parse_known_args()
    if test_args[:1] == ["--"]:
        test_args = test_args[1:]

    isa = args.isa.resolve()
    if not isa.is_file():
        raise SystemExit(f"ISA does not exist: {isa}")

    sys.path.insert(0, str(REPO))
    from my_code.isa_runner import gemm_batch_isa_runner as batch
    from my_code.isa_runner import moe_cpp_backend

    # moe_cpp_backend imports this same module and calls these values when it
    # prepares the scoped injection.  Override them only in this process; no
    # tracked production source or checked-in kernel is replaced.
    batch.MOE_CPP_ISA_BASENAME = isa.name
    batch.moe_cpp_target_isa = lambda: isa

    # The current grouped launcher forwards these two layout/scheduling facts.
    # The assembly adapter consumes tensors that are already preshuffled and
    # uses its fixed 1024-row/expert launch contract, so accept and validate the
    # facts here before calling the audited adapter implementation.
    original_call = moe_cpp_backend.MoePipelineGemm1Adapter.__call__

    def call_with_layout_contract(self, *call_args, **call_kwargs):
        a_preshuffle = int(call_kwargs.pop("a_preshuffle", 0))
        balanced_rows = int(call_kwargs.pop("balanced_rows_per_expert", 0))
        if int(call_kwargs.get("stage1_act", 0)) == 0:
            return original_call(self, *call_args, **call_kwargs)
        if a_preshuffle != 1:
            raise ValueError(
                "isolated ASM GEMM1 requires a_preshuffle=1, "
                f"got {a_preshuffle}"
            )
        if balanced_rows != moe_cpp_backend.PIPELINE_ROWS_PER_EXPERT:
            raise ValueError(
                "isolated ASM GEMM1 requires balanced_rows_per_expert="
                f"{moe_cpp_backend.PIPELINE_ROWS_PER_EXPERT}, got {balanced_rows}"
            )
        # The current Python path narrows its FlyDSL launch domain to the
        # balanced 96*1024 valid rows.  This audited assembly keeps the older
        # 122880-row physical launch contract; its extra clusters read the
        # sentinel entries in m_tile_map and exit before touching payloads.
        requested_m = int(call_kwargs.get("contiguous_m", 0))
        balanced_m = (
            moe_cpp_backend.PIPELINE_EXPERTS
            * moe_cpp_backend.PIPELINE_ROWS_PER_EXPERT
        )
        if requested_m != balanced_m:
            raise ValueError(
                f"isolated ASM GEMM1 expected balanced launch M={balanced_m}, "
                f"got {requested_m}"
            )
        call_kwargs["contiguous_m"] = moe_cpp_backend.PIPELINE_CONTIGUOUS_M
        return original_call(self, *call_args, **call_kwargs)

    moe_cpp_backend.MoePipelineGemm1Adapter.__call__ = call_with_layout_contract

    os.environ["AITER_MOE_GEMM1_LAUNCH_BACKEND"] = "cpp"
    print(f"[run_e2e_candidate] GEMM1 ISA override: {isa}", flush=True)
    sys.argv = [str(TEST), *test_args]
    runpy.run_path(str(TEST), run_name="__main__")


if __name__ == "__main__":
    main()
