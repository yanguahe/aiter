#!/usr/bin/env python3
"""Test LLVM mode-2 scheduling on GEMM1 with the original benchmark unchanged."""

import argparse
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", choices=("bench", "verify"), required=True)
    args, arguments = parser.parse_known_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from aiter.ops.flydsl.kernels.mxfp4_preshuffle_gfx1250_tdm_prefill_persistent import (
        launch_gemm_a8w4_tdm_fused_persistent,
    )

    options = launch_gemm_a8w4_tdm_fused_persistent.compile_hints["llvm_options"]
    assert options["amdgpu-expert-scheduling-mode"] is False
    options["amdgpu-expert-scheduling-mode"] = True
    print("[GEMM1 compile options] " + json.dumps(options), flush=True)
    script = repo / "my_code" / (
        "test_flydsl_grouped_gemm_gfx1250.py"
        if args.script == "bench"
        else "verify_dsr1_e65_gemm1.py"
    )
    remaining = arguments[1:] if arguments[:1] == ["--"] else arguments
    sys.argv = [str(script), *remaining]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
