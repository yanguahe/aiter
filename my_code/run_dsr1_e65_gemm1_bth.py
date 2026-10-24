#!/usr/bin/env python3
"""Exercise an existing B-only TDM hint on GEMM1 with the benchmark unchanged."""

import argparse
import functools
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", choices=("bench", "verify"), required=True)
    parser.add_argument("--tdm-b-th", type=int, choices=(0, 1, 2, 4, 5, 6), required=True)
    args, arguments = parser.parse_known_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from aiter.ops.flydsl import grouped_gemm_mxfp4 as gemm

    launch = gemm.flydsl_grouped_gemm_a8w4_masked

    @functools.wraps(launch)
    def with_gemm1_hint(*positional, **kwargs):
        if kwargs.get("stage1_act") == 1 and kwargs.get("stage1_quant_out") == 1:
            kwargs = dict(kwargs, tdm_b_th=args.tdm_b_th)
        return launch(*positional, **kwargs)

    gemm.flydsl_grouped_gemm_a8w4_masked = with_gemm1_hint
    print(f"[GEMM1 B-only TDM hint] tdm_b_th={args.tdm_b_th}; GEMM2 unchanged", flush=True)
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
