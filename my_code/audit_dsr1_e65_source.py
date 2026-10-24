#!/usr/bin/env python3
"""Audit unchanged GEMM2/math and the documented profiler-name-only edit."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]
KERNEL = "aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py"


def head_bytes(relative):
    return subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=REPO)


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == name)


def main():
    original = head_bytes(KERNEL)
    current = (REPO / KERNEL).read_bytes()
    source = ast.parse(current)
    base = ast.parse(original)
    checks = {}
    for name in ("_amax_inputs_from_bf16x16", "launch_gemm_a8w4_tdm_gemm2_persistent"):
        checks[name] = ast.dump(function(source, name)) == ast.dump(function(base, name))
    for relative in ("my_code/test_flydsl_grouped_gemm_gfx1250.py",
                     "aiter/ops/flydsl/grouped_moe_gfx1250.py",
                     "aiter/ops/flydsl/kernels/quant_utils.py",
                     "aiter/ops/flydsl/kernels/gemm_common_gfx1250.py"):
        if relative == "my_code/test_flydsl_grouped_gemm_gfx1250.py":
            # Runtime expert-count symbols must be recognized. Every other
            # benchmark statement, including timing and reference math, stays
            # bytecode-AST-equivalent to HEAD.
            allowed = {"_TDM_KERNEL_RE", "_parse_tdm_kernel_name",
                       "_extract_profiled_gemm_timings",
                       "test_profiled_shared_expert_gemm1_timing_extraction_fixture"}

            def retained_nodes(data):
                return [ast.dump(node) for node in ast.parse(data).body if not (
                    isinstance(node, ast.FunctionDef) and node.name in allowed
                    or isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id in allowed
                        for target in node.targets))]

            checks["benchmark_timing_inputs_math_and_reference_AST"] = (
                retained_nodes((REPO / relative).read_bytes())
                == retained_nodes(head_bytes(relative))
            )
        else:
            checks[relative] = (REPO / relative).read_bytes() == head_bytes(relative)
    print(json.dumps({"checks": checks, "kernel_sha256": hashlib.sha256(current).hexdigest()}, indent=2))
    if not all(checks.values()):
        raise RuntimeError("a source outside the GEMM1 optimization changed")


if __name__ == "__main__":
    main()
