#!/usr/bin/env python3

"""Run the grouped-MoE test with the in-tree 93665e GEMM1 implementation."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiter.ops.flydsl.grouped_gemm_mxfp4 import use_gemm1_baseline_93665e


use_gemm1_baseline_93665e()
test_script = REPO_ROOT / "op_tests" / "test_flydsl_grouped_gemm_gfx1250.py"
sys.argv[0] = str(test_script)
runpy.run_path(str(test_script), run_name="__main__")
