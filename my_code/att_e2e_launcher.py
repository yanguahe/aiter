#!/usr/bin/env python3
"""Minimal profiler-free MoE launcher for fused GEMM1 ATT capture."""

from __future__ import annotations

import importlib
import os

os.environ.setdefault("ENABLE_CK", "0")
os.environ.setdefault("AITER_MOE_EXPERT_BALANCE", "true")
os.environ.setdefault("AITER_LOG_MORE", "0")
os.environ.setdefault("AITER_USE_GROUPED_GEMM", "1")
os.environ.setdefault("AITER_GROUPED_DEBUG", "0")
os.environ.setdefault("AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE", "1")
os.environ.setdefault("AITER_FLYDSL_GEMM1_FUSED_QUANT", "1")

from aiter import ActivationType

# ATT owns the device while the target process initializes. Runtime discovery
# helpers that launch HIP queries can therefore observe no agents. The host
# preflight already established the target as one 256-CU gfx1250 device, so use
# those fixed values only in this trace launcher.
chip_info = importlib.import_module("aiter.jit.utils.chip_info")
chip_info.get_gfx_runtime = lambda: "gfx1250"
chip_info.get_cu_num = lambda: 256
fused_moe_module = importlib.import_module("aiter.fused_moe")
fused_moe_module.get_gfx_runtime = lambda: "gfx1250"
fused_moe_module.get_cu_num = lambda: 256
grouped_moe = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
grouped_moe.get_cu_num = lambda: 256
moe_test = importlib.import_module("my_code.test_flydsl_grouped_gemm_gfx1250")


def main() -> int:
    for launch in range(2):
        result = moe_test._run_grouped_via_fused_moe(
            experts=64,
            tokens=1536,
            topk=8,
            model_dim=7168,
            inter_dim=2048,
            data_format="a4w4",
            activation=ActivationType.Silu,
            use_bias=False,
            bench=False,
            kernel_bench=False,
            seed=0,
            warmup=0,
            iters=1,
            const_init=0.0,
        )
        out, reference = result[:2]
        logits_diff = moe_test._logits_diff(out, reference)
        rel_l2 = moe_test._rel_l2(out, reference)
        print(
            f"ATT_LAUNCH={launch} logits_diff={logits_diff:.4e} "
            f"rel_l2={rel_l2:.4e}",
            flush=True,
        )
        if logits_diff >= moe_test.LOGITS_DIFF_TOL:
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
