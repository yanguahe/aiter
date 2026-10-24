#!/usr/bin/env python3
"""Locate E64 post-profiler failures without changing production computation."""

import argparse
import functools
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from my_code.run_gemm1_e64_copy import install, elf_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    launcher, state, _ = install("copy")
    import torch
    from my_code import test_flydsl_grouped_gemm_gfx1250 as test
    from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped
    from aiter.ops.flydsl import grouped_gemm_mxfp4 as gemm
    from aiter.ops.flydsl import moe_kernels

    with (args.output / "events.jsonl").open("w", buffering=1) as events:
        def emit(event, **details):
            record = dict(event=event, time=time.monotonic(), **details)
            events.write(json.dumps(record) + "\n")
            print("[E64 diagnostic] " + json.dumps(record), flush=True)

        original_perf = test._run_perftest_with_trace_compat

        def observed_perf(*positional, **keywords):
            result = original_perf(*positional, **keywords)
            emit("profiler_returned")
            torch.cuda.synchronize()
            emit("profiler_fenced")
            return result

        test._run_perftest_with_trace_compat = observed_perf

        def wrap_capture(module, name):
            original = getattr(module, name)

            @functools.wraps(original)
            def observed(*positional, **keywords):
                active = grouped.stage_output_capture is not None
                if active:
                    emit("capture_stage_before", stage=name,
                         stage1_act=keywords.get("stage1_act"),
                         stage1_quant_out=keywords.get("stage1_quant_out"))
                result = original(*positional, **keywords)
                if active:
                    torch.cuda.synchronize()
                    emit("capture_stage_fenced", stage=name)
                return result

            setattr(module, name, observed)

        for module, name in ((gemm, "flydsl_grouped_gemm_a8w4_masked"),
                             (moe_kernels, "flydsl_moe_fused_quant_preshuffle"),
                             (grouped, "flydsl_moe_gather_reduce")):
            wrap_capture(module, name)

        def wrap_post(name):
            original = getattr(test, name)

            @functools.wraps(original)
            def observed(*positional, **keywords):
                torch.cuda.synchronize()
                emit("post_before", stage=name)
                result = original(*positional, **keywords)
                torch.cuda.synchronize()
                emit("post_fenced", stage=name)
                return result

            setattr(test, name, observed)

        for name in ("_indexed_tensor_hash128", "_tensor_hash128", "_torch_moe_ref"):
            wrap_post(name)
        emit("start", performance_data=False)
        test.set_data_format("a4w4")
        result = test._run_grouped_via_fused_moe(
            experts=64, tokens=1536, topk=8, shared_experts=0,
            model_dim=7168, inter_dim=2048, data_format="a4w4",
            activation=test.ActivationType.Silu, use_bias=False,
            bench=True, seed=0, warmup=5, iters=20)
        accuracy = test._logits_diff(result[0], result[1])
        assert accuracy < test.LOGITS_DIFF_TOL, accuracy
        assert state["calls"] and sum(result[-1]) == 12288
        cached = list(launcher._mem_cache.values())
        if not cached and launcher._last_compiled:
            cached = [launcher._last_compiled[1]]
        records = [hashlib.sha256(elf_bytes(
            (item if hasattr(item, "ir") else item[0]).ir)).hexdigest() for item in cached]
        record = dict(passed=True, performance_data=False, logits_diff=accuracy,
                      elf_sha256=records, routing_counts=result[-1])
        (args.output / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
        emit("complete", **record)


if __name__ == "__main__":
    main()
