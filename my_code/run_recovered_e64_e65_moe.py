#!/usr/bin/env python3
"""Verify and profile the original group8 E65 path and its isolated E64 copy."""

import argparse
import functools
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / ".codex_tmp/side_dual_moe_20261019"
SNAPSHOT = ROOT / "original_snapshot"
sys.path.insert(0, str(REPO))

GEMM1_PARAMETERS = (
    "arg_c", "arg_a", "arg_b", "arg_scale_a", "arg_scale_b", "i32_m", "stream",
    "N", "K", "tile_m", "tile_n", "tile_k", "m_warp", "n_warp", "out_is_f16",
    "num_buffers", "a_is_fp4", "arg_m_tile_map", "n_experts", "stage1_act",
    "has_bias", "arg_bias", "f32_swiglu_limit", "stage1_quant_out",
    "quant_wmma_rep", "arg_quant_scale", "cluster_n", "cluster_m",
    "next_stage_prefetch", "num_waves_per_tensor_tdm", "tdm_b_th",
)
TUNING = dict(persistent_n_tasks=1, staged_quant_scale=1, row_tail_predication=3,
              half_a_prefetch=1, half_scale_prefetch=1, row_tail_a_groups=2,
              pair_a_prefetch=1, task_m_order=1, task_m_group=8,
              balanced_m192=0, partial_m_direct_scale=0, partial_m_drain_output=0)
EXPECTED = {
    65: {"gemm1": "7c643f5cddb07e3e431ca7d0f28c6e78d2262588c629e16abf8083fccbe3c692",
         "gemm2": "840d2c768ad0b92ee446af222164513d8d190e75ddf5e6a664fc2b4b5a20cca0"},
    64: {"gemm1": "0c5952afb691424ddffb63736bf378a184447713a0b1d6c25506234f13da5949",
         "gemm2": "6982c0689aa1c8934ba07fa210b1b17d9210c79ffbc062e31acb5e999e84582c"},
}


def load_snapshot(name, relative):
    path = SNAPSHOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    parent, attribute = name.rsplit(".", 1)
    setattr(importlib.import_module(parent), attribute, module)
    return module


def install(experts):
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for relative, expected in manifest.items():
        if hashlib.sha256((REPO / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"source SHA256 mismatch: {relative}")
    if os.environ.get("AITER_MOE_EXPERT_BALANCE", "").lower() not in ("0", "false"):
        raise RuntimeError("random non-balanced routing requires AITER_MOE_EXPERT_BALANCE=false")
    if any(key.startswith("AITER_GFX1250_GEMM1_") for key in os.environ):
        raise RuntimeError("unrelated GEMM1 candidate environment variables are present")
    os.environ["AITER_MOE_NUM_SHARED_EXPERTS"] = str(int(experts == 65))
    import aiter.ops.flydsl.kernels
    persistent = load_snapshot(
        "aiter.ops.flydsl.kernels.mxfp4_preshuffle_gfx1250_tdm_prefill_persistent",
        "aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py")
    copied = load_snapshot(
        "aiter.ops.flydsl.kernels._recovered_gemm1_e64",
        "aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_gemm1_e64.py")
    load_snapshot("aiter.ops.flydsl.grouped_gemm_mxfp4",
                  "aiter/ops/flydsl/grouped_gemm_mxfp4.py")
    grouped = load_snapshot("aiter.ops.flydsl.grouped_moe_gfx1250",
                            "aiter/ops/flydsl/grouped_moe_gfx1250.py")
    original_gemm1 = persistent.launch_gemm_a8w4_tdm_fused_persistent
    gemm1 = copied.launch_gemm_a8w4_tdm_fused_persistent if experts == 64 else original_gemm1
    state = dict(calls=0, geometry=None, tuning=None)

    def selected(*positional, **keywords):
        arguments = dict(zip(GEMM1_PARAMETERS, positional))
        arguments.update(keywords)
        if arguments.get("stage1_act") != 1 or arguments.get("stage1_quant_out") != 1:
            return original_gemm1(*positional, **keywords)
        expected = dict(n_experts=experts, N=4096, K=7168,
                        i32_m=28672 if experts == 64 else 30464,
                        tile_m=256, tile_n=256, tile_k=256, m_warp=2, n_warp=4,
                        num_buffers=4, has_bias=0, a_is_fp4=1, cluster_n=4,
                        cluster_m=1, next_stage_prefetch=1, num_waves_per_tensor_tdm=2,
                        tdm_b_th=6 if experts == 64 else 0)
        assert all(arguments.get(key) == value for key, value in expected.items()), arguments
        launch_keywords = dict(keywords, **TUNING) if experts == 64 else keywords
        assert all(launch_keywords.get(key) == value for key, value in TUNING.items())
        state.update(calls=state["calls"] + 1, geometry=expected, tuning=TUNING)
        return gemm1(*positional, **launch_keywords)

    persistent.launch_gemm_a8w4_tdm_fused_persistent = selected
    test = load_snapshot("my_code.test_flydsl_grouped_gemm_gfx1250",
                         "my_code/test_flydsl_grouped_gemm_gfx1250.py")
    test.REPO_ROOT = str(REPO)
    snapshot_path = str(SNAPSHOT)
    if snapshot_path in sys.path:
        sys.path.remove(snapshot_path)
    if str(REPO) in sys.path:
        sys.path.remove(str(REPO))
    sys.path.insert(0, str(REPO))
    return test, grouped, gemm1, persistent.launch_gemm_a8w4_tdm_gemm2_persistent, state, manifest


def actual_bindings(gemm1, gemm2, experts, output):
    from my_code.run_gemm1_e64_copy import elf_bytes
    records = {}
    for stage, launcher in (("gemm1", gemm1), ("gemm2", gemm2)):
        cached = list(launcher._mem_cache.values())
        if not cached and launcher._last_compiled:
            cached = [launcher._last_compiled[1]]
        assert len(cached) == 1, (stage, len(cached))
        item = cached[0]
        compiled = item if hasattr(item, "ir") else item[0]
        data = elf_bytes(compiled.ir)
        digest = hashlib.sha256(data).hexdigest()
        assert digest == EXPECTED[experts][stage], (stage, digest, EXPECTED[experts][stage])
        (output / f"actual_{stage}.elf").write_bytes(data)
        records[stage] = dict(elf_sha256=digest, elf_bytes=len(data))
    return records


def check_accuracy(test, out, reference):
    import torch
    assert out.shape == reference.shape == (1536, 7168)
    assert out.dtype == reference.dtype == torch.bfloat16
    assert bool(torch.isfinite(out).all()) and bool(torch.isfinite(reference).all())
    logits_diff = test._logits_diff(out, reference)
    assert logits_diff < test.LOGITS_DIFF_TOL, logits_diff
    return dict(logits_diff=logits_diff, rel_l2=test._rel_l2(out, reference),
                original_gate=test.LOGITS_DIFF_TOL, shape=list(out.shape),
                dtype=str(out.dtype), finite=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--experts", type=int, choices=(64, 65), required=True)
    parser.add_argument("--stage", choices=("verify", "stress", "perf"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    test, grouped, gemm1, gemm2, state, manifest = install(args.experts)
    import torch
    torch.set_num_threads(8)
    test.set_data_format("a4w4")
    shape = dict(experts=args.experts, tokens=1536, topk=9 if args.experts == 65 else 8,
                 shared_experts=int(args.experts == 65), model_dim=7168, inter_dim=2048)
    record = dict(stage=args.stage, workload=shape, seed=args.seed, balanced=False,
                  random_inputs=True, source_manifest=manifest, performance_data=False)
    captured_results = []
    original_helper = test._run_grouped_via_fused_moe

    def observed_helper(**keywords):
        result = original_helper(**keywords)
        check_accuracy(test, result[0], result[1])
        assert sum(result[-1]) == shape["tokens"] * shape["topk"]
        if shape["shared_experts"]:
            assert result[-1][-1] == 1536
        captured_results.append(result)
        return result

    test._run_grouped_via_fused_moe = observed_helper
    if args.stage == "verify":
        assert args.compare and args.compare.is_file(), "a pinned effective baseline is required"
        verifier = load_snapshot("my_code._verify_recovered_dual_moe",
                                  "my_code/verify_dsr1_e65_gemm1.py")
        verifier.REPO_ROOT = REPO
        sys.argv = [str(verifier.__file__), "--experts", str(args.experts),
                    "--tokens", "1536", "--topk", str(shape["topk"]),
                    "--shared-experts", str(shape["shared_experts"]),
                    "--model-dim", "7168", "--inter-dim", "2048",
                    "--seed", str(args.seed), "--replays", "3", "--dynamic-replays", "3",
                    "--compare", str(args.compare)]
        rc = verifier.main()
        assert rc == 0, rc
        assert len(captured_results) == 1
        result = captured_results[0]
        record.update(accuracy=check_accuracy(test, result[0], result[1]),
                      route_counts=result[-1], baseline=str(args.compare))
    elif args.stage == "stress":
        with (args.output / "events.jsonl").open("w", buffering=1) as events:
            def emit(event, **details):
                item = dict(event=event, time=time.monotonic(), **details)
                events.write(json.dumps(item) + "\n")
                print("[recovered MoE stress] " + json.dumps(item), flush=True)

            full_calls = []
            original_perf = test._run_perftest_with_trace_compat

            def observed_perf(api, function, **keywords):
                full_calls.append(function)
                emit("profiler_before")
                result = original_perf(api, function, **keywords)
                emit("profiler_returned")
                return result

            test._run_perftest_with_trace_compat = observed_perf
            for name in ("_indexed_tensor_hash128", "_tensor_hash128", "_torch_moe_ref"):
                original = getattr(test, name)

                def observed_post(*positional, _original=original, _name=name, **keywords):
                    emit("post_before", operation=_name)
                    result = _original(*positional, **keywords)
                    emit("post_returned", operation=_name)
                    return result

                setattr(test, name, observed_post)
            rounds = []
            for seed in (args.seed, 17, 101):
                emit("full_workload_before", seed=seed)
                result = observed_helper(**shape, data_format="a4w4", activation=test.ActivationType.Silu,
                                         use_bias=False, bench=True, seed=seed, warmup=5, iters=20)
                accuracy = check_accuracy(test, result[0], result[1])
                full_call = full_calls[-1]
                expected = result[0].detach().clone()
                eager = []
                for repeat in range(3):
                    actual = full_call()
                    torch.cuda.synchronize()
                    assert torch.equal(actual, expected), "full e2e eager replay differs"
                    eager.append(0)
                current = torch.cuda.current_stream()
                graph_stream = torch.cuda.Stream()
                graph_stream.wait_stream(current)
                with torch.cuda.stream(graph_stream):
                    full_call()
                current.wait_stream(graph_stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=graph_stream):
                    graph_out = full_call()
                current.wait_stream(graph_stream)
                graph_differences = []
                for repeat in range(3):
                    graph_stream.wait_stream(current)
                    graph.replay()
                    current.wait_stream(graph_stream)
                    torch.cuda.synchronize()
                    assert torch.equal(graph_out, expected), "full e2e graph replay differs"
                    graph_differences.append(0)
                rounds.append(dict(seed=seed, accuracy=accuracy, route_counts=result[-1],
                                   full_eager_differing_bytes=eager,
                                   full_graph_differing_bytes=graph_differences))
                emit("full_workload_passed", seed=seed, accuracy=accuracy)
            record["rounds"] = rounds
    else:
        metrics = test.run_moe(
            "a4w4", **shape, activation=test.ActivationType.Silu, use_bias=False,
            check_aot_cache=False, raise_on_fail=True, bench=True, warmup=5, iters=20)
        assert metrics["passed"] and len(captured_results) == 1
        result = captured_results[0]
        profile = result[-2]
        assert profile is not None
        profile.to_json(args.output / "profile.json", orient="records", indent=2)
        for stage in ("gemm1", "gemm2"):
            assert metrics[stage + "_us"] > 0 and metrics[stage + "_count"] == 19
        assert metrics["us"] > 0
        record.update(performance_data=True, testGraph=False, warmup=5, iters=20,
                      gemm1_us=metrics["gemm1_us"], gemm2_us=metrics["gemm2_us"],
                      moe_e2e_us=metrics["us"], metrics=metrics,
                      accuracy=check_accuracy(test, result[0], result[1]),
                      route_counts=result[-1])
    assert state["calls"] > 0
    record.update(passed=True, binding=actual_bindings(gemm1, gemm2, args.experts, args.output),
                  gemm1_dispatch=state)
    (args.output / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print("[recovered dual MoE result] " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
