"""Shared data-independent MoE pipeline timing context.

This module deliberately imports neither gemm_isa_runner nor
gemm_batch_isa_runner, so both CLIs can call it without a circular import.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping


def _json_line(tag: str, value: Any) -> None:
    print(f"{tag}={json.dumps(value, sort_keys=True)}", flush=True)


def _cuda_rows(trace_df: Any) -> list[dict[str, Any]]:
    rows = []
    for record in trace_df.to_dict("records"):
        if str(record.get("device_type")).rsplit(".", 1)[-1] != "CUDA":
            continue
        rows.append(dict(record))
    return rows


def run_inmoe_context(
    *,
    torch_module: Any,
    device: int,
    symbol: str,
    target_shape: tuple[int, int, int],
    target_tensors: Mapping[str, Any],
    raw_target_launch: Callable[[], Any],
    profiler_target_launch: Callable[[], Any],
    target_warmup: int,
    iters: int,
    backend_label: str,
    synchronize: Callable[[], Any],
    mark_stream_synchronized: Callable[[], Any] | None,
    run_target_perftest: Callable[..., tuple[Any, float, dict[str, Any], Any]],
    select_target_profiler_timing: Callable[..., dict[str, Any]],
    timing_source: str,
    error_type: type[Exception] = RuntimeError,
    target_validation_hook: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Profile one external target inside the fixed, independent MoE context."""

    from my_code.isa_runner import moe_gemm1_pipeline_launch_compare as pipeline

    state = pipeline.FixedMoePipeline(torch_module, device)
    try:
        isolation = state.assert_external_runner_isolation(dict(target_tensors))
        _json_line(
            "INMOE_CONTEXT_CONTRACT",
            {
                "target_symbol": symbol,
                "target_shape": target_shape,
                "target_allocations": sorted(target_tensors),
                "context_shape": {
                    "tokens": pipeline.TOKENS,
                    "experts": pipeline.EXPERTS,
                    "model_dim": pipeline.MODEL_DIM,
                    "inter_dim": pipeline.INTER_DIM,
                    "random_y1_shape": pipeline.Y1_SHAPE,
                },
                "execution_order": (
                    "pipeline frontend -> independent external target -> "
                    "pipeline downstream(random_y1)"
                ),
                "tensor_data_dependency": False,
            },
        )

        # Compile and smoke only the fixed context. This is not target warmup.
        state.warm("random", iterations=1)
        print(
            "[inmoe] pipeline lazy-JIT smoke=1 "
            "(pipeline-only; not --warmup)"
        )

        for _ in range(target_warmup):
            raw_target_launch()
        synchronize()
        if mark_stream_synchronized is not None:
            mark_stream_synchronized()
        print(
            f"[inmoe] target-only warmup={target_warmup}; "
            "pipeline launches=0 during --warmup"
        )

        # This untimed trace both proves placement and rejects symbol collision:
        # exactly one device event must match the external target symbol.
        order = state.verify_external_kernel_order(
            profiler_target_launch,
            symbol,
            backend_label,
        )
        if mark_stream_synchronized is not None:
            mark_stream_synchronized()

        random_y1_hash_before = state.random_y1_sha256()

        def pipeline_callable() -> Any:
            return state.run_with_external_gemm1(profiler_target_launch)

        print(
            "[inmoe] starting formal full-pipeline run_perftest: "
            f"num_warmup=0 num_iters={iters} testGraph=False "
            "return_trace_df=True"
        )
        pipeline_output, pipeline_us, timing, trace_df = run_target_perftest(
            pipeline_callable,
            symbol=symbol,
            warmup=0,
            reported_warmup=target_warmup,
            iters=iters,
            device=device,
            synchronize=synchronize,
            mark_stream_synchronized=mark_stream_synchronized,
            timing_source=timing_source,
        )

        random_y1_hash_after = state.random_y1_sha256()
        if random_y1_hash_before != random_y1_hash_after:
            raise error_type(
                "pipeline random_y1 changed during formal timing: "
                f"before={random_y1_hash_before}, "
                f"after={random_y1_hash_after}"
            )

        gemm2_rows = [
            row
            for row in _cuda_rows(trace_df)
            if pipeline.GEMM2_RE.search(str(row.get("name"))) is not None
        ]
        if len(gemm2_rows) != 1:
            raise error_type(
                "pipeline K3072 expected exactly one CUDA profiler row, "
                f"got {gemm2_rows}"
            )
        gemm2 = select_target_profiler_timing(
            trace_df,
            symbol=str(gemm2_rows[0]["name"]),
            warmup=0,
            requested_device=device,
        )
        if pipeline_output is not state.moe_out:
            raise error_type(
                "pipeline callable did not return the harness moe_out"
            )
        expected_rows = pipeline.EXPERTS * pipeline.TILE_M
        if int(state.contiguous_m_t.item()) != expected_rows:
            raise error_type(
                "pipeline contiguous_m sanity check failed: "
                f"expected {expected_rows}"
            )
        if target_validation_hook is not None:
            target_validation_hook()

        timing.update(
            {
                "timing_context": "moe-pipeline",
                "run_perftest_num_warmup": 0,
                "pipeline_us": pipeline_us,
                "pipeline_gemm2": gemm2,
                "pipeline_output_sample_sha256": state.output_sample_sha256(),
                "random_y1_sha256_before": random_y1_hash_before,
                "random_y1_sha256_after": random_y1_hash_after,
                "random_y1_unchanged": True,
                "isolation_pairs_checked": isolation[
                    "cross_allocation_pairs_checked"
                ],
                "order": order,
                "standalone_output_consumed_by_pipeline": False,
                "pipeline_downstream_input": "random_y1",
            }
        )
        print(
            "[inmoe] formal result: "
            f"pipeline us={pipeline_us:.4f}; target avg/count="
            f"{timing['device_time_avg']:.4f}/{timing['cnt']}; "
            f"K3072 avg/count={gemm2['device_time_avg']:.4f}/"
            f"{gemm2['cnt']}"
        )
        print(
            "[inmoe] dataflow proof: target allocations="
            f"{len(target_tensors)}; range-pairs="
            f"{isolation['cross_allocation_pairs_checked']}; downstream="
            f"random_y1 ptr={int(state.random_y1.data_ptr())}; "
            f"sha256 unchanged={random_y1_hash_before}"
        )
        return timing
    finally:
        synchronize()
        if mark_stream_synchronized is not None:
            mark_stream_synchronized()
        state.close()
