#!/usr/bin/env python3
"""Verify that repeated E64/T1536/topk8 random runs produce identical topk_id.

Each parent-side repetition starts a fresh Python worker.  The worker invokes
the real grouped-MoE test setup, including every RNG draw for weights, scales,
and hidden states that precedes routing, then intercepts ``_make_topk`` and
exits before any GEMM is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys


RESULT_PREFIX = "ROUTING_CAPTURE_JSON="


class RoutingCaptured(Exception):
    """Stop the benchmark setup immediately after routing is materialized."""


def _tensor_hash128(tensor) -> str:
    import torch

    digest = hashlib.blake2b(digest_size=16)
    raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1).cpu().numpy()
    digest.update(raw.tobytes())
    return digest.hexdigest()


def _worker(seed: int) -> None:
    import torch

    from aiter import ActivationType
    from my_code import test_flydsl_grouped_gemm_gfx1250 as benchmark

    if benchmark.AITER_MOE_EXPERT_BALANCE:
        raise RuntimeError(
            "This test targets random routing; set AITER_MOE_EXPERT_BALANCE=false"
        )

    original_make_topk = benchmark._make_topk

    def capture_topk(hidden_states, experts: int, topk: int):
        topk_id, topk_weight = original_make_topk(hidden_states, experts, topk)
        topk_id_i32 = topk_id.to(torch.int32)
        counts = torch.bincount(
            topk_id_i32.reshape(-1).to(torch.long), minlength=experts
        )
        record = {
            "seed": seed,
            "shape": list(topk_id_i32.shape),
            "topk_id_hash128": _tensor_hash128(topk_id_i32),
            "topk_weight_hash128": _tensor_hash128(topk_weight),
            "route_counts": counts.cpu().tolist(),
            "first_rows": topk_id_i32[:4].cpu().tolist(),
        }
        print(RESULT_PREFIX + json.dumps(record, separators=(",", ":")), flush=True)
        raise RoutingCaptured

    benchmark._make_topk = capture_topk
    try:
        benchmark._run_grouped_via_fused_moe(
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
            seed=seed,
            const_init=None,
        )
    except RoutingCaptured:
        return
    finally:
        benchmark._make_topk = original_make_topk
    raise RuntimeError("routing capture hook was not reached")


def _run_parent(runs: int, seed: int) -> None:
    records = []
    for run in range(runs):
        command = [sys.executable, "-u", __file__, "--worker", "--seed", str(seed)]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            raise SystemExit(
                f"worker {run + 1}/{runs} failed with status {completed.returncode}"
            )
        result_lines = [
            line[len(RESULT_PREFIX) :]
            for line in completed.stdout.splitlines()
            if line.startswith(RESULT_PREFIX)
        ]
        if len(result_lines) != 1:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            raise SystemExit(
                f"worker {run + 1}/{runs} emitted {len(result_lines)} routing records"
            )
        record = json.loads(result_lines[0])
        records.append(record)
        print(
            f"run {run + 1}/{runs}: topk_id={record['topk_id_hash128']} "
            f"topk_weight={record['topk_weight_hash128']} "
            f"min/max routes={min(record['route_counts'])}/"
            f"{max(record['route_counts'])}",
            flush=True,
        )

    baseline = records[0]
    mismatches = [
        index + 1
        for index, record in enumerate(records[1:], start=1)
        if record != baseline
    ]
    summary = {
        "runs": runs,
        "seed": seed,
        "all_identical": not mismatches,
        "mismatching_runs": mismatches,
        "topk_id_hash128": baseline["topk_id_hash128"],
        "topk_weight_hash128": baseline["topk_weight_hash128"],
        "route_counts": baseline["route_counts"],
        "first_rows": baseline["first_rows"],
    }
    print(json.dumps(summary, indent=2), flush=True)
    if mismatches:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2")
    if args.worker:
        _worker(args.seed)
    else:
        _run_parent(args.runs, args.seed)


if __name__ == "__main__":
    main()
