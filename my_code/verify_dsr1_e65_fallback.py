#!/usr/bin/env python3
"""Check unchanged fallback launches and complete MoE output against HEAD."""

import argparse
import functools
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--topk", type=int, default=9)
    parser.add_argument("--shared-experts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--original-head", action="store_true")
    parser.add_argument("--save", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    import verify_dsr1_e65_gemm1 as verification

    if args.original_head:
        verification._use_original_head()
    import torch
    from aiter.ops.flydsl import grouped_gemm_mxfp4 as gemm
    from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped
    from my_code import test_flydsl_grouped_gemm_gfx1250 as workload

    launch = gemm.flydsl_grouped_gemm_a8w4_masked
    calls = []
    stages = []

    @functools.wraps(launch)
    def capture(output, a, w, sa, sw, psum, **kwargs):
        result = launch(output, a, w, sa, sw, psum, **kwargs)
        record = {k: kwargs.get(k) for k in (
            "N", "K", "tile_m", "tile_n", "tile_k", "m_warp", "n_warp",
            "num_buffers", "contiguous_m", "n_experts", "stage1_act",
            "stage1_quant_out", "a_preshuffle", "tdm_b_th",
        )}
        calls.append(record)
        # The original benchmark's untimed stage-capture call exposes its real
        # route map. Normalize to token/topk order: atomic route allocation can
        # reorder valid rows between processes even with identical input seeds.
        stage_capture = grouped.stage_output_capture
        if kwargs["stage1_act"] == 0 and stage_capture is not None:
            torch.cuda.synchronize()
            rows = stage_capture["topids_to_rows"].reshape(-1).long()
            for stage in (stage_capture["gemm1_grouped_out"], output):
                stages.append(stage.reshape(-1, stage.shape[-1]).index_select(
                    0, rows.to(stage.device)).detach().cpu())
        return result

    gemm.flydsl_grouped_gemm_a8w4_masked = capture
    workload.os.environ[workload.SHARED_EXPERTS_ENV] = str(args.shared_experts)
    out, reference, *other = workload._run_grouped_via_fused_moe(
        experts=args.experts, tokens=args.tokens, topk=args.topk,
        shared_experts=args.shared_experts, model_dim=7168, inter_dim=2048,
        data_format="a4w4", activation=verification.ActivationType.Silu,
        use_bias=False, bench=False, kernel_bench=False, seed=args.seed,
        warmup=0, iters=1,
    )
    logits_diff = workload._logits_diff(out, reference)
    if not logits_diff < workload.LOGITS_DIFF_TOL:
        raise RuntimeError(f"original MoE accuracy gate failed: logits_diff={logits_diff}")
    expected_routes = args.tokens * args.topk
    counts = other[-1]
    assert sum(counts) == expected_routes
    assert args.shared_experts != 1 or counts[-1] == args.tokens
    assert len(stages) == 2
    assert all(stage.shape[0] == expected_routes for stage in stages)
    effective = {"moe": out.detach().cpu(), "stages": stages,
                 "calls": calls, "route_counts": counts}
    if args.save:
        torch.save(effective, args.save)
    if args.compare:
        parent = torch.load(args.compare, map_location="cpu", weights_only=True)
        assert calls == parent["calls"], "fallback launch configuration changed"
        assert counts == parent["route_counts"], "fallback route histogram changed"
        assert len(stages) == len(parent["stages"])
        for name, actual, expected in (
            ("moe", effective["moe"], parent["moe"]),
            *((f"stage{i}", value, parent["stages"][i]) for i, value in enumerate(stages)),
        ):
            differing = int((actual.contiguous().view(torch.uint8)
                             != expected.contiguous().view(torch.uint8)).sum())
            print(f"[fallback HEAD exact {name}] bytes={actual.numel() * actual.element_size()} "
                  f"differing_bytes={differing}", flush=True)
            if differing:
                raise RuntimeError(f"fallback changed HEAD {name}")
    print(json.dumps({"passed": True, "experts": args.experts, "tokens": args.tokens,
                      "routes": expected_routes, "calls": calls,
                      "logits_diff": logits_diff,
                      "rel_l2": workload._rel_l2(out, reference),
                      "timing_valid": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
