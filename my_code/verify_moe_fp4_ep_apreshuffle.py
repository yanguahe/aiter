#!/usr/bin/env python3
"""Check FP4 EP layout conversion, padded scale strides, and dropped routes.

The oracle gathers logical token rows with PyTorch and uses independent
reshape/permute transforms for M16 payload and canonical M32 ScaleA. Untouched
padding and dead-tail rows must retain their initial bytes.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aiter.ops.flydsl.moe_kernels import flydsl_moe_fused_quant_preshuffle


def verify_case(feature_dim, scale_stride, experts, dropped, seed):
    device = torch.device("cuda")
    tokens, topk, max_m = 64, 8, 192
    routes, rows = tokens * topk, experts * max_m
    payload_width, scale_width = feature_dim // 2, feature_dim // 32
    generator = torch.Generator().manual_seed(seed)
    payload = torch.randint(
        0, 256, (tokens, payload_width), generator=generator, dtype=torch.uint8
    )
    scales = torch.full((tokens, scale_stride), 251, dtype=torch.uint8)
    scales[:, :scale_width] = torch.randint(
        100, 140, (tokens, scale_width), generator=generator, dtype=torch.uint8
    )
    candidate_rows = torch.cat((torch.arange(384), torch.arange(rows - 192, rows)))
    mapping = candidate_rows[
        torch.randperm(candidate_rows.numel(), generator=generator)[:routes]
    ]
    valid_routes = routes - 11 * topk if dropped else routes
    if dropped:
        mapping[:valid_routes:5] = -1
    mapping = mapping.to(torch.int32)
    inverse = torch.full((rows,), -1, dtype=torch.long)
    for route, destination in enumerate(mapping[:valid_routes].tolist()):
        if destination >= 0:
            inverse[destination] = route // topk
    live = inverse >= 0
    logical_payload = torch.full((rows, payload_width), 0xA5, dtype=torch.uint8)
    logical_scales = torch.full((rows, scale_width), 0x5A, dtype=torch.uint8)
    logical_payload[live] = payload[inverse[live]]
    logical_scales[live] = scales[inverse[live], :scale_width]
    expected_payload = (
        logical_payload.reshape(rows // 16, 16, payload_width // 16, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
        .flatten()
    )
    expected_scales = (
        logical_scales.reshape(rows // 32, 32, scale_width // 4, 4)
        .permute(0, 2, 1, 3)
        .contiguous()
        .flatten()
    )
    out_payload = torch.full(
        (experts, max_m, payload_width), 0xA5, dtype=torch.uint8, device=device
    )
    out_scale = torch.full(
        (experts, max_m // 6, scale_width * 6), 0x5A, dtype=torch.uint8, device=device
    )
    payload_gpu, scales_gpu, mapping_gpu = (
        tensor.to(device) for tensor in (payload, scales, mapping)
    )
    valid_gpu = torch.full((1,), valid_routes, dtype=torch.int32, device=device)
    for _ in range(3):
        flydsl_moe_fused_quant_preshuffle(
            payload_gpu,
            experts,
            max_m,
            wmma_rep=6,
            topids_to_rows=mapping_gpu,
            source_topk=topk,
            out_payload=out_payload,
            out_scale=out_scale,
            num_valid_routes=valid_gpu if dropped else None,
            prequantized_scale=scales_gpu,
            a_preshuffle=True,
        )
    torch.cuda.synchronize()
    actual_payload = out_payload.flatten().cpu()
    actual_scales = out_scale.flatten().cpu()
    assert torch.equal(
        actual_payload, expected_payload
    ), f"payload mismatch: {int((actual_payload != expected_payload).sum())} bytes"
    assert torch.equal(
        actual_scales, expected_scales
    ), f"ScaleA mismatch: {int((actual_scales != expected_scales).sum())} bytes"
    print(
        json.dumps(
            {
                "feature_dim": feature_dim,
                "source_scale_stride": scale_stride,
                "experts": experts,
                "dropped_and_dead_tail": dropped,
                "live_rows": int(live.sum()),
                "payload_checked_bytes": expected_payload.numel(),
                "scale_checked_bytes": expected_scales.numel(),
                "replays": 3,
                "result": "PASS",
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    cases = (
        (7168, 256, 64, True),
        (7168, 224, 4, False),
        (2048, 128, 4, True),
        (2048, 64, 4, False),
    )
    for seed, case in enumerate(cases):
        verify_case(*case, seed=seed)


if __name__ == "__main__":
    main()
