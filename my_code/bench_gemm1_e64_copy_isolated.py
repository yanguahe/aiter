#!/usr/bin/env python3
"""Time only GEMM1 on the unchanged random production inputs; verify its output."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from my_code.run_gemm1_e64_copy import install, elf_bytes, ARTIFACTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", choices=("baseline", "copy"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    launcher, state, manifest = install(args.variant)
    import torch
    from my_code import test_flydsl_grouped_gemm_gfx1250 as test
    from my_code.verify_dsr1_e65_gemm1 import _decode_payload_rows, _decode_scale_rows
    from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped
    from aiter.test_common import run_perftest

    test.set_data_format("a4w4")
    captured = []
    assert grouped.kernel_bench_callable is None
    grouped.kernel_bench_callable = captured
    try:
        result = test._run_grouped_via_fused_moe(
            experts=64, tokens=1536, topk=8, shared_experts=0,
            model_dim=7168, inter_dim=2048, data_format="a4w4",
            activation=test.ActivationType.Silu, use_bias=False,
            bench=False, seed=0, warmup=0, iters=1)
    finally:
        grouped.kernel_bench_callable = None
    accuracy = test._logits_diff(result[0], result[1])
    assert accuracy < test.LOGITS_DIFF_TOL, accuracy
    counts = result[-1]
    assert sum(counts) == 12288
    candidates = [function for name, function in captured
                  if name == "gemm1" and function.keywords.get("stage1_quant_out") == 1]
    assert len(candidates) == 1, len(candidates)
    function = candidates[0]
    output, _, _, _, _, psum = function.args
    scale = function.keywords["quant_scale"]
    ends = psum.detach().cpu().reshape(-1).long()
    padded_ends = ((ends + 255) // 256) * 256
    starts = torch.cat((torch.zeros(1, dtype=torch.long, device="cpu"), padded_ends[:-1]))
    assert (ends - starts).tolist() == counts
    rows = torch.cat([starts[expert] + torch.arange(count, dtype=torch.long, device="cpu")
                      for expert, count in enumerate(counts)]).to(output.device)

    def decode():
        torch.cuda.synchronize()
        payload = _decode_payload_rows(output, rows, feature_dim=2048)
        decoded_scale = _decode_scale_rows(scale, rows, feature_dim=2048,
                                           wmma_rep=8,
                                           a_preshuffle=True)
        return payload, decoded_scale

    # Compare only valid rows, using the same aligned expert starts as the
    # production verifier. Padding is intentionally undefined.
    function()
    before = decode()
    function()
    repeated = decode()
    assert all(torch.equal(a, b) for a, b in zip(before, repeated)), "current-input eager repeat changed output"
    # Use the repository's existing eager per-kernel profiling API, including
    # its normal argument rotation, with the same warmup/iteration counts.
    _, us, trace = run_perftest(function, num_warmup=5, num_iters=20,
                               testGraph=False, return_trace_df=True)
    after = decode()
    differences = [int((a.contiguous().view(torch.uint8) !=
                        b.contiguous().view(torch.uint8)).sum())
                   for a, b in zip(before, after)]
    assert differences == [0, 0], differences
    cached = list(launcher._mem_cache.values())
    if not cached and launcher._last_compiled:
        cached = [launcher._last_compiled[1]]
    records = []
    for entry in cached:
        compiled = entry if hasattr(entry, "ir") else entry[0]
        data = elf_bytes(compiled.ir)
        records.append(dict(elf_sha256=hashlib.sha256(data).hexdigest(), elf_bytes=len(data)))
        (args.output / f"actual_{len(records)}.elf").write_bytes(data)
    reference = json.loads((ARTIFACTS / f"compile_{args.variant}_th6_v3/compiled/summary.json").read_text())
    assert {record["elf_sha256"] for record in records} == {reference["elf_sha256"]}
    trace.to_json(args.output / "profile.json", orient="records", indent=2)
    names = [str(value) for value in trace["name"] if "a8w4_tdm_fp4" in str(value)]
    record = dict(passed=True, measurement="isolated GEMM1; no MoE e2e timing",
                  variant=args.variant, gemm1_us=us, profiler_kernels=names,
                  source_manifest=manifest, actual_ELF=records, testGraph=False,
                  warmup=5, iters=20, seed=0, random_inputs=True, balanced=False,
                  experts=64, tokens=1536, topk=8, shared_experts=0,
                  valid_routes=12288, route_counts=counts, logits_diff=accuracy,
                  payload_differing_bytes=differences[0], scale_differing_bytes=differences[1])
    (args.output / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print("[isolated GEMM1] " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
