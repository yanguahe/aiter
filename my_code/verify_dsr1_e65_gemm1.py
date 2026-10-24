#!/usr/bin/env python3
"""Compare fused GEMM1 quant output with the original standalone producer.

The grouped-MoE helper already performs a normal production launch followed by
an untimed stage-capture launch.  Installing ``quant_output_capture`` records
GEMM1's fused payload/ScaleA from the first launch and the standalone quant
payload/ScaleA from the second launch, using identical inputs and routing.

Only routed rows are compared because expert padding and the sentinel tail are
intentionally undefined in both implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# These switches are read while the AITER modules are imported.
os.environ.setdefault("ENABLE_CK", "0")
os.environ.setdefault("AITER_MOE_EXPERT_BALANCE", "true")
os.environ.setdefault("AITER_LOG_MORE", "1")
os.environ.setdefault("AITER_USE_GROUPED_GEMM", "1")
os.environ.setdefault("AITER_GROUPED_DEBUG", "0")
os.environ.setdefault("AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE", "1")

# Import only after the environment above is complete; these modules read the
# switches during initialization.
torch = importlib.import_module("torch")
ActivationType = importlib.import_module("aiter").ActivationType
grouped = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
moe_test = importlib.import_module("my_code.test_flydsl_grouped_gemm_gfx1250")


def _hash128(tensor: torch.Tensor) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    )
    return digest.hexdigest()


def _decode_payload_rows(
    payload: torch.Tensor,
    rows: torch.Tensor,
    *,
    feature_dim: int,
) -> torch.Tensor:
    """Undo ``shuffle_weight_f4`` row/MX-block ordering for selected rows."""
    payload_bytes_per_row = feature_dim // 2
    mx_blocks_per_row = feature_dim // 32
    flat = payload.detach().view(torch.uint8).reshape(-1).cpu()
    rows = rows.to(torch.long).cpu()
    decoded = torch.empty(
        (rows.numel(), payload_bytes_per_row), dtype=torch.uint8, device="cpu"
    )
    byte_lane = torch.arange(16, dtype=torch.long, device="cpu")
    row_tile16 = torch.div(rows, 16, rounding_mode="floor")
    row_in_tile16 = rows - row_tile16 * 16
    for mx_block in range(mx_blocks_per_row):
        offsets = (
            row_tile16 * (payload_bytes_per_row * 16)
            + mx_block * (16 * 16)
            + row_in_tile16 * 16
        )
        decoded[:, mx_block * 16 : (mx_block + 1) * 16] = flat[
            offsets[:, None] + byte_lane[None, :]
        ]
    return decoded


def _decode_scale_rows(
    scale: torch.Tensor,
    rows: torch.Tensor,
    *,
    feature_dim: int,
    wmma_rep: int,
    a_preshuffle: bool = False,
) -> torch.Tensor:
    """Undo ``shuffle_scale_f4(..., wmma_rep)`` for selected rows."""
    scale_bytes_per_row = feature_dim // 32
    scale_dwords_per_row = scale_bytes_per_row // 4
    if a_preshuffle:
        flat = scale.detach().view(torch.uint8).reshape(-1).cpu()
        rows = rows.to(torch.long).cpu()
        mx = torch.arange(scale_bytes_per_row, dtype=torch.long, device="cpu")
        offsets = (
            (rows[:, None] // 32) * scale_bytes_per_row * 32
            + (mx[None, :] // 4) * 128
            + (rows[:, None] % 32) * 4
            + (mx[None, :] % 4)
        )
        return flat[offsets]
    rows_per_tile = wmma_rep * 16
    flat = scale.detach().view(torch.uint8).reshape(-1).cpu()
    rows = rows.to(torch.long).cpu()
    decoded = torch.empty(
        (rows.numel(), scale_bytes_per_row), dtype=torch.uint8, device="cpu"
    )
    row_tile = torch.div(rows, rows_per_tile, rounding_mode="floor")
    row_in_tile = rows - row_tile * rows_per_tile
    wmma_row = torch.div(row_in_tile, 16, rounding_mode="floor")
    scale_lane = row_in_tile - wmma_row * 16
    for mx_block in range(scale_bytes_per_row):
        scale_dword = mx_block // 4
        byte_in_dword = mx_block & 3
        offsets = (
            (
                (
                    (row_tile * scale_dwords_per_row + scale_dword) * wmma_rep
                    + wmma_row
                )
                * 16
                + scale_lane
            )
            * 4
            + byte_in_dword
        )
        decoded[:, mx_block] = flat[offsets]
    return decoded


def _layout_self_test() -> None:
    rows = 192
    feature_dim = 2048
    wmma_rep = 6
    payload_bytes = feature_dim // 2
    scale_bytes = feature_dim // 32

    logical_payload = torch.arange(
        rows * payload_bytes, dtype=torch.int64, device="cpu"
    )
    logical_payload = (logical_payload & 0xFF).to(torch.uint8).view(rows, -1)
    shuffled_payload = torch.empty_like(logical_payload).reshape(-1)
    for row in range(rows):
        for mx_block in range(feature_dim // 32):
            src = logical_payload[row, mx_block * 16 : (mx_block + 1) * 16]
            dst = (
                (row // 16) * payload_bytes * 16
                + mx_block * 16 * 16
                + (row % 16) * 16
            )
            shuffled_payload[dst : dst + 16] = src

    logical_scale = torch.arange(
        rows * scale_bytes, dtype=torch.int64, device="cpu"
    )
    logical_scale = ((logical_scale * 17 + 3) & 0xFF).to(torch.uint8).view(rows, -1)
    shuffled_scale = torch.empty_like(logical_scale).reshape(-1)
    scale_dwords = scale_bytes // 4
    rows_per_tile = wmma_rep * 16
    for row in range(rows):
        row_tile = row // rows_per_tile
        row_in_tile = row % rows_per_tile
        wmma_row = row_in_tile // 16
        lane = row_in_tile % 16
        for mx_block in range(scale_bytes):
            dst = (
                (
                    (
                        (row_tile * scale_dwords + mx_block // 4) * wmma_rep
                        + wmma_row
                    )
                    * 16
                    + lane
                )
                * 4
                + (mx_block & 3)
            )
            shuffled_scale[dst] = logical_scale[row, mx_block]

    selected = torch.tensor(
        [0, 1, 15, 16, 31, 32, 95, 96, 191], device="cpu"
    )
    assert torch.equal(
        _decode_payload_rows(
            shuffled_payload, selected, feature_dim=feature_dim
        ),
        logical_payload[selected],
    )
    assert torch.equal(
        _decode_scale_rows(
            shuffled_scale,
            selected,
            feature_dim=feature_dim,
            wmma_rep=wmma_rep,
        ),
        logical_scale[selected],
    )


def _report(name: str, fused: torch.Tensor, reference: torch.Tensor) -> int:
    mismatch = fused != reference
    count = int(mismatch.sum())
    print(
        f"[{name}] bytes={fused.numel()} mismatch={count} "
        f"fused_hash128={_hash128(fused)} "
        f"reference_hash128={_hash128(reference)}",
        flush=True,
    )
    if count:
        first = mismatch.nonzero(as_tuple=False)[:16]
        for row, col in first.tolist():
            print(
                f"[{name}] row={row} byte={col} "
                f"fused=0x{int(fused[row, col]):02x} "
                f"reference=0x{int(reference[row, col]):02x}",
                flush=True,
            )
    return count


def _report_payload_permutation(
    fused_payload: torch.Tensor, reference_payload: torch.Tensor
) -> None:
    """Report the strongest reference-position match for each fused nibble."""

    def unpack(payload: torch.Tensor) -> torch.Tensor:
        low = payload & 0x0F
        high = payload >> 4
        return torch.stack((low, high), dim=-1).reshape(payload.shape[0], -1, 32)

    fused = unpack(fused_payload).reshape(-1, 32)[:4096]
    reference = unpack(reference_payload).reshape(-1, 32)[:4096]
    scores = (fused[:, :, None] == reference[:, None, :]).float().mean(dim=0)
    best_score, best_reference_pos = scores.max(dim=1)
    mapping = ", ".join(
        f"{fused_pos}->{int(best_reference_pos[fused_pos])}"
        f"({float(best_score[fused_pos]) * 100:.1f}%)"
        for fused_pos in range(32)
    )
    print(f"[payload permutation] {mapping}", flush=True)


def _report_fp4_semantic_mismatch(
    fused_payload: torch.Tensor, reference_payload: torch.Tensor
) -> int:
    """Compare FP4 nibbles after canonicalizing negative zero to positive zero."""

    def unpack(payload: torch.Tensor) -> torch.Tensor:
        low = payload & 0x0F
        high = payload >> 4
        return torch.stack((low, high), dim=-1).reshape(payload.shape[0], -1)

    fused = unpack(fused_payload)
    reference = unpack(reference_payload)
    fused = torch.where(fused == 0x8, 0, fused)
    reference = torch.where(reference == 0x8, 0, reference)
    mismatch = fused != reference
    count = int(mismatch.sum())
    print(
        f"[payload semantic] fp4_values={fused.numel()} "
        f"mismatch_after_zero_sign_normalization={count}",
        flush=True,
    )
    return count


def _report_tensor(name: str, fused: torch.Tensor, reference: torch.Tensor) -> None:
    fused = fused.detach().cpu()
    reference = reference.detach().cpu()
    bitwise_mismatch = int(
        (
            fused.contiguous().view(torch.uint8)
            != reference.contiguous().view(torch.uint8)
        ).sum()
    )
    diff = fused.float() - reference.float()
    rel_l2 = float(diff.norm() / reference.float().norm().clamp(min=1e-12))
    print(
        f"[{name}] bitwise_mismatch={bitwise_mismatch} "
        f"max_abs={float(diff.abs().max()):.6e} rel_l2={rel_l2:.6e} "
        f"fused_hash128={_hash128(fused)} "
        f"reference_hash128={_hash128(reference)}",
        flush=True,
    )


def _use_original_head() -> None:
    """Load the two exact production sources saved by the original benchmark."""
    snapshot = REPO_ROOT / ".codex_tmp/dsr1_e65_gemm1_20261017/baseline_settled/sources"
    for name, relative, expected_sha in (
        ("aiter.ops.flydsl.kernels.mxfp4_preshuffle_gfx1250_tdm_prefill_persistent",
         "aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py",
         "b1f0ff87c886915b737cebbb77f61ca67bd213abde41f913c7e9400a251bf6b4"),
        ("aiter.ops.flydsl.grouped_gemm_mxfp4",
         "aiter/ops/flydsl/grouped_gemm_mxfp4.py",
         "5329bb55c22d9c76850dc3c09c539fd9674bce542ad8115d8a4a5821d377013b"),
    ):
        path = snapshot / relative
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(f"original HEAD source mismatch: {path}")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        parent_name, _, attr = name.rpartition(".")
        setattr(importlib.import_module(parent_name), attr, module)
        print(f"[original HEAD] source={path} sha256={actual_sha}", flush=True)


def _original_head_gemm1_launcher():
    """Load the audited HEAD launcher under an isolated module name."""
    relative = "aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py"
    path = (REPO_ROOT / ".codex_tmp/dsr1_e65_gemm1_20261017/baseline_settled/sources"
            / relative)
    expected = "b1f0ff87c886915b737cebbb77f61ca67bd213abde41f913c7e9400a251bf6b4"
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("dynamic replay HEAD source SHA256 mismatch")
    name = "aiter.ops.flydsl.kernels._dsr1_e65_original_gemm1"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.launch_gemm_a8w4_tdm_fused_persistent


def _install_replay_checks(repeats: int, dynamic_replays: int = 0) -> None:
    """Check canaries and optional same-address input/psum updates against HEAD."""
    gemm_module = importlib.import_module("aiter.ops.flydsl.grouped_gemm_mxfp4")
    launch = gemm_module.flydsl_grouped_gemm_a8w4_masked
    original = _original_head_gemm1_launcher() if dynamic_replays else None
    call_count = 0

    def checked(output, a, w, sa, sw, psum, **kwargs):
        nonlocal call_count
        result = launch(output, a, w, sa, sw, psum, **kwargs)
        if kwargs.get("stage1_quant_out") != 1 or kwargs.get("stage1_act") != 1:
            return result
        call_count += 1
        torch.cuda.synchronize()
        ends = psum.detach().cpu().long().reshape(-1)
        alignment = kwargs["tile_m"]
        padded_ends = ((ends + alignment - 1) // alignment) * alignment
        starts = torch.cat((torch.zeros(1, dtype=torch.long, device="cpu"), padded_ends[:-1]))
        rows = torch.cat([torch.arange(int(start), int(end), device="cpu")
                          for start, end in zip(starts, ends)])
        feature_dim = kwargs["N"] // 2
        scale = kwargs["quant_scale"]
        expected_payload = _decode_payload_rows(output, rows, feature_dim=feature_dim)
        expected_scale = _decode_scale_rows(scale, rows, feature_dim=feature_dim,
                                           wmma_rep=kwargs["quant_wmma_rep"],
                                           a_preshuffle=True)

        def compare(label):
            torch.cuda.synchronize()
            actual_payload = _decode_payload_rows(output, rows, feature_dim=feature_dim)
            actual_scale = _decode_scale_rows(scale, rows, feature_dim=feature_dim,
                                             wmma_rep=kwargs["quant_wmma_rep"],
                                             a_preshuffle=True)
            failures = []
            for name, actual, expected in (("payload", actual_payload, expected_payload),
                                            ("scale", actual_scale, expected_scale)):
                mismatch = actual != expected
                differing = int(mismatch.sum())
                print(f"[replay {call_count} {label} {name}] rows={rows.numel()} "
                      f"differing_bytes={differing}", flush=True)
                if differing:
                    failures.append(name)
                    for row, col in mismatch.nonzero(as_tuple=False)[:16].tolist():
                        print(f"[replay mismatch {name}] grouped_row={int(rows[row])} "
                              f"byte={col} actual=0x{int(actual[row, col]):02x} "
                              f"expected=0x{int(expected[row, col]):02x}", flush=True)
            if failures:
                raise RuntimeError(f"GEMM1 {label} changed effective {','.join(failures)}")

        for repeat in range(repeats):
            output.fill_(0x35 if repeat % 2 == 0 else 0xA6)
            scale.fill_(0x35 if repeat % 2 == 0 else 0xA6)
            launch(output, a, w, sa, sw, psum, **kwargs)
            compare(f"eager{repeat}")
        stream = torch.cuda.current_stream()
        graph_stream = torch.cuda.Stream()
        graph_stream.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        graph_kwargs = dict(kwargs, stream=graph_stream)
        # Warm up this stream's launcher cache before capture. The graph
        # captures only the already compiled production launch.
        with torch.cuda.stream(graph_stream):
            launch(output, a, w, sa, sw, psum, **graph_kwargs)
        stream.wait_stream(graph_stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=graph_stream):
            launch(output, a, w, sa, sw, psum, **graph_kwargs)
        stream.wait_stream(graph_stream)
        for repeat in range(repeats):
            output.fill_(0x6B if repeat % 2 == 0 else 0xD2)
            scale.fill_(0x6B if repeat % 2 == 0 else 0xD2)
            graph.replay()
            compare(f"graph{repeat}")
        if dynamic_replays:
            _check_dynamic_replays(
                launch, original, graph, graph_stream, output, a, w, sa, sw, psum,
                kwargs, rows, dynamic_replays, call_count,
            )
        return result

    gemm_module.flydsl_grouped_gemm_a8w4_masked = checked


def _check_dynamic_replays(launch, original, graph, graph_stream, output,
                          a, w, sa, sw, psum, kwargs, original_rows, repeats,
                          call_count):
    """Change data at captured pointers; restore every production input afterward.

    Count updates fit inside the captured allocation, exercise empty experts
    and M16/M32/M64/M128/t256 tails, and retain the shared expert's row count.
    Recompute the production aligned prefix sum for every update. These are
    untimed direct GEMM1 ABI tests, not replacement routing or timing for the
    user's production benchmark.
    """
    if (kwargs["tile_m"], kwargs["tile_n"], kwargs["tile_k"],
        kwargs["m_warp"], kwargs["n_warp"], kwargs["N"], kwargs["K"]) != (
            256, 256, 256, 2, 4, 4096, 7168):
        raise RuntimeError("dynamic replay only covers the audited t256 GEMM1 ABI")
    if not kwargs.get("a_preshuffle") or kwargs.get("stage1_quant_out") != 1:
        raise RuntimeError("dynamic replay requires the production A-preshuffle ABI")
    device = output.device
    current = torch.cuda.current_stream()
    scale = kwargs["quant_scale"]
    backup_a, backup_sa, backup_psum = a.clone(), sa.clone(), psum.clone()
    original_ends = backup_psum.detach().cpu().long().reshape(-1)
    alignment = kwargs["tile_m"]
    padded = ((original_ends + alignment - 1) // alignment) * alignment
    starts = torch.cat((torch.zeros(1, dtype=torch.long, device="cpu"), padded[:-1]))
    capacities = padded - starts
    original_counts = original_ends - starts
    expected_output, expected_scale = torch.empty_like(output), torch.empty_like(scale)
    generator = torch.Generator(device=device).manual_seed(20261017 + call_count)
    ptrs = tuple(t.data_ptr() for t in (output, scale, a, sa, psum))
    feature_dim = kwargs["N"] // 2

    def decode(tensor, selected, is_scale):
        if is_scale:
            return _decode_scale_rows(tensor, selected, feature_dim=feature_dim,
                                      wmma_rep=kwargs["quant_wmma_rep"],
                                      a_preshuffle=True)
        return _decode_payload_rows(tensor, selected, feature_dim=feature_dim)

    def head_launch():
        # Exactly the production arguments of the saved HEAD wrapper. No
        # candidate kwargs enter this independently loaded original launcher.
        original(
            expected_output, gemm_module_ptr(a), gemm_module_ptr(w),
            sa.view(torch.int32), sw.view(torch.int32), kwargs["contiguous_m"],
            current, kwargs["N"], kwargs["K"], 256, 256, 256, 2, 4,
            kwargs["out_is_f16"], kwargs["num_buffers"], kwargs["a_is_fp4"],
            gemm_module_ptr(psum), kwargs["n_experts"], kwargs["stage1_act"],
            int(kwargs.get("bias") is not None),
            gemm_module_ptr(kwargs.get("bias") if kwargs.get("bias") is not None else a),
            float(kwargs.get("swiglu_limit", 7.0)), 1, kwargs["quant_wmma_rep"],
            expected_scale, kwargs["cluster_n"], kwargs["cluster_m"],
            kwargs["next_stage_prefetch"], 2, kwargs.get("tdm_b_th", 0),
            balanced_m192=0,
            partial_m_direct_scale=int(bool(kwargs.get("partial_m_direct_scale", 0))),
            partial_m_drain_output=int(bool(kwargs.get("partial_m_drain_output", 0))),
        )

    original_effective = (decode(output, original_rows, False),
                          decode(scale, original_rows, True))
    boundaries = (0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65,
                  127, 128, 129, 191, 192, 193, 255, 256)
    try:
        for update in range(repeats):
            # Preserve pointer, shape, stride and all FP4/E8M0 formats. The
            # first update retains real production routes with fresh payload.
            a_bytes = a.view(torch.uint8)
            sa_bytes = sa.view(torch.uint8)
            a_bytes.copy_(torch.randint(0, 256, a_bytes.shape, dtype=torch.uint8,
                                       device=device, generator=generator))
            sa_bytes.copy_(torch.randint(125, 129, sa_bytes.shape, dtype=torch.uint8,
                                        device=device, generator=generator))
            counts = original_counts.clone()
            if update:
                for expert in range(counts.numel() - 1):
                    counts[expert] = min(int(capacities[expert]),
                                         boundaries[(expert + update * 7) % len(boundaries)])
            aligned_counts = ((counts + alignment - 1) // alignment) * alignment
            updated_starts = torch.cumsum(aligned_counts, 0) - aligned_counts
            assert int(aligned_counts.sum()) <= kwargs["contiguous_m"]
            updated_ends = (updated_starts + counts).to(dtype=psum.dtype, device=psum.device)
            psum.copy_(updated_ends.reshape(psum.shape))
            selected = torch.cat([torch.arange(int(start), int(start + count), device="cpu")
                                  for start, count in zip(updated_starts, counts)])
            expected_output.fill_(0x79)
            expected_scale.fill_(0x79)
            head_launch()
            torch.cuda.synchronize()
            expected = (decode(expected_output, selected, False),
                        decode(expected_scale, selected, True))
            for mode in ("eager", "graph"):
                output.fill_(0xCA)
                scale.fill_(0xCA)
                if mode == "eager":
                    launch(output, a, w, sa, sw, psum, **kwargs)
                else:
                    graph_stream.wait_stream(current)
                    graph.replay()
                    current.wait_stream(graph_stream)
                torch.cuda.synchronize()
                assert ptrs == tuple(t.data_ptr() for t in (output, scale, a, sa, psum))
                for name, tensor, is_scale, reference in (
                    ("payload", output, False, expected[0]),
                    ("scale", scale, True, expected[1]),
                ):
                    actual = decode(tensor, selected, is_scale)
                    differing = int((actual != reference).sum())
                    print(f"[dynamic replay {call_count} update{update} {mode} {name}] "
                          f"rows={selected.numel()} zero_experts={int((counts == 0).sum())} "
                          f"same_pointers=True differing_bytes={differing}", flush=True)
                    if differing:
                        raise RuntimeError(f"dynamic {mode} changed HEAD effective {name}")
    finally:
        torch.cuda.synchronize()
        a.copy_(backup_a)
        sa.copy_(backup_sa)
        psum.copy_(backup_psum)
        launch(output, a, w, sa, sw, psum, **kwargs)
        torch.cuda.synchronize()
    # The surrounding full MoE diagnostic still receives the original input
    # and effective output after the direct-kernel update checks.
    for tensor, is_scale, reference in (
        (output, False, original_effective[0]),
        (scale, True, original_effective[1]),
    ):
        assert tensor.data_ptr() in ptrs
        if not torch.equal(decode(tensor, original_rows, is_scale), reference):
            raise RuntimeError("dynamic replay failed to restore original effective output")


def gemm_module_ptr(tensor):
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
    return ptr_arg(tensor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=65)
    parser.add_argument("--tokens", type=int, default=1536)
    parser.add_argument("--topk", type=int, default=9)
    parser.add_argument("--shared-experts", type=int, default=1)
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--const-init", type=float, default=None)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--original-head", action="store_true")
    parser.add_argument("--replays", type=int, default=0)
    parser.add_argument("--dynamic-replays", type=int, default=0,
                        help="Same-pointer random payload/ScaleA/psum updates checked against HEAD.")
    args = parser.parse_args()

    # Diagnostic launch markers distinguish runtime initialization stalls from
    # candidate compute stalls. They never change a production launch or timing.
    torch.set_num_threads(8)
    print(f"[verify] layout self-test seed={args.seed}", flush=True)

    if args.original_head:
        _use_original_head()
    if args.dynamic_replays and args.replays <= 0:
        parser.error("--dynamic-replays requires --replays > 0")
    if args.replays:
        _install_replay_checks(args.replays, args.dynamic_replays)
    _layout_self_test()
    print("[verify] building random production workload", flush=True)
    capture: dict[str, object] = {}
    if grouped.quant_output_capture is not None:
        raise RuntimeError("another quant-output capture is already active")
    grouped.quant_output_capture = capture
    os.environ[moe_test.SHARED_EXPERTS_ENV] = str(args.shared_experts)
    try:
        out, reference, *_ = moe_test._run_grouped_via_fused_moe(
            experts=args.experts,
            tokens=args.tokens,
            topk=args.topk,
            shared_experts=args.shared_experts,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
            data_format="a4w4",
            activation=ActivationType.Silu,
            use_bias=False,
            bench=False,
            kernel_bench=False,
            seed=args.seed,
            warmup=0,
            iters=1,
            const_init=args.const_init,
        )
    finally:
        grouped.quant_output_capture = None
    logits_diff = moe_test._logits_diff(out, reference)
    rel_l2 = moe_test._rel_l2(out, reference)
    passed = logits_diff < moe_test.LOGITS_DIFF_TOL

    required = {
        "fused_payload",
        "fused_scale",
        "reference_payload",
        "reference_scale",
        "fused_topids_to_rows",
        "reference_topids_to_rows",
        "contiguous_m",
        "inter_dim",
        "wmma_rep",
        "gemm1_a_preshuffle",
        "gemm2_a_preshuffle",
        "fused_gemm2_grouped_out",
        "reference_gemm2_grouped_out",
        "fused_moe_out",
        "reference_moe_out",
    }
    missing = required.difference(capture)
    if missing:
        raise RuntimeError(f"quant-output capture is incomplete: {sorted(missing)}")

    fused_route_rows = capture["fused_topids_to_rows"]
    reference_route_rows = capture["reference_topids_to_rows"]
    if fused_route_rows is None or reference_route_rows is None:
        raise RuntimeError("the diagnostic requires the non-EP routed-row layout")
    contiguous_m = int(capture["contiguous_m"])
    feature_dim = int(capture["inter_dim"])
    wmma_rep = int(capture["wmma_rep"])
    fused_route_rows = fused_route_rows.detach().reshape(-1).to(torch.long).cpu()
    reference_route_rows = (
        reference_route_rows.detach().reshape(-1).to(torch.long).cpu()
    )
    if not bool(
        ((fused_route_rows >= 0) & (fused_route_rows < contiguous_m)).all()
    ):
        raise RuntimeError("fused route map contains an invalid grouped row")
    if not bool(
        ((reference_route_rows >= 0) & (reference_route_rows < contiguous_m)).all()
    ):
        raise RuntimeError("reference route map contains an invalid grouped row")

    fused_payload = _decode_payload_rows(
        capture["fused_payload"], fused_route_rows, feature_dim=feature_dim
    )
    reference_payload = _decode_payload_rows(
        capture["reference_payload"], reference_route_rows, feature_dim=feature_dim
    )
    fused_scale = _decode_scale_rows(
        capture["fused_scale"],
        fused_route_rows,
        feature_dim=feature_dim,
        wmma_rep=wmma_rep,
        a_preshuffle=bool(capture["gemm2_a_preshuffle"]),
    )
    reference_scale = _decode_scale_rows(
        capture["reference_scale"],
        reference_route_rows,
        feature_dim=feature_dim,
        wmma_rep=wmma_rep,
        a_preshuffle=bool(capture["gemm2_a_preshuffle"]),
    )

    payload_mismatch = _report("payload", fused_payload, reference_payload)
    semantic_payload_mismatch = _report_fp4_semantic_mismatch(
        fused_payload, reference_payload
    )
    if payload_mismatch:
        _report_payload_permutation(fused_payload, reference_payload)
    scale_mismatch = _report("scale", fused_scale, reference_scale)
    fused_gemm2 = capture["fused_gemm2_grouped_out"].reshape(
        -1, args.model_dim
    ).index_select(
        0, fused_route_rows.to(capture["fused_gemm2_grouped_out"].device)
    )
    reference_gemm2 = capture["reference_gemm2_grouped_out"].reshape(
        -1, args.model_dim
    ).index_select(
        0,
        reference_route_rows.to(capture["reference_gemm2_grouped_out"].device),
    )
    _report_tensor("gemm2 valid rows", fused_gemm2, reference_gemm2)
    _report_tensor("moe output", capture["fused_moe_out"], capture["reference_moe_out"])
    effective = {"payload": fused_payload, "scale": fused_scale,
                 "gemm2": fused_gemm2.detach().cpu(),
                 "moe": capture["fused_moe_out"].detach().cpu()}
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(effective, args.save)
    if args.compare:
        parent = torch.load(args.compare, map_location="cpu", weights_only=True)
        for key, value in effective.items():
            actual = value.contiguous().view(torch.uint8)
            expected = parent[key].contiguous().view(torch.uint8)
            differing = int((actual != expected).sum())
            print(f"[parent exact {key}] bytes={actual.numel()} differing_bytes={differing}", flush=True)
            if differing:
                raise RuntimeError(f"candidate changed parent effective {key}")
    print(
        "[layout] "
        f"gemm1_a_preshuffle={capture['gemm1_a_preshuffle']} "
        f"gemm2_a_preshuffle={capture['gemm2_a_preshuffle']} "
        f"wmma_rep={wmma_rep}",
        flush=True,
    )
    print(
        f"[moe] pass={passed} logits_diff={logits_diff:.4e} rel_l2={rel_l2:.4e}",
        flush=True,
    )
    return (
        0
        if passed and semantic_payload_mismatch == 0 and scale_mismatch == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
