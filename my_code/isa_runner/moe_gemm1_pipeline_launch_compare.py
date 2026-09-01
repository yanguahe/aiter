#!/usr/bin/env python3
"""Compare decoupled C++ and Python HIP-driver GEMM1 launches in a MoE sequence.

The workload intentionally has no shape CLI.  It reproduces the A4W4 benchmark
from ``my_code/run_gemm_a4w4.sh``.  Every timed iteration executes the real MoE
frontend, optionally launches an independent standalone GEMM1, then executes
the real MoE downstream from one fixed random placeholder.

* ``random``/``none``: use one pre-generated BF16 intermediate and launch no
  K7168 GEMM.
* ``cpp``: launch the cached code object through the PyTorch C++ extension,
  using tensors returned by ``gemm_batch_isa_runner.build_moe_inputs``.
* ``hipdrv``: launch that same standalone runner input/output through ctypes
  ``hipDrvLaunchKernelEx``.

The standalone GEMM1 output is deliberately dead: downstream always reads
``random_y1``.  Thus GEMM1 is ordered inside the stream but both its inputs and
output are allocation-disjoint from the pipeline dataflow.  With
``--gemm1-backend all --rounds 2`` the measured order is
``random, cpp, hipdrv, cpp, hipdrv``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from typing import Any


# ---------------------------------------------------------------------------
# Immutable workload contract from my_code/run_gemm_a4w4.sh
# ---------------------------------------------------------------------------
SCENARIO = "bench"
DATA_FORMAT = "a4w4"
EXPERTS = 96
TOKENS = 512
TOPK = 6
DEFAULT_ITERS = 100
MODEL_DIM = 7168
INTER_DIM = 3072
ACTIVATION = "silu"
USE_BIAS = False
CHECK_AOT_CACHE = False
TILE_M = 64
TILE_N = 256
TILE_K = 256
M_WARP = 1
N_WARP = 4
NUM_BUFFERS = 2
WAVES_PER_TENSOR_TDM = 4
CLUSTER_N_REQUEST = -1
NEXT_STAGE_PREFETCH_REQUEST = 1
SWIGLU_LIMIT = 7.0
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
SEED = 0
RANDOM_PLACEHOLDER_SEED = 20260901
DEFAULT_WARMUP = 101
DEFAULT_ROUNDS = 2
SCALE_BLOCK = 32
DEFAULT_SCALE_BYTE = 127
GPU_BUSY_PERCENT_PATH = "/sys/class/drm/card1/device/gpu_busy_percent"
EXPERIMENT_SEMANTICS = (
    "decoupled runner-input GEMM1 + random pipeline placeholder"
)

ROUTES = TOKENS * TOPK
VALID_ROWS_PER_EXPERT = ROUTES // EXPERTS
MAX_M = max(TILE_M, ((ROUTES + TILE_M - 1) // TILE_M) * TILE_M)
CONTIGUOUS_M = max(
    TILE_M,
    ((ROUTES + EXPERTS * TILE_M - TOPK + TILE_M - 1) // TILE_M) * TILE_M,
)
WMMA_REP = TILE_M // M_WARP // 16
Y1_SHAPE = (1, CONTIGUOUS_M, INTER_DIM)
Y1_STRIDE = (CONTIGUOUS_M * INTER_DIM, INTER_DIM, 1)
GEMM1_N = 2 * INTER_DIM
GEMM1_GRID = (
    (CONTIGUOUS_M // TILE_M) * (GEMM1_N // TILE_N),
    1,
    1,
)
GEMM_BLOCK = (M_WARP * N_WARP * 32, 1, 1)
GEMM_CLUSTER = (1, 1, 1)
TARGET_SYMBOL = "a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4"
GEMM2_RE = re.compile(r"a8w4_tdm_fp4_.*_K3072_e96(?:_|$)")

FIXED_ENV = {
    "ENABLE_CK": "0",
    "AITER_MOE_EXPERT_BALANCE": "true",
    "AITER_LOG_MORE": "1",
    "AITER_USE_GROUPED_GEMM": "1",
    "AITER_GROUPED_DEBUG": "0",
    "AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE": "1",
    "AITER_TDM_TILE_M": "64",
}


def _apply_fixed_environment() -> None:
    for name, value in FIXED_ENV.items():
        os.environ[name] = value
    os.environ.pop("AITER_FORCE_A8W4", None)


_apply_fixed_environment()


def _json_line(tag: str, value: Any) -> None:
    print(f"{tag}={json.dumps(value, sort_keys=True)}", flush=True)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _full_tensor_sha256(tensor: Any, torch: Any) -> str:
    raw = (
        tensor.detach()
        .contiguous()
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    )
    return _sha256_bytes(raw)


def _sample_tensor_sha256(tensor: Any, torch: Any, sample_bytes: int = 4096) -> str:
    flat = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    n = int(flat.numel())
    take = min(sample_bytes, n)
    if take == n:
        sample = flat
    else:
        left = take // 3
        middle = take // 3
        right = take - left - middle
        center = max(0, n // 2 - middle // 2)
        sample = torch.cat(
            (
                flat[:left],
                flat[center : center + middle],
                flat[n - right :],
            )
        )
    metadata = (
        f"shape={tuple(tensor.shape)};stride={tuple(tensor.stride())};"
        f"dtype={tensor.dtype};numel={tensor.numel()};"
    ).encode("ascii")
    return _sha256_bytes(metadata + sample.cpu().numpy().tobytes())


def _fixed_contract() -> dict[str, Any]:
    return {
        "scenario": SCENARIO,
        "data_format": DATA_FORMAT,
        "experts": EXPERTS,
        "tokens": TOKENS,
        "topk": TOPK,
        "iters": DEFAULT_ITERS,
        "model_dim": MODEL_DIM,
        "inter_dim": INTER_DIM,
        "activation": ACTIVATION,
        "use_bias": USE_BIAS,
        "check_aot_cache": CHECK_AOT_CACHE,
        "tile_m": TILE_M,
        "seed": SEED,
        "routes": ROUTES,
        "valid_rows_per_expert": VALID_ROWS_PER_EXPERT,
        "max_m": MAX_M,
        "contiguous_m": CONTIGUOUS_M,
        "y1_shape": Y1_SHAPE,
        "y1_stride": Y1_STRIDE,
        "gemm1_symbol": TARGET_SYMBOL,
        "gemm1_grid": GEMM1_GRID,
        "gemm_block": GEMM_BLOCK,
        "gemm_cluster": GEMM_CLUSTER,
        "experiment_semantics": EXPERIMENT_SEMANTICS,
        "standalone_input_source": "gemm_batch_isa_runner.build_moe_inputs",
        "standalone_output_consumed_by_pipeline": False,
    }


def _self_test() -> None:
    assert ROUTES == 3072
    assert VALID_ROWS_PER_EXPERT == 32
    assert MAX_M == 3072
    assert CONTIGUOUS_M == 9216
    assert WMMA_REP == 4
    assert Y1_SHAPE == (1, 9216, 3072)
    assert Y1_STRIDE == (28_311_552, 3072, 1)
    assert GEMM1_GRID == (3456, 1, 1)
    valid_rows = {
        expert * TILE_M + row
        for expert in range(EXPERTS)
        for row in range(VALID_ROWS_PER_EXPERT)
    }
    assert len(valid_rows) == ROUTES
    assert max(valid_rows) == EXPERTS * TILE_M - TILE_M + 31
    assert CONTIGUOUS_M - len(valid_rows) == 6144
    assert GEMM2_RE.search(
        "a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96_wpt4"
    )
    assert "decoupled runner-input" in EXPERIMENT_SEMANTICS
    _json_line("SELF_TEST_OK", _fixed_contract())


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gemm1-backend",
        choices=("all", "random", "none", "cpp", "hipdrv"),
        default="all",
        help=(
            "GEMM1 slot implementation. 'none' aliases 'random'; 'all' first "
            "measures random, verifies cpp vs hipdrv, then alternates them."
        ),
    )
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run CPU-only fixed-contract tests and exit",
    )
    return parser


class FixedMoePipeline:
    """Own the pipeline and one allocation-disjoint standalone GEMM1 runner."""

    def __init__(self, torch: Any, device_index: int) -> None:
        self.torch = torch
        self.device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.current_stream(self.device)
        self._closed = False
        self.hipdrv_module = None
        self.hipdrv_out_ptr: int | None = None
        self.cpp_extension = None
        self.backend_artifacts = None
        self.runner_inputs = None
        self.runner_kernarg_payload: bytes | None = None

        self._check_runtime()
        self._build_shared_inputs()
        self._build_workspaces()
        self._prepare_core_launchers()

    def _check_runtime(self) -> None:
        torch = self.torch
        if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
            raise RuntimeError("a ROCm PyTorch GPU is required")
        from aiter.jit.utils.chip_info import get_gfx

        gfx = str(get_gfx())
        if gfx != "gfx1250":
            raise RuntimeError(f"this fixed experiment requires gfx1250, got {gfx!r}")
        _json_line(
            "RUNTIME",
            {
                "gfx": gfx,
                "torch": str(torch.__version__),
                "hip": str(torch.version.hip),
                "device": str(self.device),
                "stream": int(self.stream.cuda_stream),
            },
        )

    def assert_gpu_idle(self, label: str) -> None:
        import time

        self.torch.cuda.synchronize()
        samples = []
        for _ in range(50):
            with open(GPU_BUSY_PERCENT_PATH, encoding="ascii") as handle:
                busy_percent = int(handle.read().strip())
            samples.append(busy_percent)
            if busy_percent == 0:
                break
            time.sleep(0.1)
        result = {
            "label": label,
            "path": GPU_BUSY_PERCENT_PATH,
            "busy_percent": busy_percent,
            "samples": samples,
            "waited_ms": 100 * (len(samples) - 1),
            "verified_idle": busy_percent == 0,
        }
        _json_line("GPU_IDLE_CHECK", result)
        if busy_percent != 0:
            raise RuntimeError(f"GPU is busy before {label}: {busy_percent}%")

    def _random_u8(self, shape: tuple[int, ...]) -> Any:
        return self.torch.randint(
            0,
            256,
            shape,
            dtype=self.torch.uint8,
            device=self.device,
        )

    def _random_scale(self, shape: tuple[int, ...]) -> Any:
        torch = self.torch
        raw = torch.randint(
            0,
            3,
            shape,
            dtype=torch.int16,
            device=self.device,
        )
        return (raw + (DEFAULT_SCALE_BYTE - 1)).to(torch.uint8)

    def _build_shared_inputs(self) -> None:
        torch = self.torch
        from aiter.fused_moe import fused_topk
        from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

        torch.manual_seed(SEED)
        # Preserve the exact setup draw order in
        # test_flydsl_grouped_gemm_gfx1250._run_grouped_via_fused_moe.
        w1_logical = self._random_u8(
            (EXPERTS, 2 * INTER_DIM, MODEL_DIM // 2)
        )
        w2_logical = self._random_u8(
            (EXPERTS, MODEL_DIM, INTER_DIM // 2)
        )
        w1_scale_raw = self._random_scale(
            (EXPERTS, 2 * INTER_DIM, MODEL_DIM // SCALE_BLOCK)
        )
        w2_scale_raw = self._random_scale(
            (EXPERTS, MODEL_DIM, INTER_DIM // SCALE_BLOCK)
        )
        self.hidden = (
            torch.randn(
                (TOKENS, MODEL_DIM),
                dtype=torch.float32,
                device=self.device,
            )
            * 0.5
        ).to(torch.bfloat16)

        # Vectorized form of the benchmark's balanced routing score:
        # token t activates [(t*TOPK) % E, ..., +TOPK-1].
        score = torch.zeros(
            (TOKENS, EXPERTS),
            dtype=torch.float32,
            device=self.device,
        )
        route_ord = torch.arange(ROUTES, device=self.device)
        score[
            torch.div(route_ord, TOPK, rounding_mode="floor"),
            route_ord.remainder(EXPERTS),
        ] = 1.0
        topk_weight, topk_ids = fused_topk(
            self.hidden,
            score,
            TOPK,
            True,
        )
        self.topk_ids = topk_ids.to(torch.int32).contiguous()
        self.topk_weight = topk_weight.to(torch.bfloat16).contiguous()
        del score, route_ord, topk_ids, topk_weight

        route_counts = torch.bincount(
            self.topk_ids.reshape(-1).long(),
            minlength=EXPERTS,
        )
        if not bool(torch.all(route_counts == VALID_ROWS_PER_EXPERT)):
            raise AssertionError(
                "balanced routing did not produce exactly 32 rows per expert"
            )

        self.w1_u8 = moe_shuffle_weight(
            w1_logical,
            experts_cnt=EXPERTS,
            is_guinterleave=True,
            gate_up=True,
        ).view(torch.uint8).contiguous()
        self.w2_u8 = moe_shuffle_weight(
            w2_logical,
            experts_cnt=EXPERTS,
        ).view(torch.uint8).contiguous()
        w1_scale = moe_shuffle_scale(
            w1_scale_raw.contiguous(),
            experts_cnt=EXPERTS,
            is_guinterleave=True,
            gate_up=True,
        )
        w2_scale = moe_shuffle_scale(
            w2_scale_raw.contiguous(),
            experts_cnt=EXPERTS,
        )
        self.w1s_i32 = w1_scale.reshape(-1).view(torch.int32)
        self.w2s_i32 = w2_scale.reshape(-1).view(torch.int32)

        input_summary = {
            "hidden_sample_sha256": _sample_tensor_sha256(self.hidden, torch),
            "topk_ids_sha256": _full_tensor_sha256(self.topk_ids, torch),
            "topk_weight_sha256": _full_tensor_sha256(self.topk_weight, torch),
            "w1_logical_sample_sha256": _sample_tensor_sha256(w1_logical, torch),
            "w2_logical_sample_sha256": _sample_tensor_sha256(w2_logical, torch),
            "w1_scale_raw_sample_sha256": _sample_tensor_sha256(
                w1_scale_raw, torch
            ),
            "w2_scale_raw_sample_sha256": _sample_tensor_sha256(
                w2_scale_raw, torch
            ),
            "w1_grouped_sample_sha256": _sample_tensor_sha256(self.w1_u8, torch),
            "w2_grouped_sample_sha256": _sample_tensor_sha256(self.w2_u8, torch),
            "route_counts": route_counts.cpu().tolist(),
        }
        _json_line("PIPELINE_INPUTS", input_summary)

        del (
            w1_logical,
            w2_logical,
            w1_scale_raw,
            w2_scale_raw,
            w1_scale,
            w2_scale,
            route_counts,
        )
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _build_workspaces(self) -> None:
        torch = self.torch
        self.null_i32 = torch.empty(0, dtype=torch.int32, device=self.device)
        if self.null_i32.data_ptr() != 0:
            raise AssertionError("zero-length tensor must provide a null data_ptr")

        self.route_counter = torch.empty(
            EXPERTS, dtype=torch.int32, device=self.device
        )
        self.topids_to_rows = torch.empty(
            (TOKENS, TOPK), dtype=torch.int32, device=self.device
        )
        self.starts = torch.empty(EXPERTS, dtype=torch.int32, device=self.device)
        self.psum = torch.empty(EXPERTS, dtype=torch.int32, device=self.device)
        self.contiguous_m_t = torch.empty(
            1, dtype=torch.int32, device=self.device
        )

        self.a1_payload = torch.empty(
            (1, CONTIGUOUS_M, MODEL_DIM // 2),
            dtype=torch.uint8,
            device=self.device,
        )
        self.a1_scale = torch.empty(
            (
                1,
                CONTIGUOUS_M // WMMA_REP,
                (MODEL_DIM // SCALE_BLOCK) * WMMA_REP,
            ),
            dtype=torch.uint8,
            device=self.device,
        )
        self.a2_payload = torch.empty(
            (1, CONTIGUOUS_M, INTER_DIM // 2),
            dtype=torch.uint8,
            device=self.device,
        )
        self.a2_scale = torch.empty(
            (
                1,
                CONTIGUOUS_M // WMMA_REP,
                (INTER_DIM // SCALE_BLOCK) * WMMA_REP,
            ),
            dtype=torch.uint8,
            device=self.device,
        )
        self.grouped_out = torch.zeros(
            (1, CONTIGUOUS_M, MODEL_DIM),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.moe_out = torch.empty(
            (TOKENS, MODEL_DIM),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.valid_y1_rows = (
            torch.arange(EXPERTS, device=self.device)[:, None] * TILE_M
            + torch.arange(VALID_ROWS_PER_EXPERT, device=self.device)[None, :]
        ).reshape(-1)
        self.random_y1 = torch.zeros(
            Y1_SHAPE,
            dtype=torch.bfloat16,
            device=self.device,
        )
        if tuple(self.random_y1.stride()) != Y1_STRIDE:
            raise AssertionError(
                f"unexpected random_y1 stride {tuple(self.random_y1.stride())}"
            )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(RANDOM_PLACEHOLDER_SEED)
        # The route kernel assigns row positions within each expert atomically,
        # so that order can vary between otherwise identical launches.  Draw
        # one random vector per expert and repeat it over that expert's valid
        # rows: all valid values remain random, while the placeholder becomes
        # invariant to route-row permutation and permits bitwise mode checks.
        random_per_expert = torch.randn(
            (EXPERTS, INTER_DIM),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        ).to(torch.bfloat16)
        random_valid = (
            random_per_expert[:, None, :]
            .expand(EXPERTS, VALID_ROWS_PER_EXPERT, INTER_DIM)
            .reshape(ROUTES, INTER_DIM)
        )
        self.random_y1.view(CONTIGUOUS_M, INTER_DIM).index_copy_(
            0,
            self.valid_y1_rows,
            random_valid,
        )
        del random_per_expert, random_valid

        padding_nonzero = self._padding_nonzero(self.random_y1)
        if padding_nonzero != 0:
            raise AssertionError(
                f"random y1 has {padding_nonzero} nonzero padding values"
            )
        _json_line(
            "RANDOM_PLACEHOLDER",
            {
                "seed": RANDOM_PLACEHOLDER_SEED,
                "shape": tuple(self.random_y1.shape),
                "stride": tuple(self.random_y1.stride()),
                "dtype": str(self.random_y1.dtype),
                "valid_rows": ROUTES,
                "random_vectors": EXPERTS,
                "valid_row_fill": (
                    "one fixed random vector repeated across each expert"
                ),
                "route_row_permutation_invariant": True,
                "padding_rows": CONTIGUOUS_M - ROUTES,
                "padding_nonzero": padding_nonzero,
                "sha256": _full_tensor_sha256(self.random_y1, torch),
                "ptr": int(self.random_y1.data_ptr()),
                "constructed_outside_timed_loop": True,
                "consumed_by_every_pipeline_mode": True,
            },
        )

    def _prepare_core_launchers(self) -> None:
        from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped
        from aiter.ops.flydsl import moe_kernels
        from aiter.ops.flydsl.kernels.kernels_common import get_warp_size

        wave_size = int(get_warp_size())
        if wave_size != 32:
            raise RuntimeError(f"gfx1250 wave size must be 32, got {wave_size}")
        self.quant_grid = (ROUTES + (256 // wave_size) - 1) // (
            256 // wave_size
        )
        self.quant_ksplit = self.quant_grid < 512
        self.route_launch = moe_kernels._get_compiled_topids_to_rows()
        self.psum_remap_launch = (
            grouped._get_compiled_contiguous_psum_remap()
        )
        self.quant1_launch = (
            moe_kernels._get_compiled_fused_quant_preshuffle_route_ksplit(
                feat_dim=MODEL_DIM,
                wmma_rep=WMMA_REP,
                quant_mode="fp4",
                source_topk=TOPK,
                remap_rows=False,
                ksplit=self.quant_ksplit,
            )
        )
        self.quant2_launch = (
            moe_kernels._get_compiled_fused_quant_preshuffle_route_ksplit(
                feat_dim=INTER_DIM,
                wmma_rep=WMMA_REP,
                quant_mode="fp4",
                source_topk=0,
                remap_rows=False,
                ksplit=self.quant_ksplit,
            )
        )
        gather_vec = grouped._choose_gather_reduce_vec(TOKENS, MODEL_DIM)
        self.gather_launch = grouped._get_compiled_gather_reduce(
            MODEL_DIM,
            TOPK,
            "bf16",
            1,
            gather_vec,
            "bf16",
        )
        self.gather_slice_stride_dw = CONTIGUOUS_M * (MODEL_DIM // 2)
        _json_line(
            "CORE_LAUNCHERS_READY",
            {
                "route_grid": (ROUTES + 255) // 256,
                "quant_grid": self.quant_grid,
                "quant_ksplit": self.quant_ksplit,
                "gather_vec_dwords": gather_vec,
                "all_prepared_outside_timed_callable": True,
            },
        )

    def _padding_nonzero(self, tensor: Any) -> int:
        torch = self.torch
        rows = tensor.view(CONTIGUOUS_M, INTER_DIM)
        routed = rows[: EXPERTS * TILE_M].view(
            EXPERTS,
            TILE_M,
            INTER_DIM,
        )
        return int(
            torch.count_nonzero(routed[:, VALID_ROWS_PER_EXPERT:, :]).item()
            + torch.count_nonzero(rows[EXPERTS * TILE_M :]).item()
        )

    def _run_frontend(self) -> None:
        from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

        stream = self.torch.cuda.current_stream(self.device)
        self.route_counter.zero_()
        self.route_launch(
            ptr_arg(self.topk_ids.reshape(-1)),
            ptr_arg(self.route_counter),
            ptr_arg(self.topids_to_rows.reshape(-1)),
            ROUTES,
            MAX_M,
            (ROUTES + 255) // 256,
            stream=stream,
        )
        self.psum_remap_launch(
            ptr_arg(self.route_counter),
            ptr_arg(self.topids_to_rows.reshape(-1)),
            ptr_arg(self.starts),
            ptr_arg(self.psum),
            ptr_arg(self.contiguous_m_t),
            ROUTES,
            EXPERTS,
            MAX_M,
            TILE_M,
            ptr_arg(self.null_i32),
            stream=stream,
        )
        self.quant1_launch(
            ptr_arg(self.hidden.reshape(-1)),
            ptr_arg(self.a1_payload.reshape(-1)),
            ptr_arg(self.a1_scale.reshape(-1)),
            ptr_arg(self.topids_to_rows.reshape(-1)),
            ptr_arg(self.route_counter),  # unused row_starts when remap_rows=False
            1,
            ROUTES,
            ptr_arg(self.null_i32),
            self.quant_grid,
            stream=stream,
        )

    def _run_downstream(self, y1: Any) -> Any:
        from aiter.ops.flydsl import grouped_gemm_mxfp4 as grouped_gemm
        from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

        stream = self.torch.cuda.current_stream(self.device)
        self.quant2_launch(
            ptr_arg(y1.reshape(-1)),
            ptr_arg(self.a2_payload.reshape(-1)),
            ptr_arg(self.a2_scale.reshape(-1)),
            ptr_arg(self.topids_to_rows.reshape(-1)),
            ptr_arg(self.route_counter),  # unused row_starts when remap_rows=False
            1,
            ROUTES,
            ptr_arg(self.null_i32),
            self.quant_grid,
            stream=stream,
        )
        grouped_gemm.flydsl_grouped_gemm_a8w4_masked(
            self.grouped_out,
            self.a2_payload,
            self.w2_u8,
            self.a2_scale,
            self.w2s_i32,
            self.psum,
            n_experts=EXPERTS,
            contiguous_m=CONTIGUOUS_M,
            N=MODEL_DIM,
            K=INTER_DIM,
            tile_m=TILE_M,
            tile_n=TILE_N,
            tile_k=TILE_K,
            m_warp=M_WARP,
            n_warp=N_WARP,
            num_buffers=NUM_BUFFERS,
            out_is_f16=0,
            a_is_fp4=1,
            stage1_act=0,
            bias=None,
            swiglu_limit=SWIGLU_LIMIT,
            stream=stream,
            stage1_quant_out=0,
            quant_scale=None,
            quant_wmma_rep=1,
            cluster_n=CLUSTER_N_REQUEST,
            waves_per_tensor_tdm=WAVES_PER_TENSOR_TDM,
            next_stage_prefetch=NEXT_STAGE_PREFETCH_REQUEST,
            situ_beta=SITU_BETA,
            situ_linear_beta=SITU_LINEAR_BETA,
        )
        self.gather_launch(
            ptr_arg(self.grouped_out.reshape(-1)),
            ptr_arg(self.topids_to_rows),
            ptr_arg(self.topk_weight),
            ptr_arg(self.moe_out),
            TOKENS,
            self.gather_slice_stride_dw,
            ptr_arg(self.null_i32),
            stream=stream,
        )
        return self.moe_out

    def _tensor_summary(self, tensor: Any) -> dict[str, Any]:
        nbytes = int(tensor.nbytes)
        full_hash = nbytes <= 64 * 1024 * 1024
        return {
            "shape": tuple(int(value) for value in tensor.shape),
            "stride": tuple(int(value) for value in tensor.stride()),
            "dtype": str(tensor.dtype),
            "ptr": int(tensor.data_ptr()),
            "end_ptr_exclusive": int(tensor.data_ptr()) + nbytes,
            "nbytes": nbytes,
            "sha256": (
                _full_tensor_sha256(tensor, self.torch)
                if full_hash
                else _sample_tensor_sha256(tensor, self.torch)
            ),
            "sha256_scope": "full" if full_hash else "sample+metadata",
        }

    @staticmethod
    def _ranges_overlap(lhs: Any, rhs: Any) -> bool:
        lhs_begin = int(lhs.data_ptr())
        rhs_begin = int(rhs.data_ptr())
        lhs_end = lhs_begin + int(lhs.nbytes)
        rhs_end = rhs_begin + int(rhs.nbytes)
        return max(lhs_begin, rhs_begin) < min(lhs_end, rhs_end)

    def _build_standalone_runner_inputs(self) -> None:
        if self.runner_inputs is not None:
            return
        from my_code.isa_runner import gemm_batch_isa_runner as batch

        # This is the exact public input constructor used by
        # gemm_batch_isa_runner._run_moe_gemm for moe_gemm1_a4w4_v0.s.
        runner = batch.build_moe_inputs(
            self.torch,
            m_per_expert=TILE_M,
            n=GEMM1_N,
            k=MODEL_DIM,
            experts=EXPERTS,
            device=self.device,
            seed=SEED,
        )
        reference = runner.pop("reference")
        del reference
        self.runner_inputs = runner
        self.torch.cuda.empty_cache()
        self.torch.cuda.synchronize()

        names = (
            "a",
            "scale_a",
            "b",
            "scale_b",
            "m_tile_map",
            "out",
            "bias",
            "quant_scale",
        )
        _json_line(
            "STANDALONE_RUNNER_INPUTS",
            {
                "source": "gemm_batch_isa_runner.build_moe_inputs",
                "source_call": {
                    "m_per_expert": TILE_M,
                    "n": GEMM1_N,
                    "k": MODEL_DIM,
                    "experts": EXPERTS,
                    "seed": SEED,
                },
                "same_constructor_as_standalone_runner": True,
                "tensors": {
                    name: self._tensor_summary(runner[name]) for name in names
                },
            },
        )
        self._assert_runner_pipeline_isolation()

    def _assert_runner_pipeline_isolation(self) -> None:
        if self.runner_inputs is None:
            raise RuntimeError("standalone runner inputs are not built")
        runner_tensors = {
            name: self.runner_inputs[name]
            for name in (
                "a",
                "scale_a",
                "b",
                "scale_b",
                "m_tile_map",
                "out",
                "bias",
                "quant_scale",
            )
        }
        self.assert_external_runner_isolation(runner_tensors)

    def assert_external_runner_isolation(
        self,
        runner_tensors: dict[str, Any],
    ) -> dict[str, Any]:
        """Prove arbitrary external target buffers avoid pipeline ranges."""

        if not runner_tensors:
            raise AssertionError("external target tensor set must not be empty")
        for name, tensor in runner_tensors.items():
            if not hasattr(tensor, "data_ptr") or not hasattr(tensor, "nbytes"):
                raise AssertionError(
                    f"external target allocation {name!r} is not tensor-like"
                )
        pipeline_tensors = {
            "a1_payload": self.a1_payload,
            "a1_scale": self.a1_scale,
            "w1": self.w1_u8,
            "scaleB": self.w1s_i32,
            "psum": self.psum,
            "random_y1": self.random_y1,
        }
        checked: list[str] = []
        for runner_name, runner_tensor in runner_tensors.items():
            for pipeline_name, pipeline_tensor in pipeline_tensors.items():
                if self._ranges_overlap(runner_tensor, pipeline_tensor):
                    raise AssertionError(
                        "standalone runner allocation overlaps pipeline: "
                        f"{runner_name} vs {pipeline_name}"
                    )
                checked.append(f"{runner_name}!={pipeline_name}")
        result = {
            "verified": True,
            "cross_allocation_pairs_checked": len(checked),
            "runner_ptrs": {
                name: int(tensor.data_ptr())
                for name, tensor in runner_tensors.items()
            },
            "pipeline_ptrs": {
                name: int(tensor.data_ptr())
                for name, tensor in pipeline_tensors.items()
            },
            "comparison": "full byte ranges, not pointer equality only",
            "standalone_output_consumed_by_pipeline": False,
            "pipeline_downstream_input": "random_y1",
        }
        expected_pairs = len(runner_tensors) * len(pipeline_tensors)
        if len(checked) != expected_pairs:
            raise AssertionError(
                f"runner/pipeline isolation checked {len(checked)} pairs, "
                f"expected {expected_pairs}"
            )
        _json_line("RUNNER_PIPELINE_ALLOCATION_ISOLATION", result)
        return result

    def _pack_gemm1_kernarg(self) -> bytes:
        if self.runner_inputs is None:
            raise RuntimeError("standalone runner inputs are not built")
        from my_code.isa_runner import gemm_batch_isa_runner as batch

        t = self.runner_inputs
        sa_rows = int(t["scale_a"].shape[-2])
        sa_cols = int(t["scale_a"].shape[-1])
        # Keep this descriptor construction identical to
        # gemm_batch_isa_runner._run_moe_gemm.
        return batch.pack_moe_kernargs(
            ptr_c=int(t["out"].data_ptr()),
            ptr_a=int(t["a"].data_ptr()),
            ptr_b=int(t["b"].data_ptr()),
            ptr_scale_a=int(t["scale_a"].data_ptr()),
            ptr_scale_b=int(t["scale_b"].data_ptr()),
            ptr_m_tile_map=int(t["m_tile_map"].data_ptr()),
            ptr_bias=int(t["bias"].data_ptr()),
            ptr_quant_scale=int(t["quant_scale"].data_ptr()),
            c_shape=(1, CONTIGUOUS_M, INTER_DIM),
            c_strides=(CONTIGUOUS_M * INTER_DIM, INTER_DIM),
            sa_shape=(1, sa_rows, sa_cols),
            sa_strides=(sa_rows * sa_cols, sa_cols),
            sb_size0=int(t["scale_b"].shape[0]),
            qs_shape=(1, 1, 1),
            qs_strides=(1, 1),
            i32_m=CONTIGUOUS_M,
            i32_n=GEMM1_N,
        )

    def prepare_gemm1_backends(self) -> None:
        from my_code.isa_runner import gemm_batch_isa_runner as batch
        from my_code.isa_runner import moe_cpp_backend

        self._build_standalone_runner_inputs()
        isa = batch.moe_cpp_target_isa().resolve()
        batch.validate_cpp_backend_target(
            isa,
            batch.MOE_GEMM1_WPT4_KERNEL_SYMBOL,
            "moe-gemm1",
        )
        batch.single._resolve_clang(None)
        clang = batch.single.DEFAULT_CLANG.absolute()
        if batch.single._clang_uses_default_runtime_libraries(clang):
            batch.single._prepend_default_clang_runtime_libraries()

        first = moe_cpp_backend.prepare_moe_cpp_backend(
            isa=isa,
            clang=clang,
            symbol=batch.MOE_GEMM1_WPT4_KERNEL_SYMBOL,
            single_module=batch.single,
            torch_module=self.torch,
        )
        # A second content-addressed lookup is intentional: it proves and logs
        # that both code-object and extension caches hit before formal timing.
        artifacts = moe_cpp_backend.prepare_moe_cpp_backend(
            isa=isa,
            clang=clang,
            symbol=batch.MOE_GEMM1_WPT4_KERNEL_SYMBOL,
            single_module=batch.single,
            torch_module=self.torch,
        )
        if not artifacts.code.cache_hit or not artifacts.extension.cache_hit:
            raise AssertionError("second backend preparation must be a cache hit")
        if (
            first.code.code_object_sha256
            != artifacts.code.code_object_sha256
        ):
            raise AssertionError("content-addressed code object changed between loads")
        if (
            artifacts.extension.module.code_object_sha256()
            != artifacts.code.code_object_sha256
        ):
            raise AssertionError("C++ extension reports a different .co SHA256")
        self.backend_artifacts = artifacts
        self.cpp_extension = artifacts.extension.module
        self.geometry = batch.make_moe_launch_geometry(
            TILE_M,
            GEMM1_N,
            MODEL_DIM,
            EXPERTS,
        )
        if (
            self.geometry.grid != GEMM1_GRID
            or self.geometry.block != GEMM_BLOCK
            or self.geometry.cluster != GEMM_CLUSTER
        ):
            raise AssertionError(f"unexpected GEMM1 geometry: {self.geometry}")
        self.hipdrv_module = batch.single._LoadedClusterKernel(
            artifacts.code.code_object,
            TARGET_SYMBOL,
            self.device.index,
        )
        self.runner_kernarg_payload = self._pack_gemm1_kernarg()

        # Preflight both APIs on the exact standalone runner output allocation.
        self.runner_inputs["out"].zero_()
        self._launch_gemm1_cpp()
        self.torch.cuda.synchronize()
        self._configure_hipdrv()
        payload = self.runner_kernarg_payload
        if len(payload) != 184:
            raise AssertionError(f"kernarg size is {len(payload)}, expected 184")
        _json_line(
            "BACKEND_EQUIVALENCE",
            {
                "symbol": TARGET_SYMBOL,
                "code_object": str(artifacts.code.code_object),
                "code_object_sha256": artifacts.code.code_object_sha256,
                "code_object_key": artifacts.code.key,
                "code_object_cache_hit": artifacts.code.cache_hit,
                "extension_key": artifacts.extension.key,
                "extension_cache_hit": artifacts.extension.cache_hit,
                "kernarg_size": len(payload),
                "kernarg_sha256": _sha256_bytes(payload),
                "grid": self.geometry.grid,
                "block": self.geometry.block,
                "cluster": self.geometry.cluster,
                "stream": int(self.stream.cuda_stream),
                "cpp_api": "hipModuleLaunchKernel",
                "python_api": "hipDrvLaunchKernelEx",
                "same_runner_inputs_for_both_backends": True,
                "same_runner_output_pointer_for_both_backends": True,
                "runner_output_ptr": int(self.runner_inputs["out"].data_ptr()),
                "runner_output_consumed_by_pipeline": False,
                "pipeline_downstream_input": "random_y1",
            },
        )

    def _launch_gemm1_cpp(self) -> Any:
        if (
            self.cpp_extension is None
            or self.runner_inputs is None
            or self.runner_kernarg_payload is None
            or self.backend_artifacts is None
        ):
            raise RuntimeError("C++ GEMM1 backend is not prepared")
        t = self.runner_inputs
        return self.cpp_extension.launch(
            t["out"],
            t["a"],
            t["b"],
            t["scale_a"],
            t["scale_b"],
            t["m_tile_map"],
            t["bias"],
            t["quant_scale"],
            self.runner_kernarg_payload,
            list(self.geometry.grid),
            list(self.geometry.block),
            list(self.geometry.cluster),
            TILE_M,
            GEMM1_N,
            MODEL_DIM,
            EXPERTS,
            str(self.backend_artifacts.code.code_object),
        )

    def _configure_hipdrv(self) -> None:
        if self.hipdrv_module is None or self.runner_inputs is None:
            raise RuntimeError("Python HIP-driver module is not loaded")
        if self.runner_kernarg_payload is None:
            raise RuntimeError("standalone runner kernarg is not packed")
        payload = self.runner_kernarg_payload
        self.hipdrv_module.configure(
            payload,
            self.geometry,
            int(self.torch.cuda.current_stream(self.device).cuda_stream),
            184,
        )
        self.hipdrv_out_ptr = int(self.runner_inputs["out"].data_ptr())

    def _launch_gemm1_hipdrv(self) -> None:
        if self.hipdrv_module is None or self.runner_inputs is None:
            raise RuntimeError("Python HIP-driver module is not loaded")
        if self.hipdrv_out_ptr != int(self.runner_inputs["out"].data_ptr()):
            raise RuntimeError(
                "hipDrv launcher is configured for a different runner output"
            )
        self.hipdrv_module.launch()

    def run(self, backend: str) -> Any:
        self._run_frontend()
        if backend == "random":
            pass
        elif backend == "cpp":
            self._launch_gemm1_cpp()
        elif backend == "hipdrv":
            self._launch_gemm1_hipdrv()
        else:
            raise ValueError(f"unknown backend {backend!r}")
        # The standalone output is intentionally dead.  Every mode consumes
        # the same immutable random_y1 in downstream.
        return self._run_downstream(self.random_y1)

    def run_with_external_gemm1(self, target_launch: Any) -> Any:
        """Run the pipeline around one allocation-disjoint external GEMM1."""

        self._run_frontend()
        target_launch()
        return self._run_downstream(self.random_y1)

    def random_y1_sha256(self) -> str:
        return _full_tensor_sha256(self.random_y1, self.torch)

    def output_sample_sha256(self) -> str:
        return _sample_tensor_sha256(self.moe_out, self.torch)

    def verify_external_kernel_order(
        self,
        target_launch: Any,
        target_symbol: str,
        backend: str,
    ) -> dict[str, Any]:
        """Trace one untimed iteration and pin the external GEMM1 slot."""

        torch = self.torch
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            self._run_frontend()
            with torch.profiler.record_function(
                "external_standalone_gemm1_slot"
            ):
                target_launch()
            self._run_downstream(self.random_y1)
        torch.cuda.synchronize()
        events = sorted(
            (
                event
                for event in profile.events()
                if str(event.device_type).endswith("CUDA")
            ),
            key=lambda event: float(event.time_range.start),
        )
        names = [str(event.name) for event in events]
        target_indices = [
            index
            for index, name in enumerate(names)
            if target_symbol in name
        ]
        gemm2_indices = [
            index
            for index, name in enumerate(names)
            if GEMM2_RE.search(name) is not None
        ]
        quant1_indices = [
            index
            for index, name in enumerate(names)
            if "moe_fused_quant_preshuffle_routeks_fd7168_" in name
        ]
        quant2_indices = [
            index
            for index, name in enumerate(names)
            if "moe_fused_quant_preshuffle_routeks_fd3072_" in name
        ]
        if (
            len(target_indices) != 1
            or len(gemm2_indices) != 1
            or len(quant1_indices) != 1
            or len(quant2_indices) != 1
        ):
            raise AssertionError(
                "external order trace expected one quant1, external target, "
                "quant2, and K3072 kernel; got "
                f"quant1={quant1_indices}, target={target_indices}, "
                f"quant2={quant2_indices}, K3072={gemm2_indices}, "
                f"sequence={names}"
            )
        quant1_index = quant1_indices[0]
        target_index = target_indices[0]
        quant2_index = quant2_indices[0]
        gemm2_index = gemm2_indices[0]
        if not quant1_index < target_index < quant2_index < gemm2_index:
            raise AssertionError(
                "expected quant1 -> external target -> quant2 -> K3072, got "
                f"indices {quant1_index}, {target_index}, {quant2_index}, "
                f"{gemm2_index}"
            )
        result = {
            "verified": True,
            "backend": backend,
            "same_stream": int(
                torch.cuda.current_stream(self.device).cuda_stream
            ),
            "target_index": target_index,
            "target_name": names[target_index],
            "quant1_index": quant1_index,
            "quant1_name": names[quant1_index],
            "quant2_index": quant2_index,
            "quant2_name": names[quant2_index],
            "gemm2_index": gemm2_index,
            "gemm2_name": names[gemm2_index],
            "device_event_sequence": names,
            "pipeline_downstream_input": "random_y1",
            "standalone_output_consumed_by_pipeline": False,
        }
        _json_line("EXTERNAL_GEMM1_ORDER_TRACE", result)
        return result

    def warm(self, backend: str, iterations: int = 2) -> None:
        for _ in range(iterations):
            self.run(backend)
        self.torch.cuda.synchronize()
        aligned_rows = int(self.contiguous_m_t.item())
        if aligned_rows != EXPERTS * TILE_M:
            raise AssertionError(
                f"contiguous psum returned {aligned_rows}, expected 6144"
            )
        _json_line(
            "PIPELINE_WARM",
            {
                "backend": backend,
                "iterations": iterations,
                "aligned_rows": aligned_rows,
                "compilation_outside_formal_timing": True,
            },
        )

    def verify_kernel_order(self) -> dict[str, Any]:
        """Capture one untimed cpp iteration and prove GEMM1 precedes GEMM2."""
        torch = self.torch
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            self.run("cpp")
        torch.cuda.synchronize()
        events = sorted(
            (
                event
                for event in profile.events()
                if str(event.device_type).endswith("CUDA")
            ),
            key=lambda event: float(event.time_range.start),
        )
        names = [str(event.name) for event in events]
        gemm1_indices = [
            index for index, name in enumerate(names) if TARGET_SYMBOL in name
        ]
        gemm2_indices = [
            index
            for index, name in enumerate(names)
            if GEMM2_RE.search(name) is not None
        ]
        if len(gemm1_indices) != 1 or len(gemm2_indices) != 1:
            raise AssertionError(
                "order trace expected one K7168 and one K3072 kernel, got "
                f"K7168={gemm1_indices}, K3072={gemm2_indices}"
            )
        gemm1_index = gemm1_indices[0]
        gemm2_index = gemm2_indices[0]
        if gemm1_index >= gemm2_index:
            raise AssertionError(
                f"standalone GEMM1 index {gemm1_index} must precede "
                f"pipeline GEMM2 index {gemm2_index}"
            )
        result = {
            "verified": True,
            "backend": "cpp",
            "same_stream": int(
                torch.cuda.current_stream(self.device).cuda_stream
            ),
            "gemm1_index": gemm1_index,
            "gemm2_index": gemm2_index,
            "device_events_before_gemm1": gemm1_index,
            "device_events_between_gemm1_gemm2": gemm2_index - gemm1_index - 1,
            "device_events_after_gemm2": len(names) - gemm2_index - 1,
            "device_event_sequence": names,
        }
        _json_line("UNTIMED_KERNEL_ORDER_TRACE", result)
        return result

    def verify_real_backends(self) -> dict[str, Any]:
        if (
            self.cpp_extension is None
            or self.hipdrv_module is None
            or self.runner_inputs is None
        ):
            raise RuntimeError("both real backends are required for verification")
        torch = self.torch
        runner_out = self.runner_inputs["out"]
        runner_out_ptr = int(runner_out.data_ptr())

        # Layer 1: both APIs write the same physical standalone runner output.
        # Zeroing and cloning happen only in this untimed correctness phase.
        runner_out.zero_()
        self._launch_gemm1_cpp()
        runner_cpp = runner_out.clone()
        runner_out.zero_()
        self._launch_gemm1_hipdrv()
        runner_hipdrv = runner_out.clone()
        torch.cuda.synchronize()

        runner_diff = (runner_cpp.float() - runner_hipdrv.float()).abs()
        runner_max_diff = float(runner_diff.max().item())
        runner_bitwise = bool(torch.equal(runner_cpp, runner_hipdrv))
        runner_allclose = bool(
            torch.allclose(runner_cpp, runner_hipdrv, rtol=0.0, atol=0.0)
        )
        cpp_padding_nonzero = self._padding_nonzero(runner_cpp)
        hipdrv_padding_nonzero = self._padding_nonzero(runner_hipdrv)

        # Layer 2: execute all three complete modes.  GEMM1 output is never
        # selected here; run() unconditionally sends random_y1 downstream.
        random_y1_hash_before = _full_tensor_sha256(self.random_y1, torch)
        final_outputs = {}
        for backend in ("random", "cpp", "hipdrv"):
            self.run(backend)
            final_outputs[backend] = self.moe_out.clone()
        torch.cuda.synchronize()
        random_y1_hash_after = _full_tensor_sha256(self.random_y1, torch)

        final_random = final_outputs["random"]
        final_checks = {}
        for backend in ("cpp", "hipdrv"):
            candidate = final_outputs[backend]
            diff = (final_random.float() - candidate.float()).abs()
            final_checks[backend] = {
                "bitwise_equal_to_random": bool(
                    torch.equal(final_random, candidate)
                ),
                "checkAllclose_rtol0_atol0": bool(
                    torch.allclose(final_random, candidate, rtol=0.0, atol=0.0)
                ),
                "max_abs_diff": float(diff.max().item()),
            }

        result = {
            "standalone_gemm1": {
                "shared_output_ptr": runner_out_ptr,
                "cpp_sha256": _full_tensor_sha256(runner_cpp, torch),
                "hipdrv_sha256": _full_tensor_sha256(runner_hipdrv, torch),
                "bitwise_equal": runner_bitwise,
                "checkAllclose_rtol0_atol0": runner_allclose,
                "max_abs_diff": runner_max_diff,
                "cpp_padding_nonzero": cpp_padding_nonzero,
                "hipdrv_padding_nonzero": hipdrv_padding_nonzero,
                "output_consumed_by_pipeline": False,
            },
            "pipeline_final": {
                "downstream_input_for_all_modes": "random_y1",
                "random_y1_sha256_before": random_y1_hash_before,
                "random_y1_sha256_after": random_y1_hash_after,
                "random_y1_unchanged": (
                    random_y1_hash_before == random_y1_hash_after
                ),
                "output_sha256": {
                    backend: _full_tensor_sha256(output, torch)
                    for backend, output in final_outputs.items()
                },
                "comparisons_to_random": final_checks,
            },
        }
        _json_line("CORRECTNESS", result)
        if (
            not runner_bitwise
            or not runner_allclose
            or cpp_padding_nonzero
            or hipdrv_padding_nonzero
            or random_y1_hash_before != random_y1_hash_after
            or not all(
                check["bitwise_equal_to_random"]
                and check["checkAllclose_rtol0_atol0"]
                for check in final_checks.values()
            )
        ):
            raise AssertionError(f"cpp/hipdrv correctness mismatch: {result}")

        # Formal cpp/hipdrv timings keep the same output pointer and payload.
        self._configure_hipdrv()
        return result

    def mark_hipdrv_stream_synchronized(self) -> None:
        if self.hipdrv_module is not None:
            self.hipdrv_module.mark_stream_synchronized()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.hipdrv_module is not None:
            self.hipdrv_module.close()
            self.hipdrv_module = None


def _cuda_rows(trace_df: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in trace_df.to_dict("records"):
        if str(record.get("device_type")) != "CUDA":
            continue
        try:
            avg = float(record.get("device_time_avg"))
            total = float(record.get("device_time_sum"))
            count = int(record.get("cnt"))
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "name": str(record.get("name")),
                "count": count,
                "device_time_avg_us": avg,
                "device_time_sum_us": total,
            }
        )
    return rows


def _one_unique(
    rows: list[dict[str, Any]],
    predicate: Any,
    label: str,
) -> dict[str, Any]:
    matches = [row for row in rows if predicate(row["name"])]
    if len(matches) != 1:
        names = [row["name"] for row in matches]
        raise AssertionError(f"{label} expected one profiler row, got {names}")
    return matches[0]


def _measure(
    state: FixedMoePipeline,
    backend: str,
    label: str,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    from aiter.test_common import run_perftest

    def callable_() -> Any:
        return state.run(backend)

    output, full_us, trace_df = run_perftest(
        callable_,
        num_warmup=warmup,
        num_iters=iters,
        testGraph=False,
        return_trace_df=True,
    )
    state.torch.cuda.synchronize()
    state.mark_hipdrv_stream_synchronized()
    rows = _cuda_rows(trace_df)
    target_rows = [row for row in rows if TARGET_SYMBOL in row["name"]]
    any_k7168 = [
        row
        for row in rows
        if "a8w4_tdm_fp4" in row["name"] and "_K7168_e96" in row["name"]
    ]
    if backend == "random":
        if target_rows or any_k7168:
            raise AssertionError(
                f"random mode unexpectedly launched K7168 GEMM1: {any_k7168}"
            )
        gemm1 = None
    else:
        gemm1 = _one_unique(
            rows,
            lambda name: TARGET_SYMBOL in name,
            "K7168 GEMM1",
        )
    gemm2 = _one_unique(
        rows,
        lambda name: GEMM2_RE.search(name) is not None,
        "K3072 GEMM2",
    )
    major = sorted(
        rows,
        key=lambda row: row["device_time_sum_us"],
        reverse=True,
    )[:12]
    result = {
        "label": label,
        "backend": backend,
        "gemm1_role": (
            None if backend == "random" else "decoupled standalone runner"
        ),
        "pipeline_downstream_input": "random_y1",
        "standalone_output_consumed_by_pipeline": False,
        "iters_requested": iters,
        "full_pipeline_us": float(full_us),
        "gemm1": gemm1,
        "gemm2": gemm2,
        "k7168_rows": any_k7168,
        "major_kernels": major,
        "output_sample_sha256": _sample_tensor_sha256(output, state.torch),
    }
    print(
        f"\n[profiler {label}]\n{trace_df.to_string(index=True)}",
        flush=True,
    )
    _json_line("MEASUREMENT", result)
    return result


def _median(values: list[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def _summarize(
    measurements: list[dict[str, Any]],
    correctness: dict[str, Any] | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"correctness": correctness}
    random_rows = [
        row for row in measurements if row["backend"] == "random"
    ]
    if random_rows:
        summary["random"] = {
            "full_pipeline_us": random_rows[0]["full_pipeline_us"],
            "gemm1_absent": not random_rows[0]["k7168_rows"],
            "gemm2_us": random_rows[0]["gemm2"]["device_time_avg_us"],
        }

    for backend in ("cpp", "hipdrv"):
        rows = [row for row in measurements if row["backend"] == backend]
        if rows:
            summary[backend] = {
                "rounds": len(rows),
                "gemm1_median_us": _median(
                    [row["gemm1"]["device_time_avg_us"] for row in rows]
                ),
                "gemm2_median_us": _median(
                    [row["gemm2"]["device_time_avg_us"] for row in rows]
                ),
                "full_pipeline_median_us": _median(
                    [row["full_pipeline_us"] for row in rows]
                ),
            }
    if "cpp" in summary and "hipdrv" in summary:
        comparisons = {}
        for metric in (
            "gemm1_median_us",
            "gemm2_median_us",
            "full_pipeline_median_us",
        ):
            cpp = float(summary["cpp"][metric])
            hipdrv = float(summary["hipdrv"][metric])
            comparisons[metric.replace("_median_us", "_hipdrv_vs_cpp_pct")] = (
                (hipdrv - cpp) / cpp * 100.0
            )
        summary["difference"] = comparisons
    _json_line("SUMMARY", summary)
    return summary


def main() -> None:
    args = _make_parser().parse_args()
    if args.self_test:
        _self_test()
        return
    if args.iters <= 1:
        raise SystemExit("--iters must be greater than 1 for get_trace_perf")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.rounds < 1:
        raise SystemExit("--rounds must be at least 1")

    import torch

    _json_line("FIXED_CONTRACT", _fixed_contract())
    _json_line("FIXED_ENV", FIXED_ENV)
    backend = "random" if args.gemm1_backend == "none" else args.gemm1_backend
    state = FixedMoePipeline(torch, args.device)
    measurements: list[dict[str, Any]] = []
    correctness = None
    try:
        # The omission baseline is always first in the all-backend experiment.
        if backend in ("all", "random"):
            state.warm("random")
            state.assert_gpu_idle("random")
            measurements.append(
                _measure(
                    state,
                    "random",
                    "random",
                    warmup=args.warmup,
                    iters=args.iters,
                )
            )

        if backend in ("all", "cpp", "hipdrv"):
            state.prepare_gemm1_backends()
            if backend == "all":
                state.verify_kernel_order()
                correctness = state.verify_real_backends()
                order = [
                    selected
                    for _ in range(args.rounds)
                    for selected in ("cpp", "hipdrv")
                ]
            else:
                order = [backend] * args.rounds

            # Both launch mechanisms and the downstream pipeline are warmed
            # before the first formal cpp/hipdrv profile.
            if backend == "all":
                state.warm("cpp")
                state.warm("hipdrv")
            else:
                state.warm(backend)

            seen = {"cpp": 0, "hipdrv": 0}
            for selected in order:
                seen[selected] += 1
                label = f"{selected}-round-{seen[selected]}"
                state.assert_gpu_idle(label)
                measurements.append(
                    _measure(
                        state,
                        selected,
                        label,
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                )
        _summarize(measurements, correctness)
    finally:
        state.close()
        torch.cuda.synchronize()
        _json_line("CLEAN_EXIT", {"pid": os.getpid(), "hip_module_unloaded": True})


if __name__ == "__main__":
    main()
