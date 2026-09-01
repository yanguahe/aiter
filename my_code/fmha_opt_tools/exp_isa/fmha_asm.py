"""Python wrapper for MI350 FMHA assembly kernels.

Inputs use the benchmark harness layout:
    q: [B, S, H, D_QK]
    k: [B, S, H_KV, D_QK]
    v: [B, S, H_KV, D_V]
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_CODE_OBJECTS = {
    (128, 128, True): _THIS_DIR / "fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0.co",
    (128, 128, False): _THIS_DIR / "fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk0_gm0.co",
    (192, 128, True): _THIS_DIR / "fmha_fwd_hd192x128_bf16_1tg_4w_128x128_350_msk1_gm0.co",
    (192, 128, False): _THIS_DIR / "fmha_fwd_hd192x128_bf16_1tg_4w_128x128_350_msk0_gm0.co",
}


def _code_object(qk_head_dim: int, v_head_dim: int, causal: bool) -> Path:
    key = (qk_head_dim, v_head_dim, bool(causal))
    try:
        path = _CODE_OBJECTS[key]
    except KeyError as exc:
        supported = ", ".join(f"D{qk}x{vd} {'causal' if c else 'nocausal'}" for qk, vd, c in _CODE_OBJECTS)
        raise ValueError(f"unsupported MI350 fmha asm shape D{qk_head_dim}x{v_head_dim}; supported: {supported}") from exc
    if not path.is_file():
        raise RuntimeError(f"{path} not found. Build it with: cd {_THIS_DIR} && ./build.sh")
    return path


def _load_ext():
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    return importlib.import_module("fmha_asm_ext")


_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = _load_ext()
    return _EXT


def _check_tensor(name: str, tensor: torch.Tensor, dtype: torch.dtype) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA/ROCm tensor")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def forward_out(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Launch the causal or non-causal assembly kernel into ``out`` and return ``out``."""
    _check_tensor("q", q, torch.bfloat16)
    _check_tensor("k", k, torch.bfloat16)
    _check_tensor("v", v, torch.bfloat16)
    _check_tensor("out", out, torch.bfloat16)

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or out.ndim != 4:
        raise ValueError("q, k, v, and out must be 4D tensors")
    if q.device != k.device or q.device != v.device or q.device != out.device:
        raise ValueError("q, k, v, and out must be on the same device")
    b, s, h, qk_head_dim = q.shape
    bk, sk, h_kv, k_head_dim = k.shape
    bv, sv, hv_kv, v_head_dim = v.shape
    bo, so, ho, out_head_dim = out.shape
    if (bk, sk, k_head_dim) != (b, s, qk_head_dim):
        raise ValueError("q and k must share B, S, and QK head_dim")
    if (bv, sv, hv_kv) != (b, s, h_kv):
        raise ValueError("k and v must share B, S, and H_KV")
    if (bo, so, ho, out_head_dim) != (b, s, h, v_head_dim):
        raise ValueError(f"out shape {tuple(out.shape)} must be {(b, s, h, v_head_dim)}")
    if h_kv <= 0 or h % h_kv != 0:
        raise ValueError(f"MI350 fmha asm supports H % H_KV == 0; got H={h}, H_KV={h_kv}")
    code_object = _code_object(qk_head_dim, v_head_dim, causal)
    if causal and h % 8 != 0:
        raise ValueError(f"causal MI350 fmha asm requires H to be a multiple of 8, got H={h}")
    if qk_head_dim == 128 and s % 256 != 0:
        raise ValueError(f"D128 seq_len must be divisible by 256, got {s}")
    if qk_head_dim == 192 and s % 128 != 0:
        raise ValueError(f"D192x128 seq_len must be divisible by 128, got {s}")

    lse = torch.empty((b, h, s), dtype=torch.float32, device=q.device)
    _ext().forward_out(q, k, v, out, lse, bool(causal), str(code_object))
    return out


def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
    """Return the assembly FMHA output for bf16 contiguous tensors."""
    out = torch.empty((*q.shape[:3], v.shape[3]), dtype=q.dtype, device=q.device)
    return forward_out(q, k, v, out, causal=causal)
