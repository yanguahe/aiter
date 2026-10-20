#!/usr/bin/env python3
"""Replay the original four-rank Mega MoE workload for external ATT collection.

This runner disables Kineto and reports zero timings. It retains the original
model, routing, graph capture, kernel selection, and optional accuracy check.
"""

import ctypes
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
BENCH_PATH = REPO_ROOT / "op_tests/multigpu_tests/bench_mega_moe.py"
spec = importlib.util.spec_from_file_location("mega_moe_att_workload", BENCH_PATH)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


class EmptyProfile:
    def key_averages(self):
        return []


def install_isolated_diagnostic():
    """Load an explicitly requested diagnostic copy for this process only."""
    variant = os.environ.get("AITER_ATT_DIAGNOSTIC")
    if not variant:
        return
    if variant not in (
        "split_wait", "global_window", "global_split", "const_geometry",
        "oob_output", "oob_sorted", "local_output",
    ):
        raise ValueError(f"Unknown isolated ATT diagnostic: {variant}")
    directory = Path(os.environ["AITER_ATT_DIAGNOSTIC_DIR"])
    import aiter.ops.flydsl.kernels.mxfp4_preshuffle_gfx1250_tdm_prefill_persistent as production

    def load(name, path):
        module_spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)
        return module

    diagnostic = load(
        f"aiter.ops.flydsl.kernels._codex_att_{variant}",
        directory / f"persistent_{variant}.py",
    )
    if variant in ("global_window", "global_split"):
        scatter = load(
            f"aiter.ops.flydsl.kernels.mega_moe_gfx1250._codex_att_scatter_{variant}",
            directory / "scatter_global.py",
        )
        diagnostic.make_bf16_scatter_view = scatter.make_bf16_scatter_view
    elif variant in ("const_geometry", "local_output"):
        scatter = load(
            "aiter.ops.flydsl.kernels.mega_moe_gfx1250._codex_att_scatter_const_geometry",
            directory / "scatter_const_geometry.py",
        )
        diagnostic.make_bf16_scatter_view = scatter.make_bf16_scatter_view
    elif variant in ("oob_output", "oob_sorted"):
        scatter = load(
            f"aiter.ops.flydsl.kernels.mega_moe_gfx1250._codex_att_scatter_{variant}",
            directory / f"scatter_{variant}.py",
        )
        diagnostic.store_bf16_scatter_group = scatter.store_bf16_scatter_group
    launcher = diagnostic.launch_gemm_a8w4_tdm_gemm2_persistent
    dumped = False
    geometries = {}
    scratch_outputs = {}

    def resolve_geometry(handle):
        """Snapshot the live window before graph capture for an isolated ablation."""
        if handle in geometries:
            return geometries[handle]
        # Use the HIP library already loaded by this process, rather than a
        # second runtime from another toolchain. ccoWindowDevice begins with
        # winBase (u64), stride4G (u32), and lsaRank (i32).
        library_paths = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/libamdhip64.so" in line
        }
        if len(library_paths) != 1:
            raise RuntimeError(f"Expected one loaded HIP runtime: {library_paths}")
        hip = ctypes.CDLL(next(iter(library_paths)))

        class Attributes(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_int), ("device", ctypes.c_int),
                ("device_pointer", ctypes.c_void_p), ("host_pointer", ctypes.c_void_p),
                ("is_managed", ctypes.c_int), ("allocation_flags", ctypes.c_uint),
            ]

        attributes = Attributes()
        hip.hipPointerGetAttributes.argtypes = [ctypes.POINTER(Attributes), ctypes.c_void_p]
        hip.hipPointerGetAttributes.restype = ctypes.c_int
        attribute_rc = hip.hipPointerGetAttributes(ctypes.byref(attributes), handle)
        data = (ctypes.c_ubyte * 16)()
        hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        hip.hipMemcpy.restype = ctypes.c_int
        copy_rc = hip.hipMemcpy(data, handle, 16, 4)  # hipMemcpyDefault
        if copy_rc:
            raise RuntimeError(f"Live window geometry snapshot failed: hipMemcpy={copy_rc}")
        base = int.from_bytes(bytes(data[:8]), "little")
        stride4g = int.from_bytes(bytes(data[8:12]), "little")
        if not base or not stride4g:
            raise RuntimeError("Invalid live cco window geometry")
        geometries[handle] = (base, stride4g << 32)
        record = {
            "rank": int(os.environ["LOCAL_RANK"]), "arena_handle": handle,
            "win_base": base, "stride4g": stride4g,
            "lsa_rank": int.from_bytes(bytes(data[12:16]), "little", signed=True),
            "hip_pointer_attribute_rc": attribute_rc,
            "memory_type": attributes.type, "device": attributes.device,
            "device_pointer": attributes.device_pointer, "host_pointer": attributes.host_pointer,
            "is_managed": attributes.is_managed, "allocation_flags": attributes.allocation_flags,
            "hip_runtime": str(hip._name),
        }
        output = Path(os.environ["AITER_ATT_OUTPUT"]).parents[4]
        output.mkdir(parents=True, exist_ok=True)
        (output / f"window_geometry_rank{record['rank']}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        print(f"[ATT diagnostic] live window geometry {json.dumps(record)}", flush=True)
        return geometries[handle]

    def diagnostic_launch(*args, **kwargs):
        nonlocal dumped
        if variant == "const_geometry" and kwargs.get("enable_ep_scatter"):
            base, stride = resolve_geometry(int(kwargs["ep_arena_handle"]))
            kwargs = {**kwargs, "ep_window_base": base, "ep_window_stride": stride}
        elif variant == "local_output" and kwargs.get("enable_ep_scatter"):
            if os.environ.get("AITER_ATT_DIAGNOSTIC_DEFER") != "1":
                raise RuntimeError("Local-output ablation requires the original eager GEMM2 inputs")
            handle = int(kwargs["ep_arena_handle"])
            slots = int(kwargs["ep_destination_stride"])
            slot_stride = int(kwargs["ep_slot_stride_bytes"])
            world = int(kwargs["ep_world_size"])
            if handle not in scratch_outputs:
                # Each original peer/slot receives its own scratch row on the
                # local GPU. Preserve valid output byte count and avoid aliasing
                # either production windows or another route's destination.
                scratch_outputs[handle] = torch.empty(
                    slots * world * slot_stride, dtype=torch.uint8,
                    device=torch.cuda.current_device(),
                )
                print(
                    f"[ATT diagnostic] local scratch bytes={scratch_outputs[handle].numel()} "
                    f"slots={slots} world={world} slot_stride={slot_stride}", flush=True,
                )
            kwargs = {
                **kwargs, "ep_window_base": scratch_outputs[handle].data_ptr(),
                "ep_window_stride": slots * slot_stride,
            }
        result = launcher(*args, **kwargs)
        dump_root = os.environ.get("AITER_ATT_DIAGNOSTIC_DUMP")
        if dump_root and not dumped and os.environ.get("LOCAL_RANK", "0") == "0":
            artifact = launcher._last_compiled[1]
            output = Path(dump_root)
            output.mkdir(parents=True, exist_ok=True)
            (output / f"{variant}_source.mlir").write_text(artifact._source_ir)
            for serialized in re.findall(r'bin\s*=\s*"((?:[^"\\]|\\.)*)"', artifact._ir_text):
                data = bytearray()
                index = 0
                while index < len(serialized):
                    if serialized[index] == "\\":
                        pair = serialized[index + 1:index + 3]
                        if re.fullmatch("[0-9a-fA-F]{2}", pair):
                            data.append(int(pair, 16))
                            index += 3
                            continue
                        index += 1
                    data.append(ord(serialized[index]))
                    index += 1
                if data.startswith(b"\x7fELF"):
                    digest = hashlib.sha256(data).hexdigest()
                    (output / f"{variant}_{digest[:16]}.hsaco").write_bytes(data)
            dumped = True
        return result

    production.launch_gemm_a8w4_tdm_gemm2_persistent = diagnostic_launch
    print(f"[ATT diagnostic] isolated variant={variant} directory={directory}", flush=True)


def trace_replays(self, warmup=5, iters=2, prof_replays=0):
    for _ in range(warmup):
        self.graph.replay()
    torch.cuda.synchronize()
    self.comm.barrier()
    device_tool = None
    if (
        os.environ.get("AITER_ATT_ENGINE") == "device"
        and os.environ.get("LOCAL_RANK") == os.environ.get("AITER_ATT_LOCAL_RANK", "0")
    ):
        device_tool = ctypes.CDLL(os.environ["AITER_ATT_DEVICE_LIBRARY"])
        if device_tool.aiter_att_start() != 0:
            raise RuntimeError("Device ATT start failed")
        print("[device ATT workload] tracing one original graph replay", flush=True)
    for _ in range(iters):
        self.graph.replay()
        torch.cuda.synchronize()
        if device_tool is not None:
            if device_tool.aiter_att_stop() != 0:
                raise RuntimeError("Device ATT stop failed")
            device_tool = None
    self.comm.barrier()
    self._prof = EmptyProfile()
    return dict.fromkeys(("min", "median", "mean", "max"), 0.0), 0.0


def install_gemm2_region():
    """Collect one eager GEMM2 with the original EP inputs and live windows.

    The full original graph setup and warmup still run. The extra eager layer
    avoids CUDA graph interception and brackets only its GEMM2 launch. All EP
    ranks execute this extra layer and its normal combine in the same order.
    """
    import aiter.ops.flydsl.grouped_gemm_mxfp4 as grouped

    original = grouped.flydsl_grouped_gemm_a8w4_masked
    state = {"armed": False, "collected": False}
    selected_rank = os.environ.get("LOCAL_RANK") == os.environ.get(
        "AITER_ATT_LOCAL_RANK", "0"
    )

    def gemm2_region(*args, **kwargs):
        selected = state["armed"] and kwargs.get("K") == 2048
        if not selected:
            return original(*args, **kwargs)
        state["armed"] = False
        tool = None
        # Align all ranks at the identical production GEMM2 call, outside graph
        # capture. Do not change its tensors, schedule, or output window.
        torch.cuda.synchronize()
        if os.environ.get("AITER_ATT_DIAGNOSTIC_DEFER") == "1":
            # The graph and the eager dispatch/GEMM1 above used production
            # kernels. Enable the ablation only for this selected GEMM2, so its
            # inputs are the original nonzero EP tensors rather than outputs
            # propagated from an earlier diagnostic layer.
            install_isolated_diagnostic()
        if selected_rank:
            tool = ctypes.CDLL(os.environ["AITER_ATT_DEVICE_LIBRARY"])
            if tool.aiter_att_start() != 0:
                raise RuntimeError("Device GEMM2 ATT start failed")
            print("[device ATT workload] tracing one original eager GEMM2", flush=True)
        result = original(*args, **kwargs)
        torch.cuda.synchronize()
        if tool is not None and tool.aiter_att_stop() != 0:
            raise RuntimeError("Device GEMM2 ATT stop failed")
        state["collected"] = True
        return result

    grouped.flydsl_grouped_gemm_a8w4_masked = gemm2_region

    def gemm2_replays(self, warmup=5, iters=2, prof_replays=0):
        for _ in range(warmup):
            self.graph.replay()
        torch.cuda.synchronize()
        self.comm.barrier()
        state["armed"] = True
        self._layer_step(self.x0_static, 0)
        torch.cuda.synchronize()
        self.comm.barrier()
        if not state["collected"]:
            raise RuntimeError("No production GEMM2 passed through the region hook")
        self._prof = EmptyProfile()
        return dict.fromkeys(("min", "median", "mean", "max"), 0.0), 0.0

    return gemm2_replays


if __name__ == "__main__":
    print("[ATT workload] Kineto disabled; reported timings are zero.", flush=True)
    if os.environ.get("AITER_ATT_DIAGNOSTIC_DEFER") != "1":
        install_isolated_diagnostic()
    benchmark.DeviceMoEPipeline.bench = (
        install_gemm2_region()
        if os.environ.get("AITER_ATT_REGION") == "gemm2"
        else trace_replays
    )
    benchmark.main()
