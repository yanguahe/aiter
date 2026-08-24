#!/usr/bin/env python3
r"""Compile and benchmark one dense gfx1250 MXFP4 GEMM assembly kernel.

The input ``.s`` is treated as a complete AMDGPU assembly source.  This runner
reads its kernel-name directives but never synthesizes or modifies the kernel
descriptor or metadata.  It assembles and links the source, loads the resulting
code object once, and then launches the selected symbol on PyTorch's current
HIP stream.

The input/reference construction and numerical reporting deliberately reuse
``op_tests/test_f4gemm.py``.  The launch ABI and persistent 4x4-cluster geometry
mirror ``csrc/py_itfs_cu/asm_f4gemm.cu``.

Typical use from the aiter repository root::

    AITER_LOG_MORE=1 python my_code/isa_runner/gemm_isa_runner.py \
        --iters 100 \
        --isa ./my_code/f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps.s

This file can be imported on a machine without ROCm for its pure ABI/geometry
checks.  Compilation and execution require a Linux ROCm installation and a
gfx1250 GPU.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _path in (_REPO, _REPO / "op_tests"):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)


ARCH = "gfx1250"
CODE_OBJECT_VERSION = 6
DEFAULT_CLANG = Path("/opt/rocm/llvm/bin/clang")
DEFAULT_SYMBOL: str | None = None
EXPECTED_KERNEL_PREFIX = "f4gemm_bf16_mxfp4_"
EXPECTED_KERNEL_CONFIG = "256x256_4x4_ps"
EXPECTED_KERNEL_BASENAME = (
    "f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps"
)
LEGACY_MANGLED_SYMBOL = (
    "_ZN5aiter45f4gemm_bf16_mxfp4_"
    "ABpreShuffle_256x256_4x4_psE"
)
_SYMBOL_TOKEN = r"[A-Za-z_][A-Za-z0-9_]*"
_SYMBOL_RE = re.compile(rf"^{_SYMBOL_TOKEN}$")
_METADATA_BLOCK_RE = re.compile(
    r"(?ms)^[ \t]*\.amdgpu_metadata[ \t]*$"
    r"(?P<body>.*?)"
    r"^[ \t]*\.end_amdgpu_metadata[ \t]*$"
)

MXFP4_SCALE_BLOCK = 32
TILE_M = 256
TILE_N = 256
CLUSTER = (4, 4, 1)
BLOCK = (128, 1, 1)
PERSISTENT_TG = 256
PERSISTENT_GRID_Y = 4

KERNARG_SIZE = 80
KERNARG_LAYOUT = (
    ("ptr_D", 0, "Q"),
    ("ptr_A", 8, "Q"),
    ("ptr_B", 16, "Q"),
    ("ptr_ScaleA", 24, "Q"),
    ("ptr_ScaleB", 32, "Q"),
    ("strideD0", 40, "I"),
    ("strideA0", 44, "I"),
    ("strideB0", 48, "I"),
    ("ScaleA_stride0", 52, "I"),
    ("ScaleB_stride0", 56, "I"),
    ("M", 60, "I"),
    ("N", 64, "I"),
    ("K", 68, "I"),
    # For MXFP4, asm_f4gemm.cu stores these raw uint32 values in the two
    # GlobalScale float slots and ships exactly 80 bytes.
    ("log2_grid_x", 72, "I"),
    ("log2_grid_y", 76, "I"),
)

SUMMARY_COLUMNS = (
    "intype",
    "M",
    "N",
    "K",
    "apre",
    "init",
    "seed",
    "dtype",
    "gfx",
    "ref hash128",
    "gemm_a4w4 us",
    "gemm_a4w4 TFLOPS",
    "gemm_a4w4 TB/s",
    "gemm_a4w4 err",
    "gemm_a4w4 out hash128",
    "gemm_a4w4 max_abs",
    "gemm_a4w4 rel_l2",
)

EVENT_TIMING_COLUMNS = (
    "name",
    "cnt",
    "device_time_sum",
    "device_time_avg",
    "device_type",
    "device_index",
    "source",
)


class GemmIsaRunnerError(RuntimeError):
    """Expected build, validation, HIP, or launch failure."""


class CompileError(GemmIsaRunnerError):
    """A clang subprocess failed, with all diagnostics retained."""

    def __init__(
        self,
        command: Sequence[str],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        self.command = tuple(command)
        self.returncode = int(returncode)
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(
            f"command failed with exit code {self.returncode}:\n"
            f"{_format_command(self.command)}\n"
            f"--- stdout ---\n{self.stdout}"
            f"{'' if self.stdout.endswith(chr(10)) or not self.stdout else chr(10)}"
            f"--- stderr ---\n{self.stderr}"
        )


@dataclass(frozen=True)
class BuildResult:
    """One temporary code object and the commands that produced it."""

    code_object: Path
    commands: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class LaunchGeometry:
    """Persistent launch dimensions derived exactly as in asm_f4gemm.cu."""

    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    cluster: tuple[int, int, int]
    tiles: tuple[int, int]
    cluster_grid: tuple[int, int]
    log2_grid: tuple[int, int]


@dataclass(frozen=True)
class _Dependencies:
    torch: Any
    f4_test: Any
    check_allclose: Any
    run_perftest: Any
    get_gfx: Any


def _format_command(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def _parse_shape(text: str) -> tuple[int, int, int]:
    pieces = text.split(",")
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError(
            f"shape must be M,N,K (three comma-separated integers), got {text!r}"
        )
    try:
        shape = tuple(int(piece.strip()) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"shape must contain integers, got {text!r}"
        ) from exc
    if any(value <= 0 for value in shape):
        raise argparse.ArgumentTypeError(f"shape dimensions must be positive, got {shape}")
    return shape  # type: ignore[return-value]


def _resolve_isa(path_text: str) -> Path:
    path = Path(os.path.expandvars(path_text)).expanduser().resolve()
    if not path.is_file():
        raise GemmIsaRunnerError(f"ISA source does not exist or is not a file: {path}")
    if path.suffix.lower() != ".s":
        raise GemmIsaRunnerError(f"--isa must name an AMDGPU .s source, got: {path}")
    return path


def _validate_symbol_token(symbol: str, context: str) -> str:
    """Accept ordinary C identifiers and Itanium-mangled symbol names."""

    if not _SYMBOL_RE.fullmatch(symbol):
        raise GemmIsaRunnerError(
            f"{context} contains unsupported kernel symbol {symbol!r}; "
            "expected a C identifier or Itanium-mangled name"
        )
    return symbol


def _strip_asm_comment(text: str) -> str:
    return re.split(r";|//|#", text, maxsplit=1)[0].strip()


def _parse_asm_directive_symbols(
    source: str,
    directive: str,
    *,
    allow_many_per_line: bool,
) -> tuple[str, ...]:
    """Parse symbol operands from one GNU-style assembly directive."""

    lines = re.findall(
        rf"(?m)^\s*\.{re.escape(directive)}\b(?P<body>.*)$",
        source,
    )
    symbols: list[str] = []
    for body in lines:
        operand_text = _strip_asm_comment(body)
        operands = [
            item
            for item in re.split(r"[\s,]+", operand_text)
            if item
        ]
        if not operands:
            raise GemmIsaRunnerError(
                f".{directive} directive has no symbol; pass --symbol explicitly"
            )
        if not allow_many_per_line and len(operands) != 1:
            raise GemmIsaRunnerError(
                f".{directive} directive must contain exactly one symbol, got "
                f"{operands}; pass --symbol explicitly"
            )
        symbols.extend(
            _validate_symbol_token(item, f".{directive} directive")
            for item in operands
        )
    return tuple(symbols)


def _parse_function_type_symbols(source: str) -> tuple[str, ...]:
    symbols: list[str] = []
    for line in source.splitlines():
        clean = _strip_asm_comment(line)
        match = re.fullmatch(
            rf"\s*\.type\s+({_SYMBOL_TOKEN})\s*,\s*[@%]function\s*",
            clean,
        )
        if match:
            symbols.append(match.group(1))
    return tuple(symbols)


def _parse_metadata_symbols(source: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return metadata (.name values, .symbol values without the .kd suffix)."""

    names: list[str] = []
    symbols: list[str] = []
    for block_match in _METADATA_BLOCK_RE.finditer(source):
        block = block_match.group("body")
        for line in block.splitlines():
            clean = line.split("#", maxsplit=1)[0].strip()
            name_match = re.fullmatch(
                rf"\.name:\s*['\"]?({_SYMBOL_TOKEN})['\"]?\s*",
                clean,
            )
            if name_match:
                names.append(name_match.group(1))
                continue
            symbol_match = re.fullmatch(
                rf"\.symbol:\s*['\"]?({_SYMBOL_TOKEN})\.kd['\"]?\s*",
                clean,
            )
            if symbol_match:
                symbols.append(symbol_match.group(1))
    return tuple(names), tuple(symbols)


def _one_candidate(
    candidates: Sequence[str],
    source_kind: str,
) -> str | None:
    unique = tuple(dict.fromkeys(candidates))
    if len(unique) > 1:
        raise GemmIsaRunnerError(
            f"multiple kernel candidates from {source_kind}: {list(unique)}; "
            "pass --symbol explicitly"
        )
    return unique[0] if unique else None


def detect_kernel_symbol_from_text(source: str) -> str:
    """Safely detect one kernel symbol from complete AMDGPU assembly text.

    ``.amdhsa_kernel`` is authoritative.  Metadata and global/function
    directives are cross-checked when present, and are used as fallbacks only
    when no kernel descriptor directive exists.
    """

    descriptor = _one_candidate(
        _parse_asm_directive_symbols(
            source,
            "amdhsa_kernel",
            allow_many_per_line=False,
        ),
        ".amdhsa_kernel",
    )
    globals_ = _parse_asm_directive_symbols(
        source,
        "globl",
        allow_many_per_line=True,
    )
    function_types = _parse_function_type_symbols(source)
    metadata_names, metadata_symbols = _parse_metadata_symbols(source)
    metadata_name = _one_candidate(metadata_names, "metadata .name")
    metadata_symbol = _one_candidate(metadata_symbols, "metadata .symbol")
    if (
        metadata_name is not None
        and metadata_symbol is not None
        and metadata_name != metadata_symbol
    ):
        raise GemmIsaRunnerError(
            f"metadata .name {metadata_name!r} conflicts with metadata .symbol "
            f"{metadata_symbol!r}.kd; pass --symbol explicitly"
        )
    metadata = metadata_name or metadata_symbol

    if descriptor is not None:
        if metadata is not None and metadata != descriptor:
            raise GemmIsaRunnerError(
                f".amdhsa_kernel {descriptor!r} conflicts with metadata kernel "
                f"{metadata!r}; pass --symbol explicitly"
            )
        if globals_ and descriptor not in globals_:
            raise GemmIsaRunnerError(
                f".amdhsa_kernel {descriptor!r} is not declared by .globl "
                f"{list(dict.fromkeys(globals_))}; pass --symbol explicitly"
            )
        if function_types and descriptor not in function_types:
            raise GemmIsaRunnerError(
                f".amdhsa_kernel {descriptor!r} has no matching "
                f".type <symbol>,@function directive; pass --symbol explicitly"
            )
        return descriptor

    if metadata is not None:
        if globals_ and metadata not in globals_:
            raise GemmIsaRunnerError(
                f"metadata kernel {metadata!r} conflicts with .globl candidates "
                f"{list(dict.fromkeys(globals_))}; pass --symbol explicitly"
            )
        if function_types and metadata not in function_types:
            raise GemmIsaRunnerError(
                f"metadata kernel {metadata!r} has no matching "
                f".type <symbol>,@function directive; pass --symbol explicitly"
            )
        return metadata

    global_candidates = tuple(dict.fromkeys(globals_))
    if function_types:
        typed = set(function_types)
        global_candidates = tuple(
            symbol for symbol in global_candidates if symbol in typed
        )
    fallback = _one_candidate(global_candidates, ".globl/.type fallback")
    if fallback is None:
        raise GemmIsaRunnerError(
            "could not detect a kernel symbol: no .amdhsa_kernel, metadata "
            ".name/.symbol, or unique .globl function candidate; pass "
            "--symbol explicitly"
        )
    return fallback


def detect_kernel_symbol(isa: Path) -> str:
    try:
        source = isa.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise GemmIsaRunnerError(
            f"failed to read ISA source for symbol detection: {isa}: {exc}"
        ) from exc
    return detect_kernel_symbol_from_text(source)


def resolve_kernel_symbol(isa: Path, override: str | None) -> str:
    """Return an explicit override unchanged, otherwise inspect the ISA."""

    if override is not None:
        return _validate_symbol_token(override, "--symbol")
    return detect_kernel_symbol(isa)


def _resolve_clang(override: str | None) -> Path:
    """Find clang, preferring /opt/rocm/llvm/bin/clang as requested."""

    if override:
        expanded = os.path.expandvars(os.path.expanduser(override))
        candidate = Path(expanded)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
        found = shutil.which(expanded)
        if found:
            return Path(found).resolve()
        raise GemmIsaRunnerError(
            f"--clang is not an executable file and was not found on PATH: {override}"
        )

    candidates = [DEFAULT_CLANG]
    rocm_path = os.environ.get("ROCM_PATH")
    if rocm_path:
        candidates.append(Path(rocm_path) / "llvm" / "bin" / "clang")
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    found = shutil.which("clang")
    if found:
        return Path(found).resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise GemmIsaRunnerError(
        f"AMDGPU clang was not found (checked {searched}, then PATH); "
        "pass --clang explicitly"
    )


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        process = subprocess.run(
            [str(part) for part in command],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise CompileError(command, 127, "", str(exc)) from exc
    if process.returncode:
        raise CompileError(
            command,
            process.returncode,
            process.stdout,
            process.stderr,
        )
    return process


def compile_isa(isa: Path, clang: Path, work_dir: Path) -> BuildResult:
    """Assemble and link a complete AMDGPU source without changing it."""

    obj = work_dir / f"{isa.stem}.o"
    code_object = work_dir / f"{isa.stem}.co"
    common = (
        str(clang),
        "-target",
        "amdgcn-amd-amdhsa",
        f"-mcpu={ARCH}",
        f"-mcode-object-version={CODE_OBJECT_VERSION}",
    )
    assemble = (
        str(clang),
        "-x",
        "assembler",
        *common[1:],
        "-c",
        str(isa),
        "-o",
        str(obj),
    )
    link = (
        *common,
        "-nostdlib",
        "-Wl,--no-undefined",
        "-shared",
        str(obj),
        "-o",
        str(code_object),
    )
    _run_command(assemble)
    _run_command(link)
    if not code_object.is_file() or code_object.stat().st_size == 0:
        raise GemmIsaRunnerError(
            f"clang reported success but did not create a non-empty code object: "
            f"{code_object}"
        )
    return BuildResult(code_object=code_object, commands=(assemble, link))


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def make_launch_geometry(m: int, n: int, k: int) -> LaunchGeometry:
    """Validate a shape and reproduce the C++ persistent-cluster geometry."""

    for name, value in (("M", m), ("N", n), ("K", k)):
        if value <= 0:
            raise GemmIsaRunnerError(f"{name} must be positive, got {value}")
        if value > 0xFFFFFFFF:
            raise GemmIsaRunnerError(
                f"{name}={value} does not fit the uint32 preload kernarg"
            )
    if k % MXFP4_SCALE_BLOCK:
        raise GemmIsaRunnerError(
            f"K={k} must be divisible by the MXFP4 scale block "
            f"{MXFP4_SCALE_BLOCK}"
        )
    scale_shuffle_k = MXFP4_SCALE_BLOCK * 4
    if k % scale_shuffle_k:
        raise GemmIsaRunnerError(
            f"K={k} must be divisible by {scale_shuffle_k} for the "
            "test_f4gemm MXFP4 scale shuffle"
        )
    if m % TILE_M or n % TILE_N:
        raise GemmIsaRunnerError(
            f"the 256x256 persistent kernel requires M and N to tile exactly; "
            f"got M={m}, N={n}"
        )

    tiles_x = n // TILE_N
    tiles_y = m // TILE_M
    cluster_x, cluster_y, cluster_z = CLUSTER
    if tiles_x < cluster_x or tiles_x % cluster_x:
        raise GemmIsaRunnerError(
            f"cluster_x={cluster_x} requires N/256={tiles_x} tiles to be a "
            f"positive multiple of {cluster_x}; N must be a multiple of "
            f"{TILE_N * cluster_x}"
        )
    if tiles_y < cluster_y or tiles_y % cluster_y:
        raise GemmIsaRunnerError(
            f"cluster_y={cluster_y} requires M/256={tiles_y} tiles to be a "
            f"positive multiple of {cluster_y}; M must be a multiple of "
            f"{TILE_M * cluster_y}"
        )

    cluster_size = cluster_x * cluster_y * cluster_z
    if PERSISTENT_TG % cluster_size:
        raise GemmIsaRunnerError(
            f"persistent_tg={PERSISTENT_TG} is not divisible by "
            f"cluster size {cluster_size}"
        )
    persistent_clusters = PERSISTENT_TG // cluster_size
    if (
        PERSISTENT_GRID_Y <= 0
        or persistent_clusters % PERSISTENT_GRID_Y
    ):
        raise GemmIsaRunnerError(
            f"persistent grid_y={PERSISTENT_GRID_Y} must divide "
            f"{persistent_clusters} clusters"
        )
    grid_y = PERSISTENT_GRID_Y
    grid_x = persistent_clusters // grid_y
    if not (
        _is_power_of_two(persistent_clusters)
        and _is_power_of_two(grid_x)
        and _is_power_of_two(grid_y)
    ):
        raise GemmIsaRunnerError(
            f"persistent cluster count/grid must be powers of two; got "
            f"clusters={persistent_clusters}, gridX={grid_x}, gridY={grid_y}"
        )

    return LaunchGeometry(
        grid=(grid_x * cluster_x, grid_y * cluster_y, 1),
        block=BLOCK,
        cluster=CLUSTER,
        tiles=(tiles_x, tiles_y),
        cluster_grid=(grid_x, grid_y),
        log2_grid=(grid_x.bit_length() - 1, grid_y.bit_length() - 1),
    )


def _checked_unsigned(name: str, value: int, bits: int) -> int:
    maximum = (1 << bits) - 1
    if not 0 <= int(value) <= maximum:
        raise GemmIsaRunnerError(
            f"{name}={value} does not fit an unsigned {bits}-bit ABI field"
        )
    return int(value)


def _pack_kernarg_fields(fields: Mapping[str, int]) -> bytes:
    """Pack named fields at explicit C++ offsets; no implicit alignment."""

    expected = {name for name, _offset, _kind in KERNARG_LAYOUT}
    missing = expected.difference(fields)
    extra = set(fields).difference(expected)
    if missing or extra:
        raise GemmIsaRunnerError(
            f"invalid kernarg fields: missing={sorted(missing)}, extra={sorted(extra)}"
        )

    payload = bytearray(KERNARG_SIZE)
    for name, offset, kind in KERNARG_LAYOUT:
        bits = 64 if kind == "Q" else 32
        value = _checked_unsigned(name, fields[name], bits)
        struct.pack_into(f"<{kind}", payload, offset, value)
    return bytes(payload)


def pack_mxfp4_kernargs(
    *,
    ptr_d: int,
    ptr_a: int,
    ptr_b: int,
    ptr_scale_a: int,
    ptr_scale_b: int,
    m: int,
    n: int,
    k: int,
    geometry: LaunchGeometry,
) -> bytes:
    """Build the exact 80-byte preload-SGPR MXFP4 argument payload."""

    stride_d = n * 2
    stride_a = k // 2
    stride_b = k // 2
    stride_scale_a = k // MXFP4_SCALE_BLOCK
    stride_scale_b = k // MXFP4_SCALE_BLOCK
    return _pack_kernarg_fields(
        {
            "ptr_D": ptr_d,
            "ptr_A": ptr_a,
            "ptr_B": ptr_b,
            "ptr_ScaleA": ptr_scale_a,
            "ptr_ScaleB": ptr_scale_b,
            "strideD0": stride_d,
            "strideA0": stride_a,
            "strideB0": stride_b,
            "ScaleA_stride0": stride_scale_a,
            "ScaleB_stride0": stride_scale_b,
            "M": m,
            "N": n,
            "K": k,
            "log2_grid_x": geometry.log2_grid[0],
            "log2_grid_y": geometry.log2_grid[1],
        }
    )


class _Dim3(ctypes.Structure):
    _fields_ = (
        ("x", ctypes.c_uint),
        ("y", ctypes.c_uint),
        ("z", ctypes.c_uint),
    )


class _HipLaunchAttributeValue(ctypes.Union):
    # The public HIP ABI reserves 64 bytes.  The pointer member forces the same
    # 8-byte union alignment as the complete C union.
    _fields_ = (
        ("pad", ctypes.c_ubyte * 64),
        ("_pointer_alignment", ctypes.c_void_p),
        ("clusterDim", _Dim3),
    )


class _HipLaunchAttribute(ctypes.Structure):
    # hip_runtime_api.h explicitly pads the enum to an 8-byte value offset.
    _fields_ = (
        ("id", ctypes.c_int),
        ("_pad", ctypes.c_char * 4),
        ("value", _HipLaunchAttributeValue),
    )


class _HipLaunchConfig(ctypes.Structure):
    _fields_ = (
        ("gridDimX", ctypes.c_uint),
        ("gridDimY", ctypes.c_uint),
        ("gridDimZ", ctypes.c_uint),
        ("blockDimX", ctypes.c_uint),
        ("blockDimY", ctypes.c_uint),
        ("blockDimZ", ctypes.c_uint),
        ("sharedMemBytes", ctypes.c_uint),
        ("hStream", ctypes.c_void_p),
        ("attrs", ctypes.POINTER(_HipLaunchAttribute)),
        ("numAttrs", ctypes.c_uint),
    )


_HIP_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4
_HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(1)
_HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(2)
_HIP_LAUNCH_PARAM_END = ctypes.c_void_p(3)


def run_static_contract_checks() -> None:
    """CPU-only fixture for every ABI offset and the default launch geometry."""

    def symbol_fixture(symbol: str) -> str:
        return f"""
        .text
        .globl {symbol}
        .type {symbol},@function
        {symbol}:
            s_endpgm
        .amdhsa_kernel {symbol}
        .end_amdhsa_kernel
        .amdgpu_metadata
        ---
        amdhsa.kernels:
          - .name: {symbol}
            .symbol: {symbol}.kd
        ...
        .end_amdgpu_metadata
        """

    current_symbol = EXPECTED_KERNEL_BASENAME
    detected = detect_kernel_symbol_from_text(symbol_fixture(current_symbol))
    if detected != current_symbol:
        raise AssertionError(
            f"unmangled symbol fixture detected {detected!r}, "
            f"expected {current_symbol!r}"
        )
    detected = detect_kernel_symbol_from_text(
        symbol_fixture(LEGACY_MANGLED_SYMBOL)
    )
    if detected != LEGACY_MANGLED_SYMBOL:
        raise AssertionError(
            f"legacy mangled fixture detected {detected!r}, "
            f"expected {LEGACY_MANGLED_SYMBOL!r}"
        )
    _validate_mode("mxfp4", 1, "bf16", current_symbol)
    _validate_mode("mxfp4", 1, "bf16", LEGACY_MANGLED_SYMBOL)

    metadata_fallback = f"""
    .globl {current_symbol}
    .type {current_symbol},@function
    .amdgpu_metadata
    ---
    amdhsa.kernels:
      - .name: {current_symbol}
        .symbol: {current_symbol}.kd
    ...
    .end_amdgpu_metadata
    """
    if detect_kernel_symbol_from_text(metadata_fallback) != current_symbol:
        raise AssertionError("metadata fallback did not select the kernel")
    globl_fallback = (
        f".globl {current_symbol}\n"
        f".type {current_symbol},@function\n"
    )
    if detect_kernel_symbol_from_text(globl_fallback) != current_symbol:
        raise AssertionError(".globl/.type fallback did not select the kernel")

    explicit = resolve_kernel_symbol(
        Path("this-file-must-not-be-read.s"),
        "explicit_kernel_override",
    )
    if explicit != "explicit_kernel_override":
        raise AssertionError(f"explicit --symbol override changed to {explicit!r}")

    conflict_fixture = symbol_fixture(current_symbol).replace(
        f".name: {current_symbol}",
        ".name: conflicting_kernel",
    ).replace(
        f".symbol: {current_symbol}.kd",
        ".symbol: conflicting_kernel.kd",
    )
    multi_fixture = (
        symbol_fixture(current_symbol)
        + "\n.amdhsa_kernel second_kernel\n.end_amdhsa_kernel\n"
    )
    for label, text, message_fragment in (
        (
            "conflicting directives",
            conflict_fixture,
            "conflicts with metadata kernel",
        ),
        (
            "multiple kernels",
            multi_fixture,
            "multiple kernel candidates",
        ),
    ):
        try:
            detect_kernel_symbol_from_text(text)
        except GemmIsaRunnerError as exc:
            if message_fragment not in str(exc):
                raise AssertionError(
                    f"{label} fixture raised unexpected error: {exc}"
                ) from exc
        else:
            raise AssertionError(f"{label} fixture did not fail")

    perf_call: dict[str, Any] = {}

    def perf_stub(launch: Any, **kwargs: Any) -> tuple[str, float]:
        perf_call.update(kwargs)
        return launch(), 75.3293

    output_marker, event_avg = _run_cuda_event_perftest(
        perf_stub,
        lambda: "fixture-output",
        100,
    )
    if output_marker != "fixture-output" or event_avg != 75.3293:
        raise AssertionError("CUDA-event run_perftest stub returned wrong values")
    expected_perf_call = {
        "num_iters": 100,
        "testGraph": False,
        "use_cuda_event": True,
    }
    if perf_call != expected_perf_call:
        raise AssertionError(
            f"CUDA-event run_perftest kwargs are {perf_call}, "
            f"expected {expected_perf_call}"
        )
    timing_row = make_cuda_event_timing_row(
        EXPECTED_KERNEL_BASENAME,
        100,
        event_avg,
        0,
    )
    if tuple(timing_row) != EVENT_TIMING_COLUMNS:
        raise AssertionError(
            f"event timing columns are {tuple(timing_row)}, "
            f"expected {EVENT_TIMING_COLUMNS}"
        )
    if (
        timing_row["name"] != EXPECTED_KERNEL_BASENAME
        or timing_row["cnt"] != 100
        or abs(timing_row["device_time_sum"] - 7532.93) > 1e-9
        or timing_row["device_time_avg"] != 75.3293
        or timing_row["device_type"] != "CUDA/HIP"
        or timing_row["device_index"] != 0
        or timing_row["source"] != "cuda.Event"
    ):
        raise AssertionError(f"unexpected CUDA-event timing row: {timing_row}")

    fixture: dict[str, int] = {}
    for index, (name, _offset, kind) in enumerate(KERNARG_LAYOUT, start=1):
        fixture[name] = (
            0x1100000000000000 + index
            if kind == "Q"
            else 0x22000000 + index
        )
    payload = _pack_kernarg_fields(fixture)
    if len(payload) != KERNARG_SIZE:
        raise AssertionError(
            f"kernarg fixture size is {len(payload)}, expected {KERNARG_SIZE}"
        )
    for name, offset, kind in KERNARG_LAYOUT:
        actual = struct.unpack_from(f"<{kind}", payload, offset)[0]
        if actual != fixture[name]:
            raise AssertionError(
                f"kernarg field {name} at byte {offset}: "
                f"got 0x{actual:x}, expected 0x{fixture[name]:x}"
            )

    geometry = make_launch_geometry(18432, 2048, 7168)
    expected = LaunchGeometry(
        grid=(16, 16, 1),
        block=(128, 1, 1),
        cluster=(4, 4, 1),
        tiles=(8, 72),
        cluster_grid=(4, 4),
        log2_grid=(2, 2),
    )
    if geometry != expected:
        raise AssertionError(
            f"default geometry mismatch: got {geometry}, expected {expected}"
        )

    default_payload = pack_mxfp4_kernargs(
        ptr_d=0x1010101010101010,
        ptr_a=0x2020202020202020,
        ptr_b=0x3030303030303030,
        ptr_scale_a=0x4040404040404040,
        ptr_scale_b=0x5050505050505050,
        m=18432,
        n=2048,
        k=7168,
        geometry=geometry,
    )
    expected_default_fields = {
        "ptr_D": 0x1010101010101010,
        "ptr_A": 0x2020202020202020,
        "ptr_B": 0x3030303030303030,
        "ptr_ScaleA": 0x4040404040404040,
        "ptr_ScaleB": 0x5050505050505050,
        "strideD0": 4096,
        "strideA0": 3584,
        "strideB0": 3584,
        "ScaleA_stride0": 224,
        "ScaleB_stride0": 224,
        "M": 18432,
        "N": 2048,
        "K": 7168,
        "log2_grid_x": 2,
        "log2_grid_y": 2,
    }
    for name, offset, kind in KERNARG_LAYOUT:
        actual = struct.unpack_from(f"<{kind}", default_payload, offset)[0]
        expected_value = expected_default_fields[name]
        if actual != expected_value:
            raise AssertionError(
                f"default kernarg field {name} at byte {offset}: "
                f"got {actual}, expected {expected_value}"
            )

    ctypes_contract = {
        "attribute value size": (
            ctypes.sizeof(_HipLaunchAttributeValue),
            64,
        ),
        "attribute value offset": (_HipLaunchAttribute.value.offset, 8),
        "attribute size": (ctypes.sizeof(_HipLaunchAttribute), 72),
        "config stream offset": (_HipLaunchConfig.hStream.offset, 32),
        "config attrs offset": (_HipLaunchConfig.attrs.offset, 40),
        "config numAttrs offset": (_HipLaunchConfig.numAttrs.offset, 48),
        "config size": (ctypes.sizeof(_HipLaunchConfig), 56),
    }
    for label, (actual, expected_value) in ctypes_contract.items():
        if actual != expected_value:
            raise AssertionError(
                f"HIP ctypes {label} is {actual}, expected {expected_value}"
            )


class _HipRuntime:
    """Minimal checked HIP runtime/driver binding used by the dense runner."""

    def __init__(self) -> None:
        rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
        candidates = (
            rocm / "lib" / "libamdhip64.so",
            rocm / "lib64" / "libamdhip64.so",
            Path("libamdhip64.so"),
            Path("libamdhip64.so.7"),
            Path("libamdhip64.so.6"),
        )
        failures: list[str] = []
        self.lib: Any | None = None
        for candidate in candidates:
            try:
                self.lib = ctypes.CDLL(str(candidate))
                break
            except OSError as exc:
                failures.append(f"{candidate}: {exc}")
        if self.lib is None:
            raise GemmIsaRunnerError(
                "libamdhip64.so was not found; tried:\n  " + "\n  ".join(failures)
            )

        void_p = ctypes.c_void_p
        self._bind("hipGetErrorString", [ctypes.c_int], ctypes.c_char_p)
        self._bind("hipInit", [ctypes.c_uint], ctypes.c_int)
        self._bind("hipSetDevice", [ctypes.c_int], ctypes.c_int)
        self._bind(
            "hipModuleLoadData",
            [ctypes.POINTER(void_p), void_p],
            ctypes.c_int,
        )
        self._bind("hipModuleUnload", [void_p], ctypes.c_int)
        self._bind(
            "hipModuleGetFunction",
            [ctypes.POINTER(void_p), void_p, ctypes.c_char_p],
            ctypes.c_int,
        )
        self._bind(
            "hipDrvLaunchKernelEx",
            [
                ctypes.POINTER(_HipLaunchConfig),
                void_p,
                ctypes.POINTER(void_p),
                ctypes.POINTER(void_p),
            ],
            ctypes.c_int,
        )
        self._bind("hipStreamSynchronize", [void_p], ctypes.c_int)
        self.check(self.lib.hipInit(0), "hipInit")

    def _bind(
        self,
        name: str,
        argtypes: Sequence[Any],
        restype: Any,
    ) -> None:
        try:
            function = getattr(self.lib, name)
        except AttributeError as exc:
            if name == "hipDrvLaunchKernelEx":
                raise GemmIsaRunnerError(
                    "the HIP runtime does not export hipDrvLaunchKernelEx; "
                    "a 4x4 workgroup-cluster kernel cannot be launched safely"
                ) from exc
            raise GemmIsaRunnerError(
                f"the HIP runtime does not export required function {name}"
            ) from exc
        function.argtypes = list(argtypes)
        function.restype = restype

    def check(self, error: int, call: str) -> None:
        if error == 0:
            return
        raw = self.lib.hipGetErrorString(int(error))
        message = raw.decode(errors="replace") if raw else "unknown HIP error"
        raise GemmIsaRunnerError(f"{call} failed with HIP error {error}: {message}")


class _LoadedClusterKernel:
    """A loaded module/function kept resident for all timed launches."""

    def __init__(
        self,
        code_object: Path,
        symbol: str,
        device: int,
    ) -> None:
        self._hip = _HipRuntime()
        self._hip.check(self._hip.lib.hipSetDevice(device), "hipSetDevice")
        self._module = ctypes.c_void_p()
        self._function = ctypes.c_void_p()
        self._stream = ctypes.c_void_p()
        self._configured = False

        data = code_object.read_bytes()
        self._blob = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        self._hip.check(
            self._hip.lib.hipModuleLoadData(
                ctypes.byref(self._module),
                ctypes.cast(self._blob, ctypes.c_void_p),
            ),
            f"hipModuleLoadData({code_object})",
        )
        try:
            self._hip.check(
                self._hip.lib.hipModuleGetFunction(
                    ctypes.byref(self._function),
                    self._module,
                    symbol.encode(),
                ),
                f"hipModuleGetFunction({symbol})",
            )
        except BaseException:
            self._hip.lib.hipModuleUnload(self._module)
            self._module = ctypes.c_void_p()
            raise

    def configure(
        self,
        payload: bytes,
        geometry: LaunchGeometry,
        stream: int,
    ) -> None:
        if len(payload) != KERNARG_SIZE:
            raise GemmIsaRunnerError(
                f"MXFP4 kernarg payload must be {KERNARG_SIZE} bytes, "
                f"got {len(payload)}"
            )
        self._arg_buffer = (
            ctypes.c_ubyte * len(payload)
        ).from_buffer_copy(payload)
        self._arg_size = ctypes.c_size_t(len(payload))
        self._extra = (ctypes.c_void_p * 5)(
            _HIP_LAUNCH_PARAM_BUFFER_POINTER,
            ctypes.cast(self._arg_buffer, ctypes.c_void_p),
            _HIP_LAUNCH_PARAM_BUFFER_SIZE,
            ctypes.cast(ctypes.byref(self._arg_size), ctypes.c_void_p),
            _HIP_LAUNCH_PARAM_END,
        )

        self._attributes = (_HipLaunchAttribute * 1)()
        self._attributes[0].id = _HIP_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        self._attributes[0].value.clusterDim = _Dim3(*geometry.cluster)
        self._stream = ctypes.c_void_p(stream)
        self._config = _HipLaunchConfig(
            gridDimX=geometry.grid[0],
            gridDimY=geometry.grid[1],
            gridDimZ=geometry.grid[2],
            blockDimX=geometry.block[0],
            blockDimY=geometry.block[1],
            blockDimZ=geometry.block[2],
            # The assembly descriptor owns its fixed 327680-byte LDS segment;
            # asm_f4gemm.cu likewise passes zero dynamic shared memory.
            sharedMemBytes=0,
            hStream=self._stream,
            attrs=ctypes.cast(
                self._attributes,
                ctypes.POINTER(_HipLaunchAttribute),
            ),
            numAttrs=1,
        )
        self._configured = True

    def launch(self) -> None:
        """Enqueue exactly one dispatch; no loading, packing, or synchronization."""

        if not self._configured:
            raise GemmIsaRunnerError("kernel launch attempted before configure()")
        # hipExtModuleLaunchKernel uses this same five-entry extra-buffer
        # protocol, but it has no cluster-attribute parameter.  The current
        # Aiter cluster launcher therefore uses hipDrvLaunchKernelEx.
        self._hip.check(
            self._hip.lib.hipDrvLaunchKernelEx(
                ctypes.byref(self._config),
                self._function,
                None,
                ctypes.cast(
                    self._extra,
                    ctypes.POINTER(ctypes.c_void_p),
                ),
            ),
            "hipDrvLaunchKernelEx",
        )

    def synchronize(self) -> None:
        self._hip.check(
            self._hip.lib.hipStreamSynchronize(self._stream),
            "hipStreamSynchronize",
        )

    def close(self) -> None:
        if not self._module.value:
            return
        sync_error: BaseException | None = None
        if self._configured:
            try:
                self.synchronize()
            except BaseException as exc:
                sync_error = exc
        module = self._module
        self._module = ctypes.c_void_p()
        unload_error = self._hip.lib.hipModuleUnload(module)
        if sync_error is not None:
            raise sync_error
        self._hip.check(unload_error, "hipModuleUnload")

    def __enter__(self) -> "_LoadedClusterKernel":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except BaseException:
                # Preserve the original build/input/launch exception.
                pass
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


def _load_dependencies(device: int) -> _Dependencies:
    """Import the repo implementation only after CLI parsing/build."""

    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise GemmIsaRunnerError("PyTorch is required to run the GEMM") from exc
    if not torch.cuda.is_available():
        raise GemmIsaRunnerError("a ROCm PyTorch GPU device is required")
    if getattr(torch.version, "hip", None) is None:
        raise GemmIsaRunnerError(
            "this runner requires a ROCm PyTorch build (torch.version.hip is unset)"
        )
    try:
        torch.cuda.set_device(device)
    except Exception as exc:
        raise GemmIsaRunnerError(
            f"cannot select PyTorch/HIP device {device}: {exc}"
        ) from exc

    try:
        import test_f4gemm as f4_test  # type: ignore[import-not-found]
        from aiter.jit.utils.chip_info import get_gfx_runtime
        from aiter.test_common import checkAllclose, run_perftest
    except ImportError as exc:
        raise GemmIsaRunnerError(
            f"failed to import aiter from repository root {_REPO}: {exc}"
        ) from exc
    return _Dependencies(
        torch=torch,
        f4_test=f4_test,
        check_allclose=checkAllclose,
        run_perftest=run_perftest,
        get_gfx=get_gfx_runtime,
    )


def _validate_mode(
    intype: str,
    apre: int,
    dtype: str,
    symbol: str,
) -> None:
    if intype != "mxfp4":
        raise GemmIsaRunnerError(
            "this independent ISA runner supports only MXFP4.  NVFP4 uses an "
            "88-byte ABI with global scales and different scale blocking; use "
            "an NVFP4-specific runner/kernel instead"
        )
    if dtype != "bf16":
        raise GemmIsaRunnerError(
            f"this kernel writes BF16 output only, not {dtype}"
        )
    if (
        EXPECTED_KERNEL_PREFIX not in symbol
        or EXPECTED_KERNEL_CONFIG not in symbol
    ):
        raise GemmIsaRunnerError(
            f"kernel symbol {symbol!r} is not an obvious match for the "
            "BF16 MXFP4 256x256 4x4 persistent ABI used by this runner"
        )
    expected_shuffle = "ABpreShuffle" if apre else "BpreShuffle"
    if expected_shuffle not in symbol:
        raise GemmIsaRunnerError(
            f"--apre {apre} requires a {expected_shuffle} kernel, got "
            f"symbol {symbol!r}"
        )
    if apre == 0 and "ABpreShuffle" in symbol:
        raise GemmIsaRunnerError(
            f"--apre 0 is incompatible with ABpreShuffle symbol {symbol!r}; "
            "pass the BpreShuffle ISA and its symbol via --symbol"
        )


def _run_cuda_event_perftest(
    run_perftest: Any,
    launch: Any,
    iters: int,
) -> tuple[Any, float]:
    """Use run_perftest warmups/iterations but return before Kineto profiling."""

    return run_perftest(
        launch,
        num_iters=iters,
        testGraph=False,
        use_cuda_event=True,
    )


def make_cuda_event_timing_row(
    symbol: str,
    iters: int,
    avg_us: float,
    device: int,
) -> dict[str, Any]:
    """Build a profiler-like row whose source is explicitly cuda.Event."""

    avg_us = float(avg_us)
    if avg_us <= 0.0:
        raise GemmIsaRunnerError(
            f"cuda.Event returned a non-positive latency: {avg_us} us"
        )
    if iters < 1:
        raise GemmIsaRunnerError(
            f"CUDA-event timing requires at least one iteration, got {iters}"
        )
    return {
        "name": symbol,
        "cnt": int(iters),
        "device_time_sum": avg_us * iters,
        "device_time_avg": avg_us,
        "device_type": "CUDA/HIP",
        "device_index": int(device),
        "source": "cuda.Event",
    }


def _plain_markdown(
    row: Mapping[str, Any],
    columns: Sequence[str] = SUMMARY_COLUMNS,
) -> str:
    values = [str(row[column]) for column in columns]
    widths = [
        max(len(column), len(value))
        for column, value in zip(columns, values)
    ]

    def line(items: Sequence[str]) -> str:
        return "| " + " | ".join(
            item.ljust(width)
            for item, width in zip(items, widths)
        ) + " |"

    separator = ["-" * width for width in widths]
    return "\n".join((line(columns), line(separator), line(values)))


def _print_cuda_event_timing(row: Mapping[str, Any]) -> None:
    display = dict(row)
    display["device_time_sum"] = f"{float(row['device_time_sum']):.4f}"
    display["device_time_avg"] = f"{float(row['device_time_avg']):.4f}"
    print(
        "CUDA-event kernel timing "
        "(microseconds; source=cuda.Event, not torch profiler):"
    )
    print(_plain_markdown(display, EVENT_TIMING_COLUMNS))


def _print_summary(row: Mapping[str, Any], pandas_module: Any) -> None:
    frame = pandas_module.DataFrame([row], columns=SUMMARY_COLUMNS)
    try:
        markdown = frame.to_markdown(index=False)
    except ImportError:
        markdown = _plain_markdown(row)
    print("gemm_a4w4 (F4GEMM ISA runner) summary (markdown):")
    print(markdown)


def _run_gemm(
    args: argparse.Namespace,
    code_object: Path,
    symbol: str,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    m, n, k = args.shape
    geometry = make_launch_geometry(m, n, k)
    dependencies = _load_dependencies(args.device)
    torch = dependencies.torch

    gfx = dependencies.get_gfx()
    if gfx != ARCH:
        raise GemmIsaRunnerError(
            f"the supplied kernel targets {ARCH}, but the active device reports {gfx}"
        )

    dtype = torch.bfloat16
    with _LoadedClusterKernel(
        code_object,
        symbol,
        args.device,
    ) as module:
        # Re-seed every input preparation so results do not depend on run order.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # This is the exact test_f4gemm.py path: one per_1x32 quantization,
        # one FP32 decoded reference matmul, and the same A/B/scale shuffles.
        inputs, reference = dependencies.f4_test._prep_mxfp4(
            m,
            n,
            k,
            args.apre,
            dtype,
            args.init,
        )
        # The digest forces reference/input preparation to finish before timing.
        reference_hash = dependencies.f4_test._tensor_blake2b128(reference)
        output = torch.empty((m, n), dtype=dtype, device=inputs["A"].device)
        stream = torch.cuda.current_stream(args.device)
        payload = pack_mxfp4_kernargs(
            ptr_d=int(output.data_ptr()),
            ptr_a=int(inputs["A"].data_ptr()),
            ptr_b=int(inputs["B"].data_ptr()),
            ptr_scale_a=int(inputs["sA"].data_ptr()),
            ptr_scale_b=int(inputs["sB"].data_ptr()),
            m=m,
            n=n,
            k=k,
            geometry=geometry,
        )
        module.configure(payload, geometry, int(stream.cuda_stream))

        def launch() -> Any:
            module.launch()
            return output

        timed_output, microseconds = _run_cuda_event_perftest(
            dependencies.run_perftest,
            launch,
            args.iters,
        )
        module.synchronize()
        event_timing = make_cuda_event_timing_row(
            symbol,
            args.iters,
            microseconds,
            args.device,
        )
        us = float(event_timing["device_time_avg"])

    error = dependencies.check_allclose(
        reference,
        timed_output,
        rtol=1e-1,
        atol=1.0,
        msg="mxfp4 gemm_a4w4 ISA runner",
    )
    output_hash = dependencies.f4_test._tensor_blake2b128(timed_output)
    max_abs, rel_l2 = dependencies.f4_test._float32_error_metrics(
        reference,
        timed_output,
    )

    flops = 2 * m * n * k
    logical_bytes = (
        inputs["A"].nbytes
        + inputs["B"].nbytes
        + inputs["sA"].nbytes
        + inputs["sB"].nbytes
        + m * n * dtype.itemsize
    )
    row = {
        "intype": args.intype,
        "M": m,
        "N": n,
        "K": k,
        "apre": args.apre,
        "init": args.init,
        "seed": args.seed,
        "dtype": args.dtype,
        "gfx": gfx,
        "ref hash128": reference_hash,
        "gemm_a4w4 us": round(us, 2),
        "gemm_a4w4 TFLOPS": round(flops / us / 1e6, 1),
        "gemm_a4w4 TB/s": round(logical_bytes / us / 1e6, 2),
        "gemm_a4w4 err": error,
        "gemm_a4w4 out hash128": output_hash,
        "gemm_a4w4 max_abs": max_abs,
        "gemm_a4w4 rel_l2": rel_l2,
    }
    return row, event_timing, bool(error == 0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--isa",
        required=True,
        help="complete gfx1250 AMDGPU assembly source (.s)",
    )
    parser.add_argument(
        "--clang",
        help="AMDGPU clang executable (default: /opt/rocm/llvm/bin/clang, then PATH)",
    )
    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help=(
            "kernel symbol override (default: auto-detect from .amdhsa_kernel, "
            "then safely fall back to metadata/.globl)"
        ),
    )
    parser.add_argument(
        "--intype",
        choices=("mxfp4", "nvfp4"),
        default="mxfp4",
        help="input format; this 80-byte runner rejects nvfp4 explicitly",
    )
    parser.add_argument(
        "--apre",
        type=int,
        choices=(0, 1),
        default=1,
        help="1 for A preshuffle, 0 for row-major A and a matching override symbol",
    )
    parser.add_argument(
        "--init",
        choices=("constant", "random"),
        default="random",
        help="input initialization mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random input seed, reapplied before each input preparation (default: 0)",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        choices=("bf16",),
        default="bf16",
        help="output dtype (this kernel supports bf16 only)",
    )
    parser.add_argument(
        "-mnk",
        "--shape",
        type=_parse_shape,
        default=(18432, 2048, 7168),
        metavar="M,N,K",
        help="single GEMM shape (default: 18432,2048,7168)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="run_perftest timed iterations (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="PyTorch/HIP device ordinal (default: 0)",
    )
    parser.add_argument(
        "--keep-co",
        action="store_true",
        help="copy the temporary code object beside the ISA as <stem>.co",
    )
    parser.add_argument(
        "--co-out",
        help="copy the compiled code object to this path (implies --keep-co)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.iters < 1:
        parser.error("--iters must be at least 1")
    if args.device < 0:
        parser.error("--device must be non-negative")

    try:
        run_static_contract_checks()
        # Validate geometry before invoking clang or allocating device memory.
        make_launch_geometry(*args.shape)
        isa = _resolve_isa(args.isa)
        symbol = resolve_kernel_symbol(isa, args.symbol)
        _validate_mode(args.intype, args.apre, args.dtype, symbol)
        clang = _resolve_clang(args.clang)

        with tempfile.TemporaryDirectory(prefix="gemm_isa_runner_") as temp:
            result = compile_isa(isa, clang, Path(temp))
            for command in result.commands:
                print(f"[gemm_isa_runner] {_format_command(command)}")
            print(f"[gemm_isa_runner] loading kernel symbol: {symbol}")

            if args.co_out or args.keep_co:
                destination = (
                    Path(os.path.expandvars(args.co_out)).expanduser().resolve()
                    if args.co_out
                    else isa.with_suffix(".co")
                )
                if destination == isa:
                    raise GemmIsaRunnerError(
                        "--co-out must not overwrite the input ISA source"
                    )
                if destination.suffix.lower() != ".co":
                    raise GemmIsaRunnerError(
                        f"--co-out must use a .co suffix, got: {destination}"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(result.code_object, destination)
                print(f"[gemm_isa_runner] kept code object: {destination}")

            row, event_timing, passed = _run_gemm(
                args,
                result.code_object,
                symbol,
            )
            _print_cuda_event_timing(event_timing)
            _print_summary(row, _load_pandas_from_row_context())
            return 0 if passed else 3
    except CompileError as exc:
        print(f"[gemm_isa_runner] compile failed:\n{exc}", file=sys.stderr)
        return 2
    except GemmIsaRunnerError as exc:
        print(f"[gemm_isa_runner] error: {exc}", file=sys.stderr)
        return 1


def _load_pandas_from_row_context() -> Any:
    """Import the repository's existing pandas dependency for final rendering."""

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise GemmIsaRunnerError(
            "pandas is required by test_f4gemm.py and for summary rendering"
        ) from exc
    return pd


if __name__ == "__main__":
    sys.exit(main())
