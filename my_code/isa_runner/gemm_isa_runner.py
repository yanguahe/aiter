#!/usr/bin/env python3
r"""Compile and benchmark one dense gfx1250 MXFP4 GEMM assembly kernel.

The input ``.s`` is treated as a complete AMDGPU assembly source.  This runner
reads its kernel-name directives but never synthesizes or modifies the kernel
descriptor or metadata.  It assembles and links the source, loads the resulting
code object once, and then launches the selected symbol on PyTorch's current
HIP stream.

The input/reference construction and numerical reporting deliberately reuse
``op_tests/test_f4gemm.py``.  The launch ABI and each profile's persistent
cluster geometry mirror ``csrc/py_itfs_cu/asm_f4gemm.cu``.

Typical use from the aiter repository root::

    AITER_LOG_MORE=1 python my_code/isa_runner/gemm_isa_runner.py \
        --iters 100 \
        --isa ./my_code/f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps.s

The exact ``256x256_4x4`` and ``128x128_4x4`` symbols remain supported with
the same command line; symbols with unrelated suffixes are rejected.

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
DEFAULT_CLANG = Path(
    "/data/yanguahe/code/wk_sp1/llvm-project/mlir_install/bin/clang"
)
DEFAULT_CLANG_RUNTIME_LIBRARIES = (
    Path(
        "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/"
        "rocm_sysdeps/lib"
    ),
    Path("/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib"),
    Path("/opt/rocm/lib"),
)
DEFAULT_SYMBOL: str | None = None
KERNEL_SYMBOL_256 = "f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps"
KERNEL_SYMBOL_128 = "f4gemm_bf16_mxfp4_ABpreShuffle_128x128_4x4_ps"
KERNEL_SYMBOL_64X256 = "f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps"
# Retain this public constant for importers that used the original runner.
EXPECTED_KERNEL_BASENAME = KERNEL_SYMBOL_256
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


@dataclass(frozen=True)
class KernelProfile:
    """Exact symbol, geometry, ABI, and resource contract for one ISA."""

    name: str
    primary_symbol: str
    symbols: tuple[str, ...]
    wg_tile: tuple[int, int]
    wave_tile: tuple[int, int]
    output_quadrants: tuple[int, int]
    cluster: tuple[int, int, int]
    block: tuple[int, int, int]
    k_multiple: int
    persistent_tg: int
    persistent_grid_y: int
    apre: int
    abi_name: str
    kernarg_size: int
    kernarg_layout: tuple[tuple[str, int, str], ...]
    group_segment_fixed_size: int
    next_free_vgpr: int
    next_free_sgpr: int
    metadata_vgpr_count: int
    metadata_sgpr_count: int


KERNEL_PROFILE_256 = KernelProfile(
    name="bf16-mxfp4-wg256-wave128-4x4-persistent",
    primary_symbol=KERNEL_SYMBOL_256,
    symbols=(KERNEL_SYMBOL_256, LEGACY_MANGLED_SYMBOL),
    wg_tile=(256, 256),
    wave_tile=(128, 128),
    output_quadrants=(2, 2),
    cluster=(4, 4, 1),
    block=(128, 1, 1),
    # Preserve the original runner's accepted K domain.  The old ISA document
    # establishes K%128 for input shuffling but does not establish that every
    # K%128 path is a complete K256 body.
    k_multiple=128,
    persistent_tg=256,
    persistent_grid_y=4,
    apre=1,
    abi_name="bf16-mxfp4-preload-v1",
    kernarg_size=KERNARG_SIZE,
    kernarg_layout=KERNARG_LAYOUT,
    group_segment_fixed_size=327680,
    next_free_vgpr=1024,
    next_free_sgpr=104,
    metadata_vgpr_count=1024,
    metadata_sgpr_count=106,
)
KERNEL_PROFILE_128 = KernelProfile(
    name="bf16-mxfp4-wg128-wave64-4x4-persistent",
    primary_symbol=KERNEL_SYMBOL_128,
    symbols=(KERNEL_SYMBOL_128,),
    wg_tile=(128, 128),
    wave_tile=(64, 64),
    output_quadrants=(2, 2),
    cluster=(4, 4, 1),
    block=(128, 1, 1),
    k_multiple=256,
    persistent_tg=256,
    persistent_grid_y=4,
    apre=1,
    abi_name="bf16-mxfp4-preload-v1",
    kernarg_size=KERNARG_SIZE,
    kernarg_layout=KERNARG_LAYOUT,
    group_segment_fixed_size=172032,
    next_free_vgpr=384,
    next_free_sgpr=104,
    metadata_vgpr_count=384,
    metadata_sgpr_count=106,
)
KERNEL_PROFILE_64X256 = KernelProfile(
    name="bf16-mxfp4-wg64x256-wave64-1x4-persistent",
    primary_symbol=KERNEL_SYMBOL_64X256,
    symbols=(KERNEL_SYMBOL_64X256,),
    wg_tile=(64, 256),
    wave_tile=(64, 64),
    output_quadrants=(1, 4),
    # HIP cluster dimensions are (x=N, y=M, z).
    cluster=(4, 1, 1),
    block=(128, 1, 1),
    k_multiple=256,
    persistent_tg=256,
    persistent_grid_y=4,
    apre=1,
    abi_name="bf16-mxfp4-preload-v1",
    kernarg_size=KERNARG_SIZE,
    kernarg_layout=KERNARG_LAYOUT,
    group_segment_fixed_size=206848,
    next_free_vgpr=384,
    next_free_sgpr=104,
    metadata_vgpr_count=384,
    metadata_sgpr_count=106,
)
SUPPORTED_KERNEL_PROFILES = (
    KERNEL_PROFILE_256,
    KERNEL_PROFILE_128,
    KERNEL_PROFILE_64X256,
)


def _build_kernel_profile_lookup(
    profiles: Sequence[KernelProfile],
) -> dict[str, KernelProfile]:
    """Build an exact-symbol map and reject ambiguous registrations."""

    lookup: dict[str, KernelProfile] = {}
    for profile in profiles:
        for symbol in profile.symbols:
            if symbol in lookup:
                raise ValueError(
                    f"kernel symbol {symbol!r} is registered by both "
                    f"{lookup[symbol].name!r} and {profile.name!r}"
                )
            lookup[symbol] = profile
    return lookup


_KERNEL_PROFILE_BY_SYMBOL = _build_kernel_profile_lookup(
    SUPPORTED_KERNEL_PROFILES
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
    "warmup",
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


def select_kernel_profile(symbol: str) -> KernelProfile:
    """Select a contract only for an explicitly supported, exact symbol."""

    profile = _KERNEL_PROFILE_BY_SYMBOL.get(symbol)
    if profile is None:
        supported = ", ".join(
            repr(profile.primary_symbol)
            for profile in SUPPORTED_KERNEL_PROFILES
        )
        raise GemmIsaRunnerError(
            f"unsupported BF16 MXFP4 ISA kernel symbol {symbol!r}; "
            f"supported exact basenames are {supported}"
        )
    return profile


@dataclass(frozen=True)
class BuildResult:
    """One temporary code object and the commands that produced it."""

    code_object: Path
    commands: tuple[tuple[str, ...], ...]
    patches: tuple[str, ...] = ()


@dataclass(frozen=True)
class LaunchGeometry:
    """Persistent launch dimensions derived exactly as in asm_f4gemm.cu."""

    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    cluster: tuple[int, int, int]
    tiles: tuple[int, int]
    cluster_grid: tuple[int, int]
    log2_grid: tuple[int, int]
    logical_cluster_grid: tuple[int, int]
    logical_wg_tasks: int
    logical_cluster_tasks: int
    persistent_stride: int

    @property
    def logical_wg_grid(self) -> tuple[int, int, int]:
        return (*self.tiles, 1)

    @property
    def logical_cluster_grid_3d(self) -> tuple[int, int, int]:
        return (*self.logical_cluster_grid, 1)

    @property
    def physical_cluster_grid(self) -> tuple[int, int, int]:
        return (*self.cluster_grid, 1)


@dataclass(frozen=True)
class _Dependencies:
    torch: Any
    f4_test: Any
    check_allclose: Any
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


def _read_isa_source(isa: Path) -> str:
    try:
        return isa.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise GemmIsaRunnerError(
            f"failed to read ISA source: {isa}: {exc}"
        ) from exc


def detect_kernel_symbol(isa: Path) -> str:
    return detect_kernel_symbol_from_text(_read_isa_source(isa))


def resolve_kernel_symbol_from_text(source: str, override: str | None) -> str:
    """Resolve a symbol and require an override to match the actual ISA."""

    detected = detect_kernel_symbol_from_text(source)
    if override is None:
        return detected

    explicit = _validate_symbol_token(override, "--symbol")
    if explicit != detected:
        raise GemmIsaRunnerError(
            f"--symbol {explicit!r} does not match the actual ISA kernel "
            f"symbol {detected!r}"
        )
    return detected


def resolve_kernel_symbol(isa: Path, override: str | None) -> str:
    return resolve_kernel_symbol_from_text(_read_isa_source(isa), override)


def _required_int(
    text: str,
    pattern: str,
    label: str,
) -> int:
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise GemmIsaRunnerError(
            f"ISA contract requires exactly one {label}, found {len(matches)}"
        )
    try:
        return int(matches[0], 0)
    except ValueError as exc:
        raise GemmIsaRunnerError(
            f"ISA contract has invalid integer for {label}: {matches[0]!r}"
        ) from exc


def _descriptor_body(source: str, symbol: str) -> str:
    matches = re.findall(
        rf"(?ms)^[ \t]*\.amdhsa_kernel[ \t]+{re.escape(symbol)}[ \t]*$"
        rf"(?P<body>.*?)"
        rf"^[ \t]*\.end_amdhsa_kernel[ \t]*$",
        source,
    )
    if len(matches) != 1:
        raise GemmIsaRunnerError(
            f"ISA contract requires exactly one .amdhsa_kernel block for "
            f"{symbol!r}, found {len(matches)}"
        )
    return matches[0]


def _metadata_body(source: str) -> str:
    matches = [
        match.group("body")
        for match in _METADATA_BLOCK_RE.finditer(source)
    ]
    if len(matches) != 1:
        raise GemmIsaRunnerError(
            "ISA contract requires exactly one .amdgpu_metadata block, "
            f"found {len(matches)}"
        )
    return matches[0]


def _descriptor_int(body: str, directive: str) -> int:
    return _required_int(
        body,
        rf"(?m)^[ \t]*\.{re.escape(directive)}[ \t]+"
        rf"(0[xX][0-9A-Fa-f]+|[0-9]+)[ \t]*(?:[;#].*)?$",
        f".{directive}",
    )


def _metadata_int(body: str, field: str) -> int:
    return _required_int(
        body,
        rf"(?m)^[ \t]*\.{re.escape(field)}:[ \t]*"
        rf"(0[xX][0-9A-Fa-f]+|[0-9]+)[ \t]*(?:#.*)?$",
        f"metadata .{field}",
    )


def _metadata_arg_layout(body: str) -> tuple[tuple[int, int], ...]:
    match = re.search(
        r"(?ms)^[ \t]*-[ \t]+\.args:[ \t]*$"
        r"(?P<args>.*?)"
        r"(?=^[ ]{4}\.[A-Za-z_])",
        body,
    )
    if match is None:
        raise GemmIsaRunnerError("metadata has no parseable .args list")
    args = match.group("args")
    offsets = [
        int(value, 0)
        for value in re.findall(
            r"(?m)^[ \t]*(?:-[ \t]+)?\.offset:[ \t]*"
            r"(0[xX][0-9A-Fa-f]+|[0-9]+)[ \t]*(?:#.*)?$",
            args,
        )
    ]
    sizes = [
        int(value, 0)
        for value in re.findall(
            r"(?m)^[ \t]*\.size:[ \t]*"
            r"(0[xX][0-9A-Fa-f]+|[0-9]+)[ \t]*(?:#.*)?$",
            args,
        )
    ]
    if not offsets or len(offsets) != len(sizes):
        raise GemmIsaRunnerError(
            "metadata .args offsets/sizes are incomplete: "
            f"offsets={offsets}, sizes={sizes}"
        )
    return tuple(zip(offsets, sizes))


def validate_assembly_contract_from_text(
    source: str,
    symbol: str,
    profile: KernelProfile,
) -> None:
    """Validate ABI and resource metadata before compiling or launching."""

    actual_symbol = detect_kernel_symbol_from_text(source)
    if actual_symbol != symbol:
        raise GemmIsaRunnerError(
            f"selected symbol {symbol!r} does not match actual ISA symbol "
            f"{actual_symbol!r}"
        )
    selected_profile = select_kernel_profile(actual_symbol)
    if selected_profile != profile:
        raise GemmIsaRunnerError(
            f"kernel profile {profile.name!r} does not match symbol "
            f"{actual_symbol!r}"
        )

    descriptor = _descriptor_body(source, symbol)
    metadata = _metadata_body(source)
    descriptor_expected = {
        "amdhsa_group_segment_fixed_size": profile.group_segment_fixed_size,
        "amdhsa_private_segment_fixed_size": 0,
        "amdhsa_kernarg_size": profile.kernarg_size,
        "amdhsa_user_sgpr_count": 22,
        "amdhsa_user_sgpr_kernarg_segment_ptr": 1,
        "amdhsa_user_sgpr_kernarg_preload_length": (
            profile.kernarg_size // 4
        ),
        "amdhsa_user_sgpr_kernarg_preload_offset": 0,
        "amdhsa_wavefront_size32": 1,
        "amdhsa_next_free_vgpr": profile.next_free_vgpr,
        "amdhsa_next_free_sgpr": profile.next_free_sgpr,
    }
    for directive, expected in descriptor_expected.items():
        actual = _descriptor_int(descriptor, directive)
        if actual != expected:
            raise GemmIsaRunnerError(
                f"{symbol}: .{directive} is {actual}, expected {expected} "
                f"for profile {profile.name}"
            )

    metadata_expected = {
        "group_segment_fixed_size": profile.group_segment_fixed_size,
        "kernarg_segment_align": 8,
        "kernarg_segment_size": profile.kernarg_size,
        "max_flat_workgroup_size": (
            profile.block[0] * profile.block[1] * profile.block[2]
        ),
        "private_segment_fixed_size": 0,
        "sgpr_count": profile.metadata_sgpr_count,
        "vgpr_count": profile.metadata_vgpr_count,
        "wavefront_size": 32,
    }
    for field, expected in metadata_expected.items():
        actual = _metadata_int(metadata, field)
        if actual != expected:
            raise GemmIsaRunnerError(
                f"{symbol}: metadata .{field} is {actual}, expected "
                f"{expected} for profile {profile.name}"
            )

    expected_args = tuple(
        (offset, struct.calcsize(f"<{kind}"))
        for _name, offset, kind in profile.kernarg_layout
    )
    actual_args = _metadata_arg_layout(metadata)
    if actual_args != expected_args:
        raise GemmIsaRunnerError(
            f"{symbol}: metadata kernarg layout {actual_args} does not match "
            f"profile ABI {expected_args}"
        )
    if not re.search(
        rf"(?m)^amdhsa\.target:[ \t]+"
        rf"amdgcn-amd-amdhsa--{re.escape(ARCH)}[ \t]*$",
        metadata,
    ):
        raise GemmIsaRunnerError(
            f"{symbol}: metadata target is not amdgcn-amd-amdhsa--{ARCH}"
        )


def _resolve_clang(override: str | None) -> Path:
    """Resolve explicit ``--clang`` or require the fixed default clang.

    An explicit override is interpreted as a path first and then as a command
    name for ``shutil.which``.  Without an override, no ROCm or PATH fallback
    is attempted.
    """

    if override is not None:
        expanded = os.path.expandvars(os.path.expanduser(override))
        candidate = Path(expanded)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
        found = shutil.which(expanded)
        if found:
            return Path(found).resolve()
        raise GemmIsaRunnerError(
            f"--clang is neither an executable file nor a command found on "
            f"PATH: {override}"
        )

    if DEFAULT_CLANG.is_file() and os.access(DEFAULT_CLANG, os.X_OK):
        return DEFAULT_CLANG.resolve()
    raise GemmIsaRunnerError(
        f"fixed default AMDGPU clang is missing or not executable: "
        f"{DEFAULT_CLANG}; pass --clang explicitly"
    )


def _clang_uses_default_runtime_libraries(clang: Path) -> bool:
    """Return whether clang is the fixed build with extra runtime dependencies."""

    return clang.resolve() == DEFAULT_CLANG.resolve()


def _prepend_default_clang_runtime_libraries() -> None:
    """Prepend fixed-clang runtime dependencies without dropping user entries."""

    requested = [str(path) for path in DEFAULT_CLANG_RUNTIME_LIBRARIES]
    existing = [
        entry
        for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if entry
    ]
    combined: list[str] = []
    for entry in (*requested, *existing):
        if entry not in combined:
            combined.append(entry)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(combined)


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


# gfx1250 divides the 384 KiB per-WGP SRAM between the LDS partition and the
# vector cache, and COMPUTE_PGM_RSRC3.TCP_SPLIT selects the division.  The
# clang 22 assembler in the ROCm container exposes no directive for that field,
# so an ISA source asks for a value with the absolute symbol below and the
# linked code object is rewritten here, before ROCr loads it.  ROCr does not
# validate RSRC3; the field is forwarded to CP/SPI as written.
TCP_SPLIT_REQUEST_SYMBOL = "__aiter_tcp_split"

KERNEL_DESCRIPTOR_SIZE = 64
_KD_GROUP_SEGMENT_FIXED_SIZE_OFFSET = 0
_KD_COMPUTE_PGM_RSRC3_OFFSET = 44
# Field positions match LLVM's AMDHSAKernelDescriptor.h for gfx12.
_RSRC3_TCP_SPLIT_SHIFT = 18
_RSRC3_TCP_SPLIT_MASK = 0x7
_RSRC3_INST_PREF_SIZE_SHIFT = 4
_RSRC3_INST_PREF_SIZE_MASK = 0xFF
_ELF64_SECTION_HEADER_SIZE = 64
_ELF64_SYMBOL_SIZE = 24
_SHT_SYMTAB = 2
_SHT_STRTAB = 3
_SHT_NOBITS = 8
_SHT_DYNSYM = 11


def parse_requested_tcp_split(source: str) -> int | None:
    """Read the optional ``.set __aiter_tcp_split, N`` resource request."""

    matches = re.findall(
        rf"(?m)^[ \t]*\.set[ \t]+{re.escape(TCP_SPLIT_REQUEST_SYMBOL)}[ \t]*,"
        rf"[ \t]*(0[xX][0-9A-Fa-f]+|[0-9]+)[ \t]*(?:(?:[;#]|//).*)?$",
        source,
    )
    if not matches:
        return None
    values = {int(value, 0) for value in matches}
    if len(values) != 1:
        raise GemmIsaRunnerError(
            f"ISA sets {TCP_SPLIT_REQUEST_SYMBOL} to conflicting values "
            f"{sorted(values)}"
        )
    value = values.pop()
    if not 0 <= value <= _RSRC3_TCP_SPLIT_MASK:
        raise GemmIsaRunnerError(
            f".set {TCP_SPLIT_REQUEST_SYMBOL} must be in 0..."
            f"{_RSRC3_TCP_SPLIT_MASK} because COMPUTE_PGM_RSRC3.TCP_SPLIT is "
            f"bits [20:18], got {value}"
        )
    return value


@dataclass(frozen=True)
class _ElfSection:
    index: int
    name_offset: int
    type: int
    addr: int
    offset: int
    size: int
    link: int
    entsize: int


def _elf64_sections(data: bytes, label: str) -> tuple[_ElfSection, ...]:
    if len(data) < _ELF64_SECTION_HEADER_SIZE or data[:4] != b"\x7fELF":
        raise GemmIsaRunnerError(f"{label} is not an ELF file")
    if data[4] != 2:
        raise GemmIsaRunnerError(
            f"{label} is not ELFCLASS64 (EI_CLASS={data[4]})"
        )
    if data[5] != 1:
        raise GemmIsaRunnerError(
            f"{label} is not ELFDATA2LSB (EI_DATA={data[5]})"
        )
    (e_shoff,) = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x3A)
    if e_shentsize != _ELF64_SECTION_HEADER_SIZE:
        raise GemmIsaRunnerError(
            f"{label} has unexpected e_shentsize {e_shentsize}, "
            f"expected {_ELF64_SECTION_HEADER_SIZE}"
        )
    if e_shoff == 0 or e_shnum == 0:
        raise GemmIsaRunnerError(f"{label} has no section header table")
    if e_shoff + e_shnum * e_shentsize > len(data):
        raise GemmIsaRunnerError(f"{label} section header table is truncated")
    sections = []
    for index in range(e_shnum):
        (
            name_offset,
            kind,
            _flags,
            addr,
            offset,
            size,
            link,
            _info,
            _align,
            entsize,
        ) = struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + index * e_shentsize)
        if kind != _SHT_NOBITS and offset + size > len(data):
            raise GemmIsaRunnerError(
                f"{label} section {index} extends past the end of the file"
            )
        sections.append(
            _ElfSection(index, name_offset, kind, addr, offset, size, link, entsize)
        )
    return tuple(sections)


def _elf64_string(
    data: bytes,
    table: _ElfSection,
    offset: int,
    label: str,
) -> str:
    if table.type != _SHT_STRTAB:
        raise GemmIsaRunnerError(
            f"{label} section {table.index} is not a string table"
        )
    if offset >= table.size:
        raise GemmIsaRunnerError(
            f"{label} string offset {offset} is outside section {table.index}"
        )
    start = table.offset + offset
    end = data.find(b"\x00", start, table.offset + table.size)
    if end < 0:
        raise GemmIsaRunnerError(
            f"{label} string at offset {offset} is unterminated"
        )
    return data[start:end].decode("utf-8", errors="replace")


def _find_kernel_descriptor_offset(
    data: bytes,
    sections: Sequence[_ElfSection],
    kd_name: str,
    label: str,
) -> tuple[int, str]:
    """Map ``<kernel>.kd`` to the single file offset holding its 64 bytes."""

    origins: dict[int, str] = {}
    symbol_tables = 0
    for section in sections:
        if section.type not in (_SHT_SYMTAB, _SHT_DYNSYM):
            continue
        if section.entsize != _ELF64_SYMBOL_SIZE:
            raise GemmIsaRunnerError(
                f"{label} symbol table {section.index} has sh_entsize "
                f"{section.entsize}, expected {_ELF64_SYMBOL_SIZE}"
            )
        if section.link >= len(sections):
            raise GemmIsaRunnerError(
                f"{label} symbol table {section.index} links to missing "
                f"string table {section.link}"
            )
        strtab = sections[section.link]
        symbol_tables += 1
        for index in range(section.size // _ELF64_SYMBOL_SIZE):
            base = section.offset + index * _ELF64_SYMBOL_SIZE
            st_name, _info, _other, st_shndx, st_value, st_size = (
                struct.unpack_from("<IBBHQQ", data, base)
            )
            if st_name == 0:
                continue
            if _elf64_string(data, strtab, st_name, label) != kd_name:
                continue
            if st_size < KERNEL_DESCRIPTOR_SIZE:
                raise GemmIsaRunnerError(
                    f"{label}: {kd_name} has st_size {st_size}, expected at "
                    f"least {KERNEL_DESCRIPTOR_SIZE}"
                )
            if st_shndx == 0 or st_shndx >= len(sections):
                raise GemmIsaRunnerError(
                    f"{label}: {kd_name} has unusable st_shndx {st_shndx}"
                )
            host = sections[st_shndx]
            if host.type == _SHT_NOBITS:
                raise GemmIsaRunnerError(
                    f"{label}: {kd_name} lives in SHT_NOBITS section "
                    f"{host.index}, which has no file bytes to patch"
                )
            if st_value < host.addr:
                raise GemmIsaRunnerError(
                    f"{label}: {kd_name} st_value 0x{st_value:x} is below "
                    f"section {host.index} sh_addr 0x{host.addr:x}"
                )
            delta = st_value - host.addr
            if delta + KERNEL_DESCRIPTOR_SIZE > host.size:
                raise GemmIsaRunnerError(
                    f"{label}: {kd_name} does not fit inside section "
                    f"{host.index}"
                )
            origins.setdefault(
                host.offset + delta,
                f"section {host.index} via symbol table {section.index}",
            )
    if symbol_tables == 0:
        raise GemmIsaRunnerError(
            f"{label} has neither .symtab nor .dynsym, so {kd_name} cannot be "
            f"located"
        )
    if not origins:
        raise GemmIsaRunnerError(f"{label} has no symbol named {kd_name}")
    if len(origins) != 1:
        found = ", ".join(
            f"0x{offset:x} ({origin})" for offset, origin in sorted(origins.items())
        )
        raise GemmIsaRunnerError(
            f"{label} resolves {kd_name} to {len(origins)} distinct file "
            f"offsets: {found}"
        )
    return next(iter(origins.items()))


def patch_code_object_tcp_split(
    code_object: Path,
    kernel_symbol: str,
    tcp_split: int,
    expected_group_segment_fixed_size: int,
) -> str:
    """Rewrite COMPUTE_PGM_RSRC3.TCP_SPLIT in a linked code object."""

    if not 0 <= tcp_split <= _RSRC3_TCP_SPLIT_MASK:
        raise GemmIsaRunnerError(
            f"TCP_SPLIT must be in 0...{_RSRC3_TCP_SPLIT_MASK}, got {tcp_split}"
        )
    label = str(code_object)
    try:
        data = bytearray(code_object.read_bytes())
    except OSError as exc:
        raise GemmIsaRunnerError(
            f"failed to read code object {label}: {exc}"
        ) from exc
    sections = _elf64_sections(bytes(data), label)
    kd_name = f"{_validate_symbol_token(kernel_symbol, 'kernel symbol')}.kd"
    offset, origin = _find_kernel_descriptor_offset(
        bytes(data),
        sections,
        kd_name,
        label,
    )

    (group_segment,) = struct.unpack_from(
        "<I",
        data,
        offset + _KD_GROUP_SEGMENT_FIXED_SIZE_OFFSET,
    )
    if group_segment != expected_group_segment_fixed_size:
        raise GemmIsaRunnerError(
            f"{label}: {kd_name} declares group_segment_fixed_size "
            f"{group_segment}, but the ISA descriptor declares "
            f"{expected_group_segment_fixed_size}; refusing to patch a "
            f"descriptor that is not the one that was assembled"
        )
    (old_rsrc3,) = struct.unpack_from(
        "<I",
        data,
        offset + _KD_COMPUTE_PGM_RSRC3_OFFSET,
    )
    old_split = (old_rsrc3 >> _RSRC3_TCP_SPLIT_SHIFT) & _RSRC3_TCP_SPLIT_MASK
    inst_pref_size = (
        old_rsrc3 >> _RSRC3_INST_PREF_SIZE_SHIFT
    ) & _RSRC3_INST_PREF_SIZE_MASK
    new_rsrc3 = (
        old_rsrc3 & ~(_RSRC3_TCP_SPLIT_MASK << _RSRC3_TCP_SPLIT_SHIFT)
    ) | (tcp_split << _RSRC3_TCP_SPLIT_SHIFT)
    struct.pack_into(
        "<I",
        data,
        offset + _KD_COMPUTE_PGM_RSRC3_OFFSET,
        new_rsrc3,
    )
    try:
        code_object.write_bytes(bytes(data))
        verified = code_object.read_bytes()
    except OSError as exc:
        raise GemmIsaRunnerError(
            f"failed to rewrite code object {label}: {exc}"
        ) from exc
    if verified != bytes(data):
        raise GemmIsaRunnerError(
            f"{label} does not read back byte-identical after patching"
        )
    (read_back,) = struct.unpack_from(
        "<I",
        verified,
        offset + _KD_COMPUTE_PGM_RSRC3_OFFSET,
    )
    if read_back != new_rsrc3:
        raise GemmIsaRunnerError(
            f"{label}: RSRC3 reads back 0x{read_back:08x}, expected "
            f"0x{new_rsrc3:08x}"
        )
    return (
        f"patched TCP_SPLIT {old_split} -> {tcp_split}, "
        f"RSRC3 0x{old_rsrc3:08x} -> 0x{new_rsrc3:08x} "
        f"(INST_PREF_SIZE {inst_pref_size} and every other RSRC3 bit "
        f"unchanged; group_segment_fixed_size {group_segment}; {kd_name} at "
        f"file offset 0x{offset:x}, {origin})"
    )


def compile_isa(
    isa: Path,
    clang: Path,
    work_dir: Path,
    symbol: str | None = None,
) -> BuildResult:
    """Assemble and link a complete AMDGPU source without changing it."""

    artifact_stem = (
        _validate_symbol_token(symbol, "compile symbol")
        if symbol is not None
        else isa.stem
    )
    obj = work_dir / f"{artifact_stem}.o"
    code_object = work_dir / f"{artifact_stem}.co"
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
    source = _read_isa_source(isa)
    requested_tcp_split = parse_requested_tcp_split(source)
    patches: tuple[str, ...] = ()
    if requested_tcp_split is not None:
        kernel_symbol = resolve_kernel_symbol_from_text(source, symbol)
        patches = (
            patch_code_object_tcp_split(
                code_object,
                kernel_symbol,
                requested_tcp_split,
                _descriptor_int(
                    _descriptor_body(source, kernel_symbol),
                    "amdhsa_group_segment_fixed_size",
                ),
            ),
        )
    return BuildResult(
        code_object=code_object,
        commands=(assemble, link),
        patches=patches,
    )


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def make_launch_geometry(
    m: int,
    n: int,
    k: int,
    profile: KernelProfile = KERNEL_PROFILE_256,
) -> LaunchGeometry:
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
    if k % profile.k_multiple:
        raise GemmIsaRunnerError(
            f"profile {profile.name} requires K to be divisible by "
            f"{profile.k_multiple} with no K tail; got K={k}"
        )

    tile_m, tile_n = profile.wg_tile
    if m % tile_m or n % tile_n:
        raise GemmIsaRunnerError(
            f"profile {profile.name} requires exact {tile_m}x{tile_n} WG "
            f"tiles with no M/N boundary tiles; got M={m}, N={n}"
        )

    tiles_x = n // tile_n
    tiles_y = m // tile_m
    cluster_x, cluster_y, cluster_z = profile.cluster
    if cluster_z != 1:
        raise GemmIsaRunnerError(
            f"profile {profile.name} requires unsupported cluster_z={cluster_z}; "
            "this dense GEMM launcher is two-dimensional"
        )
    if tiles_x < cluster_x or tiles_x % cluster_x:
        raise GemmIsaRunnerError(
            f"profile {profile.name} cluster_x={cluster_x} requires "
            f"N/{tile_n}={tiles_x} tiles to be a "
            f"positive multiple of {cluster_x}; N must be a multiple of "
            f"{tile_n * cluster_x}"
        )
    if tiles_y < cluster_y or tiles_y % cluster_y:
        raise GemmIsaRunnerError(
            f"profile {profile.name} cluster_y={cluster_y} requires "
            f"M/{tile_m}={tiles_y} tiles to be a "
            f"positive multiple of {cluster_y}; M must be a multiple of "
            f"{tile_m * cluster_y}"
        )

    cluster_size = cluster_x * cluster_y * cluster_z
    if profile.persistent_tg % cluster_size:
        raise GemmIsaRunnerError(
            f"persistent_tg={profile.persistent_tg} is not divisible by "
            f"cluster size {cluster_size}"
        )
    persistent_clusters = profile.persistent_tg // cluster_size
    if (
        profile.persistent_grid_y <= 0
        or persistent_clusters % profile.persistent_grid_y
    ):
        raise GemmIsaRunnerError(
            f"persistent grid_y={profile.persistent_grid_y} must divide "
            f"{persistent_clusters} clusters"
        )
    grid_y = profile.persistent_grid_y
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

    log2_grid = (grid_x.bit_length() - 1, grid_y.bit_length() - 1)
    persistent_stride = 1 << sum(log2_grid)
    if persistent_stride != persistent_clusters:
        raise GemmIsaRunnerError(
            f"profile {profile.name} persistent stride {persistent_stride} "
            f"does not equal physical cluster count {persistent_clusters}"
        )
    logical_cluster_grid = (
        tiles_x // cluster_x,
        tiles_y // cluster_y,
    )
    return LaunchGeometry(
        grid=(grid_x * cluster_x, grid_y * cluster_y, 1),
        block=profile.block,
        cluster=profile.cluster,
        tiles=(tiles_x, tiles_y),
        cluster_grid=(grid_x, grid_y),
        log2_grid=log2_grid,
        logical_cluster_grid=logical_cluster_grid,
        logical_wg_tasks=tiles_x * tiles_y,
        logical_cluster_tasks=(
            logical_cluster_grid[0] * logical_cluster_grid[1]
        ),
        persistent_stride=persistent_stride,
    )


def _checked_unsigned(name: str, value: int, bits: int) -> int:
    maximum = (1 << bits) - 1
    if not 0 <= int(value) <= maximum:
        raise GemmIsaRunnerError(
            f"{name}={value} does not fit an unsigned {bits}-bit ABI field"
        )
    return int(value)


def _pack_kernarg_fields(
    fields: Mapping[str, int],
    *,
    layout: tuple[tuple[str, int, str], ...] = KERNARG_LAYOUT,
    size: int = KERNARG_SIZE,
) -> bytes:
    """Pack named fields at explicit C++ offsets; no implicit alignment."""

    expected = {name for name, _offset, _kind in layout}
    missing = expected.difference(fields)
    extra = set(fields).difference(expected)
    if missing or extra:
        raise GemmIsaRunnerError(
            f"invalid kernarg fields: missing={sorted(missing)}, extra={sorted(extra)}"
        )

    payload = bytearray(size)
    for name, offset, kind in layout:
        field_size = struct.calcsize(f"<{kind}")
        if offset < 0 or offset + field_size > size:
            raise GemmIsaRunnerError(
                f"kernarg field {name} at {offset} with size {field_size} "
                f"does not fit the {size}-byte ABI"
            )
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
    profile: KernelProfile = KERNEL_PROFILE_256,
) -> bytes:
    """Build the profile's exact preload-SGPR MXFP4 argument payload."""

    if geometry.block != profile.block or geometry.cluster != profile.cluster:
        raise GemmIsaRunnerError(
            f"launch geometry block/cluster "
            f"{geometry.block}/{geometry.cluster} does not match profile "
            f"{profile.block}/{profile.cluster}"
        )

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
        },
        layout=profile.kernarg_layout,
        size=profile.kernarg_size,
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


def _make_hip_launch_config(
    geometry: LaunchGeometry,
    stream: ctypes.c_void_p,
) -> tuple[Any, _HipLaunchConfig]:
    """Build profile-derived HIP dimensions and the cluster attribute."""

    attributes = (_HipLaunchAttribute * 1)()
    attributes[0].id = _HIP_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    attributes[0].value.clusterDim = _Dim3(*geometry.cluster)
    config = _HipLaunchConfig(
        gridDimX=geometry.grid[0],
        gridDimY=geometry.grid[1],
        gridDimZ=geometry.grid[2],
        blockDimX=geometry.block[0],
        blockDimY=geometry.block[1],
        blockDimZ=geometry.block[2],
        # The assembly descriptor owns its fixed LDS segment; no dynamic LDS.
        sharedMemBytes=0,
        hStream=stream,
        attrs=ctypes.cast(
            attributes,
            ctypes.POINTER(_HipLaunchAttribute),
        ),
        numAttrs=1,
    )
    return attributes, config


def run_static_contract_checks() -> None:
    """CPU-only fixture for ABI, geometry, and the fixed clang policy."""

    expected_clang = Path(
        "/data/yanguahe/code/wk_sp1/llvm-project/mlir_install/bin/clang"
    )
    if DEFAULT_CLANG != expected_clang:
        raise AssertionError(
            f"default clang changed to {DEFAULT_CLANG}, expected {expected_clang}"
        )
    clang_action = next(
        action for action in _build_parser()._actions if action.dest == "clang"
    )
    clang_help = clang_action.help or ""
    if (
        str(DEFAULT_CLANG) not in clang_help
        or "no ROCm/PATH fallback" not in clang_help
    ):
        raise AssertionError(
            f"--clang help does not describe the fixed-only policy: {clang_help}"
        )

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

    def assembly_contract_fixture(profile: KernelProfile) -> str:
        symbol = profile.primary_symbol
        args = "\n".join(
            (
                f"      - .offset:         {offset}\n"
                f"        .size:           {struct.calcsize(f'<{kind}')}\n"
                "        .value_kind:     by_value"
            )
            for _name, offset, kind in profile.kernarg_layout
        )
        return f"""
.amdgcn_target "amdgcn-amd-amdhsa--{ARCH}"
.text
.globl {symbol}
.type {symbol},@function
{symbol}:
    s_endpgm
.amdhsa_kernel {symbol}
    .amdhsa_group_segment_fixed_size {profile.group_segment_fixed_size}
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size {profile.kernarg_size}
    .amdhsa_user_sgpr_count 22
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_user_sgpr_kernarg_preload_length {profile.kernarg_size // 4}
    .amdhsa_user_sgpr_kernarg_preload_offset 0
    .amdhsa_wavefront_size32 1
    .amdhsa_next_free_vgpr {profile.next_free_vgpr}
    .amdhsa_next_free_sgpr {profile.next_free_sgpr}
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
{args}
    .group_segment_fixed_size: {profile.group_segment_fixed_size}
    .kernarg_segment_align: 8
    .kernarg_segment_size: {profile.kernarg_size}
    .max_flat_workgroup_size: {profile.block[0] * profile.block[1] * profile.block[2]}
    .name: {symbol}
    .private_segment_fixed_size: 0
    .sgpr_count: {profile.metadata_sgpr_count}
    .symbol: {symbol}.kd
    .vgpr_count: {profile.metadata_vgpr_count}
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--{ARCH}
amdhsa.version:
  - 1
  - 2
...
.end_amdgpu_metadata
"""

    current_symbol = EXPECTED_KERNEL_BASENAME
    for profile in SUPPORTED_KERNEL_PROFILES:
        for profile_symbol in profile.symbols:
            detected = detect_kernel_symbol_from_text(
                symbol_fixture(profile_symbol)
            )
            if detected != profile_symbol:
                raise AssertionError(
                    f"symbol fixture detected {detected!r}, "
                    f"expected {profile_symbol!r}"
                )
            if select_kernel_profile(profile_symbol) != profile:
                raise AssertionError(
                    f"wrong profile selected for {profile_symbol!r}"
                )
            if (
                _validate_mode("mxfp4", profile.apre, "bf16", profile_symbol)
                != profile
            ):
                raise AssertionError(
                    f"mode validation selected the wrong profile for "
                    f"{profile_symbol!r}"
                )
        contract_source = assembly_contract_fixture(profile)
        validate_assembly_contract_from_text(
            contract_source,
            profile.primary_symbol,
            profile,
        )
        bad_descriptor = contract_source.replace(
            f".amdhsa_next_free_vgpr {profile.next_free_vgpr}",
            f".amdhsa_next_free_vgpr {profile.next_free_vgpr + 1}",
            1,
        )
        try:
            validate_assembly_contract_from_text(
                bad_descriptor,
                profile.primary_symbol,
                profile,
            )
        except GemmIsaRunnerError as exc:
            if ".amdhsa_next_free_vgpr" not in str(exc):
                raise AssertionError(
                    f"{profile.name} descriptor mismatch raised unexpected "
                    f"error: {exc}"
                ) from exc
        else:
            raise AssertionError(
                f"{profile.name} accepted mismatched descriptor resources"
            )
        bad_metadata = contract_source.replace(
            f".vgpr_count: {profile.metadata_vgpr_count}",
            f".vgpr_count: {profile.metadata_vgpr_count + 1}",
            1,
        )
        try:
            validate_assembly_contract_from_text(
                bad_metadata,
                profile.primary_symbol,
                profile,
            )
        except GemmIsaRunnerError as exc:
            if "metadata .vgpr_count" not in str(exc):
                raise AssertionError(
                    f"{profile.name} metadata mismatch raised unexpected "
                    f"error: {exc}"
                ) from exc
        else:
            raise AssertionError(
                f"{profile.name} accepted mismatched metadata resources"
            )

    try:
        _build_kernel_profile_lookup(
            (*SUPPORTED_KERNEL_PROFILES, KERNEL_PROFILE_64X256)
        )
    except ValueError as exc:
        if "is registered by both" not in str(exc):
            raise AssertionError(
                f"profile-symbol collision raised unexpected error: {exc}"
            ) from exc
    else:
        raise AssertionError("duplicate profile-symbol registration did not fail")

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

    explicit = resolve_kernel_symbol_from_text(
        symbol_fixture(current_symbol),
        current_symbol,
    )
    if explicit != current_symbol:
        raise AssertionError(f"explicit --symbol override changed to {explicit!r}")
    try:
        resolve_kernel_symbol_from_text(
            symbol_fixture(current_symbol),
            KERNEL_SYMBOL_128,
        )
    except GemmIsaRunnerError as exc:
        if "does not match the actual ISA kernel symbol" not in str(exc):
            raise AssertionError(
                f"mismatched --symbol raised unexpected error: {exc}"
            ) from exc
    else:
        raise AssertionError("mismatched --symbol override did not fail")
    for profile in SUPPORTED_KERNEL_PROFILES:
        for profile_symbol in profile.symbols:
            try:
                select_kernel_profile(f"{profile_symbol}_unrelated")
            except GemmIsaRunnerError as exc:
                if "supported exact basenames" not in str(exc):
                    raise AssertionError(
                        f"unrelated symbol raised unexpected error: {exc}"
                    ) from exc
            else:
                raise AssertionError(
                    f"unrelated suffix for {profile_symbol!r} did not fail"
                )

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

    if parse_requested_tcp_split(".text\n    s_endpgm\n") is not None:
        raise AssertionError("a source without the request declared one")
    requested = parse_requested_tcp_split(
        f"\t.set {TCP_SPLIT_REQUEST_SYMBOL}, 5 ; 256 KiB LDS partition\n"
    )
    if requested != 5:
        raise AssertionError(f"TCP_SPLIT request parsed as {requested!r}")
    for label, text, message_fragment in (
        (
            "an out-of-range request",
            f".set {TCP_SPLIT_REQUEST_SYMBOL}, 8\n",
            "bits [20:18]",
        ),
        (
            "two disagreeing requests",
            f".set {TCP_SPLIT_REQUEST_SYMBOL}, 5\n"
            f".set {TCP_SPLIT_REQUEST_SYMBOL}, 4\n",
            "conflicting values",
        ),
    ):
        try:
            parse_requested_tcp_split(text)
        except GemmIsaRunnerError as exc:
            if message_fragment not in str(exc):
                raise AssertionError(
                    f"{label} raised unexpected error: {exc}"
                ) from exc
        else:
            raise AssertionError(f"{label} was accepted")

    def kernel_descriptor_elf(kd_name: str, group_segment: int, rsrc3: int) -> bytes:
        """A minimal ELF64 shared object holding one kernel descriptor."""

        descriptor = bytearray(KERNEL_DESCRIPTOR_SIZE)
        struct.pack_into(
            "<I",
            descriptor,
            _KD_GROUP_SEGMENT_FIXED_SIZE_OFFSET,
            group_segment,
        )
        struct.pack_into("<I", descriptor, _KD_COMPUTE_PGM_RSRC3_OFFSET, rsrc3)
        strtab = b"\x00" + kd_name.encode("utf-8") + b"\x00"
        shstrtab = b"\x00.rodata\x00.symtab\x00.strtab\x00.shstrtab\x00"
        rodata_addr = 0x1000
        rodata_offset = _ELF64_SECTION_HEADER_SIZE
        symtab_offset = rodata_offset + len(descriptor)
        symbols = bytearray(_ELF64_SYMBOL_SIZE)
        symbols += struct.pack(
            "<IBBHQQ",
            1,
            (1 << 4) | 10,
            0,
            1,
            rodata_addr,
            KERNEL_DESCRIPTOR_SIZE,
        )
        strtab_offset = symtab_offset + len(symbols)
        shstrtab_offset = strtab_offset + len(strtab)
        shoff = shstrtab_offset + len(shstrtab)
        header = bytearray(_ELF64_SECTION_HEADER_SIZE)
        header[0:4] = b"\x7fELF"
        header[4] = 2  # ELFCLASS64
        header[5] = 1  # ELFDATA2LSB
        header[6] = 1  # EV_CURRENT
        struct.pack_into("<HH", header, 0x10, 3, 0xE0)  # ET_DYN, EM_AMDGPU
        struct.pack_into("<Q", header, 0x28, shoff)
        struct.pack_into(
            "<HHH",
            header,
            0x3A,
            _ELF64_SECTION_HEADER_SIZE,
            5,
            4,
        )

        def section(
            name_offset: int,
            kind: int,
            addr: int,
            offset: int,
            size: int,
            link: int,
            entsize: int,
        ) -> bytes:
            return struct.pack(
                "<IIQQQQIIQQ",
                name_offset,
                kind,
                0,
                addr,
                offset,
                size,
                link,
                0,
                1,
                entsize,
            )

        headers = (
            section(0, 0, 0, 0, 0, 0, 0)
            + section(
                shstrtab.index(b".rodata"),
                1,
                rodata_addr,
                rodata_offset,
                len(descriptor),
                0,
                0,
            )
            + section(
                shstrtab.index(b".symtab"),
                _SHT_SYMTAB,
                0,
                symtab_offset,
                len(symbols),
                3,
                _ELF64_SYMBOL_SIZE,
            )
            + section(
                shstrtab.index(b".strtab"),
                _SHT_STRTAB,
                0,
                strtab_offset,
                len(strtab),
                0,
                0,
            )
            + section(
                shstrtab.index(b".shstrtab"),
                _SHT_STRTAB,
                0,
                shstrtab_offset,
                len(shstrtab),
                0,
                0,
            )
        )
        return bytes(header + descriptor + symbols + strtab + shstrtab + headers)

    with tempfile.TemporaryDirectory(prefix="gemm_isa_runner_selftest_") as temp:
        fixture = Path(temp) / "fixture.co"
        original = kernel_descriptor_elf(f"{current_symbol}.kd", 131072, 0x00000410)
        fixture.write_bytes(original)
        message = patch_code_object_tcp_split(fixture, current_symbol, 5, 131072)
        if "TCP_SPLIT 0 -> 5" not in message:
            raise AssertionError(f"unexpected TCP_SPLIT report: {message}")
        if "0x00000410 -> 0x00140410" not in message:
            raise AssertionError(f"unexpected RSRC3 report: {message}")
        patched = fixture.read_bytes()
        rsrc3_at = _ELF64_SECTION_HEADER_SIZE + _KD_COMPUTE_PGM_RSRC3_OFFSET
        if (
            patched[:rsrc3_at] != original[:rsrc3_at]
            or patched[rsrc3_at + 4:] != original[rsrc3_at + 4:]
        ):
            raise AssertionError("the patch changed bytes outside RSRC3")
        (patched_rsrc3,) = struct.unpack_from("<I", patched, rsrc3_at)
        if patched_rsrc3 != 0x00140410:
            raise AssertionError(f"patched RSRC3 is 0x{patched_rsrc3:08x}")
        repeat = patch_code_object_tcp_split(fixture, current_symbol, 5, 131072)
        if "TCP_SPLIT 5 -> 5" not in repeat or fixture.read_bytes() != patched:
            raise AssertionError(f"re-patching was not idempotent: {repeat}")

        not_elf = Path(temp) / "not_elf.co"
        not_elf.write_bytes(b"this is not a code object")
        for label, arguments, message_fragment in (
            (
                "a descriptor whose LDS size disagrees with the ISA",
                (fixture, current_symbol, 5, 206848),
                "refusing to patch",
            ),
            (
                "a code object without the requested kernel",
                (fixture, KERNEL_SYMBOL_128, 5, 131072),
                "has no symbol named",
            ),
            (
                "an out-of-range TCP_SPLIT",
                (fixture, current_symbol, 8, 131072),
                "TCP_SPLIT must be in",
            ),
            (
                "a file that is not an ELF object",
                (not_elf, current_symbol, 5, 131072),
                "is not an ELF file",
            ),
        ):
            try:
                patch_code_object_tcp_split(*arguments)
            except GemmIsaRunnerError as exc:
                if message_fragment not in str(exc):
                    raise AssertionError(
                        f"{label} raised unexpected error: {exc}"
                    ) from exc
            else:
                raise AssertionError(f"{label} was patched anyway")
        if fixture.read_bytes() != patched:
            raise AssertionError("a rejected patch still modified the fixture")

    timing_calls: list[tuple[Any, ...]] = []

    class EventStub:
        def __init__(self, label: str) -> None:
            self.label = label

        def record(self, stream: Any) -> None:
            timing_calls.append(("event.record", self.label, stream))

        def synchronize(self) -> None:
            timing_calls.append(("event.synchronize", self.label))

        def elapsed_time(self, other: Any) -> float:
            timing_calls.append(
                ("event.elapsed_time", self.label, other.label)
            )
            return 1.5  # milliseconds for the complete two-launch batch

    class CudaStub:
        def __init__(self) -> None:
            self.event_count = 0

        def synchronize(self) -> None:
            timing_calls.append(("cuda.synchronize",))

        def Event(self, *, enable_timing: bool) -> EventStub:
            if not enable_timing:
                raise AssertionError("timing event was not enabled")
            label = "start" if self.event_count == 0 else "end"
            self.event_count += 1
            timing_calls.append(("event.create", label))
            return EventStub(label)

    class TorchStub:
        def __init__(self) -> None:
            self.cuda = CudaStub()

    launch_count = 0

    def launch_stub() -> str:
        nonlocal launch_count
        launch_count += 1
        timing_calls.append(("launch", launch_count))
        return f"fixture-output-{launch_count}"

    stream_marker = object()
    output_marker, event_avg = _run_batched_cuda_event_timing(
        TorchStub(),
        launch_stub,
        stream_marker,
        num_warmup=3,
        num_iters=2,
    )
    expected_timing_calls = [
        ("launch", 1),
        ("launch", 2),
        ("launch", 3),
        ("cuda.synchronize",),
        ("event.create", "start"),
        ("event.create", "end"),
        ("event.record", "start", stream_marker),
        ("launch", 4),
        ("launch", 5),
        ("event.record", "end", stream_marker),
        ("event.synchronize", "end"),
        ("event.elapsed_time", "start", "end"),
    ]
    if timing_calls != expected_timing_calls:
        raise AssertionError(
            f"batched CUDA-event call order is {timing_calls}, "
            f"expected {expected_timing_calls}"
        )
    if output_marker != "fixture-output-5" or event_avg != 750.0:
        raise AssertionError(
            f"batched CUDA-event result is {(output_marker, event_avg)}, "
            "expected ('fixture-output-5', 750.0)"
        )

    defaults = _build_parser().parse_args(["--isa", "fixture.s"])
    if defaults.warmup != 101 or defaults.iters != 100:
        raise AssertionError(
            f"timing CLI defaults are warmup={defaults.warmup}, "
            f"iters={defaults.iters}; expected 101/100"
        )
    zero_warmup = _build_parser().parse_args(
        ["--isa", "fixture.s", "--warmup", "0"]
    )
    if zero_warmup.warmup != 0:
        raise AssertionError("--warmup 0 was not preserved")
    timing_row = make_cuda_event_timing_row(
        EXPECTED_KERNEL_BASENAME,
        2,
        3,
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
        or timing_row["cnt"] != 2
        or timing_row["warmup"] != 3
        or timing_row["device_time_sum"] != 1500.0
        or timing_row["device_time_avg"] != 750.0
        or timing_row["device_type"] != "CUDA/HIP"
        or timing_row["device_index"] != 0
        or timing_row["source"] != "cuda.Event batched"
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

    expected_geometries = {
        KERNEL_PROFILE_256: LaunchGeometry(
            grid=(16, 16, 1),
            block=(128, 1, 1),
            cluster=(4, 4, 1),
            tiles=(8, 72),
            cluster_grid=(4, 4),
            log2_grid=(2, 2),
            logical_cluster_grid=(2, 18),
            logical_wg_tasks=576,
            logical_cluster_tasks=36,
            persistent_stride=16,
        ),
        KERNEL_PROFILE_128: LaunchGeometry(
            grid=(16, 16, 1),
            block=(128, 1, 1),
            cluster=(4, 4, 1),
            tiles=(16, 144),
            cluster_grid=(4, 4),
            log2_grid=(2, 2),
            logical_cluster_grid=(4, 36),
            logical_wg_tasks=2304,
            logical_cluster_tasks=144,
            persistent_stride=16,
        ),
        KERNEL_PROFILE_64X256: LaunchGeometry(
            grid=(64, 4, 1),
            block=(128, 1, 1),
            cluster=(4, 1, 1),
            tiles=(8, 288),
            cluster_grid=(16, 4),
            log2_grid=(4, 2),
            logical_cluster_grid=(2, 288),
            logical_wg_tasks=2304,
            logical_cluster_tasks=576,
            persistent_stride=64,
        ),
    }
    geometries: dict[KernelProfile, LaunchGeometry] = {}
    for profile, expected_geometry in expected_geometries.items():
        if profile.wg_tile != (
            profile.wave_tile[0] * profile.output_quadrants[0],
            profile.wave_tile[1] * profile.output_quadrants[1],
        ):
            raise AssertionError(
                f"profile {profile.name} WG/wave arrangement is invalid"
            )
        if (
            profile.output_quadrants[0] * profile.output_quadrants[1] != 4
            or profile.block != (4 * 32, 1, 1)
        ):
            raise AssertionError(
                f"profile {profile.name} is not block128/four wave32"
            )
        geometry = make_launch_geometry(18432, 2048, 7168, profile)
        geometries[profile] = geometry
        if geometry != expected_geometry:
            raise AssertionError(
                f"{profile.name} default geometry mismatch: got {geometry}, "
                f"expected {expected_geometry}"
            )

    # Preserve the original profile's K%128 host gate while enforcing full
    # K256 bodies for both newer profiles.
    make_launch_geometry(1024, 1024, 128, KERNEL_PROFILE_256)
    shape_rejections = {
        KERNEL_PROFILE_256: (
            (768, 1024, 128),
            (1024, 768, 128),
            (1024, 1024, 64),
        ),
        KERNEL_PROFILE_128: (
            (384, 512, 256),
            (512, 384, 256),
            (512, 512, 128),
        ),
        KERNEL_PROFILE_64X256: (
            (65, 1024, 256),
            (64, 768, 256),
            (64, 1024, 128),
        ),
    }
    for profile, invalid_shapes in shape_rejections.items():
        for invalid_shape in invalid_shapes:
            try:
                make_launch_geometry(*invalid_shape, profile)
            except GemmIsaRunnerError:
                pass
            else:
                raise AssertionError(
                    f"profile {profile.name} accepted invalid shape "
                    f"{invalid_shape}"
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
    }
    profile_payloads: dict[KernelProfile, bytes] = {}
    for profile, geometry in geometries.items():
        if (
            profile.kernarg_size != KERNARG_SIZE
            or profile.kernarg_layout != KERNARG_LAYOUT
        ):
            raise AssertionError(
                f"{profile.name} does not use the exact {KERNARG_SIZE}-byte "
                "BF16 MXFP4 preload ABI"
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
            profile=profile,
        )
        profile_payloads[profile] = default_payload
        if len(default_payload) != profile.kernarg_size:
            raise AssertionError(
                f"{profile.name} kernarg size is {len(default_payload)}, "
                f"expected {profile.kernarg_size}"
            )
        for name, offset, kind in profile.kernarg_layout:
            actual = struct.unpack_from(f"<{kind}", default_payload, offset)[0]
            if name == "log2_grid_x":
                expected_value = geometry.log2_grid[0]
            elif name == "log2_grid_y":
                expected_value = geometry.log2_grid[1]
            else:
                expected_value = expected_default_fields[name]
            if actual != expected_value:
                raise AssertionError(
                    f"{profile.name} kernarg field {name} at byte {offset}: "
                    f"got {actual}, expected {expected_value}"
                )
    # Pointer, runtime-stride, and shape fields are byte-identical.  The last
    # eight bytes intentionally carry each physical cluster grid's log2 pair.
    if len({payload[:72] for payload in profile_payloads.values()}) != 1:
        raise AssertionError(
            "the three source-declared BF16 MXFP4 ABI prefixes packed differently"
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
    for profile, geometry in geometries.items():
        attributes, config = _make_hip_launch_config(
            geometry,
            ctypes.c_void_p(0x1234),
        )
        config_grid = (
            config.gridDimX,
            config.gridDimY,
            config.gridDimZ,
        )
        config_block = (
            config.blockDimX,
            config.blockDimY,
            config.blockDimZ,
        )
        cluster_dim = attributes[0].value.clusterDim
        config_cluster = (cluster_dim.x, cluster_dim.y, cluster_dim.z)
        if (
            config_grid != geometry.grid
            or config_block != geometry.block
            or config_cluster != geometry.cluster
            or attributes[0].id
            != _HIP_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            or config.numAttrs != 1
        ):
            raise AssertionError(
                f"{profile.name} HIP launch config is grid={config_grid}, "
                f"block={config_block}, cluster={config_cluster}"
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
                    "a workgroup-cluster kernel cannot be launched safely"
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
        self._stream_synchronized = True

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
        expected_kernarg_size: int,
    ) -> None:
        if len(payload) != expected_kernarg_size:
            raise GemmIsaRunnerError(
                f"MXFP4 kernarg payload must be {expected_kernarg_size} bytes, "
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

        self._stream = ctypes.c_void_p(stream)
        self._attributes, self._config = _make_hip_launch_config(
            geometry,
            self._stream,
        )
        self._configured = True
        self._stream_synchronized = True

    def launch(self) -> None:
        """Enqueue exactly one dispatch; no loading, packing, or synchronization."""

        if not self._configured:
            raise GemmIsaRunnerError("kernel launch attempted before configure()")
        self._stream_synchronized = False
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
        self._stream_synchronized = True

    def mark_stream_synchronized(self) -> None:
        """Record that an event on this same stream has completed."""

        self._stream_synchronized = True

    def close(self) -> None:
        if not self._module.value:
            return
        sync_error: BaseException | None = None
        if self._configured and not self._stream_synchronized:
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
        from aiter.test_common import checkAllclose
    except ImportError as exc:
        raise GemmIsaRunnerError(
            f"failed to import aiter from repository root {_REPO}: {exc}"
        ) from exc
    return _Dependencies(
        torch=torch,
        f4_test=f4_test,
        check_allclose=checkAllclose,
        get_gfx=get_gfx_runtime,
    )


def _validate_mode(
    intype: str,
    apre: int,
    dtype: str,
    symbol: str,
) -> KernelProfile:
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
    profile = select_kernel_profile(symbol)
    if apre != profile.apre:
        raise GemmIsaRunnerError(
            f"--apre {apre} is incompatible with profile {profile.name}; "
            f"exact symbol {symbol!r} requires --apre {profile.apre}"
        )
    return profile


def _run_batched_cuda_event_timing(
    torch_module: Any,
    launch: Any,
    stream: Any,
    *,
    num_warmup: int,
    num_iters: int,
) -> tuple[Any, float]:
    """Time one continuously queued launch batch with two CUDA events.

    Warmups are enqueued back-to-back and followed by one device sync.  The
    measured launches are then enclosed by one start/end pair on the exact
    PyTorch stream also passed to ``hipDrvLaunchKernelEx``.  Only the end event
    is synchronized; there is no per-iteration synchronization or cache flush.
    """

    if num_warmup < 0:
        raise GemmIsaRunnerError(
            f"batched CUDA-event warmup must be non-negative, got {num_warmup}"
        )
    if num_iters < 1:
        raise GemmIsaRunnerError(
            f"batched CUDA-event iterations must be at least one, got {num_iters}"
        )

    for _ in range(num_warmup):
        launch()
    torch_module.cuda.synchronize()

    start = torch_module.cuda.Event(enable_timing=True)
    end = torch_module.cuda.Event(enable_timing=True)
    start.record(stream)
    output = None
    for _ in range(num_iters):
        output = launch()
    end.record(stream)
    end.synchronize()
    avg_us = float(start.elapsed_time(end)) * 1000.0 / num_iters
    return output, avg_us


def make_cuda_event_timing_row(
    symbol: str,
    iters: int,
    warmup: int,
    avg_us: float,
    device: int,
) -> dict[str, Any]:
    """Build a profiler-like row explicitly sourced from batched CUDA events."""

    avg_us = float(avg_us)
    if avg_us <= 0.0:
        raise GemmIsaRunnerError(
            f"cuda.Event returned a non-positive latency: {avg_us} us"
        )
    if iters < 1:
        raise GemmIsaRunnerError(
            f"CUDA-event timing requires at least one iteration, got {iters}"
        )
    if warmup < 0:
        raise GemmIsaRunnerError(
            f"CUDA-event timing warmup must be non-negative, got {warmup}"
        )
    return {
        "name": symbol,
        "cnt": int(iters),
        "warmup": int(warmup),
        "device_time_sum": avg_us * iters,
        "device_time_avg": avg_us,
        "device_type": "CUDA/HIP",
        "device_index": int(device),
        "source": "cuda.Event batched",
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
        "CUDA-event batched kernel timing "
        "(microseconds; source=cuda.Event batched, not torch profiler):"
    )
    print(_plain_markdown(display, EVENT_TIMING_COLUMNS))


def _print_summary(
    row: Mapping[str, Any],
    pandas_module: Any,
    *,
    symbol: str | None = None,
    profile: KernelProfile | None = None,
) -> None:
    frame = pandas_module.DataFrame([row], columns=SUMMARY_COLUMNS)
    try:
        markdown = frame.to_markdown(index=False)
    except ImportError:
        markdown = _plain_markdown(row)
    identity = (
        f"; profile={profile.name}; symbol={symbol}"
        if profile is not None and symbol is not None
        else ""
    )
    print(
        "gemm_a4w4 (F4GEMM ISA runner) summary "
        f"(markdown{identity}):"
    )
    print(markdown)


def _print_selected_contract(
    isa: Path,
    symbol: str,
    profile: KernelProfile,
    geometry: LaunchGeometry,
) -> None:
    cluster_size = (
        profile.cluster[0] * profile.cluster[1] * profile.cluster[2]
    )
    physical_clusters = geometry.cluster_grid[0] * geometry.cluster_grid[1]
    print(
        f"[gemm_isa_runner] selected profile: {profile.name}; "
        f"WG tile={profile.wg_tile}; wave tile={profile.wave_tile}; "
        f"output wave arrangement (M,N)={profile.output_quadrants}"
    )
    print(
        f"[gemm_isa_runner] logical WG grid="
        f"{geometry.logical_wg_grid}; tasks={geometry.logical_wg_tasks}; "
        f"logical cluster grid={geometry.logical_cluster_grid_3d}; "
        f"cluster tasks={geometry.logical_cluster_tasks}"
    )
    print(
        f"[gemm_isa_runner] physical launch={geometry.grid}; "
        f"block={geometry.block}; cluster={geometry.cluster}; "
        f"physical cluster grid={geometry.physical_cluster_grid}; "
        f"physical clusters={physical_clusters}; cluster size={cluster_size}; "
        f"persistent WGs={profile.persistent_tg}; "
        f"persistent stride={geometry.persistent_stride}; "
        f"log2 grid={geometry.log2_grid}"
    )
    print(
        f"[gemm_isa_runner] exact shape contract: "
        f"M%{profile.wg_tile[0] * profile.cluster[1]}=0; "
        f"N%{profile.wg_tile[1] * profile.cluster[0]}=0; "
        f"K%{profile.k_multiple}=0; no tails or partial clusters"
    )
    print(
        f"[gemm_isa_runner] ABI={profile.abi_name}; "
        f"kernarg={profile.kernarg_size} bytes; "
        f"fixed LDS={profile.group_segment_fixed_size} bytes; "
        f"next-free VGPR/SGPR={profile.next_free_vgpr}/"
        f"{profile.next_free_sgpr}; metadata VGPR/SGPR="
        f"{profile.metadata_vgpr_count}/{profile.metadata_sgpr_count}"
    )
    print(f"[gemm_isa_runner] source: {isa}")
    print(f"[gemm_isa_runner] symbol: {symbol}")


def _run_gemm(
    args: argparse.Namespace,
    code_object: Path,
    symbol: str,
    profile: KernelProfile,
    geometry: LaunchGeometry,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    m, n, k = args.shape
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
            profile=profile,
        )
        module.configure(
            payload,
            geometry,
            int(stream.cuda_stream),
            profile.kernarg_size,
        )

        def launch() -> Any:
            module.launch()
            return output

        timed_output, microseconds = _run_batched_cuda_event_timing(
            torch,
            launch,
            stream,
            num_warmup=args.warmup,
            num_iters=args.iters,
        )
        # end.synchronize() completed every launch on this exact stream.
        module.mark_stream_synchronized()
        event_timing = make_cuda_event_timing_row(
            symbol,
            args.iters,
            args.warmup,
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
        help=(
            "complete gfx1250 AMDGPU assembly source (.s); its symbol must "
            "exactly match a registered 256x256_4x4, 128x128_4x4, or "
            "64x256_1x4 profile"
        ),
    )
    parser.add_argument(
        "--clang",
        help=(
            "AMDGPU clang executable (explicit --clang path/name; otherwise "
            f"fixed {DEFAULT_CLANG}; no ROCm/PATH fallback)"
        ),
    )
    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help=(
            "kernel symbol override (default: auto-detect from .amdhsa_kernel, "
            "then safely fall back to metadata/.globl); an override must "
            "exactly match both the source and a registered profile"
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
        help="A-preshuffle mode (supported exact ISA profiles require 1)",
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
        "--warmup",
        type=int,
        default=101,
        help="continuously enqueued untimed warmup launches (default: 101)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="continuously enqueued batched event iterations (default: 100)",
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
        help="copy the temporary code object beside the ISA as <symbol>.co",
    )
    parser.add_argument(
        "--co-out",
        help="copy the compiled code object to this path (implies --keep-co)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iters < 1:
        parser.error("--iters must be at least 1")
    if args.device < 0:
        parser.error("--device must be non-negative")

    try:
        run_static_contract_checks()
        isa = _resolve_isa(args.isa)
        source = _read_isa_source(isa)
        symbol = resolve_kernel_symbol_from_text(source, args.symbol)
        profile = _validate_mode(args.intype, args.apre, args.dtype, symbol)
        validate_assembly_contract_from_text(source, symbol, profile)
        # Validate profile-specific geometry before clang or device allocation.
        geometry = make_launch_geometry(*args.shape, profile=profile)
        _print_selected_contract(isa, symbol, profile, geometry)
        clang = _resolve_clang(args.clang)
        if _clang_uses_default_runtime_libraries(clang):
            _prepend_default_clang_runtime_libraries()

        with tempfile.TemporaryDirectory(prefix="gemm_isa_runner_") as temp:
            result = compile_isa(isa, clang, Path(temp), symbol)
            for command in result.commands:
                print(f"[gemm_isa_runner] {_format_command(command)}")
            for patch in result.patches:
                print(f"[gemm_isa_runner] {patch}")
            print(f"[gemm_isa_runner] loading kernel symbol: {symbol}")

            if args.co_out or args.keep_co:
                destination = (
                    Path(os.path.expandvars(args.co_out)).expanduser().resolve()
                    if args.co_out
                    else isa.with_name(f"{symbol}.co")
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
                profile,
                geometry,
            )
            _print_cuda_event_timing(event_timing)
            _print_summary(
                row,
                _load_pandas_from_row_context(),
                symbol=symbol,
                profile=profile,
            )
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
