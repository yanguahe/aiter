#!/usr/bin/env python3
"""Content-addressed build/cache support for the MoE GEMM1 C++ launcher.

This module intentionally has no import-time PyTorch dependency so its key and
cache logic can be tested on CPU-only development machines.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


_LOG_PREFIX = "[moe_cpp_backend]"
_CACHE_SCHEMA = 1
_CO_SCHEMA = "moe-gemm1-code-object-v1"
_EXTENSION_SCHEMA = "moe-gemm1-cpp-extension-v1"
_CPP_SOURCE = Path(__file__).with_name("moe_gemm1_cpp_launcher.cpp")
_CPP_EXTRA_CFLAGS = ("-O3", "-std=c++17")
_CPP_EXTRA_CUDA_CFLAGS: tuple[str, ...] = ()
_CPP_EXTRA_LDFLAGS = ("-Wl,--no-as-needed",)
_LOCK_TIMEOUT_SECONDS = 30 * 60
_STALE_LOCK_SECONDS = 60 * 60

PIPELINE_LAUNCH_BACKEND_ENV = "AITER_MOE_GEMM1_LAUNCH_BACKEND"
PIPELINE_LAUNCH_BACKEND_CPP = "cpp"
PIPELINE_TARGET_SYMBOL = (
    "a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4"
)
PIPELINE_CONTIGUOUS_M = 9216
PIPELINE_VALID_ROWS_PER_EXPERT = 32
PIPELINE_EXPERTS = 96
PIPELINE_N = 6144
PIPELINE_K = 7168
PIPELINE_TILE_M = 64
PIPELINE_TILE_N = 256
PIPELINE_TILE_K = 256
PIPELINE_BLOCK = (128, 1, 1)
PIPELINE_CLUSTER = (1, 1, 1)


@dataclass(frozen=True)
class CacheProbe:
    hit: bool
    reason: str
    manifest: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CachedCodeObject:
    object_file: Path
    code_object: Path
    code_object_sha256: str
    key: str
    cache_hit: bool
    commands: tuple[tuple[str, ...], ...]
    patches: tuple[str, ...]


@dataclass(frozen=True)
class CachedExtension:
    module: Any
    key: str
    cache_hit: bool
    shared_library: Path


@dataclass(frozen=True)
class MoeCppBackendArtifacts:
    code: CachedCodeObject
    extension: CachedExtension
    cache_root: Path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_build_key(layer: str, inputs: Mapping[str, Any]) -> str:
    """Return a stable SHA256 key for JSON-compatible build inputs."""

    payload = {
        "cache_schema": _CACHE_SCHEMA,
        "layer": str(layer),
        "inputs": inputs,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return _sha256_bytes(encoded)


def default_cache_root() -> Path:
    override = os.environ.get("AITER_MOE_CPP_CACHE_DIR")
    if override:
        return Path(os.path.expandvars(override)).expanduser().resolve()
    base = os.environ.get("XDG_CACHE_HOME")
    root = (
        Path(os.path.expandvars(base)).expanduser()
        if base
        else Path.home() / ".cache"
    )
    return (root / "aiter" / "moe_gemm1_cpp_launcher").resolve()


def _safe_artifact_name(name: Any) -> str | None:
    if not isinstance(name, str) or not name:
        return None
    path = Path(name)
    if path.name != name or name in (".", ".."):
        return None
    return name


def _cache_entry_status(
    entry: Path,
    expected_key: str,
    expected_artifacts: Sequence[str] | None = None,
) -> CacheProbe:
    """Validate a complete cache entry, including every artifact digest."""

    manifest_path = entry / "manifest.json"
    if not manifest_path.is_file():
        return CacheProbe(False, "manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return CacheProbe(False, f"manifest is unreadable: {exc}")
    if not isinstance(manifest, dict):
        return CacheProbe(False, "manifest root is not an object")
    if manifest.get("key") != expected_key:
        return CacheProbe(False, "manifest build key differs", manifest)

    names = (
        tuple(expected_artifacts)
        if expected_artifacts is not None
        else tuple(manifest.get("required_artifacts", ()))
    )
    records = manifest.get("artifacts")
    if not isinstance(records, dict):
        return CacheProbe(False, "manifest has no artifact records", manifest)
    for raw_name in names:
        name = _safe_artifact_name(raw_name)
        if name is None:
            return CacheProbe(False, f"unsafe artifact name {raw_name!r}", manifest)
        record = records.get(name)
        if not isinstance(record, dict):
            return CacheProbe(False, f"manifest lacks artifact {name}", manifest)
        path = entry / name
        if not path.is_file():
            return CacheProbe(False, f"artifact {name} is missing", manifest)
        try:
            size = path.stat().st_size
        except OSError as exc:
            return CacheProbe(False, f"artifact {name} cannot be stated: {exc}", manifest)
        if size <= 0 or size != record.get("size"):
            return CacheProbe(False, f"artifact {name} size changed", manifest)
        if _sha256_file(path) != record.get("sha256"):
            return CacheProbe(False, f"artifact {name} sha256 changed", manifest)
    return CacheProbe(True, "complete manifest and artifact hashes match", manifest)


def _write_cache_manifest(
    directory: Path,
    *,
    key: str,
    inputs: Mapping[str, Any],
    artifact_names: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    names: list[str] = []
    for raw_name in artifact_names:
        name = _safe_artifact_name(raw_name)
        if name is None:
            raise ValueError(f"unsafe cache artifact name: {raw_name!r}")
        path = directory / name
        size = path.stat().st_size
        if size <= 0:
            raise RuntimeError(f"cache artifact is empty: {path}")
        names.append(name)
        artifacts[name] = {
            "size": size,
            "sha256": _sha256_file(path),
        }
    manifest: dict[str, Any] = {
        "cache_schema": _CACHE_SCHEMA,
        "key": key,
        "inputs": inputs,
        "required_artifacts": names,
        "artifacts": artifacts,
    }
    if extra:
        manifest.update(extra)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _input_differences(
    previous: Any,
    current: Any,
    prefix: str = "",
) -> list[str]:
    if isinstance(previous, dict) and isinstance(current, dict):
        changed: list[str] = []
        for key in sorted(set(previous) | set(current)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in previous or key not in current:
                changed.append(child)
            else:
                changed.extend(
                    _input_differences(previous[key], current[key], child)
                )
        return changed
    return [] if previous == current else [prefix or "<root>"]


def _cache_miss_reason(layer_root: Path, inputs: Mapping[str, Any], fallback: str) -> str:
    latest = layer_root / "latest.json"
    try:
        previous = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return fallback
    changed = _input_differences(previous.get("inputs"), inputs)
    if not changed:
        return fallback
    shown = ", ".join(changed[:8])
    if len(changed) > 8:
        shown += f", ... (+{len(changed) - 8})"
    return f"build inputs changed: {shown}"


class _DirectoryLock:
    """Portable mkdir lock with stale-lock recovery."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    def __enter__(self) -> "_DirectoryLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        while True:
            try:
                self.path.mkdir()
                self.acquired = True
                _atomic_write_json(
                    self.path / "owner.json",
                    {
                        "pid": os.getpid(),
                        "host": platform.node(),
                        "created_unix": time.time(),
                    },
                )
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > _STALE_LOCK_SECONDS:
                    stale = self.path.with_name(
                        f"{self.path.name}.stale.{uuid.uuid4().hex}"
                    )
                    try:
                        os.replace(self.path, stale)
                    except (FileNotFoundError, OSError):
                        pass
                    else:
                        shutil.rmtree(stale, ignore_errors=True)
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for build lock {self.path}")
                time.sleep(0.2)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self.acquired:
            shutil.rmtree(self.path, ignore_errors=True)
            self.acquired = False
        return False


def _publish_directory(temporary: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(temporary, destination)


def _command_version(command: Sequence[str]) -> str:
    try:
        process = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return f"unavailable: {exc}"
    return (
        f"exit={process.returncode}\nstdout:\n{process.stdout}"
        f"stderr:\n{process.stderr}"
    )


def _code_object_inputs(
    *,
    isa: Path,
    source_bytes: bytes,
    clang: Path,
    symbol: str,
    single_module: Any,
) -> Mapping[str, Any]:
    placeholder_isa = Path("<ISA>")
    placeholder_dir = Path("<BUILD>")
    assemble, link, _obj, _co = single_module.make_compile_isa_commands(
        placeholder_isa,
        clang,
        placeholder_dir,
        symbol,
    )
    source_text = source_bytes.decode("utf-8")
    return {
        "schema": _CO_SCHEMA,
        "isa": {
            "basename": isa.name,
            "size": len(source_bytes),
            "sha256": _sha256_bytes(source_bytes),
        },
        "kernel_symbol": symbol,
        "arch": single_module.ARCH,
        "code_object_version": single_module.CODE_OBJECT_VERSION,
        "clang": {
            "absolute_path": str(clang.resolve()),
            "version": _command_version((str(clang), "--version")),
            "runtime_libraries": [
                str(path) for path in single_module.DEFAULT_CLANG_RUNTIME_LIBRARIES
            ],
        },
        "commands": {
            "assemble": list(assemble),
            "link": list(link),
        },
        "tcp_split_request": single_module.parse_requested_tcp_split(source_text),
    }


def _build_or_get_code_object(
    *,
    cache_root: Path,
    isa: Path,
    clang: Path,
    symbol: str,
    single_module: Any,
) -> CachedCodeObject:
    source_bytes = isa.read_bytes()
    inputs = _code_object_inputs(
        isa=isa,
        source_bytes=source_bytes,
        clang=clang,
        symbol=symbol,
        single_module=single_module,
    )
    key = make_build_key("code_object", inputs)
    layer_root = cache_root / "code_objects"
    entry = layer_root / key
    object_name = f"{symbol}.o"
    code_name = f"{symbol}.co"
    expected = (object_name, code_name)
    probe = _cache_entry_status(entry, key, expected)
    if probe.hit:
        print(
            f"{_LOG_PREFIX} code object cache hit: key={key}; "
            f"path={entry / code_name}"
        )
        assert probe.manifest is not None
        return CachedCodeObject(
            object_file=entry / object_name,
            code_object=entry / code_name,
            code_object_sha256=str(
                probe.manifest["artifacts"][code_name]["sha256"]
            ),
            key=key,
            cache_hit=True,
            commands=(),
            patches=tuple(probe.manifest.get("patches", ())),
        )

    layer_root.mkdir(parents=True, exist_ok=True)
    reason = _cache_miss_reason(layer_root, inputs, probe.reason)
    print(f"{_LOG_PREFIX} code object cache miss: key={key}; reason={reason}")
    with _DirectoryLock(cache_root / "locks" / f"co-{key}.lock"):
        probe = _cache_entry_status(entry, key, expected)
        if probe.hit:
            print(
                f"{_LOG_PREFIX} code object cache hit after lock: "
                f"key={key}; path={entry / code_name}"
            )
            assert probe.manifest is not None
            return CachedCodeObject(
                object_file=entry / object_name,
                code_object=entry / code_name,
                code_object_sha256=str(
                    probe.manifest["artifacts"][code_name]["sha256"]
                ),
                key=key,
                cache_hit=True,
                commands=(),
                patches=tuple(probe.manifest.get("patches", ())),
            )

        temporary_root = cache_root / "tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f"co-{key[:12]}-", dir=temporary_root)
        )
        commands: tuple[tuple[str, ...], ...] = ()
        patches: tuple[str, ...] = ()
        try:
            result = single_module.compile_isa(
                isa,
                clang,
                temporary,
                symbol,
            )
            if result.object_file is None:
                raise RuntimeError("compile_isa did not report its .o artifact")
            commands = result.commands
            patches = result.patches
            for command in commands:
                print(
                    f"{_LOG_PREFIX} code object build command: "
                    f"{single_module._format_command(command)}"
                )
            for patch in patches:
                print(f"{_LOG_PREFIX} code object patch: {patch}")
            if result.object_file.name != object_name:
                raise RuntimeError(
                    f"unexpected object filename {result.object_file.name!r}"
                )
            if result.code_object.name != code_name:
                raise RuntimeError(
                    f"unexpected code-object filename {result.code_object.name!r}"
                )
            manifest = _write_cache_manifest(
                temporary,
                key=key,
                inputs=inputs,
                artifact_names=expected,
                extra={
                    "commands": [list(command) for command in commands],
                    "patches": list(patches),
                },
            )
            _publish_directory(temporary, entry)
            _atomic_write_json(
                layer_root / "latest.json",
                {"key": key, "inputs": inputs},
            )
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)

    print(
        f"{_LOG_PREFIX} code object build complete: key={key}; "
        f"path={entry / code_name}"
    )
    return CachedCodeObject(
        object_file=entry / object_name,
        code_object=entry / code_name,
        code_object_sha256=str(manifest["artifacts"][code_name]["sha256"]),
        key=key,
        cache_hit=False,
        commands=commands,
        patches=patches,
    )


def _extension_inputs(
    *,
    source_bytes: bytes,
    code: CachedCodeObject,
    symbol: str,
    torch_module: Any,
    cpp_extension: Any,
    single_module: Any,
) -> Mapping[str, Any]:
    cxx_raw = os.environ.get("CXX", "c++")
    cxx_path = shutil.which(cxx_raw) or cxx_raw
    # Newer PyTorch calls this positional parameter ``device_type`` while
    # older releases called it the boolean ``cuda``.  The string is accepted
    # by both (truthy in the legacy implementation).
    include_paths = [str(path) for path in cpp_extension.include_paths("cuda")]
    library_paths = [str(path) for path in cpp_extension.library_paths("cuda")]
    return {
        "schema": _EXTENSION_SCHEMA,
        "cpp_source": {
            "basename": _CPP_SOURCE.name,
            "size": len(source_bytes),
            "sha256": _sha256_bytes(source_bytes),
        },
        "code_object": {
            "build_key": code.key,
            "sha256": code.code_object_sha256,
            "arch": single_module.ARCH,
            "version": single_module.CODE_OBJECT_VERSION,
        },
        "kernel_symbol": symbol,
        "build": {
            "with_cuda": True,
            "extra_cflags": list(_CPP_EXTRA_CFLAGS),
            "extra_cuda_cflags": list(_CPP_EXTRA_CUDA_CFLAGS),
            "extra_ldflags": list(_CPP_EXTRA_LDFLAGS),
            "include_paths": include_paths,
            "library_paths": library_paths,
            "cxx": {
                "requested": cxx_raw,
                "resolved": str(Path(cxx_path).resolve())
                if Path(cxx_path).exists()
                else cxx_path,
                "version": _command_version((cxx_path, "--version")),
            },
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "CC",
                    "CXX",
                    "CFLAGS",
                    "CXXFLAGS",
                    "LDFLAGS",
                    "ROCM_HOME",
                    "ROCM_PATH",
                    "CUDA_HOME",
                    "PYTORCH_ROCM_ARCH",
                    "TORCH_CUDA_ARCH_LIST",
                )
            },
        },
        "python_abi": {
            "version": sys.version,
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
            "soabi": sysconfig.get_config_var("SOABI"),
            "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        },
        "pytorch_hip_abi": {
            "torch_version": str(torch_module.__version__),
            "torch_hip_version": str(getattr(torch_module.version, "hip", None)),
            "glibcxx_use_cxx11_abi": bool(
                getattr(torch_module._C, "_GLIBCXX_USE_CXX11_ABI", False)
            ),
            "cuda_home": str(getattr(cpp_extension, "CUDA_HOME", None)),
            "rocm_home": str(getattr(cpp_extension, "ROCM_HOME", None)),
        },
    }


def _render_cpp_source(
    template: bytes,
    *,
    extension_key: str,
    code_object_sha256: str,
) -> bytes:
    text = template.decode("utf-8")
    replacements = {
        "@MOE_CPP_EXTENSION_BUILD_KEY@": extension_key,
        "@MOE_CODE_OBJECT_SHA256@": code_object_sha256,
    }
    for marker, value in replacements.items():
        if text.count(marker) != 1:
            raise RuntimeError(
                f"C++ launcher template must contain {marker!r} exactly once"
            )
        text = text.replace(marker, value)
    return text.encode("utf-8")


def _import_extension(module_name: str, shared_library: Path) -> Any:
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, shared_library)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for {shared_library}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _verify_extension_identity(
    module: Any,
    *,
    extension_key: str,
    code_object_sha256: str,
) -> None:
    if module.build_key() != extension_key:
        raise RuntimeError(
            f"C++ extension build key {module.build_key()!r} does not match "
            f"{extension_key!r}"
        )
    if module.code_object_sha256() != code_object_sha256:
        raise RuntimeError(
            "C++ extension code-object sha256 does not match the cached .co"
        )


def _load_cached_extension(
    entry: Path,
    key: str,
    code_sha256: str,
    probe: CacheProbe,
) -> tuple[Any, Path]:
    if not probe.hit or probe.manifest is None:
        raise RuntimeError("attempted to load an invalid extension cache entry")
    module_name = str(probe.manifest.get("module_name"))
    shared_name = _safe_artifact_name(probe.manifest.get("shared_library"))
    if not module_name or shared_name is None:
        raise RuntimeError("extension manifest lacks module/shared-library names")
    shared = entry / shared_name
    module = _import_extension(module_name, shared)
    _verify_extension_identity(
        module,
        extension_key=key,
        code_object_sha256=code_sha256,
    )
    return module, shared


def _forget_cached_extension(probe: CacheProbe) -> None:
    if probe.manifest is None:
        return
    module_name = probe.manifest.get("module_name")
    if isinstance(module_name, str):
        sys.modules.pop(module_name, None)


def _build_or_get_extension(
    *,
    cache_root: Path,
    code: CachedCodeObject,
    symbol: str,
    torch_module: Any,
    single_module: Any,
) -> CachedExtension:
    from torch.utils import cpp_extension

    template = _CPP_SOURCE.read_bytes()
    inputs = _extension_inputs(
        source_bytes=template,
        code=code,
        symbol=symbol,
        torch_module=torch_module,
        cpp_extension=cpp_extension,
        single_module=single_module,
    )
    key = make_build_key("cpp_extension", inputs)
    module_name = f"aiter_moe_gemm1_cpp_{key[:20]}"
    layer_root = cache_root / "extensions"
    entry = layer_root / key
    probe = _cache_entry_status(entry, key)
    if probe.hit:
        try:
            module, shared = _load_cached_extension(
                entry,
                key,
                code.code_object_sha256,
                probe,
            )
        except Exception as exc:
            _forget_cached_extension(probe)
            probe = CacheProbe(False, f"cached extension import failed: {exc}")
        else:
            print(
                f"{_LOG_PREFIX} C++ extension cache hit: key={key}; "
                f"path={shared}"
            )
            return CachedExtension(module, key, True, shared)

    layer_root.mkdir(parents=True, exist_ok=True)
    reason = _cache_miss_reason(layer_root, inputs, probe.reason)
    print(f"{_LOG_PREFIX} C++ extension cache miss: key={key}; reason={reason}")
    with _DirectoryLock(cache_root / "locks" / f"ext-{key}.lock"):
        probe = _cache_entry_status(entry, key)
        if probe.hit:
            try:
                module, shared = _load_cached_extension(
                    entry,
                    key,
                    code.code_object_sha256,
                    probe,
                )
            except Exception as exc:
                _forget_cached_extension(probe)
                print(
                    f"{_LOG_PREFIX} invalid cached extension after lock: {exc}; "
                    "rebuilding"
                )
                shutil.rmtree(entry, ignore_errors=True)
            else:
                print(
                    f"{_LOG_PREFIX} C++ extension cache hit after lock: "
                    f"key={key}; path={shared}"
                )
                return CachedExtension(module, key, True, shared)

        temporary_root = cache_root / "tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f"ext-{key[:12]}-", dir=temporary_root)
        )
        source_name = "moe_gemm1_cpp_launcher.generated.cpp"
        generated_source = temporary / source_name
        try:
            generated_source.write_bytes(
                _render_cpp_source(
                    template,
                    extension_key=key,
                    code_object_sha256=code.code_object_sha256,
                )
            )
            print(
                f"{_LOG_PREFIX} building C++ extension: module={module_name}; "
                f"source={generated_source}; cflags={list(_CPP_EXTRA_CFLAGS)}; "
                f"ldflags={list(_CPP_EXTRA_LDFLAGS)}; with_cuda=True"
            )
            module = cpp_extension.load(
                name=module_name,
                sources=[str(generated_source)],
                extra_cflags=list(_CPP_EXTRA_CFLAGS),
                extra_cuda_cflags=list(_CPP_EXTRA_CUDA_CFLAGS),
                extra_ldflags=list(_CPP_EXTRA_LDFLAGS),
                build_directory=str(temporary),
                verbose=True,
                with_cuda=True,
                is_python_module=True,
            )
            built_shared = Path(module.__file__).resolve()
            if built_shared.parent != temporary.resolve():
                raise RuntimeError(
                    f"extension was built outside the atomic staging directory: "
                    f"{built_shared}"
                )
            shared_name = built_shared.name
            _verify_extension_identity(
                module,
                extension_key=key,
                code_object_sha256=code.code_object_sha256,
            )
            _write_cache_manifest(
                temporary,
                key=key,
                inputs=inputs,
                artifact_names=(source_name, shared_name),
                extra={
                    "module_name": module_name,
                    "shared_library": shared_name,
                    "generated_source": source_name,
                },
            )
            _publish_directory(temporary, entry)
            shared = entry / shared_name
            module.__file__ = str(shared)
            _atomic_write_json(
                layer_root / "latest.json",
                {"key": key, "inputs": inputs},
            )
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)

    print(
        f"{_LOG_PREFIX} C++ extension build complete: key={key}; path={shared}"
    )
    return CachedExtension(module, key, False, shared)


def prepare_moe_cpp_backend(
    *,
    isa: Path,
    clang: Path,
    symbol: str,
    single_module: Any,
    torch_module: Any,
    cache_root: Path | None = None,
) -> MoeCppBackendArtifacts:
    """Build/load the target .co and its exact C++ launcher extension."""

    root = (cache_root or default_cache_root()).resolve()
    print(f"{_LOG_PREFIX} cache root: {root}")
    try:
        code = _build_or_get_code_object(
            cache_root=root,
            isa=isa,
            clang=clang,
            symbol=symbol,
            single_module=single_module,
        )
        extension = _build_or_get_extension(
            cache_root=root,
            code=code,
            symbol=symbol,
            torch_module=torch_module,
            single_module=single_module,
        )
    except single_module.CompileError:
        raise
    except Exception as exc:
        raise single_module.GemmIsaRunnerError(
            f"failed to prepare the MoE C++ launch backend: {exc}"
        ) from exc
    return MoeCppBackendArtifacts(code, extension, root)


def selected_pipeline_launch_backend() -> str | None:
    """Parse the explicit pipeline-only launch-backend opt-in."""

    raw = os.environ.get(PIPELINE_LAUNCH_BACKEND_ENV, "")
    value = raw.strip().lower()
    if value in ("", "flydsl"):
        return None
    if value == PIPELINE_LAUNCH_BACKEND_CPP:
        return value
    raise ValueError(
        f"{PIPELINE_LAUNCH_BACKEND_ENV} must be unset, 'flydsl', or "
        f"'{PIPELINE_LAUNCH_BACKEND_CPP}', got {raw!r}"
    )


def validate_pipeline_gemm1_case(
    *,
    experts: int,
    tokens: int,
    topk: int,
    model_dim: int,
    inter_dim: int,
    data_format: str,
    activation: str,
    use_bias: bool,
    expert_balance: bool,
    num_expert_activated: int,
    tile_m_override: str | None,
    swiglu_limit: float,
    situ_beta: float,
    situ_linear_beta: float,
) -> None:
    """Reject every pipeline setting outside the audited C++ launch contract."""

    actual = {
        "experts": int(experts),
        "tokens": int(tokens),
        "topk": int(topk),
        "model_dim": int(model_dim),
        "inter_dim": int(inter_dim),
        "data_format": str(data_format),
        "activation": str(activation).lower(),
        "use_bias": bool(use_bias),
        "expert_balance": bool(expert_balance),
        "num_expert_activated": int(num_expert_activated),
        "AITER_TDM_TILE_M": tile_m_override,
        "swiglu_limit": float(swiglu_limit),
        "situ_beta": float(situ_beta),
        "situ_linear_beta": float(situ_linear_beta),
    }
    expected = {
        "experts": 96,
        "tokens": 512,
        "topk": 6,
        "model_dim": 7168,
        "inter_dim": 3072,
        "data_format": "a4w4",
        "activation": "silu",
        "use_bias": False,
        "expert_balance": True,
        "num_expert_activated": 0,
        "AITER_TDM_TILE_M": "64",
        "swiglu_limit": 7.0,
        "situ_beta": 4.0,
        "situ_linear_beta": 25.0,
    }
    mismatches = [
        f"{name}={actual[name]!r} (required {required!r})"
        for name, required in expected.items()
        if actual[name] != required
    ]
    if mismatches:
        raise ValueError(
            f"{PIPELINE_LAUNCH_BACKEND_ENV}=cpp is restricted to the audited "
            f"{PIPELINE_TARGET_SYMBOL!r} pipeline case; "
            + "; ".join(mismatches)
        )


def pack_pipeline_moe_kernargs(
    *,
    packer: Callable[..., bytes],
    ptr_c: int,
    ptr_a: int,
    ptr_b: int,
    ptr_scale_a: int,
    ptr_scale_b: int,
    ptr_m_tile_map: int,
    c_shape: tuple[int, int, int],
    c_strides: tuple[int, int],
    sa_shape: tuple[int, int, int],
    sa_strides: tuple[int, int],
    sb_size0: int,
    i32_m: int,
    i32_n: int,
    swiglu_limit: float,
    situ_beta: float,
    situ_linear_beta: float,
) -> bytes:
    """Pack the real no-bias/no-quant pipeline aliases into the 184-byte ABI."""

    return packer(
        ptr_c=ptr_c,
        ptr_a=ptr_a,
        ptr_b=ptr_b,
        ptr_scale_a=ptr_scale_a,
        ptr_scale_b=ptr_scale_b,
        ptr_m_tile_map=ptr_m_tile_map,
        # The FlyDSL wrapper aliases an absent bias to A.
        ptr_bias=ptr_a,
        # It likewise aliases an absent quant-scale tensor to C.
        ptr_quant_scale=ptr_c,
        c_shape=c_shape,
        c_strides=c_strides,
        sa_shape=sa_shape,
        sa_strides=sa_strides,
        sb_size0=sb_size0,
        qs_shape=c_shape,
        qs_strides=c_strides,
        i32_m=i32_m,
        i32_n=i32_n,
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )


def _tensor_contract(name: str, tensor: Any) -> str:
    return (
        f"{name}=ptr:{tensor.data_ptr():#x},shape:{tuple(tensor.shape)},"
        f"stride:{tuple(tensor.stride())},dtype:{tensor.dtype}"
    )


def _unpack_kernarg_fields(
    payload: bytes,
    layout: Sequence[tuple[str, int, str]],
) -> dict[str, int]:
    return {
        name: int(struct.unpack_from("<" + fmt, payload, offset)[0])
        for name, offset, fmt in layout
    }


class MoePipelineGemm1Adapter:
    """Replace only the audited stage-1 launch and delegate stage 2 unchanged."""

    def __init__(
        self,
        *,
        torch_module: Any,
        original_launcher: Callable[..., Any],
        launcher_module: Any,
        extension: Any,
        code_object: Path,
        packer: Callable[..., bytes],
        kernarg_layout: Sequence[tuple[str, int, str]],
    ) -> None:
        self.torch = torch_module
        self.original_launcher = original_launcher
        self.launcher_module = launcher_module
        self.extension = extension
        self.code_object = code_object
        self.packer = packer
        self.kernarg_layout = tuple(kernarg_layout)
        self.injected_calls = 0
        self.delegated_calls = 0
        self.debug_checked = False
        self.padding_nonzero: int | None = None

    def _delegate(
        self,
        out: Any,
        a: Any,
        w: Any,
        a_scales: Any,
        w_scales: Any,
        m_tile_map: Any,
        **kwargs: Any,
    ) -> Any:
        self.delegated_calls += 1
        return self.original_launcher(
            out,
            a,
            w,
            a_scales,
            w_scales,
            m_tile_map,
            **kwargs,
        )

    @staticmethod
    def _require(conditions: Sequence[tuple[bool, str]]) -> None:
        failures = [message for condition, message in conditions if not condition]
        if failures:
            raise ValueError(
                "C++ pipeline GEMM1 launch contract mismatch: "
                + "; ".join(failures)
            )

    def __call__(
        self,
        out: Any,
        a: Any,
        w: Any,
        a_scales: Any,
        w_scales: Any,
        m_tile_map: Any,
        *,
        n_experts: int,
        contiguous_m: int,
        N: int,
        K: int,
        tile_m: int = 64,
        tile_n: int = 256,
        tile_k: int = 256,
        m_warp: int = 1,
        n_warp: int = 4,
        num_buffers: int = 3,
        out_is_f16: int = 0,
        a_is_fp4: int = 0,
        stage1_act: int = 0,
        bias: Any = None,
        swiglu_limit: float = 7.0,
        stream: Any = None,
        stage1_quant_out: int = 0,
        quant_scale: Any = None,
        quant_wmma_rep: int = 1,
        cluster_n: int = -1,
        waves_per_tensor_tdm: int = -1,
        next_stage_prefetch: int = 0,
        situ_beta: float = 1.0,
        situ_linear_beta: float = 1.0,
    ) -> Any:
        if int(stage1_act) == 0:
            return self._delegate(
                out,
                a,
                w,
                a_scales,
                w_scales,
                m_tile_map,
                n_experts=n_experts,
                contiguous_m=contiguous_m,
                N=N,
                K=K,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                m_warp=m_warp,
                n_warp=n_warp,
                num_buffers=num_buffers,
                out_is_f16=out_is_f16,
                a_is_fp4=a_is_fp4,
                stage1_act=stage1_act,
                bias=bias,
                swiglu_limit=swiglu_limit,
                stream=stream,
                stage1_quant_out=stage1_quant_out,
                quant_scale=quant_scale,
                quant_wmma_rep=quant_wmma_rep,
                cluster_n=cluster_n,
                waves_per_tensor_tdm=waves_per_tensor_tdm,
                next_stage_prefetch=next_stage_prefetch,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
            )

        effective_num_buffers = min(
            int(num_buffers),
            max(1, int(K) // int(tile_k)),
        )
        n_tiles = (int(N) + int(tile_n) - 1) // int(tile_n)
        effective_cluster_n = self.launcher_module._select_cluster_n(
            n_tiles, int(cluster_n)
        )
        effective_waves = self.launcher_module._select_num_waves_per_tensor_tdm(
            int(waves_per_tensor_tdm)
        )
        effective_prefetch = self.launcher_module._select_next_stage_prefetch(
            int(next_stage_prefetch)
        )
        next_stage_on = int(
            bool(effective_prefetch and effective_num_buffers >= 3)
        )
        current_stream = self.torch.cuda.current_stream(out.device)
        requested_stream = current_stream if stream is None else stream

        expected_scale_b_bytes = PIPELINE_EXPERTS * PIPELINE_N * (PIPELINE_K // 32)
        conditions = (
            (int(n_experts) == PIPELINE_EXPERTS, f"n_experts={n_experts}"),
            (
                int(contiguous_m) == PIPELINE_CONTIGUOUS_M,
                f"contiguous_m={contiguous_m}",
            ),
            (int(N) == PIPELINE_N, f"N={N}"),
            (int(K) == PIPELINE_K, f"K={K}"),
            (int(tile_m) == PIPELINE_TILE_M, f"tile_m={tile_m}"),
            (int(tile_n) == PIPELINE_TILE_N, f"tile_n={tile_n}"),
            (int(tile_k) == PIPELINE_TILE_K, f"tile_k={tile_k}"),
            (int(m_warp) == 1, f"m_warp={m_warp}"),
            (int(n_warp) == 4, f"n_warp={n_warp}"),
            (effective_num_buffers == 2, f"num_buffers={effective_num_buffers}"),
            (int(out_is_f16) == 0, f"out_is_f16={out_is_f16}"),
            (int(a_is_fp4) == 1, f"a_is_fp4={a_is_fp4}"),
            (int(stage1_act) == 1, f"stage1_act={stage1_act}"),
            (bias is None, "bias must be None"),
            (int(stage1_quant_out) == 0, f"stage1_quant_out={stage1_quant_out}"),
            (quant_scale is None, "quant_scale must be None"),
            (int(quant_wmma_rep) == 1, f"quant_wmma_rep={quant_wmma_rep}"),
            (effective_cluster_n == 1, f"cluster_n={effective_cluster_n}"),
            (effective_waves == 4, f"waves_per_tensor_tdm={effective_waves}"),
            (
                next_stage_on == 0,
                "next_stage_on="
                f"{next_stage_on} (requested={effective_prefetch}, "
                f"num_buffers={effective_num_buffers})",
            ),
            (float(swiglu_limit) == 7.0, f"swiglu_limit={swiglu_limit}"),
            (float(situ_beta) == 4.0, f"situ_beta={situ_beta}"),
            (
                float(situ_linear_beta) == 25.0,
                f"situ_linear_beta={situ_linear_beta}",
            ),
            (
                int(requested_stream.cuda_stream) == int(current_stream.cuda_stream),
                "an explicit non-current stream is unsupported",
            ),
            (
                tuple(out.shape) == (1, PIPELINE_CONTIGUOUS_M, PIPELINE_N // 2),
                f"out.shape={tuple(out.shape)}",
            ),
            (
                tuple(a.shape) == (1, PIPELINE_CONTIGUOUS_M, PIPELINE_K // 2),
                f"a.shape={tuple(a.shape)}",
            ),
            (
                int(w.numel())
                == PIPELINE_EXPERTS * PIPELINE_N * (PIPELINE_K // 2),
                f"w.numel()={w.numel()}",
            ),
            (
                tuple(a_scales.shape)
                == (
                    1,
                    PIPELINE_CONTIGUOUS_M // 4,
                    (PIPELINE_K // 32) * 4,
                ),
                f"a_scales.shape={tuple(a_scales.shape)}",
            ),
            (
                int(w_scales.numel()) * int(w_scales.element_size())
                == expected_scale_b_bytes,
                "w_scales byte footprint differs",
            ),
            (
                tuple(m_tile_map.shape) == (PIPELINE_EXPERTS,),
                f"m_tile_map.shape={tuple(m_tile_map.shape)}",
            ),
            (out.dtype == self.torch.bfloat16, f"out.dtype={out.dtype}"),
            (a.dtype == self.torch.uint8, f"a.dtype={a.dtype}"),
            (w.dtype == self.torch.uint8, f"w.dtype={w.dtype}"),
            (
                a_scales.dtype == self.torch.uint8,
                f"a_scales.dtype={a_scales.dtype}",
            ),
            (
                w_scales.dtype == self.torch.int32,
                f"w_scales.dtype={w_scales.dtype}",
            ),
            (
                m_tile_map.dtype == self.torch.int32,
                f"m_tile_map.dtype={m_tile_map.dtype}",
            ),
            (out.is_contiguous(), "out must be contiguous"),
            (a.is_contiguous(), "a must be contiguous"),
            (w.is_contiguous(), "w must be contiguous"),
            (a_scales.is_contiguous(), "a_scales must be contiguous"),
            (w_scales.is_contiguous(), "w_scales must be contiguous"),
            (m_tile_map.is_contiguous(), "m_tile_map must be contiguous"),
        )
        self._require(conditions)

        scale_a_i32 = a_scales.view(self.torch.int32)
        scale_b_i32 = w_scales.view(self.torch.int32)
        scale_b_u8 = scale_b_i32.view(self.torch.uint8)
        c_shape = tuple(int(value) for value in out.shape)
        c_strides = tuple(int(value) for value in out.stride()[:2])
        sa_shape = tuple(int(value) for value in scale_a_i32.shape)
        sa_strides = tuple(int(value) for value in scale_a_i32.stride()[:2])
        payload = pack_pipeline_moe_kernargs(
            packer=self.packer,
            ptr_c=int(out.data_ptr()),
            ptr_a=int(a.data_ptr()),
            ptr_b=int(w.data_ptr()),
            ptr_scale_a=int(a_scales.data_ptr()),
            ptr_scale_b=int(scale_b_u8.data_ptr()),
            ptr_m_tile_map=int(m_tile_map.data_ptr()),
            c_shape=c_shape,
            c_strides=c_strides,
            sa_shape=sa_shape,
            sa_strides=sa_strides,
            sb_size0=int(scale_b_i32.shape[0]),
            i32_m=int(contiguous_m),
            i32_n=int(N),
            swiglu_limit=float(swiglu_limit),
            situ_beta=float(situ_beta),
            situ_linear_beta=float(situ_linear_beta),
        )
        grid = (
            ((int(contiguous_m) + int(tile_m) - 1) // int(tile_m))
            * ((int(N) + int(tile_n) - 1) // int(tile_n)),
            1,
            1,
        )

        if not self.debug_checked:
            fields = _unpack_kernarg_fields(payload, self.kernarg_layout)
            print(
                f"{_LOG_PREFIX} pipeline GEMM1 contract: "
                f"symbol={PIPELINE_TARGET_SYMBOL}; grid={grid}; "
                f"block={PIPELINE_BLOCK}; cluster={PIPELINE_CLUSTER}; "
                f"stream={int(current_stream.cuda_stream)}; "
                f"code_object={self.code_object}; "
                f"kernarg_size={len(payload)}; "
                f"kernarg_sha256={_sha256_bytes(payload)}",
                flush=True,
            )
            print(
                f"{_LOG_PREFIX} pipeline tensors: "
                + "; ".join(
                    (
                        _tensor_contract("out", out),
                        _tensor_contract("a", a),
                        _tensor_contract("b", w),
                        _tensor_contract("scale_a_u8", a_scales),
                        _tensor_contract("scale_a_i32_desc", scale_a_i32),
                        _tensor_contract("scale_b_i32_desc", scale_b_i32),
                        _tensor_contract("m_tile_map", m_tile_map),
                    )
                ),
                flush=True,
            )
            print(
                f"{_LOG_PREFIX} pipeline kernarg fields: {fields}",
                flush=True,
            )

            expected_psum = (
                self.torch.arange(
                    PIPELINE_EXPERTS,
                    dtype=self.torch.int32,
                    device=m_tile_map.device,
                )
                * PIPELINE_TILE_M
                + PIPELINE_VALID_ROWS_PER_EXPERT
            )
            if not bool(self.torch.equal(m_tile_map, expected_psum)):
                raise ValueError(
                    "C++ pipeline GEMM1 requires balanced psum "
                    "[32, 96, ..., 6112]"
                )
            # This diagnostic call is run by run_perftest's untimed memory probe.
            # A zero sentinel proves that the ISA leaves every padding row intact.
            out.zero_()

        result = self.extension.launch(
            out,
            a,
            w,
            a_scales,
            scale_b_u8,
            m_tile_map,
            a,
            out,
            payload,
            list(grid),
            list(PIPELINE_BLOCK),
            list(PIPELINE_CLUSTER),
            PIPELINE_TILE_M,
            PIPELINE_N,
            PIPELINE_K,
            PIPELINE_EXPERTS,
            str(self.code_object),
        )
        self.injected_calls += 1

        if not self.debug_checked:
            current_stream.synchronize()
            out_rows = out.view(PIPELINE_CONTIGUOUS_M, PIPELINE_N // 2)
            routed_rows = PIPELINE_EXPERTS * PIPELINE_TILE_M
            aligned_padding = out_rows[:routed_rows].view(
                PIPELINE_EXPERTS,
                PIPELINE_TILE_M,
                PIPELINE_N // 2,
            )[:, PIPELINE_VALID_ROWS_PER_EXPERT :, :]
            tail_padding = out_rows[routed_rows:]
            self.padding_nonzero = int(
                self.torch.count_nonzero(aligned_padding).item()
                + self.torch.count_nonzero(tail_padding).item()
            )
            if self.padding_nonzero != 0:
                raise AssertionError(
                    f"C++ GEMM1 wrote {self.padding_nonzero} nonzero padding values"
                )
            print(
                f"{_LOG_PREFIX} pipeline preflight: balanced psum matched; "
                f"padding_nonzero={self.padding_nonzero}; module loaded on "
                "the current PyTorch stream",
                flush=True,
            )
            self.debug_checked = True
        return result

    def assert_complete(self) -> None:
        if not self.debug_checked or self.injected_calls == 0:
            raise AssertionError("C++ pipeline GEMM1 adapter was never exercised")
        if self.injected_calls != self.delegated_calls:
            raise AssertionError(
                "pipeline launch counts differ: "
                f"gemm1_cpp={self.injected_calls}, "
                f"gemm2_flydsl={self.delegated_calls}"
            )
        print(
            f"{_LOG_PREFIX} pipeline adapter summary: "
            f"gemm1_cpp_calls={self.injected_calls}; "
            f"gemm2_flydsl_calls={self.delegated_calls}; "
            f"padding_nonzero={self.padding_nonzero}",
            flush=True,
        )


class MoePipelineGemm1Injection:
    """Temporarily install the stage-selective adapter in the grouped module."""

    def __init__(
        self,
        module: Any,
        original: Callable[..., Any],
        adapter: MoePipelineGemm1Adapter,
    ) -> None:
        self.module = module
        self.original = original
        self.adapter = adapter

    def __enter__(self) -> MoePipelineGemm1Adapter:
        current = self.module.flydsl_grouped_gemm_a8w4_masked
        if current is not self.original:
            raise RuntimeError("grouped GEMM launcher changed before C++ injection")
        self.module.flydsl_grouped_gemm_a8w4_masked = self.adapter
        return self.adapter

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.module.flydsl_grouped_gemm_a8w4_masked = self.original
        return False


def prepare_pipeline_gemm1_injection(
    *,
    torch_module: Any,
) -> MoePipelineGemm1Injection:
    """Build/load the cached backend before timing and return a scoped patch."""

    import logging

    aiter_python_logger = logging.getLogger("aiter")
    previous_aiter_log_level = aiter_python_logger.level
    try:
        try:
            from . import gemm_batch_isa_runner as batch
        except ImportError:
            import gemm_batch_isa_runner as batch
    finally:
        # The standalone CLI intentionally quiets aiter at import time. This
        # adapter is embedded in an existing benchmark, so preserve its logger.
        aiter_python_logger.setLevel(previous_aiter_log_level)

    from aiter.ops.flydsl import grouped_gemm_mxfp4 as grouped_launcher

    isa = batch.moe_cpp_target_isa().resolve()
    batch.validate_cpp_backend_target(
        isa,
        batch.MOE_GEMM1_WPT4_KERNEL_SYMBOL,
        "moe-gemm1",
    )
    # Resolve first to fail clearly, then preserve the configured path spelling
    # in the content-addressed manifest, matching the standalone runner.
    batch.single._resolve_clang(None)
    clang = batch.single.DEFAULT_CLANG.absolute()
    if batch.single._clang_uses_default_runtime_libraries(clang):
        batch.single._prepend_default_clang_runtime_libraries()
    artifacts = prepare_moe_cpp_backend(
        isa=isa,
        clang=clang,
        symbol=batch.MOE_GEMM1_WPT4_KERNEL_SYMBOL,
        single_module=batch.single,
        torch_module=torch_module,
    )
    if artifacts.extension.module.kernel_symbol() != PIPELINE_TARGET_SYMBOL:
        raise RuntimeError("loaded C++ extension reports the wrong kernel symbol")
    print(
        f"{_LOG_PREFIX} pipeline backend ready before timed callable: "
        f"co_key={artifacts.code.key}; "
        f"extension_key={artifacts.extension.key}; "
        f"co_cache_hit={artifacts.code.cache_hit}; "
        f"extension_cache_hit={artifacts.extension.cache_hit}; "
        f"co_sha256={artifacts.code.code_object_sha256}",
        flush=True,
    )
    original = grouped_launcher.flydsl_grouped_gemm_a8w4_masked
    adapter = MoePipelineGemm1Adapter(
        torch_module=torch_module,
        original_launcher=original,
        launcher_module=grouped_launcher,
        extension=artifacts.extension.module,
        code_object=artifacts.code.code_object,
        packer=batch.pack_moe_kernargs,
        kernarg_layout=batch.MOE_KERNARG_LAYOUT,
    )
    return MoePipelineGemm1Injection(grouped_launcher, original, adapter)
