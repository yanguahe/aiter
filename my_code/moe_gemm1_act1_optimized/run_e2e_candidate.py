#!/usr/bin/env python3
"""Run the gfx1250 grouped-MoE test with an isolated GEMM1 ISA candidate."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SNAPSHOT_ROOT = HERE / "repo_snapshot"
DEFAULT_ISA = (
    HERE
    / "moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_opt.s"
)
TEST = SNAPSHOT_ROOT / "op_tests" / "test_flydsl_grouped_gemm_gfx1250.py"


def _audit_repository_module_paths() -> None:
    repo = REPO.resolve()
    allowed = (REPO / "my_code").resolve()
    seen: set[Path] = set()
    violations: list[Path] = []
    for module in tuple(sys.modules.values()):
        raw = getattr(module, "__file__", None)
        if not raw:
            continue
        path = Path(raw).resolve()
        # torch.library creates synthetic modules such as _ops.py and
        # _classes.py whose __file__ is relative to cwd but does not exist.
        if not path.exists():
            continue
        if path in seen:
            continue
        seen.add(path)
        if (path == repo or repo in path.parents) and not (
            path == allowed or allowed in path.parents
        ):
            violations.append(path)
    if violations:
        rendered = "\n".join(f"  {path}" for path in sorted(violations))
        raise RuntimeError(
            "repository modules resolved outside my_code:\n" + rendered
        )
    print(
        f"[run_e2e_candidate] repository module audit passed: "
        f"{sum(path == allowed or allowed in path.parents for path in seen)} "
        "files under my_code",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Patch the existing C++ GEMM1 injection in-process so the normal "
            "grouped-MoE e2e test loads an ISA from this experiment directory."
        )
    )
    parser.add_argument("--isa", type=Path, default=DEFAULT_ISA)
    parser.add_argument("--grid-x", type=int)
    parser.add_argument("--grid-y", type=int)
    args, test_args = parser.parse_known_args()
    if test_args[:1] == ["--"]:
        test_args = test_args[1:]

    isa = args.isa.resolve()
    if not isa.is_file():
        raise SystemExit(f"ISA does not exist: {isa}")
    if not (SNAPSHOT_ROOT / "aiter" / "__init__.py").is_file():
        raise SystemExit(
            "self-contained HEAD snapshot is missing; run on the host/local checkout: "
            f"{HERE / 'sync_head_repo_snapshot.py'}"
        )
    if not TEST.is_file():
        raise SystemExit(f"snapshotted e2e test does not exist: {TEST}")
    if (args.grid_x is None) != (args.grid_y is None):
        raise SystemExit("--grid-x and --grid-y must be provided together")
    persistent_grid = None
    if args.grid_x is not None:
        if args.grid_x <= 0 or args.grid_y <= 0:
            raise SystemExit("persistent grid dimensions must be positive")
        persistent_grid = (args.grid_x, args.grid_y, 1)

    # my_code stays at the repository root, while every non-my_code Python
    # dependency is resolved from the committed-HEAD snapshot below.
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(SNAPSHOT_ROOT))
    os.environ["AITER_META_DIR"] = str(SNAPSHOT_ROOT)
    from my_code.isa_runner import gemm_batch_isa_runner as batch
    from my_code.isa_runner import moe_cpp_backend

    import aiter

    aiter_path = Path(aiter.__file__).resolve()
    if SNAPSHOT_ROOT.resolve() not in aiter_path.parents:
        raise RuntimeError(
            f"aiter resolved outside the self-contained snapshot: {aiter_path}"
        )

    if persistent_grid is not None:
        moe_cpp_backend._CPP_SOURCE = (
            HERE / "moe_gemm1_cpp_launcher_persistent.cpp"
        )

    # moe_cpp_backend imports this same module and calls these values when it
    # prepares the scoped injection.  Override them only in this process; no
    # tracked production source or checked-in kernel is replaced.
    batch.MOE_CPP_ISA_BASENAME = isa.name
    batch.moe_cpp_target_isa = lambda: isa

    # The current grouped launcher forwards these two layout/scheduling facts.
    # The assembly adapter consumes tensors that are already preshuffled and
    # uses its fixed 1024-row/expert launch contract, so accept and validate the
    # facts here before calling the audited adapter implementation.
    original_call = moe_cpp_backend.MoePipelineGemm1Adapter.__call__

    class _GridOverrideExtension:
        def __init__(self, extension, grid):
            self._extension = extension
            self._grid = list(grid)

        def __getattr__(self, name):
            return getattr(self._extension, name)

        def launch(self, *launch_args, **launch_kwargs):
            launch_args = list(launch_args)
            # C++ launch signature: tensors[0:8], kernarg[8], grid[9], ...
            launch_args[9] = self._grid
            return self._extension.launch(*launch_args, **launch_kwargs)

    def call_with_layout_contract(self, *call_args, **call_kwargs):
        raw_a_preshuffle = call_kwargs.pop("a_preshuffle", None)
        raw_balanced_rows = call_kwargs.pop("balanced_rows_per_expert", None)
        if int(call_kwargs.get("stage1_act", 0)) == 0:
            return original_call(self, *call_args, **call_kwargs)
        # Some checked-in grouped-MoE revisions produce the preshuffled
        # payload but do not forward these two newer contract keywords.  The
        # exact E96/T16384 path below can infer both values unambiguously.
        a_preshuffle = (
            1 if raw_a_preshuffle is None else int(raw_a_preshuffle)
        )
        requested_m = int(call_kwargs.get("contiguous_m", 0))
        balanced_m = (
            moe_cpp_backend.PIPELINE_EXPERTS
            * moe_cpp_backend.PIPELINE_ROWS_PER_EXPERT
        )
        balanced_rows = (
            moe_cpp_backend.PIPELINE_ROWS_PER_EXPERT
            if raw_balanced_rows is None
            and requested_m
            in (balanced_m, moe_cpp_backend.PIPELINE_CONTIGUOUS_M)
            else int(raw_balanced_rows or 0)
        )
        if a_preshuffle != 1:
            raise ValueError(
                "isolated ASM GEMM1 requires a_preshuffle=1, "
                f"got {a_preshuffle}"
            )
        if balanced_rows != moe_cpp_backend.PIPELINE_ROWS_PER_EXPERT:
            raise ValueError(
                "isolated ASM GEMM1 requires balanced_rows_per_expert="
                f"{moe_cpp_backend.PIPELINE_ROWS_PER_EXPERT}, got {balanced_rows}"
            )
        # The current Python path narrows its FlyDSL launch domain to the
        # balanced 96*1024 valid rows.  This audited assembly keeps the older
        # 122880-row physical launch contract; its extra clusters read the
        # sentinel entries in m_tile_map and exit before touching payloads.
        if requested_m not in (
            balanced_m,
            moe_cpp_backend.PIPELINE_CONTIGUOUS_M,
        ):
            raise ValueError(
                "isolated ASM GEMM1 expected balanced or padded launch M in "
                f"({balanced_m}, {moe_cpp_backend.PIPELINE_CONTIGUOUS_M}), "
                f"got {requested_m}"
            )
        call_kwargs["contiguous_m"] = moe_cpp_backend.PIPELINE_CONTIGUOUS_M
        if persistent_grid is None:
            return original_call(self, *call_args, **call_kwargs)
        original_extension = self.extension
        self.extension = _GridOverrideExtension(
            original_extension, persistent_grid
        )
        try:
            return original_call(self, *call_args, **call_kwargs)
        finally:
            self.extension = original_extension

    moe_cpp_backend.MoePipelineGemm1Adapter.__call__ = call_with_layout_contract

    os.environ["AITER_MOE_GEMM1_LAUNCH_BACKEND"] = "cpp"
    print(f"[run_e2e_candidate] GEMM1 ISA override: {isa}", flush=True)
    if persistent_grid is not None:
        print(
            f"[run_e2e_candidate] persistent grid override: {persistent_grid}",
            flush=True,
        )
    sys.argv = [str(TEST), *test_args]
    runpy.run_path(str(TEST), run_name="__main__")
    _audit_repository_module_paths()


if __name__ == "__main__":
    main()
