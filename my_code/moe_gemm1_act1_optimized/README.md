# Optimized MoE GEMM1 experiment

This directory is intentionally self-contained so it does not modify the
existing assembly kernel, runner, or `aiter/` package files while other work is
in progress.

## Contents

- `baseline_act1_independent.s`: known-correct independent-load baseline.
- `build_optimized.py`: restores the original 4x4 multicast descriptors and
  generates the packed/software-pipelined SiLU epilogue.
- `moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_opt.s`: generated
  optimized assembly. It deliberately retains the canonical kernel symbol so
  the isolated runner can use the audited 184-byte ABI/profile.
- `build_double_output_lds_variant.py`: derives the retained double-output-LDS
  implementation from the optimized-v1 assembly.
- `moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_double_lds.s`:
  retained output-LDS baseline with disjoint output staging.
- `build_persistent_variants.py`: derives the persistent and cross-task
  output-drain-overlap variants from the validated double-output-LDS kernel.
- `moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent.s`:
  cluster-granular persistent task loop with a safe full drain at every task
  boundary.
- `moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent_overlap.s`:
  final candidate; overlaps the prior output TDM drain with independent setup
  for the next logical tile and waits before the first input TDM/LDS reuse.
- `moe_gemm1_cpp_launcher_persistent.cpp`: isolated C++ launch adapter that
  accepts the persistent physical `grid=(16,16,1)` with `cluster=(4,4,1)`.
- `benchmark_persistent.sh`: one-command, same-machine e2e comparison of
  double-LDS, persistent, and persistent-overlap.
- `sync_head_repo_snapshot.py`: copies the complete tracked `aiter/` package,
  `csrc/` JIT sources, and the grouped-MoE e2e test from one committed revision
  into `repo_snapshot/`, with an aggregate tree digest and source commit
  manifest. An existing snapshot remains pinned unless `--commit` is given.
- `repo_snapshot/`: self-contained pinned-commit repository dependencies used
  by both history scripts. No Python source under the top-level `aiter/` or
  `op_tests/` trees is imported by these benchmark runs.
- `gemm_batch_isa_runner.py`, `gemm_isa_runner.py`: isolated copies of the
  validated runner used for testing older remote checkouts.
- `compare_asm_variants.py`, `run_e2e_candidate.py`: standalone comparison and
  full grouped-MoE e2e adapters for the three retained ISA versions.
- `test_optimized.sh`: quick correctness and formal benchmark commands.
- `benchmark_history.sh`: one-command comparison of the complete five-version
  chain: safe baseline, optimized v1, double-output-LDS, persistent full-drain,
  and persistent output-drain-overlap. It records the host, snapshot commit, clocks,
  SHA256 values, validation, and timings under `history_runs/`.
- `benchmark_att_history.sh`: builds the same history set and collects/analyzes
  one ATT capture for each version under a timestamped `att_history/` directory.
- `run_att_const0.sh`: collect and analyze one const0 ATT capture with
  `get_isa_runner_att.sh --ana-att`; trace output stays under this directory.
- `att_launch_opt.py`: minimal one-launch ATT target that loads the precompiled
  `act1_opt.co`, avoiding clang/COMGR execution inside rocprof.
- `HARDWARE_CYCLE_LIMIT.md`: derives the MXFP4 compute, memory/TDM/LDS, and
  exact-kernel instruction/occupancy cycle ceilings for the const0 workload.
- `PERSISTENT_MODE_THREAD_TRACE_ANALYSIS.md`: d01-3 all-SIMD ATT phase/stall
  breakdown and persistent-mode performance upper-bound estimate.
- `att_const0_analyze.log`: preserved `--ana-att` output used by the cycle-limit
  analysis.

## Regenerate

```bash
python my_code/moe_gemm1_act1_optimized/build_optimized.py
python my_code/moe_gemm1_act1_optimized/build_persistent_variants.py
python my_code/moe_gemm1_act1_optimized/sync_head_repo_snapshot.py --verify
python my_code/moe_gemm1_act1_optimized/sync_head_repo_snapshot.py
```

The first command verifies the pinned snapshot without reading Git. The second
recreates the same pinned commit recorded in `SOURCE_COMMIT`; it must run on the
host or local checkout, where Git is available. Only an explicit
`--commit <revision>` changes the pinned revision. Benchmark commands run inside
the `hyg_fyd1` container and never run Git.

## Test

```bash
bash my_code/moe_gemm1_act1_optimized/test_optimized.sh quick-random
bash my_code/moe_gemm1_act1_optimized/test_optimized.sh perf-random
bash my_code/moe_gemm1_act1_optimized/test_optimized.sh perf-const0
bash my_code/moe_gemm1_act1_optimized/run_att_const0.sh
bash my_code/moe_gemm1_act1_optimized/benchmark_persistent.sh e2e-const0
bash my_code/moe_gemm1_act1_optimized/benchmark_persistent.sh e2e-random
bash my_code/moe_gemm1_act1_optimized/benchmark_att_history.sh
AITER_ATT_VALIDATE_ONLY=1 \
  bash my_code/moe_gemm1_act1_optimized/benchmark_att_history.sh
```

After reconnecting to a machine or after any system reconfiguration, recreate
the local performance baseline before judging a new candidate:

```bash
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh perf-const0
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh perf-random
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-random
```

The canonical performance number for this task is the `gemm1` profiler row
from the full MoE const0 flow:

```bash
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-const0
```

This is also the script's default mode when no mode argument is supplied.
Standalone timing is retained only as a faster diagnostic and tuning signal.

The stable historical chain remains these three assembly versions:

1. `baseline_act1_independent.s`
2. `moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_opt.s`
3. `moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_double_lds.s`

The persistent experiment adds two generated descendants of item 3:

4. `moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent.s`
5. `moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent_overlap.s`

`benchmark_history.sh e2e-const0`, `e2e-random`, and `e2e-both` now run all
five versions automatically. The first three use the standard production grid;
the two persistent variants use `grid=(16,16,1)`. Standalone modes retain the
original three-version interleaved comparison and then run a second interleaved
comparison for the two persistent-grid variants.

`benchmark_att_history.sh` uses the identical five-version list. Its ATT launch
helper selects the standard grid for the first three versions and
`grid=(16,16,1)` for both persistent versions.
Set `AITER_ATT_VALIDATE_ONLY=1` to compile and launch all five code objects once
without collecting rocprof ATT data; this validates the version list, launch
geometry, snapshot imports, and kernel correctness path before a long trace run.

Both scripts export `PYTHONPATH` and `AITER_META_DIR` to `repo_snapshot/`.
Consequently, all repository Python/config/JIT-source dependencies are read
from paths below `my_code/moe_gemm1_act1_optimized/`; only system dependencies
such as PyTorch, FlyDSL, ROCm, clang, and rocprof remain external.

The current acceptance number is the `gemm1` profiler row produced by
`benchmark_persistent.sh e2e-const0`. The script records the host, current
clocks, ISA/launcher SHA256 values, exact test output, and all three timings in
one timestamped `history_runs/` log.

`perf-const0` first reproduces the exact historical command against
`moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_opt.s` (the version that
measured `515.603 us` on the earlier b8-3 configuration), then runs the
same-input interleaved comparison.  Set
`AITER_HISTORY_RUN_REFERENCE_COMMAND=0` to skip this duplicate measurement.

Add one experimental ISA to the same interleaved comparison without editing
the script:

```bash
AITER_HISTORY_CANDIDATE=my_code/moe_gemm1_act1_optimized/candidate.s \
  bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh perf-const0
```

Collect comparable GFXCLK-cycle traces for the stable history set:

```bash
bash my_code/moe_gemm1_act1_optimized/benchmark_att_history.sh
```

`test_optimized.sh` also accepts `AITER_OPT_ISA=/path/to/candidate.s`, so its
standalone and e2e modes can be reused without replacing the default kernel.

The optimized epilogue swaps each `G0,U0,G1,U1` accumulator group into
contiguous gate/up pairs, uses `v_pk_mul_f32` and `v_dual_*`, and pipelines the
next eight-output batch through the current batch's `EXP/RCP` latency slots.
The first `tensor_store_from_lds` remains followed by `s_wait_tensorcnt 0x0`
before LDS reuse.
