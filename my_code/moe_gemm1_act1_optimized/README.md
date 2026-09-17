# Retained MoE GEMM1 optimization chain

This directory is the self-contained workspace for the gfx1250 E96/T16384
MoE GEMM1 assembly experiment. It retains the correctness-qualified optimization
lineage, selected diagnostic cases, and the runner/snapshot files required to
reproduce MoE e2e and ATT runs.

## Active cases

### `tensor_wait_after_wmma`

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_
static_state_hoist_descriptor_b64_tensor_wait_after_wmma.s
SHA256=02320b67ae67a8349f7942a31f89e3f2f3de6efe130e937d04d1d4bdb9273b2d
```

This is the current correctness-qualified all-`NT_RT` endpoint. It retains the
production ABI and numerical sequence while overlapping each K-ring
`s_wait_tensorcnt 0x2` with four independent WMMAs. Two clean nine-round a07-3
comparisons reproduced a `0.39-0.40%` GEMM1 improvement over `descriptor_b64`.

### `ab4_scale_half_tdm_full_setup_wait6`

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_
ab4_scale_half_tdm_full_setup_loop_wait6.s
SHA256=02dffd3a7f6ca015a25d52e8a2273c6cd49deed57bde4f0f758c1a97c7c70c06
```

This is the correctness-qualified resident 2+3, full-setup, wait6 kernel.

### `ab4_no_scale_tdm_wait4`

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_
ab4_no_scale_tdm_full_setup_loop_wait4.s
SHA256=bcf2675b551551462f9073e2936a25187da031321a9a649a8a88fb5234709393
```

This is a performance diagnostic. It removes the 56 ScaleA/ScaleB TDM loads
and retimes `s_wait_tensorcnt 0x6 -> 0x4` and `0x3 -> 0x2`. Const0 happens to
produce the reference zero output, but this does not establish random-data
correctness.

## Retained performance lineage

| case | assembly | reason retained |
|---|---|---|
| `baseline` | `baseline_act1_independent.s` | safe starting point |
| `optimized_v1` | `moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_opt.s` | multicast and packed/pipelined SiLU gain |
| `double_lds` | `moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_double_lds.s` | output-LDS overlap gain |
| `persistent` | `moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent.s` | structural parent of persistent overlap |
| `persistent_overlap` | `moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent_overlap.s` | overlaps task setup with output drain |
| `persistent_overlap_pad8` | `persistent_overlap_output_pad8.s` | output LDS bank-padding gain |
| `persistent_overlap_pad8_prefetch_stage0` | `persistent_overlap_pad8_prefetch_stage0.s` | next-task stage-0 prefetch gain |
| `persistent_overlap_pad8_prefetch_stage0_b64_clear` | `persistent_overlap_pad8_prefetch_stage0_b64_clear.s` | packed accumulator clear |
| `persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full` | `persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full.s` | full SQC instruction prefetch gain |
| `persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt` | `persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt.s` | accepted all-input `NT_RT` gain |
| `incremental_all_ptr` | `persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_incremental_all_ptr.s` | exact persistent tensor-pointer recurrence |
| `static_state_hoist` | `persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist.s` | hoists immutable per-wave state |
| `descriptor_b64` | `persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64.s` | compact next-task descriptor initialization |
| `tensor_wait_after_wmma` | `persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64_tensor_wait_after_wmma.s` | overlaps K-ring TDM completion with four WMMAs |
| `ab4_scale_half_tdm_full_setup_wait6` | active kernel above | retained correct 2+3 target |
| `ab4_no_scale_tdm_wait4` | diagnostic above | retained faster const0 endpoint |

Rejected probes, incorrect kernels, slower tail variants, raw historical ATT
directories, and old per-run logs were removed. Their conclusions remain in
the retained Markdown reports.

## Rebuild

Run from the repository root:

```bash
python my_code/moe_gemm1_act1_optimized/build_optimized.py
python my_code/moe_gemm1_act1_optimized/build_double_output_lds_variant.py
python my_code/moe_gemm1_act1_optimized/build_persistent_variants.py
python my_code/moe_gemm1_act1_optimized/build_persistent_overlap_pad8.py
python my_code/moe_gemm1_act1_optimized/build_next_task_prefetch_variants.py
python my_code/moe_gemm1_act1_optimized/build_b64_accum_clear_variant.py
python my_code/moe_gemm1_act1_optimized/build_b64_instruction_prefetch_variants.py
python my_code/moe_gemm1_act1_optimized/build_combined_tdm_hint_variants.py
python my_code/moe_gemm1_act1_optimized/build_incremental_all_ptr_variant.py
python my_code/moe_gemm1_act1_optimized/build_static_state_hoist_variant.py
python my_code/moe_gemm1_act1_optimized/build_descriptor_b64_variant.py
python my_code/moe_gemm1_act1_optimized/build_tensor_wait_after_wmma_variant.py
python my_code/moe_gemm1_act1_optimized/build_ab_quarter_scale_half_tdm_variant.py
python my_code/moe_gemm1_act1_optimized/build_no_scale_tdm_wait4_variant.py
```

Each generator checks the expected parent SHA256 before writing its retained
output. `SHA256SUMS` records the retained artifact hashes.

## Benchmark

The default `benchmark_history.sh` case list contains the correctness-qualified
retained lineage through `tensor_wait_after_wmma`. The resident 2+3 cases remain
explicitly selectable.

Compare the current endpoint with its immediate parent:

```bash
AITER_HISTORY_CASE_LIST=descriptor_b64,tensor_wait_after_wmma \
ROUNDS=9 RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-const0
```

Run random MoE e2e verification for the current endpoint:

```bash
AITER_HISTORY_CASE_LIST=tensor_wait_after_wmma \
ROUNDS=1 RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-random
```

Compare the two active cases with const0 input:

```bash
AITER_HISTORY_CASE_LIST=ab4_scale_half_tdm_full_setup_wait6,ab4_no_scale_tdm_wait4 \
ROUNDS=3 RUN_VERIFY=0 RUN_ATT=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-const0
```

Run only the correct wait6 case with random verification:

```bash
AITER_HISTORY_CASE_LIST=ab4_scale_half_tdm_full_setup_wait6 \
ROUNDS=3 RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-random
```

The canonical timing is the GEMM1 profiler row from the full MoE e2e run.
Performance numbers are valid only when all GPUs/KFD are idle before and after
the command. Correctness-only runs do not require an idle GPU.

## ATT

Validate the launch without tracing:

```bash
AITER_HISTORY_CASE_LIST=ab4_scale_half_tdm_full_setup_wait6 \
RUN_VERIFY=1 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh att-validate
```

Capture all four SIMD owner paths for either active case:

```bash
AITER_HISTORY_CASE_LIST=ab4_scale_half_tdm_full_setup_wait6 \
AITER_ATT_SIMD_LIST=0,1,2,3 \
RUN_VERIFY=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh att
```

Replace the case name with `ab4_no_scale_tdm_wait4` for the diagnostic.

The ATT path:

- uses the normal MoE e2e launcher;
- performs an untraced preflight build and launch;
- selects the known eighth GEMM1 invocation directly;
- captures SIMD0-3 sequentially with `att_shader_engine_mask=0x1`;
- isolates clang from rocprof injection;
- applies a hard timeout to each capture;
- verifies the `.att`, `code.json`, and wave JSON outputs;
- repairs output permissions on success or failure.

## Retained reports

- `RESULTS.md`: initial optimized-v1 result.
- `PERSISTENT_MODE_THREAD_TRACE_ANALYSIS.md`: persistent-mode analysis.
- `PERSISTENT_OVERLAP_PAD8_PORT_README.md` and
  `PERSISTENT_OVERLAP_PAD8_PORT_RESULTS.md`: output-pad8 implementation/results.
- `PERSISTENT_NEXT_TASK_INPUT_PREFETCH_ANALYSIS.md` and
  `PERSISTENT_NEXT_TASK_INPUT_PREFETCH_RESULTS.md`: stage-0 prefetch design/results.
- `PERSISTENT_OVERLAP_PAD8_PREFETCH_STAGE0_THREAD_TRACE_OPTIMIZATION_PLAN.md`:
  stage-0 trace-derived plan.
- `PERSISTENT_OVERLAP_PAD8_PREFETCH_STAGE0_PERFORMANCE_HISTORY.md`: accepted and
  rejected optimization history.
- `PERSISTENT_OVERLAP_PAD8_PREFETCH_STAGE0_ALL_NT_RT_THREAD_TRACE_OPTIMIZATION_PLAN.md`:
  four-SIMD ATT analysis and Experiments 1-28 leading to the retained endpoint.
- `RESIDENT_2PLUS3_FULL_SETUP_WAIT6_THREAD_TRACE_ANALYSIS.md`: wait6 all-SIMD
  bottleneck analysis.
- `RESIDENT_2PLUS3_NO_SCALE_TDM_WAIT4_THREAD_TRACE_ANALYSIS.md`: no-Scale-TDM
  performance and all-SIMD bottleneck analysis.

## Self-contained runtime dependencies

- `repo_snapshot/` contains the pinned repository payload used by the e2e
  runner.
- `repo_overlay/` contains the experiment-owned A/ScaleA preshuffle overlay.
- `sync_head_repo_snapshot.py` verifies or deliberately rebuilds that snapshot.
- `run_e2e_candidate.py`, `compare_asm_variants.py`, the runner copies, and
  `moe_gemm1_cpp_launcher_persistent.cpp` provide the isolated launch path.
