# Persistent next-task input prefetch implementation and results

## Scope

This change implements the two variants proposed in
`PERSISTENT_NEXT_TASK_INPUT_PREFETCH_ANALYSIS.md` on top of
`persistent_overlap_output_pad8.s`:

- Version A: `persistent_overlap_pad8_prefetch_stage0.s`
  - Prefetches the next persistent task's K256 stage 0.
- Version B: `persistent_overlap_pad8_prefetch_stage01.s`
  - Prefetches the next persistent task's K256 stages 0 and 1.

The implementation and all supporting changes remain under
`my_code/moe_gemm1_act1_optimized/`. No source under `aiter/` or `op_tests/`
was modified.

## Implementation

The prefetch is issued after the current task's final hotloop cluster
synchronization, `s_wait_idle`, output descriptor construction, and output LDS
address materialization, but before the first SiLU instruction. At that point
the accumulator VGPRs are still live, so the implementation uses only SGPR
descriptor state and writes the prefetched payload directly to LDS.

The persistent stride is fixed at 16 logical cluster tasks. For this schedule:

- local M remains unchanged;
- when `task_id % 24 < 8`, N advances by 16 in the same expert;
- otherwise N wraps by -8 and the expert advances by four.

The early path updates the complete expert-local C/A/B/ScaleA/ScaleB scalar
state and materializes the four tensor bases for the next task. The following
task enters through `.Lmoe_persistent_task_prefetched` and reuses this state,
avoiding duplicate kernarg loads, task swizzle, expert rebasing, tensor-base
calculation, and stage-0 descriptor construction.

The original descriptor-building instruction sequences are retained. An early
constant-folded descriptor prototype compiled but failed random-data
validation, so it was discarded. Reusing the original sequences avoids relying
on undocumented assumptions about packed TDM descriptor fields.

Version A uses this per-wave TDM order across the task boundary:

```text
I0(next), O0(current), O1(current), I1(next), I2(next), ...
```

At the next task entry, `s_wait_tensorcnt 0x2` establishes stage-0 readiness.
Before stage 2 overwrites the output-overlapping LDS banks,
`s_wait_tensorcnt 0x1` drains both old output stores while allowing stage 1 to
remain outstanding. The maximum intended cross-boundary window is three TDM
operations per wave.

Version B uses:

```text
I0(next), I1(next), O0(current), O1(current), I2(next), ...
```

At the next task entry, `s_wait_tensorcnt 0x3` establishes stage-0 readiness.
Both original stage-0 and stage-1 loads are skipped. Before stage 2 reuses the
overlapping LDS banks, `s_wait_tensorcnt 0x0` drains the two old output stores.
The maximum intended cross-boundary window is four TDM operations per wave,
below the task-specific limit of 10 requested for this experiment.

Both variants keep:

- `grid=(16,16,1)`, `cluster=(4,4,1)`, and `block=(128,1,1)`;
- the 184-byte production ABI;
- 320 KiB LDS;
- `.amdhsa_next_free_vgpr 1024`;
- `.amdhsa_next_free_sgpr 104`;
- the existing 4x4 multicast and cluster-barrier protocol;
- the output-pad8 SiLU epilogue and double-output-LDS layout.

`s101`, which is unused by the baseline kernel, records whether the current
task entered with prefetched input state. No new numbered SGPR or VGPR is
allocated.

## Files

- `build_next_task_prefetch_variants.py`: deterministic generator for A and B.
- `audit_next_task_prefetch_variants.py`: static ABI/resource/TDM/barrier audit.
- `persistent_overlap_pad8_prefetch_stage0.s`: version A.
- `persistent_overlap_pad8_prefetch_stage01.s`: version B.
- `benchmark_history.sh`: version A is a named benchmark case. Version B is
  retained as an experimental artifact and can be run through the generic
  `AITER_HISTORY_CANDIDATE` path.

## Reproduction commands

Regenerate both assembly files from the retained output-pad8 baseline:

```bash
python3 my_code/moe_gemm1_act1_optimized/build_next_task_prefetch_variants.py
```

Run the static contract audit. This checks the ABI, LDS allocation, register
metadata, TDM counts, wait protocol markers, unique labels, and the requested
per-wave in-flight window bound:

```bash
python3 my_code/moe_gemm1_act1_optimized/audit_next_task_prefetch_variants.py
```

Run standalone random correctness for both variants using the persistent
`16x16` physical grid:

```bash
python3 my_code/moe_gemm1_act1_optimized/compare_asm_variants.py \
  my_code/moe_gemm1_act1_optimized/persistent_overlap_pad8_prefetch_stage0.s \
  my_code/moe_gemm1_act1_optimized/persistent_overlap_pad8_prefetch_stage01.s \
  --seed 0 --warmup 1 --rounds 3 --launches-per-sample 1 \
  --validation-repeats 3 --grid-x 16 --grid-y 16
```

Run full MoE random correctness with each assembly replacing the FlyDSL GEMM1
inside the normal e2e flow:

```bash
AITER_HISTORY_CASE_LIST=persistent_overlap_pad8_prefetch_stage0,candidate \
AITER_HISTORY_CANDIDATE=my_code/moe_gemm1_act1_optimized/persistent_overlap_pad8_prefetch_stage01.s \
AITER_HISTORY_CANDIDATE_GRID_X=16 AITER_HISTORY_CANDIDATE_GRID_Y=16 \
ROUNDS=1 RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-random
```

Run the canonical same-machine const0 performance comparison against the
output-pad8 baseline:

```bash
AITER_HISTORY_CASE_LIST=persistent_overlap_pad8,persistent_overlap_pad8_prefetch_stage0,candidate \
AITER_HISTORY_CANDIDATE=my_code/moe_gemm1_act1_optimized/persistent_overlap_pad8_prefetch_stage01.s \
AITER_HISTORY_CANDIDATE_GRID_X=16 AITER_HISTORY_CANDIDATE_GRID_Y=16 \
ROUNDS=3 RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-const0
```

Run the corresponding random-data comparison:

```bash
AITER_HISTORY_CASE_LIST=persistent_overlap_pad8,persistent_overlap_pad8_prefetch_stage0,candidate \
AITER_HISTORY_CANDIDATE=my_code/moe_gemm1_act1_optimized/persistent_overlap_pad8_prefetch_stage01.s \
AITER_HISTORY_CANDIDATE_GRID_X=16 AITER_HISTORY_CANDIDATE_GRID_Y=16 \
ROUNDS=3 RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-random
```

Collect and analyze ATT for the two new variants:

```bash
AITER_HISTORY_CASE_LIST=persistent_overlap_pad8_prefetch_stage0,candidate \
AITER_HISTORY_CANDIDATE=my_code/moe_gemm1_act1_optimized/persistent_overlap_pad8_prefetch_stage01.s \
AITER_HISTORY_CANDIDATE_GRID_X=16 AITER_HISTORY_CANDIDATE_GRID_Y=16 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh att
```

Before and after each GPU test on d01-3, check GPU/KFD ownership with:

```bash
bash /data/yanguahe/code/gpu_users.sh
```

## Correctness

Final random-data validation on d01-3:

- Both versions passed three consecutive standalone launches with
  `err=0` and `rel_l2=9.367674611323901e-04`.
- Both versions passed the full MoE e2e flow with
  `logits_diff=3.3980e-06`, `rel_l2=2.6069e-03`, and identical output SHA256:
  `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2`.
- Both versions passed const0 with `logits_diff=0`, `rel_l2=0`, and output
  SHA256 `bc3404a6147c932d6ff7c322658fc3258b7a0ca1b81adf2a381f73ed1348020e`.

## d01-3 performance

The canonical three-round interleaved MoE e2e const0 run is stored remotely at:

```text
my_code/moe_gemm1_act1_optimized/history_runs/
heliosr-1b114-d01-3_20260912T200916Z_e2e-const0/
```

| Version | GEMM1 samples | GEMM1 median | Gain vs output-pad8 | Fused MoE median | Correctness |
|---|---|---:|---:|---:|---|
| `persistent_overlap_pad8` | `542.509, 539.985, 539.652 us` | `539.985 us` | baseline | `1778.07 us` | exact const0 |
| Version A, stage 0 | `524.558, 521.380, 521.134 us` | `521.380 us` | `3.45%` | `1762.62 us` | exact const0 |
| Version B, stage 0+1 | `519.797, 523.569, 518.037 us` | `519.797 us` | `3.74%` | `1762.78 us` | exact const0 |

The final three-round interleaved random comparison is stored remotely at:

```text
my_code/moe_gemm1_act1_optimized/history_runs/
heliosr-1b114-d01-3_20260912T201410Z_e2e-random/
```

| Version | GEMM1 samples | GEMM1 median | Gain vs output-pad8 | Fused MoE median | Correctness |
|---|---|---:|---:|---:|---|
| `persistent_overlap_pad8` | `658.978, 660.315, 661.574 us` | `660.315 us` | baseline | `2085.91 us` | pass |
| Version A, stage 0 | `644.494, 640.804, 639.273 us` | `640.804 us` | `2.95%` | `2080.21 us` | pass |
| Version B, stage 0+1 | `635.564, 634.527, 639.164 us` | `635.564 us` | `3.75%` | `2062.96 us` | pass |

## ATT cycle measurements

The baseline trace was collected in:

```text
history_runs/heliosr-1b114-d01-3_20260912T200038Z_att/
```

The final A/B traces were collected in:

```text
history_runs/heliosr-1b114-d01-3_20260912T201043Z_att/
```

Using the maximum `occupancy.json` shader timestamp span across the captured
SEs:

| Version | Max GFXCLK cycles | Change vs output-pad8 | Matched wall span |
|---|---:|---:|---:|
| `persistent_overlap_pad8` | `1,089,335` | baseline | `534.040 us` |
| Version A, stage 0 | `1,062,649` | `-26,686` (`-2.45%`) | `521.800 us` |
| Version B, stage 0+1 | `1,074,879` | `-14,456` (`-1.33%`) | `525.920 us` |

Version B has the best median e2e profiler time in the const0 comparison, while
version A has the lowest captured GFXCLK cycle span. The difference is small
enough that both variants remain useful historical benchmark points on these
machines with variable clock and platform behavior.

## Final SHA256

```text
c96f89fccf4ee385d4b3ce5f54a8d0c5efee4f6c05ff90f3ce4e271ff7a51796  persistent_overlap_pad8_prefetch_stage0.s
b7b119219daf069a826b94ea9088f6ab851a788cfae4cf727581c5ac1829f2b6  persistent_overlap_pad8_prefetch_stage01.s
```
