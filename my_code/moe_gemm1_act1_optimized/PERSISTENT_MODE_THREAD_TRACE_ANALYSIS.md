# Double-LDS GEMM1 thread-trace and persistent-mode analysis

## Scope

Target kernel:

```text
my_code/moe_gemm1_act1_optimized/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_double_lds.s
SHA256=33034bb76ada72f9fbc895c1edbf56ee339d0ed6eb28d496196b68323ceed331
```

Machine and capture:

```text
host=d01-3
capture date=2026-09-10 Asia/Shanghai (2026-09-09 UTC)
ATT selected SIMD=0,1,2,3 in four independent captures
ATT target CU=1
data=const0
```

The full remote capture is stored at:

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/moe_gemm1_act1_optimized/att_threadtrace/d01_double_lds_20260910/
```

The compact local evidence under `att_threadtrace/d01_double_lds_20260910/`
contains the four instruction-stat CSV files, `occupancy.json`,
`realtime.json`, the kernel-trace log, and the existing ATT summary.

## Capture command

The double-LDS assembly was compiled for `gfx1250`, then captured with the
existing launcher and all four SIMD selections:

```bash
AITER_ATT_CODE_OBJECT=/tmp/codex_moe_gemm1_double_lds_att.co \
TRACE_ROOT=my_code/moe_gemm1_act1_optimized/att_threadtrace \
HIP_VISIBLE_DEVICES=0 \
bash my_code/get_isa_runner_att.sh \
  moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1 \
  d01_double_lds_20260910 \
  "python my_code/moe_gemm1_act1_optimized/att_launch_opt.py" \
  --all-simd --ana-att
```

All four captures completed successfully. The maximum measured dispatch span
was `1,195,699 GFXCLK cycles`, corresponding to `575.120 us` in the same ATT
capture. The REALTIME-weighted mean clock was `2061.361 MHz`.

## Phase boundaries

The phase boundaries use the instruction indices and actual virtual addresses
from the generated `code.json`. They do not use stale address comments in the
edited `.s` file.

| Phase | Instruction index | Actual PC | Contents |
|---|---:|---:|---|
| prologue | `0..4863` | `0x1900..0x71f8` | ABI loads, tile/expert addressing, descriptors, first TDM loads, barriers, accumulator initialization |
| hotloop | `4864..6382` | `0x71fc..0xac14` | eight path-specialized K loops, LDS operand loads and `v_wmma_scale_f32_32x16x128_f4` |
| epilogue | `6383..8290` | `0xac18..0xdba0` | SiLU(gate)*up, BF16 conversion, LDS staging, two `tensor_store_from_lds`, final drain |
| end | `8291` | `0xdba4` | `s_endpgm` |

An active traced wave contains `7,589` mapped dynamic instructions. An
early-exit wave contains `29` instructions.

## Phase proportions

The aggregate percentages use the decoder's attributed `Latency + Idle`
columns across all four SIMD captures. The median cycles come directly from
the instruction timestamps of 576 complete active-wave traces.

| Phase | Aggregate attributed cycles | Aggregate share | Stall / phase | Median active-wave cycles | Median-wave share |
|---|---:|---:|---:|---:|---:|
| prologue | `2,141,919` | `9.55%` | `63.49%` | `3,464` | `11.96%` |
| hotloop | `14,974,043` | `66.78%` | `24.13%` | `18,994.5` | `65.60%` |
| epilogue + end | `5,308,274` | `23.67%` | `44.97%` | `6,315.5` | `21.81%` |

The typical active-wave lifetime is `28,954 cycles`. The 10th-to-90th
percentile ranges are:

| Phase | P10 | Median | P90 |
|---|---:|---:|---:|
| entry | `430` | `436` | `442` |
| tile prologue after entry | `2,820` | `3,028` | `3,517` |
| hotloop | `17,664` | `18,994.5` | `20,722` |
| epilogue | `6,092` | `6,312.5` | `6,736` |

The mean values are higher because a small number of waves contain long cold or
system-interference outliers. The medians describe steady-state behavior more
reliably; the aggregate percentages preserve the decoder's complete accounting.

## Memory and synchronization stalls

Across the four captures, `Stall` accounts for `32.82%` of attributed kernel
cycles. Explicit memory waits plus cluster barriers and LDS-issue stalls account
for approximately `25.8%` of all attributed cycles. ATT cannot directly prove
whether a TDM wait was served by HBM, GL2, GL1, or WGP cache, so the table calls
these memory/synchronization stalls rather than HBM stalls.

| Phase | Main stall source | Share of that phase | Interpretation |
|---|---|---:|---|
| prologue | `s_wait_tensorcnt` | `33.95%` | waiting for the initial TDM global-to-LDS tiles |
| prologue | `s_barrier_wait` | `22.82%` | cluster peers wait for multicast/input readiness |
| prologue | LDS issue stalls | `2.48%` | LDS/TDM traffic shares the WGP-side memory path |
| hotloop | `s_barrier_wait` | `12.12%` | synchronization between cluster pipeline stages |
| hotloop | `s_wait_dscnt` | `5.39%` | operand `ds_load_b32`/`ds_load_b128` completion |
| hotloop | `s_wait_tensorcnt` | `1.08%` | next TDM tile is not completely hidden by WMMA work |
| hotloop | LDS issue stalls | `0.47%` | LDS issue/backpressure |
| epilogue | `s_wait_idle` | `22.44%` | chiefly drains the two output TDM stores |
| epilogue | VALU stalls | `14.17%` | SiLU EXP/RCP and dependent packed arithmetic |
| epilogue | `s_wait_dscnt` | `5.39%` | waits for BF16 output staging stores to LDS |
| epilogue | `s_barrier_wait` | `2.19%` | output-side workgroup synchronization |

The largest single stall site is the final `s_wait_idle` at instruction index
`8290`, PC `0xdba0`. Across the four captures it contributes `1,171,811`
stall cycles, or `5.23%` of all attributed cycles. The per-active-wave median
is `1,920.5 cycles` and P10/P90 are `1,759/2,209 cycles`. Its ATT dependency
list contains both `tensor_store_from_lds` operations and the output
`global_prefetch_b8`, confirming that this is an output-memory drain rather
than SiLU arithmetic latency.

The hotloop's largest stalls are cluster `s_barrier_wait 0xfffd`, followed by
`s_wait_dscnt` on the LDS operand loads. Persistent scheduling does not remove
these waits because every new tile still needs its K-loop TDM/LDS pipeline and
cluster synchronization.

## Persistent-mode upper bound

The occupancy trace reports:

- `45` sequential wave lifetimes per physical wave slot;
- one simultaneously active slot per physical SIMD for this kernel;
- `36` active waves and `9` early-exit waves per captured SE/SIMD sequence;
- median aggregate gap between consecutive wave lifetimes of `10,847.5 cycles`;
- completion imbalance median `0.000092` (`0.0092%`).

The full `320 KiB` LDS allocation permits only one resident workgroup per WGP.
The `4x4` cluster therefore occupies 16 WGPs on one shader engine. A valid
persistent implementation must allocate and advance work at cluster granularity
so that all 16 workgroups execute matching `s_barrier_signal -3` and
`s_barrier_wait 0xfffd` operations.

### Scheduler-only ideal

Assume a zero-cost persistent task scheduler, no loss of locality, and no
change to the per-tile prologue, hotloop, epilogue, or final output drain. It
can remove:

- inter-workgroup allocation gaps: median `10,847.5 cycles` per traced WGP run;
- nine early-exit workgroups: median `1,329 cycles`;
- repeated kernel-entry work for 35 of 36 useful tasks: median `15,230 cycles`.

Across the 16 independent SIMD-selection/SE observations, the ideal latency
reduction is:

```text
median = 2.34%
range  = 2.12% .. 3.05%
```

Applied to the current e2e profiler result of `554.139 us`, this is about
`541.2 us`, a `12.97 us` latency reduction and approximately `2.40%` throughput
speedup. A real task queue, task-ID broadcast, and loop control consume part of
this budget, so `2.34%` is an upper bound for scheduler-only persistence.

### Aggressive cross-iteration ideal

A persistent loop also creates an opportunity to start the next task's scalar
address work and accumulator initialization while the previous task's output
TDM store drains. If all final output waits except the last can be hidden, the
additional median removable amount is `68,378 cycles` per physical WGP trace.

The measured upper bound then becomes:

```text
median latency reduction = 8.26%
observed range           = 7.87% .. 9.56%
median throughput gain   = 9.00%
```

Mapped to the `554.139 us` e2e GEMM1 result, the central ideal is approximately
`508.4 us`; the best observed envelope fraction corresponds to approximately
`501.2 us`. Mapped to the ATT maximum span, the central estimate is about
`1.097 million GFXCLK cycles`, down from `1.196 million`.

This aggressive bound requires cross-iteration scheduling. The next tile must
perform useful register/scalar work before touching LDS regions still consumed
by the prior `tensor_store_from_lds`. TDM ordering and the shared SIMD-pair
VMEM/LDS path can prevent full overlap, so this is not the expected result of
merely adding a persistent task loop.

## Conclusion

Persistent scheduling by itself has a small ceiling: approximately `2%–3%`
latency reduction. With a carefully pipelined cross-tile output drain, the
ideal ceiling rises to approximately `8%–10%`; `8.26%` is the central estimate.

Persistent mode cannot close the full gap to the previously derived
`541,900.8-cycle` target. Even the aggressive estimate leaves approximately
`1.097 million cycles`, about `2.02x` that target. As an additional sanity
bound, the hotloop alone occupies about `65.6%–66.8%` of the trace, corresponding
to roughly `0.78–0.80 million cycles` even if all prologue and epilogue work
were unrealistically free. Further progress therefore requires reducing the
hotloop cluster-barrier, LDS dependency, and residual TDM waits in addition to
persistent scheduling.

## Implemented result

The follow-up implementation uses 16 physical 4x4 clusters and a fixed
cluster-task stride of 16. It was evaluated on d01-3 with the exact same MoE
e2e const0 process for all three versions:

| Version | GEMM1 | Relative to double-LDS |
|---|---:|---:|
| double-output-LDS | `557.354 us` | baseline |
| persistent with per-task full drain | `557.287 us` | `-0.012%` |
| persistent with cross-task output drain | `542.360 us` | `-2.690%` |

The near-zero scheduler-only change confirms that the trace's `2.34%` figure
was an upper bound rather than an expected gain: the saved dispatch gaps are
largely replaced by task-loop setup and barrier work. Moving the TENSORcnt wait
from the task boundary to the next task's first input TDM path realizes a
`2.690%` GEMM1 reduction. This captures part of the `8.26%` aggressive ideal;
the remaining gap is consistent with the independent setup window being much
shorter than the full output-drain stall and with the shared TDM/VMEM/LDS path
limiting overlap.

Correctness remained stable for random seeds 0, 1, and 2 with three repeated
standalone validations per seed (`err=0`), and in the complete MoE pipeline:

```text
random: logits_diff=3.3980e-06, rel_l2=2.6069e-03
const0: logits_diff=0, rel_l2=0
```

The final implementation and detailed logs are documented in `RESULTS.md`.

## Hardware references

- `mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming#65.txt:1445-1453`: WGP$/LDS and one TDM per SIMD-pair.
- `MI400_Shader_Programming#65.txt:1533-1539`: LDS and vector-memory instructions share the SIMD-pair bus and may stall each other.
- `MI400_Shader_Programming#65.txt:1699-1708`: workgroup-cluster placement and cluster-wide barriers.
- `MI400_Shader_Programming#65.txt:14080-14104`: asynchronous TDM movement, multicast, ordering, and `TENSORcnt` completion.
- `mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/system/PerformanceCounters.txt:8154-8226`: wave wait-state counter definitions.
- `PerformanceCounters.txt:8248-8306`: VMEM arbiter, instruction-issue, waitcnt, and barrier stall definitions.
