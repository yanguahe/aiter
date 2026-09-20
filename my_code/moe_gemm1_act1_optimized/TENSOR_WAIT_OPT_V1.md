# tensor_wait_after_wmma optimization v1: scale LDS request fusion

Started 2026-09-18 (Asia/Shanghai), after completing
[V0](TENSOR_WAIT_OPT_V0.md). All GPU work remains on a07-3.

## Parent and motivation

V0 produced no reliably faster retained kernel. The parent remains:

```text
case=tensor_wait_after_wmma
ISA=persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64_tensor_wait_after_wmma.s
SHA256=02320b67ae67a8349f7942a31f89e3f2f3de6efe130e937d04d1d4bdb9273b2d
```

V0 reduced some local waits, but other DScnt/barrier costs absorbed the saving.
V1 tests the LDS instruction/request representation while keeping every input
byte, output byte, WMMA/SiLU arithmetic instruction, and arithmetic order intact.
The 184-byte ABI, 16x16 persistent grid, 4x4 cluster, LDS allocation, and TDM
multicast protocol are preserved.

## V1-A: fuse low-offset scale B32 pairs

Static inspection found 16 adjacent scale-load pairs in the two hotloop
families whose offsets are directly representable by `DS_LOAD_2ADDR_B32`.
For example:

```asm
ds_load_b32 v96, v81 offset:512
ds_load_b32 v97, v81 offset:640
```

becomes:

```asm
ds_load_2addr_b32 v[96:97], v81 offset0:128 offset1:160
```

CDNA5 ISA section 11.2, VDS instruction fields and double-address instructions
(local text lines 10743-10758 and 10820-10826), specifies two unsigned 8-bit
offsets scaled by four for B32. Therefore each byte offset must be divisible
by four and at most 1020. Pairs with larger offsets are excluded, avoiding
additional address VGPRs or address arithmetic.

Each selected pair has consecutive destination VGPRs and the same address
VGPR/bank state. Expanding the new instruction into its two B32 reads must
exactly reproduce the original ordered memory-access list. The instruction
count changes, but the data transferred and register contents do not.

### DScnt retirement contract

A new paired instruction represents two old DS instructions. For each
affected wait, translate the permitted outstanding suffix from old instruction
IDs into complete new request groups. A pair crossing the old retirement
frontier must complete; it cannot be counted as wholly outstanding.

For this parent, the expected changes are:

- four `wait20 -> wait18` sites after a fused scale group and sixteen payload
  loads;
- one `wait10 -> wait8` site after a fused scale group and six payload loads;
- one `wait4 -> wait2` site directly after the other family's fused scale group;
- existing payload-tail `wait8`, other `wait4` sites, and full drains remain unchanged.

The generator must derive these counts and reject any unexpected site or
retirement mapping. No TENSORcnt threshold or barrier is changed.

Expected evidence: fewer LDS instructions/requests, with reduced LDS issue or
DScnt exposure. Possible failure: the paired request completes only when both
addresses are ready, or its service cost cancels the instruction saving.

## V1-B: remove overwritten bank resets at hotloop entry

Eight hotloop-entry sites have:

```text
s_set_vgpr_msb 0
s_wait_dscnt 0x8
s_set_vgpr_msb <complete next setting>
first vector instruction
```

There is no vector instruction between the two settings. Test removing only
the first setting. The same wait remains immediately before the complete
next bank setting, and all subsequent vector operands see the same bank bits.

The ISA definition of `S_SET_VGPR_MSB` (section 15.5; local text around
19514-19543) sets all four two-bit operand-bank fields for subsequent vector
instructions. `s_wait_dscnt` is a scalar dependency-counter operation. The
candidate must preserve every arithmetic/memory/barrier instruction and wait
value and match the expected eight-site pattern exactly.

The maximum direct instruction saving is small. The experiment may simply
expose one more cycle of an existing LDS wait, yielding no net benefit. It is
included as a bounded, directly auditable test, not an assumed speedup.

## Validation and retention

- Copy the currently accepted parent into a distinct new ISA for each idea.
- Verify unchanged arithmetic order, expanded memory semantics, metadata,
  and the applicable counter-retirement invariants before GPU launch.
- Run repeated random MoE e2e and compare both output hashes and numerical
  metrics with the matching parent; const0 must match exactly.
- Use full MoE e2e GEMM1 profiler timing only after an all-GPU/KFD idle check.
  Mark any overlapping external workload invalid, stop only this task's run,
  and retry in an idle window.
- Capture parent/candidate ATT for every runnable idea. Use the canonical
  `trace_segment_cycles.py` through the existing complete-wave analyzer;
  require all WMMA/EXP/RCP counts and 36 task boundaries to be present.
- Record expected and unexpected effects here, including first task, all
  35 body intervals, full-wave totals, and relevant wait families. Do not
  infer wall-time gains by adding independently sampled medians.
- Confirm possible small improvements with alternating multi-round runs.
  Add only retained improvements to `benchmark_history.sh`. Remove rejected
  candidate ISA and dedicated code after retaining their evidence.

If V1-A is accepted, V1-B is derived from it; otherwise V1-B uses the original
parent. After V1, remaining measured bottlenecks determine whether another
new plan has a justified experiment.

## Execution record

Planning and static eligibility inspection are complete. GPU results will be
recorded below as each candidate is evaluated.

The first retirement audit distinguished the two family-specific sites:
the A-side waits after four scale reads plus six payload reads, while the
B-side wait is directly after four scale reads. Consequently their translated
counts are 10->8 and 4->2 respectively. These are derived separately; no global
wait-value replacement is used.

### 2026-09-18 restart and baseline remeasurement

The host rebooted at 08:20:11 UTC. The container had originally started before
the GPU driver initialized, leaving its private `/dev` without `/dev/kfd` and
the GPU render nodes. At this continuation, `hyg_fyd1` reports a new start time
of 09:31:16 UTC; `/dev/kfd`, the render nodes, and `rocminfo` enumeration of
`gfx1250` are restored. No driver, container, or GPU reset was performed by
this optimization continuation.

The user requested a fresh performance starting point before further GPU
optimization experiments. Earlier V0 timing is retained as historical
evidence only. A new nine-round, 20-iteration const0 measurement of the
unchanged `tensor_wait_after_wmma` is planned before V1-A GPU evaluation;
every later candidate also gets a same-session interleaved parent comparison.

Local/remote parent, V1-A, and `benchmark_history.sh` SHA256 values match.
The fixed snapshot verifies at `23c2caaafa5f1c6e6d5d9f756980fe004af4202c`,
with 1,854 payload files and tree SHA256
`d2e5c94ee4ee98997c72e3138ef26d0ac08aa5af18ef2acb9fa33d7a2cbb129d6`.
The unrelated production-package modifications are preserved, and the
snapshot has not been refreshed.

Initial idle checks found external GPU jobs in `xiangxli_1250_0911` and later
`xudong_dsl_450_att`. No performance run was started during those checks.
Host `rocm-smi` also aborted during GPU-metrics conversion, so the gate uses
`gpu_users.sh`, privileged `fuser`, KFD debugfs process entries, all available
DRM-device utilization/VRAM sysfs counters, and explicit host/container
device-node checks. The observed idle allocation remains 173,154,304 bytes;
unexplained higher VRAM usage is not accepted as idle.

New continuation evidence is stored under
`history_runs/tensor_wait_opt_rebaseline_20260918/`.

### Fresh performance starting point

The all-GPU/KFD idle gate passed before and after the nine-round baseline.
No external GPU owner was observed during the intermediate checks. The run
used 20 e2e iterations per round (19 measured GEMM1 instances), the original
ISA, the fixed snapshot, and the unchanged `three_kernel` producer.

Run: `heliosr-1b114-a07-3_20260918T095155Z_e2e-const0`.

| Metric | Samples, us | Median, us |
|---|---|---:|
| GEMM1 profiler | 479.957, 481.382, 480.494, 481.484, 481.579, 481.922, 480.077, 480.652, 480.193 | **480.652** |
| MoE e2e | 1309.67, 1312.07, 1310.04, 1305.63, 1305.95, 1309.51, 1307.38, 1306.40, 1311.91 | **1309.51** |

All nine const0 runs passed with zero numerical error and identical
MoE/reference hash `21291d9023c8af8a6324fe20f346a967`. The parent source has
not changed. The difference from the pre-reboot 507-512 us measurements is
an environment-era difference, **not an ISA optimization gain**. Candidate
decisions below use fresh interleaved parent/candidate runs.

V1-A also assembled and linked successfully with the pinned gfx1250 clang.
The link command emitted the existing non-fatal `-shared` unused-argument
warning and exited zero. This is only a CPU compilation check; it does not
establish GPU correctness or performance.

### V1-A repeated random correctness

Run: `heliosr-1b114-a07-3_20260918T095408Z_e2e-random`.
All three repetitions of both parent and candidate passed with identical:

```text
logits_diff=3.39799e-06
rel_l2=0.00260689
MoE hash=1556fc617347e2dabc9cff19dbfd822b
reference hash=1a5d22911ba167160b4f2c12092a5193
```

The random run is a correctness result only. An external job appeared during
this period, and its reported timing is not used for performance selection.
The following performance gate at 09:55:40 UTC initially found that external
job, but it exited before the later `fuser`/sysfs checks. Those later checks
passed, and a timing run did start at 09:55:41 UTC. This corrects the initial
progress interpretation made before the entire guard had returned.

The resulting run (`heliosr-1b114-a07-3_20260918T095541Z_e2e-const0`, raw log
`v1a_perf.log`) is **excluded from performance selection** because the complete
precheck was not consistently idle. Its correctness results remain valid.
This is conservative invalidation of an inconsistent precheck, not a claim
that concurrent execution was proved for every sample. The guard now also
requires `gpu_users.sh` itself to report no GPU process before proceeding to
the independent KFD/sysfs checks. No external process was signaled, and the
completed run left no test process behind.

### V1-A first valid performance screen

Run: `heliosr-1b114-a07-3_20260918T100116Z_e2e-const0`.
All three rounds passed exact const0 correctness.

| Case | GEMM1 samples, us | GEMM1 median | MoE median |
|---|---|---:|---:|
| parent | 480.093, 480.057, 480.403 | 480.093 us | 1312.92 us |
| V1-A | 481.004, 480.091, 479.991 | 480.091 us | 1312.51 us |

The separate-median difference is only -0.002 us. V1-A wins 1/3 paired rounds;
paired median delta is +0.034 us and paired mean delta is +0.178 us
(`candidate - parent`). This screen provides no evidence of a gain.

The strict precheck passed; no external KFD owner was observed at the
intermediate check. The benchmark finished at 10:01:56.502 UTC. A later
postcheck at 10:02:35 found a new external job approximately two seconds old,
so that observed job started after this benchmark had completed. No process
from this test remained. A new fully idle check passed at 10:04:03 before
starting the required parent/candidate four-SIMD ATT comparison.

### V1-A first nine-round confirmation

Run: `heliosr-1b114-a07-3_20260918T100731Z_e2e-const0`.
The initial case order is candidate then parent, reversed in even rounds.
The strict all-GPU precheck and the postcheck both passed. Intermediate KFD
checks did not identify an external GPU owner. The CPU-only trace analyzers
were also running for part of this interval; they did not open GPU devices.

| Case | GEMM1 samples, us | GEMM1 median | MoE median |
|---|---|---:|---:|
| parent | 481.137, 479.891, 481.183, 482.684, 479.372, 481.826, 480.953, 480.124, 479.905 | 480.953 us | 1314.45 us |
| V1-A | 480.616, 480.772, 479.566, 481.573, 479.764, 479.732, 480.192, 480.464, 480.090 | 480.192 us | 1312.24 us |

The separate medians suggest a 0.158% latency reduction. Paired median delta
is -0.521 us, paired mean delta is -0.478 us, and the candidate wins 5/9
rounds. All correctness checks pass. A second nine-round confirmation with
the opposite initial case order is needed before treating this small
difference as repeatable.

### First V1-A ATT pair: incomplete, excluded

The capture at `heliosr-1b114-a07-3_20260918T100404Z_att` completed all eight
capture commands successfully and produced four nonempty ATT files, one
`code.json`, and four wave JSONs per SIMD-select. The occupancy analyzer ran.
However, successful capture/decode return codes did not establish complete
instruction traces:

| Case / wave | Task boundaries | WMMA | EXP / RCP |
|---|---:|---:|---:|
| parent SIMD2 / SE2 | 7 | 13,574 | 1,792 / 1,792 |
| V1-A SIMD0 / SE2 | 19 | 34,290 | 4,864 / 4,864 |
| required per wave | 36 | 64,512 | 9,216 / 9,216 |

The other 15 waves of each case pass those counts. The affected waves still
have full-looking timestamp envelopes; dropping only a tail region or
loosening the task-count check would not repair their decoded instruction
stream. Both complete-case cycle comparisons are rejected. The raw ATT files
are about 1.6 MiB per SE, far below the configured 256 MiB buffer; no explicit
overflow/lost-data error was found in the capture logs. The precise decoder
or trace-packet cause is not established.

The new `audit_tensor_wait_capture_integrity.py` reports every wave, including
all incomplete ones, without accepting partial traces. Evidence:
`v1a_integrity_tensor_wait_after_wmma.json`, `v1a_integrity_candidate.json`,
and the two failed strict-analysis logs under the continuation evidence root.
Fresh capture and the unchanged strict checks are required before drawing any
cycle-budget conclusion for V1-A.

### V1-A further confirmation and performance decision

Two additional strict-idle nine-round runs used opposite initial case orders.
All const0 checks passed and postchecks found no residual test process.

| Run | Parent / candidate median, us | Paired median delta | Paired mean delta | Candidate wins | MoE parent / candidate, us |
|---|---:|---:|---:|---:|---:|
| `20260918T101150Z` | 480.353 / 480.150 | -0.494 us | -0.513 us | 8/9 | 1313.20 / 1313.37 |
| `20260918T102031Z` | 480.475 / 480.413 | +0.272 us | +0.531 us | 4/9 | 1313.38 / 1313.79 |

Together with the first valid screen and confirmation, there are 30 valid
paired observations (the inconsistent-precheck run is excluded):

```text
pooled parent median=480.378 us
pooled candidate median=480.285 us
paired median delta=-0.229 us
paired mean delta=-0.120467 us
candidate wins=18/30
```

The small independent-median reduction is only about 0.019%. Paired mean
changes sign in the last repeat, and the pooled mean saving is weak relative
to the variation. V1-A is **not promoted as a reliable performance gain**.
This is not a claim that every sample regressed. Its required complete-trace
comparison is still being completed before final cleanup.

The identical-config ATT repeat at `20260918T101620Z` has a complete candidate
(all 16 waves) but an incomplete parent: parent SIMD0/SE2 has 0 boundaries and
836 WMMAs, SIMD1/SE2 has 16 boundaries and 29,170 WMMAs, and SIMD1/SE3 has
64,511 rather than 64,512 WMMAs. The incomplete parent is excluded. A new
paired capture changes only `AITER_ATT_TARGET_CU` from 1 to 0 for **both**
cases; the kernel, mask `0xf`, SIMD list, and completeness conditions are
unchanged. This is a change in the sampled instruction-trace location, not a
kernel optimization or a claimed fix to the decoder.

### V1-B preparation

Since V1-A did not earn promotion, V1-B was copied from the unchanged accepted
`tensor_wait_after_wmma`, not from V1-A:

```text
ISA=tensor_wait_v1_dead_bank_reset.s
SHA256=239bc84b2fd604f8f56db7fe838774daf10c17f38cd32b90db05994878add697
removed static overwritten bank resets=8
```

The generator's arithmetic/memory/bank/barrier/metadata checks pass. GPU
validation will follow the active V1-A capture; this preparation is not yet a
performance result.

### V1-A complete CU0 trace comparison and cleanup

The paired CU0 capture at `heliosr-1b114-a07-3_20260918T102310Z_att` passes
every completeness gate for both cases: 16 instruction waves, 36 task
boundaries, 64,512 WMMAs, and 9,216 each of EXP and RCP per wave. All eight
canonical normal/representative analyses per case completed. Independent
aggregation of all four statistics CSVs exactly matches every decoded
whole-wave wait-family total. The occupancy analysis uses
`my_code/analyze_att_capture.py`.

| Metric, cycles per complete wave unless stated | Parent | V1-A | Change |
|---|---:|---:|---:|
| Entire decoded wave | 1,035,344.875 | 1,034,745.500 | -0.058% |
| Initial setup + first task | 33,266.125 | 32,525.813 | -2.23% |
| All following 35 tasks | 999,839.188 | 998,553.875 | -0.129% |
| Final boundary/drain | 2,239.563 | 3,665.813 | +63.68% |
| Mean consecutive body task | 28,566.834 | 28,530.111 | -0.129% |
| DScnt stall | 92,924.250 (8.98%) | 89,990.188 (8.70%) | -3.16% |
| Barrier stall | 176,007.563 (17.00%) | 176,912.938 (17.10%) | +0.51% |
| TENSORcnt stall | 35,959.813 (3.47%) | 39,179.813 (3.79%) | +8.95% |
| `s_wait_idle` stall | 4,985.813 (0.48%) | 7,529.875 (0.73%) | +51.03% |
| LDS-read issue-attributed timeline | 122,683.875 | 121,198.000 | -1.21% |

Percentages in parentheses use each case's complete-wave denominator.
First/body/drain closes exactly, including all 35 adjacent intervals. These
are per-wave observations, not additive wall-time savings across GPU waves.
Mean capture clocks were approximately 2028.0 and 2025.0 MHz. The change of
instruction sampling point to CU0 and the post-reboot environment mean these
numbers must not be compared directly with the older CU1 report as an
optimization gain.

What matched the hypothesis:

- There are 2,016 fewer scale LDS instructions per wave: parent has 16,416
  B32 reads; the candidate has 12,384 B32 plus 2,016 two-address B32 reads.
  The expanded read count remains 16,416. All 65,664 B128 reads and the
  numerical instruction counts are unchanged.
- `wait8` exposure falls from 69,763.250 to 66,086.188 cycles/wave, and total
  DScnt stall falls by 2,934.063 cycles/wave. The LDS-read issue timeline also
  decreases.

What did not match the hoped-for kernel improvement:

- TENSORcnt stall grows by 3,220 cycles/wave, already exceeding the total
  DScnt saving. Workgroup barrier wait grows from 74,947.313 to 77,732.938
  cycles/wave, while cluster wait falls from 101,060.250 to 99,180.000.
- Final drain and `s_wait_idle` grow, leaving only a 0.058% complete-wave
  difference. The tiny untraced gain did not reproduce in paired means.
- Owner body means change unevenly: A 28,375.71 -> 28,633.09, B 28,213.05 ->
  28,443.60, ScaleA 27,990.74 -> 29,097.31, and ScaleB 29,687.84 -> 27,946.44.
  These independent captures show phase-cost redistribution; they are not
  simultaneous barrier-arrival timestamps and do not prove a particular late
  producer on each dynamic barrier.

The largest non-wait observations still occur on LDS operations. Parent
`ds_load_b128` at PC `0x8278` has maximum latency 9,327 and mean 6.53 cycles;
candidate `ds_store_b64` at `0xabf8` has maximum 12,222 and mean 36.34 cycles.
Their issue-gap shares are 0.0795% and 0.1229% respectively. Isolated maxima
are not multiplied by hit count or added to wait-cycle percentages. The
complete PC and latency records remain in the metrics artifacts.

For the 64 sampled physical WGPs (four SEs, repeated across four independent
SIMD-select captures), mean/max completion imbalance is 0.0385%/0.2342% for
parent and 0.2496%/1.4113% for V1-A. This is a larger observed completion tail,
not evidence of balanced improvement, and still not a full-256-WGP claim.
The analyzer's raw-clock envelope metric is not a measurement of FLOPs or
REALTIME-aligned cross-SIMD wall duration.

Decision: reject V1-A for lack of a reliable untraced gain. Archive its source
transformation as `v1a_source_audit.json`, keep raw captures, performance logs,
`v1a_cu0_parent_metrics.json`, `v1a_cu0_candidate_metrics.json`, and
`v1a_trace_comparison.json`, and remove its top-level experimental ISA/audit
duplicate and standalone assembly/link outputs. The accepted parent and the
shared V1 generator needed for V1-B are preserved.

### V1-B correctness, initial timing, and measurement exclusions

Random run `20260918T102842Z_e2e-random` used three repetitions per case and
two e2e iterations per repetition. Parent and V1-B matched the established
random hashes, `logits_diff=3.39799e-06`, and `rel_l2=0.00260689` in every
repetition. An external GPU user existed around this correctness-only run;
none of its timings is used for selection.

The first strict-idle nine-round const0 comparison completed at
`20260918T103148Z_e2e-const0`:

| Case | GEMM1 samples, us | GEMM1 median | MoE median |
|---|---|---:|---:|
| parent | 480.016, 480.593, 482.097, 480.254, 480.836, 479.680, 481.110, 479.750, 480.184 | 480.254 us | 1310.60 us |
| V1-B | 481.405, 480.406, 480.324, 479.427, 480.618, 480.441, 480.011, 480.281, 479.888 | 480.324 us | 1310.92 us |

There is no positive independent-median result in this batch. All const0
checks pass. The subsequent `20260918T104211Z_e2e-const0` batch contains an
unexplained parent sample of 1028.906 us, with MoE e2e 11209.94 us. That entire
batch is excluded from optimization selection, rather than deleting only
the outlier. The raw log and all samples are retained. An external overlap
was not established by the available before/after checks; the cause of the
large excursion is unknown.

A host-side, 0.5-second KFD monitor was added for subsequent performance
work. The benchmark receives a unique `AITER_TW_RUN_TOKEN`. On interference,
the monitor may send SIGINT only to a process group whose leader is verified
to belong to `hyg_fyd1` and to carry that exact run token. It never signals a
foreign process and never changes device or driver settings.

The monitor's first trial exposed an exit race: PID 667819 was correctly
recognized as this run's process, but its environment became empty during
exit and it was then misclassified. The monitor conservatively interrupted
only this run's verified process group 664834. The partial batch is excluded;
no external process was affected. The monitor was corrected to cache verified
ownership by `(PID, process start time)` and to check a tagged group leader
for same-container subprocesses. It also handles an interrupted shell that
does not append the usual completion marker. The original monitor reached
its bounded timeout and no monitor/test process remained at continuation.

### V1-B trace integrity record

The CU0 pair at `20260918T103456Z_att` is incomplete: parent SIMD3/SE3 has
13 task boundaries and 24,557 WMMAs; candidate SIMD1/SE2 and SIMD1/SE3 have
29 boundaries with 52,139 and 52,137 WMMAs. The repeat at
`20260918T104440Z_att` has a complete candidate but an incomplete parent
SIMD2/SE2 (32 boundaries, 58,129 WMMAs). These incomplete whole-case comparisons
are not used for cycle attribution. Changing CU selection did not eliminate
the intermittent decoding problem; the exact underlying cause remains
unknown. Per-wave integrity JSONs and raw captures are retained.

At the 13:02 UTC continuation, the host boot, container start, branch, HEAD,
parent hash, and V1-B hash were rechecked and unchanged. An external `felix`
container trace job was active at the first check, so no new performance test
was started at that check. A monitored clean repeat and a complete paired
trace remain required for the V1-B decision.

### V1-B complete-trace reference comparison and decision

The corrected monitor observed two simultaneous PIDs at 13:04:34 UTC:
1243340 carried this run's token, while 1243693 did not. It interrupted only
the verified run group 1241677. That run ended with code 130, and its entire
timing batch is excluded. Subsequent prechecks found other `felix` trace work
and a separate GEMM2 e2e task in the same `hyg_fyd1` container. None of those
tasks was interrupted. Container membership alone is deliberately not used
as proof that a process belongs to this optimization run.

To retain a useful complete-wave comparison without treating corrupt traces
as valid, the complete V1-B candidate at `20260918T104440Z_att` is compared
with the already complete unchanged-parent CU0 capture at
`20260918T102310Z_att`. **These are not a contemporaneous paired capture**:
their start times differ by about 21 minutes. They share the host boot,
unchanged parent/workload, mask `0xf`, target CU0, all four SIMD selections,
and all completeness checks. Their mean capture clocks are 2028.0/2031.1 MHz.
The comparison is diagnostic context, not evidence of an ISA speedup.

Both sides contain 16 complete waves, all 560 adjacent body-task intervals,
and the exact WMMA/EXP/RCP totals. Canonical normal and representative modes
and `analyze_att_capture.py` completed; all CSV wait-family totals match the
decoded whole-wave accounting exactly.

| Reference metric, cycles/wave unless stated | Parent reference | V1-B | Change |
|---|---:|---:|---:|
| Entire decoded wave | 1,035,344.875 | 1,032,161.250 | -0.307% |
| Mean body task | 28,566.834 | 28,542.889 | -0.084% |
| First setup/task | 33,266.125 | 31,773.000 | -4.49% |
| Final drain | 2,239.563 | 1,387.125 | -38.06% |
| DScnt stall | 92,924.250 | 101,190.688 | +8.90% |
| Barrier stall | 176,007.563 | 171,807.250 | -2.39% |
| TENSORcnt stall | 35,959.813 | 35,234.875 | -2.02% |
| SALU/control issue-attributed timeline | 64,989.000 | 66,199.063 | +1.86% |

The static removal and numerical invariance behaved as intended, and memory
instruction counts are unchanged. The hoped-for control-timeline reduction
is not observed. `wait8` exposure is higher (69,763.250 -> 77,448.188
cycles/wave), consistent with an earlier wait exposing an existing dependency
rather than removing that dependency. The non-contemporaneous reference does
not establish the causal size of this effect. First-task/drain variation
accounts for much of the small full-wave difference.

V1-B's whole-wave shares remain dominated by barrier stall (16.65%), DScnt
(9.80%), and TENSORcnt (3.41%). `wait8` alone is 7.50%. The occupancy analyzer
reports mean/max completion imbalance 0.0937%/0.7293% for its sampled 64 WGPs,
versus 0.0385%/0.2342% for the reference. Neither the scope nor the raw-clock
interpretation is extended to the other 192 WGPs.

The valid untraced nine-round batch has paired median -0.218 us, mean
-0.191 us, and 6/9 wins, but its independent median regresses by 0.070 us.
The abnormal and interrupted batches cannot be used to strengthen the claim.
There is no reproducible positive result, so V1-B is **not promoted**. This is
a conservative lack-of-evidence decision, not proof that no sub-microsecond
benefit is possible. Further attempts at this eight-instruction edit are
stopped in favor of a larger, auditable follow-up.

Evidence: `v1b_perf_e2e.tsv`, `v1b_candidate_metrics.json`,
`v1b_reference_trace_comparison.json`, `v1b_source_audit.json`, and raw logs
under `history_runs/tensor_wait_opt_rebaseline_20260918/`. Remove the
experimental V1-B ISA, its top-level audit duplicate, and the now-unused V1
generator. No new named case is added to `benchmark_history.sh` in V1.

The accepted endpoint remains `tensor_wait_after_wmma`, with unchanged SHA256
`02320b67ae67a8349f7942a31f89e3f2f3de6efe130e937d04d1d4bdb9273b2d`.
Next: [V2](TENSOR_WAIT_OPT_V2.md) tests uniform scale-read pairing across all
four K subsegments using otherwise unused allocated address registers.
