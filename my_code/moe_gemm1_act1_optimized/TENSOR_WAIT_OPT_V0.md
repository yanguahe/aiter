# tensor_wait_after_wmma optimization v0: scale-load islands and LDS consumer frontiers

Started 2026-09-18 (Asia/Shanghai). GPU execution is restricted to a07-3.

## Objective and starting point

Improve the complete E96/T16384/topk6, D7168/I3072 MXFP4 MoE GEMM1 without
changing its function, floating-point arithmetic sequence, result precision,
184-byte ABI, persistent mapping, or input/output layouts.

Starting case: `tensor_wait_after_wmma`.

```text
parent ISA:
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64_tensor_wait_after_wmma.s
parent SHA256:
02320b67ae67a8349f7942a31f89e3f2f3de6efe130e937d04d1d4bdb9273b2d
```

The [fresh trace report](TENSOR_WAIT_AFTER_WMMA_THREAD_TRACE_ANALYSIS.md)
measured 38.47% explicit wait stall over complete decoded wave lifetimes:
19.72% barrier, 13.16% DScnt, and 5.14% TENSORcnt. `wait8` accounts for 10.59%,
while `wait20` accounts for 1.84%. The sampled WGPs were balanced at completion.
The historical 511.608 us untraced median is context only; every candidate
will be compared with its parent in the same idle-machine run.

## Constraints and previous negative results

- Copy the parent ISA to a distinct file before applying each new idea.
- Keep the parent immutable, preserve all WMMA/SiLU instructions in their
  original arithmetic order, and verify random MoE output hashes against the
  unchanged parent as well as the production numerical gate.
- Correctness may run while other GPU users exist. Performance and ATT wait
  for all GPUs/KFD to be idle. Do not reset/reboot the GPU/host or affect
  another user's process.
- Use the fixed snapshot and the normal MoE e2e ATT launcher, not the older
  standalone ATT launch path. Do not modify the production `aiter/` package.
- For every runnable idea, collect parent/candidate ATT, including unsuccessful
  performance ideas. Keep whole-wave, all 35 consecutive body intervals,
  first-task, and final-drain accounting separate from sampled medians.
- Preserve observations and the transformation description here before
  removing a rejected candidate and its dedicated generator code.
- Register only reproducible winners in `benchmark_history.sh`; do not commit
  without a separate user request.

The earlier all-NT-RT search already rejected broad `wait8` motion,
`dscnt12_then8`, payload tail-load advance, NOP removal, WPT2 ownership, and
several descriptor rewrites. This iteration does not simply repeat those
transformations. The static audit also finds that payload loads are already
ordered to supply early consumers, so a blind address-sorted load permutation
has no established dependency benefit.

## Experiment V0-A: coalesce existing B32 scale-load islands

Many hotloop sites contain:

```text
set VGPR_MSB to LDS-load mode
two ds_load_b32 scale loads
set VGPR_MSB to compute mode
one independent WMMA
set VGPR_MSB to the same LDS-load mode
two more ds_load_b32 scale loads
set VGPR_MSB to the same compute mode
```

Proposed ordering:

```text
set VGPR_MSB to LDS-load mode
all four original ds_load_b32 instructions, in their original order
set VGPR_MSB to compute mode
the same independent WMMA
```

This removes two bank-selector instructions per matched site and advances two
small scale reads across one WMMA. Unlike the earlier payload-tail experiment,
it advances two B32 scale requests, not four B128 payload requests. It does not
change the DS issue order relative to other DS instructions, request counts,
wait immediates, TDM sequence, or barriers.

Static gates:

1. The two LDS-load modes and the two compute modes must match exactly.
2. Moved LDS destination registers must not intersect any operand/destination
   of the crossed WMMA. The LDS address register must not be written by it.
3. Every non-selector instruction must retain its original effective
   VGPR_MSB value. All arithmetic instructions retain their original order.
4. DS/TDM/WMMA/EXP/RCP/barrier counts remain unchanged; only selector count falls.

Expected evidence: fewer selector/SALU issue cycles, possibly less scale-read
completion exposure. Failure modes: a denser LDS burst shifts contention to a
payload owner or changes barrier arrival, erasing the instruction-count saving.

## Experiment V0-B: split wait20 at its actual operand frontier

At audited hotloop sites, entry `wait8` leaves at most the youngest eight
payload reads outstanding. The body issues another twenty LDS reads. Current
`s_wait_dscnt 0x14` then retires all eight older reads before any tail consumer.

The first consumers need only the first four of those eight reads. Test:

```text
s_wait_dscnt 0x18   # <= 24: first four older reads are complete
independent first consumers of that ready fragment
s_wait_dscnt 0x14   # <= 20: remaining older reads are complete
remaining original consumers
```

For the A-fragment path, the first two WMMAs use `v[40:55]`; the third first
uses `v[56:71]`. For the B-fragment path, the first four WMMAs use
`v[40:47]` / `v[48:55]`; later WMMAs first require `v[56:71]`.
The physical bank and the older-load order must be verified per site before
the transformation is enabled. No DS instruction is inserted between these
two waits, so their frontier interpretation is stable.

This is distinct from the previously rejected `wait12 -> wait8` experiment:
it targets the older-buffer tail guarded by `wait20`, not the first
next-buffer consumer guarded by `wait8`. Nevertheless, the extra wait may
cost more than the overlap saves; measured performance decides.

## Measurement and decisions

Each candidate first runs random e2e. Require the same MoE output hash and
the same numerical metrics as its parent for matching input, and exact const0
output. The changed kernel must execute the same full WMMA/EXP/RCP counts.

Screen with same-run alternating parent/candidate e2e const0 rounds; use nine
rounds and an independent repeat for a possible small win. Report both separate
medians and matched-round differences, together with MoE e2e time. Capture
four SIMD paths for each runnable idea and compare complete-wave cycle
budgets and the relevant wait/selector sites with its measured parent.

A rejected idea leaves its measurements and explanation in this document,
while its experimental ISA and dedicated code are removed. A winner becomes
the parent for the next idea and receives a named benchmark case. After this
plan is evaluated, the remaining measured bottleneck will define v1 in a new
document; a new plan is not evidence that an untested idea improves performance.

## Execution record

Preparation: local and a07-3 HEAD are `3b0ad10a`; the target branch is correct.
a07-3 has another user's test-preparation processes, with zero GPU queues at
the initial check. Static preparation proceeds; performance/ATT are gated on
a fresh idle check. No destructive synchronization has been performed.

### V0-A preparation and correctness

Copied the starting parent and generated `tensor_wait_v0_scale_batch.s`:

```text
SHA256=3b1bc1741841e550f5db2053df925bf419cb9bb3a2fdc933a41ee6b1606b7509
matched islands=24
static instruction reduction=48 bank-selector instructions
```

The generator verifies the effective bank state of every non-selector
instruction, unchanged arithmetic order, unchanged DS/TDM order, and unchanged
barriers. The independent static audit for V0-B identified 16 `wait20` sites:
eight two-WMMA frontiers and eight four-WMMA frontiers.

On a07-3, V0-A and its parent both passed random MoE e2e with identical:

```text
logits_diff=3.39799e-06
rel_l2=0.00260689
MoE hash=1556fc617347e2dabc9cff19dbfd822b
reference hash=1a5d22911ba167160b4f2c12092a5193
```

Random validation run: `heliosr-1b114-a07-3_20260917T161627Z_e2e-random`.
Its two-iteration timing is not used as the optimization acceptance result.

The other user's GPU test processes exited before formal timing. A remaining
CPU-only `llvm-symbolizer` had zero GPU queues/VRAM and no `/dev/kfd` handle;
the GPU was at 0% utilization with its approximately 165 MiB idle allocation.
No process belonging to that user was changed.

Two independent nine-round const0 comparisons completed:

| run | parent median | V0-A median | change | MoE parent / candidate |
|---|---:|---:|---:|---:|
| `20260917T161753Z` | 507.440 us | 506.766 us | +0.13% | 1342.25 / 1341.73 us |
| `20260917T162818Z` | 507.772 us | 507.483 us | +0.06% | 1342.09 / 1341.19 us |

The small timing result has not earned promotion. The first ATT pair was
captured at `20260917T162303Z`; strict analysis found only 64,511 WMMAs in
candidate SIMD2/SE3 instead of 64,512. That entire candidate capture is not
accepted as a complete-wave comparison. This is an observed trace-integrity
failure, not evidence that the statically unchanged arithmetic was removed.
A new parent/candidate ATT pair completed at `20260917T164050Z` and passed
the same strict counts for all 16 waves of each case.

| Complete-wave ATT metric | parent | V0-A | change |
|---|---:|---:|---:|
| Mean wave cycles | 1,251,645.563 | 1,211,376.750 | -3.22% |
| Mean consecutive body task cycles | 34,808.111 | 33,511.200 | -3.73% |
| DScnt stall cycles/wave | 155,295.188 | 148,695.688 | -4.25% |
| Barrier stall cycles/wave | 238,568.250 | 240,123.313 | +0.65% |
| TENSORcnt stall cycles/wave | 57,417.750 | 59,190.688 | +3.09% |
| SALU/control issue-attributed cycles/wave | 73,915.063 | 68,922.625 | -6.75% |
| LDS-read issue-attributed cycles/wave | 138,450.125 | 120,095.125 | -13.26% |

The reduction in DScnt and control exposure matches the intended direction.
Barrier and TENSORcnt costs move in the opposite direction, and the much
larger ATT change does not translate into a clear untraced performance gain.
The two untraced runs' matched-round median differences are -0.704 us (6/9
wins) and -0.116 us (5/9 wins). Their matched-round mean differences are only
about -0.288 and -0.254 us. This is insufficient evidence to promote V0-A.
The accepted parent remains unchanged while the next idea is evaluated.

Decision: V0-A is rejected for lack of a convincing untraced gain. Its new
uncommitted ISA, top-level audit duplicate, and dedicated generator branch
are removed. The immutable parent, transformation audit, performance samples,
and ATT metrics remain; the transformation can be reconstructed from those
records if a future experiment has a specific reason to revisit it.

Compact metrics are retained in
`history_runs/tensor_wait_opt_20260918/v0a_parent_metrics.json` and
`v0a_candidate_metrics.json`. The source transformation audit is retained as
`v0a_source_audit.json` in that directory.

### V0-B pre-implementation dependency proof

The queue audit uses the actual parent disassembly to recover VGPR bank state;
bank-selector control records can be absent from the dynamic instruction
stream. Assuming mode zero from that absence would produce a false hazard.

With the recovered physical banks, all **32,256** observed parent `wait20`
instances have conservative pending depth 28. The audited partial frontiers
are two WMMAs at 16,128 instances and four WMMAs at 16,128 instances. There are
**zero intersections** between the extra LDS destinations left pending by
`wait24` and any read/write register of those consumers. The original `wait20`
is restored before the remaining consumers, with no intervening DS issue.

Proof artifact: `history_runs/tensor_wait_opt_20260918/v0b_wait20_audit.json`.
Since V0-A is not promoted, V0-B is copied from the unchanged accepted starting
parent. Its arithmetic, memory-instruction order, and barriers are unchanged.

```text
V0-B ISA: tensor_wait_v0_wait20_split.s
SHA256: aa3b7bb17e6dfd86fe2d114a9a1b451b064fb61ac5c211464154206fb446533d
16 static frontiers, 16 additional wait instructions
```

After reconnecting on 2026-09-18, the host had not rebooted and both ISA hashes
still matched. Other GPU users were active, so the next run is correctness-only:
three repetitions of deterministic random MoE e2e for parent and V0-B. Its
timings must not be used for performance acceptance.

V0-B's three repeated random e2e runs completed at
`heliosr-1b114-a07-3_20260917T173640Z_e2e-random`. Every parent and candidate run
had `logits_diff=3.39799e-06`, `rel_l2=0.00260689`, and identical MoE/reference
hashes to their matching baseline. Formal performance and ATT remain pending
an idle-machine check; no timing from the concurrent correctness run is used.

The first formal V0-B run started after an idle check, but an external KFD
user (`bwd_dqdkdv.out`, host PID 1018173) appeared during measurement at
2026-09-17 17:47:55 UTC. The assistant interrupted its own benchmark. All
performance samples from this run are invalid and excluded; they remain in
`history_runs/tensor_wait_opt_20260918/v0b_perf.log` with a separate invalidation
note. No external process was signaled.

After the GPU returned to idle, a shorter three-round screen completed at
`heliosr-1b114-a07-3_20260917T175119Z_e2e-const0`:

| case | GEMM1 samples, us | GEMM1 median | MoE median |
|---|---|---:|---:|
| parent | 510.766, 508.578, 509.210 | 509.210 us | 1343.00 us |
| V0-B | 506.984, 508.767, 510.779 | 508.767 us | 1342.62 us |

The separate median suggests +0.09%, but the matched-round median delta is
+0.189 us (candidate slower), with one win out of three. This screen is
inconclusive. Its correctness hashes match exactly. The next step is the
required ATT comparison and additional idle batches if the trace supports
further confirmation.

The parent/candidate ATT pair completed at `20260917T175458Z`. Both cases
passed strict full-wave WMMA/EXP/RCP counts and the complete task-ledger check.

| Complete-wave ATT metric | parent | V0-B | change |
|---|---:|---:|---:|
| Mean wave cycles | 1,172,714.750 | 1,203,834.750 | +2.65% |
| Mean consecutive body task cycles | 32,448.191 | 33,277.496 | +2.56% |
| DScnt stall cycles/wave | 149,385.250 | 149,227.938 | -0.11% |
| Barrier stall cycles/wave | 228,162.188 | 243,720.500 | +6.82% |
| TENSORcnt stall cycles/wave | 65,858.875 | 60,041.000 | -8.83% |
| `s_wait_idle` stall cycles/wave | 4,320.500 | 6,558.875 | +51.81% |

The targeted frontier partially behaves as intended:

```text
parent wait20:                26,258.500 cycles/wave
candidate wait24 + wait20:    14,252.250 + 8,893.000 = 23,145.250
targeted reduction:           3,113.250 cycles/wave (11.86%)
```

However, `wait8` grows from 114,304.875 to 117,619.250 cycles/wave, absorbing
slightly more than the targeted saving. Total DScnt is effectively unchanged.
All four owner families have longer body-task means; B and ScaleB worsen most
(+4.68% and +3.25%). The new early consumer frontier does not shorten the
whole traced execution, despite locally reducing the targeted wait.

Compact metrics: `v0b_parent_metrics.json`, `v0b_candidate_metrics.json`, and
`v0b_wait_forms.json` under `history_runs/tensor_wait_opt_20260918/`.

### Additional V0-B timing confirmation

Two further idle three-round batches used opposite initial case orders:

| run | parent GEMM1 samples, us | V0-B GEMM1 samples, us | parent / V0-B median | gain |
|---|---|---|---:|---:|
| `20260917T180743Z` | 510.519, 511.234, 510.494 | 508.691, 508.265, 512.187 | 510.519 / 508.691 us | +0.36% |
| `20260917T181211Z` | 515.057, 510.888, 511.209 | 528.323, 509.305, 509.494 | 511.209 / 509.494 us | +0.34% |

Across the three batches (nine matched rounds), the pooled medians are
510.766 us parent and 509.305 us candidate; paired median is -1.583 us,
but candidate wins only 5/9 rounds and paired mean is +0.538 us because of
the retained 528.323 us observation. No outlier is removed. A fresh continuous
nine-round run is required before deciding whether the median improvement
is reproducible; the negative ATT result is reported independently.

### V0 completion and decision

The continuous nine-round confirmation at `20260917T181600Z` passed correctness
and completed without an observed external GPU user:

```text
parent GEMM1 median=508.900 us; MoE median=1344.14 us
V0-B GEMM1 median=508.200 us; MoE median=1343.92 us
GEMM1 separate-median improvement=0.14%
```

Combining it with the nine earlier matched rounds gives 18 observations:

```text
candidate wins=10/18
paired median delta=-0.932 us
paired mean delta=-0.151 us
```

Small separate-median gains were observed, but their paired consistency and
mean saving remain weak relative to the observed variation. ATT also does not
show a shortened full execution. V0-B is therefore **not promoted as a reliable
improvement**; this is an uncertainty/selection decision, not a claim that all
its untraced samples regressed. V0 keeps the original `tensor_wait_after_wmma`
as the accepted parent and does not add a new benchmark case.

Both V0 experimental ISA files and their dedicated generator/auditor code are
removed after archiving the transformation audits and measurements. An unused
standalone-verification wrapper is also removed; it was never executed, and
all GPU correctness/ATT work in V0 used the established MoE e2e path.

Next: [V1 plan](TENSOR_WAIT_OPT_V1.md) tests changes to the number of LDS scale
instructions and redundant bank-state writes while preserving the computation.
