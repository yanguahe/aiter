# tensor_wait_after_wmma: optimization iteration summary

Closed on 2026-09-19 (Asia/Shanghai; measurements use 2026-09-18 UTC).
All kernel execution in this iteration was on `a07-3`, in `hyg_fyd1`.

## Outcome

Keep the original **`tensor_wait_after_wmma`**. Seven ideas across V0-V3
were implemented and analyzed; none demonstrated a reproducible untraced
performance improvement sufficient for promotion. Numerical checks passed,
but correctness alone was not treated as a performance win. No new named
case was added to `benchmark_history.sh`.

The latest V3-B comparison has six valid, alternating-order pairs:

```text
GEMM1 parent median:       481.0585 us
GEMM1 candidate median:    481.3895 us
candidate latency change: +0.068807%
paired median delta:      +0.1325 us
paired mean delta:        -0.107833 us
candidate wins:           2/6
```

The favorable mean is tiny and does not agree with the median or win count.
There is no basis to replace the accepted ISA with V3-B. This does not prove
that every rejected transformation is intrinsically slower on every run.

## Remeasured starting point and invariants

After the container/device-access issue was resolved, a fresh nine-round
baseline completed at `20260918T095155Z_e2e-const0`:

| Metric | Remeasured median |
|---|---:|
| GEMM1 profiler device time | 480.652 us |
| Complete MoE e2e | 1309.510 us |

All nine const0 results were exact. The accepted source did not change
across the environment transition. The earlier 507-512 us measurements
belong to the pre-reboot environment; the lower rebaseline is **not an ISA
optimization gain**. Each later candidate was judged against its freshly
interleaved unchanged parent, not against that historical timing.

```text
ISA:
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64_tensor_wait_after_wmma.s
SHA256:
02320b67ae67a8349f7942a31f89e3f2f3de6efe130e937d04d1d4bdb9273b2d

branch / HEAD: hyg_gfx1250_gemm_a4w4 / 3b0ad10a
snapshot: 23c2caaafa5f1c6e6d5d9f756980fe004af4202c
producer: three_kernel
experts=96 tokens=16384 topk=6 model_dim=7168 inter_dim=3072
format=a4w4 activation=SiLU
grid=(16,16,1) cluster=(4,4,1) block=(128,1,1)
LDS=327680 B VGPR=1024 scratch=0 kernarg=184 B
numbered SGPR=104; metadata SGPR=106
```

The candidate random runs retained the accepted parent's output and error
metrics. The parent is not bit-exact to the numerical reference on random
data, so these two kinds of equality are deliberately distinguished:

| Data | `logits_diff` / `rel_l2` | MoE hash | Reference hash |
|---|---|---|---|
| random | `3.39799e-06` / `0.00260689` | `1556fc617347e2dabc9cff19dbfd822b` | `1a5d22911ba167160b4f2c12092a5193` |
| const0 | `0` / `0` | `21291d9023c8af8a6324fe20f346a967` | `21291d9023c8af8a6324fe20f346a967` |

No change to arithmetic precision, operation order, interface, production
package, or input/output layout is retained.

## Experiments and lessons

Every runnable experiment used a separately copied ISA and received thread
trace analysis even if its timing did not improve. V3-B was copied from
V3-A as an immediate follow-up; both were measured against the immutable
accepted parent. Other rejected candidates were not accumulated into a
supposedly improved baseline.

| Experiment | Change | Untraced result | Trace lesson and decision |
|---|---|---|---|
| [V0-A](TENSOR_WAIT_OPT_V0.md) | Coalesce B32 scale-load islands | Two 9-round screens showed only 0.13% / 0.06% separate-median reductions, with weak paired means | DScnt/control attribution fell, but barrier/TENSORcnt rose; the larger traced change did not translate into a reliable benefit. Reject. |
| [V0-B](TENSOR_WAIT_OPT_V0.md) | Split wait20 into a proven wait24/consumer/wait20 frontier | 18 pairs, 10 wins; paired mean -0.151 us | A local early-consumer opportunity did not shorten the full traced execution. Reject. |
| [V1-A](TENSOR_WAIT_OPT_V1.md) | Fuse 16 low-offset B32 pairs with `ds_load_2addr_b32` | 30 pairs; medians 480.378 / 480.285 us; last independent batch reversed direction | DScnt -3.16%, TENSORcnt +8.95%, whole wave only -0.058%. Reject. |
| [V1-B](TENSOR_WAIT_OPT_V1.md) | Remove eight overwritten bank resets | Valid 9-round medians 480.254 / 480.324 us | No reproducible end-to-end improvement; abnormal/interrupted batches do not strengthen the evidence. Reject. |
| [V2-A](TENSOR_WAIT_OPT_V2.md) | Fuse all 64 scale pairs using existing unused VGPRs for high-offset addresses | Six valid pairs; medians 480.194 / 481.4165 us; one win | LDS-read issue attribution -3.60%, but DScnt +1.29% and whole wave +0.029%. Reject. |
| [V3-A](TENSOR_WAIT_OPT_V3.md) | Specialize the last tail and suppress 28 unused B128 loads/task | Six valid pairs; medians 480.762 / 481.3545 us; two wins | Removed reads exactly as audited, but DScnt +13.74% and control attribution +7.45%. Reject. |
| [V3-B](TENSOR_WAIT_OPT_V3.md) | Retain that payload suppression and reuse the original loop increment/exit predicate | Six valid pairs; medians 481.0585 / 481.3895 us; two wins | B128 count falls by 1,008/wave, yet DScnt +16.19% and whole wave +0.189%. Reject. |

V0 timings are from the earlier environment and must not be pooled with
post-reboot V1-V3 observations. Trace comparisons are within each documented
experiment/reference configuration, not across different target CUs or
clock eras. V1-B and V3 use explicitly identified earlier same-ISA reference
captures; they are not contemporaneous paired ATT measurements.

## Remaining observed bottleneck

The [original analysis](TENSOR_WAIT_AFTER_WMMA_THREAD_TRACE_ANALYSIS.md)
identified barrier and LDS readiness as the main exposed costs. The complete
post-reboot parent reference supports the same ordering, with different
absolute shares due to the different capture/environment:

| Complete-wave exposed stall | Post-reboot parent reference | V3-B |
|---|---:|---:|
| `s_barrier_wait` | 17.3665% | 17.0234% |
| `s_wait_dscnt` | 8.5807% | 9.9514% |
| `s_wait_tensorcnt` | 3.5034% | 3.0231% |
| Other explicit waits | 0.7512% | 0.6194% |
| All explicit waits | 30.2018% | 30.6172% |

In V3-B, `s_wait_dscnt 0x8` alone accounts for 7.9232%, and the two repeated
K-ring cluster-barrier PCs for 7.3035%, of the observed complete-wave span.
These are subsets of the rows above, not additional costs. Long individual
LDS latencies remain, but the leading non-wait sites each have much smaller
aggregate issue-gap shares: for example, a B64 store has maximum latency
13,511 cycles but only 0.1428% issue-gap attribution.

The experiments support a **coupled readiness/synchronization bottleneck**:
reducing an instruction count or one exposed wait often shifts exposure to
another wait or owner. They do not establish whether a specific physical
cache, LDS bank conflict pattern, or TDM bandwidth limit is saturated.
Instruction latency, issue-gap attribution, and explicit wait stall are
different measurements and must not be summed as independent recoverable
wall time.

`analyze_att_capture.py` shows no large sampled-WGP completion tail that
explains this budget. Parent mean/max completion imbalance is
0.01971% / 0.03414%; V3-B is 0.07525% / 0.40247%. However:

- mask `0xf` observes four SEs / 64 WGPs, not all 256 device WGPs;
- four SIMD-select captures repeat those WGPs rather than extending coverage;
- uniform final completion does not prove internal producer/consumer balance;
- raw cross-SIMD/SE shader timestamps are not a synchronized global clock.

## Why iteration stops here

There is currently no specific, evidence-backed new local transformation
that is both dependency-safe and likely to improve this fixed workload.

1. **The obvious local variants have been tested.** Earlier history already
   rejected broad wait8 motion, `dscnt12_then8`, payload-tail advance, NOP
   removal, WPT2 ownership, and descriptor variants. V0-V3 add scale-island
   scheduling, dual-address loads, bank-state pruning, and final-tail
   specialization. Repeating those patterns without new evidence is not a
   useful next experiment.
2. **Readiness waits protect real dependencies.** CDNA5 ISA section 5.7.1.4
   specifies that DScnt completion of an LDS read makes its result available
   in VGPRs. Section 11.2 describes LDS instruction counting and same-wave
   ordering. A TENSORcnt wait or cluster barrier is not a substitute for that
   data-availability proof. Shader Programming Guide section 4.3.7.4.1 also
   exposes additional software-managed hazards under `SCHED_MODE 2`.
3. **Resource changes need a different design.** The existing 320 KiB LDS and
   1,024-VGPR allocation accompanies one resident wave slot per sampled SIMD.
   Adding useful occupancy or another complete input stage is not supported
   by simply changing a launch parameter. A smaller tile, new LDS layout, or
   different producer schedule needs a full storage/lifetime and arithmetic
   equivalence design, which the present traces do not yet justify.
4. **The unresolved cause needs more discriminating evidence.** A useful
   next research phase would obtain applicable LDS/TDM/cache/scheduler
   counters and synchronized producer-readiness observations, then formulate
   a specific layout or scheduling hypothesis. It would also benefit from a
   reserved idle test window and broader SE coverage where decoder/tool
   support permits. These are prerequisites to a new experiment, not an
   untested optimization plan claimed to work.

This stopping point does **not** claim theoretical optimality or a hardware
performance limit. It records the limit of the present safe, evidence-backed
optimization ideas while preserving the user's correctness requirement.

## Evidence, measurement quality, and handoff

Detailed plans, proofs, exceptions, timings, and expected/unexpected outcomes
are in [V0](TENSOR_WAIT_OPT_V0.md), [V1](TENSOR_WAIT_OPT_V1.md),
[V2](TENSOR_WAIT_OPT_V2.md), and [V3](TENSOR_WAIT_OPT_V3.md).

The latest compact evidence is under
[tensor_wait_opt_rebaseline_20260918](history_runs/tensor_wait_opt_rebaseline_20260918/):

- [baseline_e2e.tsv](history_runs/tensor_wait_opt_rebaseline_20260918/baseline_e2e.tsv)
- [v3b_trace_comparison.json](history_runs/tensor_wait_opt_rebaseline_20260918/v3b_trace_comparison.json)
- [v3b_candidate_metrics.json](history_runs/tensor_wait_opt_rebaseline_20260918/v3b_candidate_metrics.json)
- [v3b_capture_units.json](history_runs/tensor_wait_opt_rebaseline_20260918/v3b_capture_units.json)
- [v3b_source_audit.json](history_runs/tensor_wait_opt_rebaseline_20260918/v3b_source_audit.json)

Remote raw runs and full analyzer output remain under
`/data/yanguahe/code/wk_sp1/aiter/my_code/moe_gemm1_act1_optimized/history_runs/`.
The local mirror contains compact metrics, audits, and selected logs, not
all large raw ATT files. Absolute input paths in metrics JSON are remote.

Both requested trace tools were used, with the canonical
`trace_segment_cycles.py` checksum enforced. Every retained comparison has
complete task/arithmetic counts and independently reconciled CSV wait totals.
Interrupted, incomplete, and insufficiently monitored units are retained as
excluded evidence, never combined into a valid partial wave. A late-starting
monitor was detected during V3-B; subsequent launches explicitly waited for
the monitor's first empty KFD sample. No foreign GPU process was stopped.

Rejected active ISA files and experiment-only builders are cleaned after
source-audit preservation; raw evidence and reusable analysis/monitor tools
are retained. The accepted parent and benchmark SHA256 stay unchanged. No
commit, destructive Git synchronization, device reset, driver reload, or
change to another task's production files is part of this closure.

Final host check at **2026-09-18 16:53:27 UTC**: no GPU/KFD owner, busy=0,
VRAM=173,154,304 bytes; no experiment monitor/analyzer process remained.
See [final_postcheck_20260918.log](history_runs/tensor_wait_opt_rebaseline_20260918/final_postcheck_20260918.log).
Local and remote accepted-ISA checksums still match the value above;
`benchmark_history.sh` remains
`9b86d89d6bec0070da9368a2a622d15aadb2706770f6b3a879e26f9e61f8131f`.
The V3-B source audit and compact metrics also match their remote originals.
