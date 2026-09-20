# tensor_wait_after_wmma optimization v2: uniform scale-read pairing

Started 2026-09-18 after [V1](TENSOR_WAIT_OPT_V1.md). GPU work stays on a07-3.

## Parent and motivation

V0 and V1 retained no reliable improvement. Copy the unchanged accepted
`tensor_wait_after_wmma` ISA for this experiment:

```text
parent SHA256=02320b67ae67a8349f7942a31f89e3f2f3de6efe130e937d04d1d4bdb9273b2d
candidate=tensor_wait_v2_all_scale_pairs.s
```

V1-A paired only 16 low-offset static scale-load pairs. It removed 2,016 LDS
instructions per wave and reduced measured DScnt exposure, but TENSORcnt and
other waits absorbed the saving. Three confirmation batches did not establish
a stable untraced gain. V2 is a larger, symmetric request-stream experiment,
not an assumption that the rejected partial version was faster.

The remaining pairs have byte offsets in the 2048, 4096, and 6144 groups.
Six precomputed bases allow all four K subsegments to use the same paired
representation without adding address arithmetic to the hotloop itself.

## V2-A transformation

Use these previously unreferenced logical VGPRs in every bank:

| New address register | Value |
|---|---|
| v244 | v80 + 2048 |
| v245 | v80 + 4096 |
| v246 | v80 + 6144 |
| v247 | v81 + 2048 |
| v248 | v81 + 4096 |
| v249 | v81 + 6144 |

The source's highest explicitly referenced logical VGPR is 243. A full
operand-range scan, not just a search for single-register spellings, must
confirm that 244-249 are unused. Relative GPR indexing must also be absent.
There are 24 new live address values across the four banks, but the largest
physical index is 1017, within the existing 1024-VGPR allocation. Metadata,
occupancy allocation, LDS size, scratch, ABI, and grid are not changed.

After the existing v80/v81 initialization and bank copies, compute the six
bases in bank 0 with `v_add_nc_u32_e32`, then copy them to banks 1-3. Restore
the original bank state before continuing. No SCC/VCC/EXEC changes are added.
This is **per-task setup**, not one-time-per-kernel setup: the existing
`Lmoe_persistent_state_ready` label precedes the address initialization. The
cost is 6 adds, 18 copies, and 4 bank settings, or 28 extra instructions/task.

Then fuse every eligible adjacent B32 scale pair in the two hotloop families.
For example:

```asm
; Original, with identical address bank state:
ds_load_b32 v96, v81 offset:2560
ds_load_b32 v97, v81 offset:2688

; New: v247 holds the corresponding bank's v81 + 2048.
ds_load_2addr_b32 v[96:97], v247 offset0:128 offset1:160
```

Low-offset pairs continue to use v80/v81 directly. The expected static count
is 64 fused pairs. The expanded read list must exactly match the parent,
including destination VGPR banks and byte addresses. Expected full-wave
counts are 288 remaining B32 instructions plus 8,064 two-address instructions,
representing the same 16,416 B32 reads. This saves 224 LDS instructions/task;
after the 28 setup instructions, the expected net reduction is 196/task.
All B128 payload reads, WMMA/SiLU operations, and their arithmetic order stay
unchanged.

## Counter and lifetime proof

The CDNA5 ISA, section 11.2 (local text lines 10739-10758 and 10820-10826),
defines DScnt per LDS instruction, in-order LDS retirement, and two unsigned
8-bit offsets scaled by four for B32. Section 5.7.1.4 states that read
completion means data is available in VGPRs. Section 15.5, the
`S_SET_VGPR_MSB` mapping table (local text around 19540-19616), maps VDS ADDR
to SRC0 bank bits and VDST to destination bank bits.

Translate each affected wait's old outstanding suffix into complete new
instruction groups. A fused pair crossing the old retirement frontier must
retire. Derive the new thresholds per site; do not replace immediates globally.
The expected mapping is sixteen 20->18 sites, four 10->8 sites, and four 4->2
sites. Entry/tail `wait8`, full drains, TENSORcnt, and barriers remain intact.
The generator must reject unexpected counts or changed waits that cannot be
proved from a single basic block's issued suffix.

Further static gates:

- preserve the complete original instruction stream when paired reads are
  expanded and only audited wait-immediate translations are normalized;
- verify the new-register set is disjoint from all original explicit ranges;
- verify all four original address-bank copies exist and that v80/v81 are not
  modified between the added setup and the end of the transformed hotloop;
- do not cross an outstanding `s_delay_alu` skip contract or rely on dynamic
  register indexing;
- preserve the parent's file and check its SHA256 before and after generation.

## Measurement and decision

Run repeated random MoE e2e against the same parent and require identical
hashes/error metrics. Const0 must be exact. All arithmetic instructions and
their order are immutable; no reduced precision or approximate replacement
is allowed.

Performance requires a strict all-GPU precheck, same-run alternating cases,
and the per-run KFD monitor. Interrupted/contended batches are recorded but
excluded, without selecting individual favorable samples. Use shorter matched
batches when the machine has only intermittent idle periods. Only a repeatable
improvement is registered in `benchmark_history.sh`.

Capture every runnable experiment and use the canonical
`trace_segment_cycles.py` plus `analyze_att_capture.py`. Preserve complete-wave
counts, all 35 body intervals, first/drain accounting, clock/sampling context,
and the limits of the 64-WGP sample. Do not accept decoder-corrupted waves or
infer wall-time gains by summing overlapping instruction latencies.

Expected: fewer LDS requests/issue overhead and a more uniform scale-load
stream across K subsegments. Possible failure: paired service/retirement or
changed producer arrival absorbs the saving, as in V1. Inspect DScnt, TENSORcnt,
workgroup/cluster waits, owner phases, and output drain before retaining it.
If rejected, archive the transformation and evidence, then remove the new
ISA and dedicated builder. Further work must be justified by these results.

## Execution record

The immutable parent was copied and the V2-A ISA generated after a dry audit:

```text
candidate SHA256=72e1ee7de41f0edce285de118ea4c62d5033897f250266c84382cb07b7088c33
fused static pairs=64
wait translations=16 x (20->18), 4 x (10->8), 4 x (4->2)
extra setup instructions/task=28
new physical VGPRs=244-249, 500-505, 756-761, 1012-1017
```

The full expanded instruction-stream comparison passes, including original
VALU order/bank state, LDS address/data semantics, scalar/control sequence,
barriers, and metadata. The parent hash is unchanged. No relative-register
indexing or `s_delay_alu` contract exists in this source. Base-register writes
after the setup and before the transformed hotloop ends are rejected by the
builder; none was found. Local and remote candidate SHA256 values match.

CPU assembly/link and three repeated random parent/candidate comparisons are
the next checks. Random timings will not be used as performance evidence;
formal timing and ATT still require the strict idle/monitor gates.

### Assembly and repeated random correctness

Assembly/link completed with exit code zero using the pinned gfx1250 clang.
The existing non-fatal `-shared` unused-argument warning was unchanged.
Run `heliosr-1b114-a07-3_20260918T133251Z_e2e-random` completed three
repetitions per case, with two e2e iterations per repetition. Every parent
and candidate result matched:

```text
logits_diff=3.39799e-06
rel_l2=0.00260689
MoE hash=1556fc617347e2dabc9cff19dbfd822b
reference hash=1a5d22911ba167160b4f2c12092a5193
```

These are correctness results only. The candidate has not yet earned
performance promotion. New evidence uses the existing continuation root
`history_runs/tensor_wait_opt_rebaseline_20260918/` with the `v2a_` prefix.

### Measurement granularity under shared GPU use

The first three-round screen was interrupted with exit code 130 when the
monitor observed unrelated GPU PID 1374494 alongside this run at 13:36:20
UTC. Its entire batch is excluded. Only this run's verified process group
was signaled; no unrelated job was affected.

Subsequent measurements use one complete parent/candidate pair per timing
unit, alternating the initial order between units. Each unit has its own idle
check, run token, monitor log, and completion result. This is not selection of
favorable prefixes from an interrupted batch: a unit is defined before it
starts and accepted only in its entirety.

ATT is likewise split into one case / one SIMD-select unit, with all four SEs
still enabled by mask `0xf`. `analyze_att_capture.py` natively supports
directories containing an existing subset of SIMD captures. Each accepted
unit must have four complete SE0-SE3 waves, the exact 36 task and WMMA/EXP/RCP
counts per wave, and no monitored interference. The first qualifying capture
for each case/SIMD is retained. Final analysis assembles all four independently
captured SIMD directories into a provenance-recorded view and still requires
16 complete, balanced-owner waves per case. No corrupted wave fragment is
spliced into that view. Separate windows and their clock variation are
reported; they are not treated as simultaneous hardware state.

The first valid timing unit (`20260918T134911Z_e2e-const0`) is
481.112 us parent versus 482.099 us candidate, delta +0.987 us. Both const0
results are exact, and all 27 monitor samples were clear. One unit is not a
performance decision. Parent SIMD0 at `20260918T135051Z_att` completed with
four fully checked waves and 77 clear monitor samples.

### Expert-scheduling considerations

The Shader Programming Guide section 4.3.7.4.1 documents the cross-VALU/VMEM
hazards exposed by `SCHED_MODE 2`; elapsed cycles alone are not a dependency
proof. Pairing adjacent LDS reads crosses no VALU instruction and preserves
the original VALU ordering/count between those reads and old producers.
The DScnt retirement translations protect every original LDS consumer.
The added address producers are in the setup preceding the existing
accumulator-clear routine and hotloop. Their new VGPRs are otherwise unused,
and subsequent task reuse follows the original full DScnt drain at the task
boundary. The original `s_wait_alu depctr_va_vdst(0)` drains in the epilogue
are unchanged; no nonzero VM_VSRC threshold or delay-skip contract is relaxed.

### Unit-collection progress

The second valid timing unit (`20260918T135907Z_e2e-const0`, candidate first)
is 481.210 us parent versus 480.481 us candidate, delta -0.729 us. Its 27 KFD
monitor samples were clear. The first two deltas have opposite signs; no
improvement is claimed from them.

Single-SIMD capture and collection work with the existing occupancy analyzer.
The additional memory-count gate verifies every retained parent wave has
16,416 B32 loads, 65,664 B128 loads, and 2,304 B64 stores. Every retained V2-A
wave has 288 B32 plus 8,064 two-address B32 loads, with B128 loads and B64 stores
unchanged. Counts are checked per wave, not only as an average.

Parent SIMD2's first attempt (`20260918T142032Z_att`) was uncontended but had
a corrupt SE3 instruction stream (18 boundaries, 33,131 WMMAs); it was not
registered. Its retry (`20260918T143844Z_att`) passes all gates. Candidate
SIMD2's first attempt was interrupted on unrelated GPU PID 1679552 and is
excluded. Previously accepted units are never replaced to obtain a more
favorable timing. The authoritative source list is
`history_runs/tensor_wait_opt_rebaseline_20260918/v2a_capture_units.json`.

The third valid one-pair timing unit (`20260918T145033Z_e2e-const0`) is
479.840 us parent versus 482.423 us candidate, delta +2.583 us, with a clear
monitor and exact const0 results. The first three valid paired deltas are
+0.987, -0.729, and +2.583 us. This is not a repeatable performance improvement.
Trace completion and additional bounded confirmation remain in progress.

A separate conservative payload-use audit on the complete parent SIMD0
capture found 28 B128 load candidates per task whose results were not used
before another load overwrote them or the wave ended. It treats all payload
registers in each WMMA source bank as potential reads, including operand-cache
reuse, and treats other vector mentions as reads in every bank. It therefore
intentionally under-identifies dead loads. This is a lead for subsequent
static CFG/counter verification, **not** a current ISA deletion or a claim
that the 32-load difference from an idealized byte count is all removable.
Evidence is `payload_use_parent_simd0.json`; V2-A does not remove these reads.

### Final uncontended timing and decision

Six complete, separately gated timing units finished with exact const0
correctness and no foreign KFD process observed by their 0.5-second monitors.
Each value averages 19 measured GEMM1 instances out of 20 e2e iterations.
The initial case order alternates between units. These are six paired
observations, not 114 independent observations per case.

| Run UTC | Initial order | Parent, us | V2-A, us | Candidate - parent |
|---|---|---:|---:|---:|
| `20260918T134911Z` | parent first | 481.112 | 482.099 | +0.987 |
| `20260918T135907Z` | candidate first | 481.210 | 480.481 | -0.729 |
| `20260918T145033Z` | parent first | 479.840 | 482.423 | +2.583 |
| `20260918T150227Z` | candidate first | 479.520 | 481.586 | +2.066 |
| `20260918T150559Z` | parent first | 480.408 | 481.247 | +0.839 |
| `20260918T150746Z` | candidate first | 479.980 | 480.707 | +0.727 |

Parent/candidate medians are 480.194/481.4165 us: about 0.255% higher candidate
latency. Paired median is +0.913 us, paired mean +1.078833 us, and the candidate
wins only 1/6 pairs. The last postcheck found no GPU/KFD user or residual test.
The interrupted earlier screen is not included in these statistics.

### Complete trace comparison

All eight selected case/SIMD units pass the contention, ISA-checksum, four-SE,
per-wave task/arithmetic, and LDS-count gates. The final views have 16 waves
per case and four waves per owner. Both canonical parser modes ran for all
four SIMD selections, and `analyze_att_capture.py` ran on each capture and
the assembled view. CSV wait totals independently match decoded totals.

These captures were taken in separate idle windows. Their mean capture
clocks are 2030.52/2026.01 MHz. They are not a simultaneous view of all waves,
and sub-percent cycle differences alone are not proof of a wall-time gain.

| Metric, cycles/wave unless stated | Parent | V2-A | Change |
|---|---:|---:|---:|
| Entire decoded wave | 1,037,212.313 | 1,037,508.000 | +0.029% |
| First setup + task | 31,798.750 | 34,370.500 | +8.09% |
| All 35 following tasks | 1,002,714.500 | 1,000,834.875 | -0.187% |
| Final drain | 2,699.063 | 2,302.625 | -14.69% |
| Mean / median body task | 28,648.986 / 27,212.0 | 28,595.282 / 27,444.5 | -0.187% / +0.854% |
| DScnt stall | 88,999.750 (8.58%) | 90,147.875 (8.69%) | +1.29% |
| Barrier stall | 180,127.188 (17.37%) | 174,720.188 (16.84%) | -3.00% |
| TENSORcnt stall | 36,337.875 (3.50%) | 35,186.000 (3.39%) | -3.17% |
| LDS-read issue-attributed timeline | 123,712.500 | 119,253.125 | -3.60% |
| LDS-write issue-attributed timeline | 19,556.813 | 24,979.063 | +27.73% |
| SALU/control issue-attributed timeline | 64,157.000 | 65,734.063 | +2.46% |

What matched the hypothesis: all 8,064 intended paired reads per wave occur;
the expanded scale-read count and every B128 payload read are unchanged.
LDS-read issue attribution decreases, and barrier/TENSORcnt exposure is lower
in the sample.

What did not match: `wait8` increases from 64,875.188 (6.25%) to 67,241.875
(6.48%) cycles/wave. Total DScnt does not improve. Setup, control, WMMA, and
output-write attributions absorb the apparent saving; whole-wave duration is
essentially flat and robust untraced timing regresses. A/B owner body means
increase by about 142/627 cycles, while ScaleA/ScaleB decrease by about
648/337 cycles. This is not a uniformly shortened pipeline. Those separate
owner captures do not identify the simultaneous last arrival at every barrier.

Other large observations remain on LDS: the parent B128 read at `0x655c` has
maximum/mean latency 9,285/7.54 cycles and issue-gap share 0.0916%; the candidate
B128 read at `0x9038` has 12,825/7.43 cycles and share 0.0902%. Candidate B64
store `0xb5c0` has mean 118.18 and maximum 10,236 cycles, with share 0.4066%.
These isolated latency maxima are not architectural costs and are not added
to explicit wait percentages.

Across the sampled 64 WGPs, parent mean/max completion imbalance is
0.0197%/0.0341%; candidate is 0.0498%/0.3749%. There is no large sampled-WGP
straggler that explains the main wait budget. The other 192 WGPs are not
covered, and raw cross-SIMD clock envelopes are not REALTIME-aligned wall
durations.

Decision: **reject V2-A**. Preserve `v2a_source_audit.json`, all capture-unit
provenance and logs, `v2a_parent_metrics.json`, `v2a_candidate_metrics.json`,
and `v2a_trace_comparison.json`. Remove the experimental ISA, top-level audit
duplicate, standalone object/code object, and its dedicated builder. No named
benchmark case is added. The accepted parent remains unchanged.

The next direction is not another scale issue-count reduction. The
conservative payload-use audit now covers all 16 complete parent waves and
finds the same 28 B128 candidates per task in both role families. The
[V3 plan](TENSOR_WAIT_OPT_V3.md) will first validate the final-iteration CFG,
register use, and DScnt frontier before any payload read is removed.
