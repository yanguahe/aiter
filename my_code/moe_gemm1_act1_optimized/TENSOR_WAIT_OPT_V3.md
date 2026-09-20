# tensor_wait_after_wmma optimization v3: final-iteration payload prefetch

Started 2026-09-18 after [V2](TENSOR_WAIT_OPT_V2.md). No V0/V1/V2 candidate
was promoted. The accepted parent remains unchanged:

```text
case=tensor_wait_after_wmma
SHA256=02320b67ae67a8349f7942a31f89e3f2f3de6efe130e937d04d1d4bdb9273b2d
```

## Evidence and hypothesis

The complete parent capture contains 65,664 B128 loads per wave. A
conservative register-use audit identifies 1,008 candidate unused loads per
wave, exactly 28 per logical task, in all 16 observed waves. It counts all
payload registers in each WMMA source bank as possible reads to avoid
misinterpreting operand-cache reuse, and treats other vector mentions as
reads in every bank. It must also verify that each unused instance is the
last occurrence of its static PC in that task before this hypothesis is
implemented.

The candidates are the first fourteen B128 loads of each of two final
prefetch groups. They write logical v8-v63; v64-v71 are intentionally left
alone because the conservative audit does not prove those loads unused.
The corresponding source regions are the final two compute micro-stages of
the fourth unrolled K subsegment in each role family.

Unlike V1/V2 pairing, this proposal removes requested LDS payload bytes:
28 instructions x 32 lanes x 16 bytes = 14,336 bytes/task. It does not change
TDM global-memory transfers, output stores, or numerical operations.

## V3-A: specialize only the final tail

Copy the immutable parent to a new ISA. Keep the normal loop body intact.
At the last K-ring wait/compute boundary of each family, use the same exit
predicate that the original tail will evaluate:

```text
next_progress = s58 + 0x100
if not (signed(next_progress) < signed(s59)):
    branch to a specialized final tail
otherwise:
    execute the original tail
```

Use existing scratch s24 only if static checking proves its old value is
dead at this point. Prove that incoming SCC is dead before adding the
predicate. No new SGPR/VGPR allocation is permitted. The original s58 update
still executes once in the tail. The predicate uses the runtime bound, not
a hardcoded K=7168 last-index constant. Earlier exits from the first three
unrolled subsegments are unchanged.

Copy the final-tail instruction sequence out of line, rename its internal
labels, remove only the 28 proved-unconsumed B128 reads, and translate the
affected DScnt frontier. Expected A-family change is wait10 -> wait4 after
removing the first six future payload loads; derive this from the original
issued suffix and reject any unexpected mapping. Keep every TENSORcnt wait,
cluster/workgroup barrier, scalar descriptor update, WMMA/SiLU operation,
and output operation in its original execution order.

The specialized path must reach the same original exit label (which sets
s92=3 before the common epilogue). It must not fall through into another
tail. Place the copies in `.text` before `.Lfunc_end0` so that the ELF function
size covers all executable code and ATT can decode them.

## Mandatory proofs and verification

- Last-occurrence dataflow evidence must hold for every observed task/owner.
- Static comparison must map the selected removed instructions to that
  evidence and preserve all remaining instruction order and effective banks.
- The guard must match the original signed loop-exit predicate; s58/s59 may
  not be modified between the guard and the original increment except as
  explicitly audited.
- Scratch s24 and SCC must not be read before their original redefinitions.
- Removed loads cannot justify deleting or relaxing a dependent wait. Derive
  the DScnt count from surviving issued operations, retaining full drains.
- The instruction-count changes are expected to be B128 65,664 -> 64,656 per
  wave; B32 scales, WMMA (64,512), EXP/RCP (9,216 each), and 36 task boundaries
  remain unchanged. The new early-exit path must be checked dynamically too.
- Run repeated random and exact const0 against the unchanged parent; require
  matching hashes and numerical metrics. Never change arithmetic precision.
- Measure only fully qualified, uncontended parent/candidate timing units.
  Capture and analyze every runnable attempt, including failures, using both
  requested trace tools and complete per-wave/task accounting.

Possible failure: extra guard branches or code footprint can cost more than
the avoided reads, or the reads may already be hidden under useful work.
Do not infer a speedup from reduced instruction/byte counts alone. Keep a
winner only after reproducible timing and benchmark integration; otherwise
record the result and remove the experimental ISA/builder.

## Execution record

The final-occurrence audit passes in all 16 parent waves: each candidate PC
executes seven times per task, and only its last occurrence is unconsumed.
The full static B128 instruction/bank stream from the parent ISA exactly
matches the audited disassembly, providing an ordinal mapping back to source.
Each family's 28 removed tail instructions maps to exactly eight audited waves.

The dry run confirms scratch and control safety:

- both families first redefine s24 with `s_add_co_u32 s24, s58, 0x500`, without
  reading its incoming value;
- A first redefines SCC with `s_cmp_eq_u32 s22, 0`, while B first redefines it
  with the above independent add; no earlier SCC consumer exists;
- s58 is changed only by the original `s_addk_co_i32 s58, 0x100`, and s59 is
  unchanged in each copied tail;
- both tails retain 32 WMMAs and every tensor/barrier operation in order;
- A has exactly one wait10 -> wait4 translation; B has no wait translation;
- internal branches are local to the copied tail, and the same original exit
  labels are reached.

The immutable parent was copied and the new candidate generated:

```text
ISA=tensor_wait_v3_final_payload_tail.s
SHA256=2bee22b4b5c2c41da6ceb96deeaf6aa735ac4daba8fe8eba6c583727ef59e0e8
payload proof SHA256=167f4f82ce67999ad00a55fbe2d66bb9ca4b19bf3ad865779bb8313770ce1cc2
```

Local/remote candidate hashes match. CPU assembly/link and three repeated
random parent/candidate comparisons are now the next gates; no performance
improvement is claimed before those checks and uncontended measurement.

### Assembly and correctness

Assembly/link passed on a07-3. Random run
`heliosr-1b114-a07-3_20260918T153420Z_e2e-random` completed three repetitions
of each case (two e2e iterations per repetition). All hashes and error metrics
match the parent:

```text
logits_diff=3.39799e-06
rel_l2=0.00260689
MoE hash=1556fc617347e2dabc9cff19dbfd822b
reference hash=1a5d22911ba167160b4f2c12092a5193
```

Random timings are not used for performance acceptance. Dedicated uncontended
timing units and complete trace units follow. The trace-integrity checker now
accepts an explicit expected B128-load count per task; parent remains 1824,
and V3-A must have 1796. This changes only the expected memory-count invariant
for this experiment, not the WMMA/EXP/RCP/task completeness requirements.

The first uncontended timing unit (`20260918T153656Z_e2e-const0`) gives
480.474 us parent and 482.262 us candidate, delta +1.788 us. Const0 is exact,
and the KFD monitor passes. One pair is insufficient for a final decision.
Candidate trace collection uses independently qualified single-SIMD units.
The unchanged parent's complete, uncontended V2 capture view remains a
diagnostic reference; its earlier capture times will be reported explicitly,
and only fresh paired untraced timing can establish a speedup.

The second valid timing unit (`20260918T154350Z_e2e-const0`, candidate first)
gives 480.956/481.606 us parent/candidate, delta +0.650 us. The first complete
candidate SIMD0 capture (`20260918T153938Z_att`) passes every gate, including
the expected 64,656 B128 loads and unchanged arithmetic/task totals in each
of four SE waves. It is the first registered `v3a` capture unit.

The first candidate SIMD1 attempt was interrupted when unrelated GPU PID
2082737 appeared; that entire unit is excluded and is being recaptured.

### V3-A completed result

Six uncontended, exact-const0 timing units completed, alternating initial case
order. Interrupted capture attempts are not included in these timings.

| UTC run | Parent, us | V3-A, us | Candidate - parent |
|---|---:|---:|---:|
| `20260918T153656Z` | 480.474 | 482.262 | +1.788 |
| `20260918T154350Z` | 480.956 | 481.606 | +0.650 |
| `20260918T154547Z` | 479.628 | 481.483 | +1.855 |
| `20260918T155846Z` | 482.389 | 480.093 | -2.296 |
| `20260918T160045Z` | 480.568 | 481.226 | +0.658 |
| `20260918T160208Z` | 481.942 | 480.924 | -1.018 |

Parent/candidate medians are 480.762/481.3545 us. Paired median is +0.654 us,
mean +0.272833 us, with 2/6 candidate wins. MoE medians are
1312.875/1314.100 us. No reliable gain is established; the postcheck found no
remaining GPU/KFD process from this work.

The four retained candidate SIMD units all have 36 tasks, 64,512 WMMAs,
9,216 each of EXP/RCP, 16,416 B32 loads, 64,656 B128 loads, and 2,304 B64
stores per wave. Thus the actual instruction stream removes exactly 1,008
B128 loads/wave while preserving the computation. Both canonical parser
modes and the occupancy tool complete, and CSV waits match decoded totals.
The input provenance is `v3a_capture_units.json`.

The following comparison uses the earlier complete V2 parent view. It is a
same-ISA/same-boot diagnostic reference, not a contemporaneous paired capture.
Mean reference/candidate capture clocks are 2030.52/2037.91 MHz; these cycle
differences are not a substitute for the fresh paired timings above.

| Cycles per complete wave unless stated | Parent reference | V3-A | Change |
|---|---:|---:|---:|
| Entire wave | 1,037,212.313 | 1,055,746.438 | +1.79% |
| Mean body task | 28,648.986 | 29,249.427 | +2.10% |
| First setup/task | 31,798.750 | 30,993.125 | -2.53% |
| Final drain | 2,699.063 | 1,023.375 | -62.08% |
| DScnt stall | 88,999.750 | 101,228.563 | +13.74% |
| Barrier stall | 180,127.188 | 185,642.313 | +3.06% |
| TENSORcnt stall | 36,337.875 | 37,511.813 | +3.23% |
| `s_wait_idle` stall | 5,765.313 | 2,966.500 | -48.55% |
| LDS-read issue attribution | 123,712.500 | 121,298.938 | -1.95% |
| SALU/control issue attribution | 64,157.000 | 68,937.688 | +7.45% |

What matched: the audited reads disappear, LDS-read issue attribution and
final idle/drain exposure decline, and accuracy is unchanged. What did not:
body duration and DScnt/barrier exposure increase, and the reduction does not
produce a reliable untraced benefit. Control attribution includes scheduling
gaps; it is not a measurement of standalone SALU execution cost. The added
per-iteration predicate and out-of-line code footprint are plausible follow-up
targets, not a proved explanation of every cycle change.

Candidate whole-wave wait shares are barrier 17.58%, DScnt 9.59%, TENSORcnt
3.55%, idle 0.28%, and KMcnt 0.08%. The 64 sampled WGPs have mean/max completion
imbalance 0.0330%/0.1417%, still not a large sampled distribution tail. The
same sampling and cross-clock limitations as V2 apply.

Decision: V3-A is not promoted. Its ISA is temporarily retained only to copy
the immediately following V3-B experiment, then will be removed after its
source audit is archived. `v3a_trace_comparison.json` and
`v3a_candidate_metrics.json` retain the detailed instruction/occupancy results.

## V3-B: reuse the original loop update and exit predicate

Copy V3-A to a new ISA; compare performance and precision against the accepted
`tensor_wait_after_wmma`, not against an unpromoted intermediate.

Move the existing `s_addk_co_i32 s58, 0x100` and signed exit comparison from
the end of each affected tail to its entry. The early branch then selects
the final tail without computing a second predicate in s24. Remove the now
redundant late update/comparison and, on the known non-final path, the late
conditional exit branch. Keep the existing cluster wait and loop-back branch.

Every intervening read of s58 must be audited. The expected only address use
is `s_add_co_u32 s24, s58, 0x500`; after moving the increment it becomes
`s_add_co_u32 s24, s58, 0x400`, which gives the identical 32-bit sum. Apply that
adjustment to both normal and specialized tails. Prove that outgoing SCC is
dead at the loop target and original epilogue entry before removing the late
comparison. Preserve all numerical/memory/barrier/TDM operations and the V3-A
payload removal exactly.

This targets redundant control work while retaining the same byte reduction.
It can still fail if the code footprint or memory/producer phase changes
dominate. Static equivalence, random hashes, exact const0, uncontended timing,
and a complete trace comparison are required before any promotion.

V3-B's dry audit passes. In both normal and specialized tails, the only s58/s59
uses are the descriptor-offset add, original increment, and original signed
comparison. After retiming, the loop targets overwrite SCC with an independent
add before reading it; the epilogue overwrites SCC with `s_cmp_eq_u32 s92, 3`.
The complete VALU, memory, wait, barrier, and resource-metadata instruction
lists are unchanged from V3-A. The copied candidate is:

```text
ISA=tensor_wait_v3_retimed_payload_tail.s
parent SHA256=2bee22b4b5c2c41da6ceb96deeaf6aa735ac4daba8fe8eba6c583727ef59e0e8
candidate SHA256=70b07da392eac63b3975c93e971bf64ff099a5950e583a5c7537966ae8fbfd60
```

Local/remote hashes match. Assembly and repeated random verification are the
next gates. The accepted performance baseline is still the original
`tensor_wait_after_wmma`; V3-A was only an intermediate source for this copy.

V3-B assembly/link and random run `20260918T161049Z_e2e-random` pass. All
three repetitions of both cases retain the exact established random output
hashes and numerical metrics. The V3-A experimental ISA, top-level audit
duplicate, and standalone `.o`/`.co` were then removed after archiving
`v3a_source_audit.json`; its generator is temporarily retained as shared
support for the V3-B validation code. Performance is evaluated against the
unchanged accepted parent, not against V3-A.

The first V3-B timing unit (`20260918T161247Z_e2e-const0`) is
482.034/480.773 us parent/candidate, delta -1.261 us. It is uncontended and
const0 is exact, but one favorable pair is not a promotion decision.
Candidate SIMD0 (`20260918T161415Z_att`) passes the full arithmetic/task and
64,656-B128-load checks for all four SE waves. It is the first registered
unit in `v3b_capture_units.json`.

The second V3-B pair (`20260918T161614Z_e2e-const0`, candidate first) is
480.521/480.651 us parent/candidate, delta +0.130 us. Both timing-unit monitors
pass and const0 is exact. The first two paired differences have opposite
signs, so no speedup is inferred yet. Candidate SIMD1 at `20260918T161744Z_att`
also passes all completeness and memory-count gates.

The third V3-B unit (`20260918T161939Z_e2e-const0`) is
481.191/481.326 us, delta +0.135 us; the monitor passes and const0 remains
exact. SIMD2 at `20260918T162103Z_att` also passes all four-SE wave checks.
All accepted units retain 64,656 B128 loads/wave and unchanged arithmetic.

V3-B SIMD3's first attempt (`20260918T162429Z_att`) was uncontended but had an
incomplete SE2 wave (34 boundaries, 62,023 WMMAs) and was rejected. Its second
attempt was interrupted on unrelated GPU PID 2313988; it is also excluded.
The already accepted SIMD0-2 units are preserved unchanged while only SIMD3
is retried. No partial/corrupt wave is included in the final view.

### V3-B completed timing result

The six retained const0 timing units below alternate the initial case order.
Each number is the profiler's average of 19 measured GEMM1 instances from
20 e2e iterations. The statistical unit is one parent/candidate pair, not
each of those 19 correlated instances. All twelve case results have zero
`logits_diff` and `rel_l2`, pass the numerical gate, and retain the exact
const0 output/reference hash `21291d9023c8af8a6324fe20f346a967`.

| Unit | UTC run | Initial order | Parent, us | V3-B, us | Candidate - parent |
|---|---|---|---:|---:|---:|
| `v3b_pair01` | `20260918T161247Z` | parent first | 482.034 | 480.773 | -1.261 |
| `v3b_pair02` | `20260918T161614Z` | candidate first | 480.521 | 480.651 | +0.130 |
| `v3b_pair03` | `20260918T161939Z` | parent first | 481.191 | 481.326 | +0.135 |
| `v3b_pair04` | `20260918T162256Z` | candidate first | 480.881 | 481.683 | +0.802 |
| `v3b_pair05` | `20260918T162632Z` | parent first | 482.707 | 481.727 | -0.980 |
| `v3b_pair07` | `20260918T164559Z` | candidate first | 480.926 | 481.453 | +0.527 |

```text
parent / candidate GEMM1 medians = 481.0585 / 481.3895 us
independent-median latency change = +0.068807%
paired median delta = +0.1325 us
paired mean delta = -0.107833 us
candidate wins = 2/6
parent / candidate MoE medians = 1313.980 / 1314.525 us
```

The mean's small favorable sign is inconsistent with the paired median,
win count, and separate medians. It does not justify promotion. This is a
lack-of-reproducible-gain decision, not a claim that every sample regresses.
`v3b_pair06` was interrupted when unrelated GPU PID 2321640 appeared; its
entire unit is excluded. `v3b_pair07` replaces it without overwriting its log.
The final timing postcheck at 16:46:51 UTC found no GPU/KFD owner, busy=0,
and the established 173,154,304-byte idle allocation.

### Capture exclusions and monitor startup audit

The first complete qualifying SIMD3 unit is `v3b_att_c_s3_06`, captured at
`20260918T164427Z_att`. The complete candidate view is:

| SIMD select | Retained unit | UTC run |
|---|---|---|
| 0 | `v3b_att_c_s0_01` | `20260918T161415Z_att` |
| 1 | `v3b_att_c_s1_01` | `20260918T161744Z_att` |
| 2 | `v3b_att_c_s2_01` | `20260918T162103Z_att` |
| 3 | `v3b_att_c_s3_06` | `20260918T164427Z_att` |

SIMD3 exclusions are preserved, not repaired by splicing wave fragments:

- `_01`: decode exited successfully, but SE2 had only 34 boundaries and
  62,023 WMMAs. The full unit fails completeness.
- `_02`: unrelated GPU PID 2313988 appeared; the monitor interrupted only
  this experiment's verified process groups.
- `_03`: unrelated GPU PID 2334836 appeared. The run was interrupted, and
  no completion marker was produced. It is not a valid capture.
- `_04` (`20260918T163946Z_att`): the benchmark exited zero and its monitor
  reported a clear final interval, but the monitor did not start until
  16:40:19.751 UTC, 33.751 seconds after the second-resolution run header.
  Only 6.505 seconds / 13 samples were monitored. This is insufficient
  coverage; the unit is excluded despite `kfd_gate_passed=true`.
- `_05`: the monitor was started first, but foreign GPU processes appeared
  before the benchmark launch. Launch was refused, with exit marker 125;
  no kernel experiment was run under this label.

For `_06` and the last timing pair, launch waited for the monitor's first
empty KFD sample and checked that no foreign-process event had appeared.
The accepted SIMD3 monitor covers 39.522 seconds / 79 samples, including
preflight and the traced process, with no foreign process or monitor error.
The earlier five timing units and retained SIMD0-2 captures were reviewed:
their first monitor event is empty, followed by the experiment's processes;
none has the late-start pattern of `_04`. In particular, timing monitor
startup precedes the first timestamped aiter initialization log in all five.
Periodic 0.5-second sampling is still not proof against arbitrarily short,
unobserved external activity. Prechecks and postchecks remain required.

### Complete V3-B trace comparison

All sixteen retained waves have exactly 36 tasks, 64,512 WMMAs, 9,216 EXPs,
9,216 RCPs, 16,416 B32 loads, 64,656 B128 loads, and 2,304 B64 stores.
The payload reduction is exactly 1,008 B128 loads/wave, or 28/task. Neither
the arithmetic nor the scale-read/output-store counts change.

`analyze_tensor_wait_after_wmma_trace.py` ran both modes of the canonical
`trace_segment_cycles.py` for all four captures and used
`my_code/analyze_att_capture.py` for occupancy analysis. The comparison
independently verifies that CSV wait-stall totals exactly equal decoded
totals. It covers all 35 adjacent body-task intervals plus first task and
final drain, not just the canonical parser's 18 alternating pairs/wave.

The reference is the earlier complete, same-boot V2 parent view
(`v2a_parent_analysis/metrics.json`), with identical ISA SHA, target CU0,
mask `0xf`, and workload. **It is not a contemporaneous paired ATT capture.**
Mean reference/candidate capture clocks are 2030.517/2027.416 MHz. Trace
deltas are diagnostic observations; the fresh untraced pairs above govern
the performance decision.

| Cycles per complete wave unless stated | Parent reference | V3-B | Change |
|---|---:|---:|---:|
| Entire wave | 1,037,212.313 | 1,039,170.813 | +0.189% |
| Mean body task | 28,648.986 | 28,767.295 | +0.413% |
| Median body task | 27,212 | 27,584 | +1.367% |
| First setup/task | 31,798.750 | 29,931.500 | -5.872% |
| Final drain | 2,699.063 | 2,384.000 | -11.673% |
| DScnt stall | 88,999.750 | 103,411.625 | +16.193% |
| Barrier stall | 180,127.188 | 176,902.250 | -1.790% |
| TENSORcnt stall | 36,337.875 | 31,414.938 | -13.548% |
| `s_wait_idle` stall | 5,765.313 | 5,345.063 | -7.289% |
| LDS-read issue attribution | 123,712.500 | 122,521.813 | -0.962% |
| LDS-write issue attribution | 19,556.813 | 14,529.500 | -25.706% |
| SALU/control issue attribution | 64,157.000 | 68,093.438 | +6.136% |

What matched the hypothesis: the unconsumed payload reads disappear exactly;
the retimed loop preserves numerical results; LDS-read, TENSORcnt, and final
drain exposure are lower than in the reference. The control attribution is
also lower than V3-A's 68,937.688 cycles/wave, but those independent captures
do not establish the causal size of a control-cost reduction.

What did not match: the body does not shorten, and `s_wait_dscnt 0x8` grows
from 64,875.188 to 82,335.563 cycles/wave (+26.914%). Total DScnt increases
enough to absorb the apparent reductions elsewhere. SALU/control attribution
remains above the original parent despite eliminating the duplicated exit
predicate. Fewer source instructions/bytes therefore do not establish a
shorter readiness-critical path. Code footprint and producer/consumer phase
changes are plausible contributors, not individually proven causes.

V3-B's explicit wait total is 318,165.063 cycles/wave, **30.6172%** of its
complete-wave span. The parent reference is 30.2018%.

| V3-B wait family/form | Cycles/wave | Whole-wave share |
|---|---:|---:|
| All barrier stalls | 176,902.250 | 17.0234% |
| All DScnt stalls | 103,411.625 | 9.9514% |
| All TENSORcnt stalls | 31,414.938 | 3.0231% |
| `s_wait_idle` | 5,345.063 | 0.5144% |
| KMcnt | 1,091.188 | 0.1050% |
| Cluster `s_barrier_wait 0xfffd` (subset) | 96,448.063 | 9.2813% |
| Workgroup `s_barrier_wait 0xffff` (subset) | 80,454.188 | 7.7422% |
| `s_wait_dscnt 0x8` (subset) | 82,335.563 | 7.9232% |
| `s_wait_dscnt 0x14` (subset) | 12,947.750 | 1.2460% |
| `s_wait_tensorcnt 0x2` (subset) | 28,578.938 | 2.7502% |

The two K-ring barrier PCs `0x7c84` and `0x998c` contribute 39,518.688 and
36,377.000 cycles/wave, together **7.3035%**. Their maximum observed
latencies are 14,076 and 11,316 cycles. DScnt PC `0x9408` contributes
21,250.563 cycles/wave (2.0450%, maximum latency 14,871); PC `0x76f4`
contributes 15,695.750 (1.5104%, maximum 14,057). These PCs belong to the
fresh V3-B code object and are not inferred by matching source line numbers.

Other long observations remain primarily on LDS instructions:

| V3-B PC | Instruction | Mean / maximum latency | Whole-wave issue-gap share |
|---|---|---:|---:|
| `0xb778` | `ds_store_b64 v91, v[118:119] offset:4624` | 42.227 / 13,511 | 0.1428% |
| `0x82f8` | `ds_load_b128 v[56:59], v73 offset:6144` | 5.847 / 8,664 | 0.0709% |
| `0xcdb4` | `ds_store_b64 v91, v[106:107] offset:48` | 34.622 / 7,838 | 0.1165% |
| `0xcda4` | `ds_store_b64 v91, v[102:103] offset:16` | 34.988 / 3,051 | 0.1177% |

These latency maxima are not architectural costs. Latency is not multiplied
by hit count and added to wait shares; issue-gap attribution includes
scheduling/dependency delays and is not functional-unit busy time.

### V3-B physical-WGP and owner balance

Each capture has 256 complete occupancy wave lifetimes, 64 physical WGPs,
and maximum concurrent wave-slot count one. Every sampled WGP has one
aggregate active episode and zero inter-episode idle gap.

| SIMD-select capture | WGPs | Per-SE final-end-span range, cycles | Mean imbalance | Maximum imbalance |
|---|---:|---:|---:|---:|
| 0 | 64 | 116-632 | 0.02887% | 0.06150% |
| 1 | 64 | 107-2,522 | 0.07111% | 0.24378% |
| 2 | 64 | 92-4,204 | 0.10923% | 0.40247% |
| 3 | 64 | 259-2,372 | 0.09178% | 0.22815% |

Mean/max imbalance across capture x SE rows is **0.07525% / 0.40247%**,
versus 0.01971% / 0.03414% for the parent reference. The candidate has a
larger sampled completion tail, but it remains small relative to the
recurring barrier/DScnt exposure. This does not identify an all-GPU load
imbalance as the dominant cause. Mask `0xf` covers only SE0-3 / 64 WGPs;
the four captures repeat that coverage, not 256 distinct WGPs.

| Owner | Parent reference body cycles/task | V3-B body cycles/task | V3-B barrier / DScnt / TENSORcnt shares |
|---|---:|---:|---|
| A | 28,996.800 | 28,398.586 | 15.36% / 9.62% / 5.05% |
| B | 28,534.307 | 29,054.764 | 15.39% / 9.63% / 4.20% |
| ScaleA | 28,476.600 | 28,561.086 | 20.40% / 9.08% / 0.66% |
| ScaleB | 28,588.236 | 29,054.743 | 16.99% / 11.45% / 2.16% |

Owner effects are not uniform. Barriers can propagate late-arrival costs
while keeping final WGP completion close. Sequential captures and raw
cross-SIMD shader clocks do not establish simultaneous producer arrival
times or the identity of the last wave at every barrier.

### V3 decision, cleanup, and stopping point

**Neither V3-A nor V3-B is promoted.** The accepted ISA and
`benchmark_history.sh` remain unchanged, and no new named case is added.
Random verification remains identical to the accepted parent; no arithmetic,
precision mode, ABI, resource allocation, or production package change is
retained from these experiments.

V3-B's source audit is archived as `v3b_source_audit.json`, SHA256
`68955c46772b289dc81e9afe80f39260c85c8d9b594862117a74025fc168cfa0`.
After validating that archive, the rejected ISA, its top-level audit
duplicate, the standalone `.o`/`.co`, both V3-only builders, and their local
bytecode were removed. No matching remote builder bytecode remained.
Retained artifacts include raw run directories, code objects belonging to captured evidence,
capture manifests/integrity records, timing/monitor logs, and reusable
analysis tools. Those archives permit later reconstruction without leaving
a rejected candidate in the active implementation set.

The complete evidence is under
`history_runs/tensor_wait_opt_rebaseline_20260918/`, notably
`v3b_candidate_metrics.json`, `v3b_trace_comparison.json`,
`v3b_capture_units.json`, and `v3b_pair*_e2e.tsv` in the local mirror.
The remote full analysis is `v3b_candidate_analysis/metrics.json`; its input
paths in the JSON refer to the remote raw captures.

The iteration is paused after seven tested ideas across V0-V3. Repeating
wait motion, scale issue-count reduction, bank-reset removal, or final-tail
read suppression has no new evidence-backed benefit. Larger changes would
need a new LDS/register-layout or producer-scheduling design with explicit
dependency proofs and performance-counter evidence; no safe, specific next
implementation is established by the current traces. This is not a claim
that the hardware or kernel has reached a theoretical limit. See the
[iteration summary](TENSOR_WAIT_OPT_SUMMARY.md) for the retained starting
point, all decisions, and requirements for a useful next experiment.
