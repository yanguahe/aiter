# `--trigger-lazy-else` Performance Analysis

## Scope

This report analyzes why `--trigger-lazy-else` runs faster for:

```text
B=1, S=8192, H=16, Hkv=16, D=128, dtype=bf16, causal, kv_sp=1, dense
```

Environment and sources:

- GPU: AMD Instinct MI355X
- Timing source: `FlyDSL/tt2.log`
- Base occupancy trace: `FlyDSL/thread_trace/fyd_fmha_dws_b1_s8192_h16_d128_causal/ui_output_agent_7739_dispatch_26/occupancy.json`
- Trigger occupancy trace: `FlyDSL/thread_trace/fyd_fmha_dws_b1_s8192_h16_d128_causal_lazy_rescale_o/ui_output_agent_31649_dispatch_29/occupancy.json`
- Kernel options:

```python
{
    "waves_per_eu": 2,
    "daz": True,
    "dualwave_swp_lazy_rescale": True,
    "dualwave_swp_setprio": True,
    "dualwave_swp_debug_lazy_counts": False,
    "dualwave_swp_enable_stagger": True,
}
```

## Executive Summary

`--trigger-lazy-else` is faster, but not because FlyDSL's lazy-rescale branch skips more work, and not because occupancy/load balance improves.

The strongest evidence is:

- FlyDSL, `aiter_ck`, and `aiter_asm` all get faster on the trigger input.
- `aiter_ck` and `aiter_asm` do not use FlyDSL's lazy-rescale implementation.
- Base random input already has `all_below_false_count = 0`, meaning it always takes the lazy skip path in the measured debug-count run.
- Trigger input has `all_below_false_count = 3968`, so it actually executes some additional lazy-rescale else work.
- Occupancy/load-balance metrics are almost unchanged between the two traces.

The most consistent explanation is data-dependent hardware behavior:

`--trigger-lazy-else` creates a zero-heavy / highly sparse attention distribution. Most K tiles are zero, and after the high-score K tile is seen, most P values become zero or effectively zero. MFMA instruction count stays the same, and this does not require dense BF16 MFMA to skip zero operands. The narrower hypothesis is that operand bit switching and related data activity are lower, reducing dynamic power pressure and allowing a higher effective clock or less power throttling in normal wall-clock benchmark timing.

This DVFS/power explanation is an inference from the data here. The direct validation would be to repeat the benchmark with fixed SCLK/MCLK or collect power/SCLK counters.

## Timing Data

Latest valid timing table from `tt2.log`:

| kernel | random/base | trigger-lazy-else | speedup |
|---|---:|---:|---:|
| FlyDSL | 270.6 us | 214.4 us | 1.26x |
| aiter_ck | 338.5 us | 287.4 us | 1.18x |
| aiter_asm | 243.9 us | 186.1 us | 1.31x |

Interpretation:

- The speedup is not FlyDSL-specific.
- The trigger input speeds up three independent implementations.
- Therefore the primary cause is the input data distribution, not a FlyDSL-only lazy-rescale shortcut.

## Trigger Input Distribution

`--trigger-lazy-else` constructs:

```python
q_t.fill_(1.0)
k_t.zero_()
k_t[:, 64:128, :, :].fill_(80.0)
```

For `D=128`, the score for the high-K tile is approximately:

```text
Q dot K / sqrt(D) = 128 * 80 / sqrt(128) ~= 905
```

Thus:

- K tiles outside `[64, 128)` are zero.
- The `[64, 128)` K tile dominates the row maximum once it is visible.
- Later score-0 tiles have probability near zero after softmax.
- Fast-math / denormal-flush behavior can make some tiny intermediate probabilities effectively zero in the softmax/VALU path.
- QK GEMM has many zero K operands.
- PV/GEMM2 has many zero P operands.

The nominal FLOP count is unchanged, and MFMA instructions still issue, but the actual data activity in the datapath is likely much lower.

Hardware caveat:

- CDNA4 exposes denormal-control behavior, and some F32 transcendental operations flush denormals.
- CDNA4 BF16/F16 dense MFMA operands are documented as less-than-32-bit floating inputs whose denormal behavior does not follow `MODE.denorm` flush control.
- Therefore, this report does not assume dense BF16 MFMA automatically flushes tiny operands or skips zero operands. The power/DVFS hypothesis only requires lower operand/data switching activity while the same MFMA instructions issue.

## Lazy-Rescale Branch Counts

Debug command used earlier:

```bash
python3 tests/kernels/test_flash_attn_fwd.py \
  --causal --dtype bf16 \
  --batch 1 --seq_len 8192 \
  --num_heads 16 --num_kv_heads 16 \
  --head_dim 128 --num_kv_splits 1 \
  --debug-lazy-counts
```

With and without `--trigger-lazy-else`:

| input mode | all_below_true_count | all_below_false_count | false share |
|---|---:|---:|---:|
| base random input | 253952 | 0 | 0.00% |
| `--trigger-lazy-else` | 249984 | 3968 | 1.56% |

Interpretation:

- Base random input already takes the lazy skip branch everywhere in this debug-count run.
- Trigger input takes the else branch 3968 times.
- Therefore trigger mode is not faster because it skips more lazy-rescale work.
- If anything, this branch-count evidence alone would predict trigger mode to be slightly slower.

## Occupancy Summary

The two occupancy traces have nearly the same load-balance shape:

| metric | base occupancy trace | trigger-lazy-else occupancy trace | delta |
|---|---:|---:|---:|
| trace ts span | 497436 | 503736 | +6300 / +1.27% |
| busy CUs | 26 | 26 | same |
| max slots/SIMD | 2 | 2 | same |
| over-subscribed CUs | 26 | 26 | same |
| duration median | 381034 | 384068 | +3034 / +0.80% |
| duration max | 495120 | 501252 | +6132 / +1.24% |
| end-span | 230884 | 234368 | +3484 / +1.51% |
| imbalance | 0.61 | 0.61 | same |
| max summed active cycles | 495044 | 501116 | +6072 / +1.23% |
| max start-to-end span | 495100 | 501236 | +6136 / +1.24% |

Interpretation:

- Occupancy and load balance are not better in trigger mode.
- Trigger mode has a slightly longer maximum active-cycle tail in the ATT occupancy trace.
- This is opposite to the normal wall-clock timing, where trigger mode is faster.
- Therefore occupancy active cycles do not explain the speedup.

Important distinction:

- `occupancy.json` timestamps are shader-clock-domain wave-slot lifetime timestamps under ATT profiling.
- `tt2.log` timing is normal wall-clock kernel timing.
- These are not the same metric.

## Full Base Slot Active-Cycle Table

Each row reports summed active cycles for the eight `(SIMD, slot)` entries in that sampled CU, followed by min/median/max gap between the two active intervals in that row.

| SE | CU | s0.sl0 | s0.sl1 | s1.sl0 | s1.sl1 | s2.sl0 | s2.sl1 | s3.sl0 | s3.sl1 | gap_min | gap_med | gap_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SE0 | CU0 | 453616 | 457428 | 453632 | 457632 | 453416 | 457480 | 453540 | 457600 | 60 | 1142 | 2184 |
| SE0 | CU1 | 490368 | 494440 | 490388 | 494448 | 490336 | 494388 | 490448 | 494504 | 312 | 1434 | 2560 |
| SE0 | CU2 | 223260 | 227348 | 223136 | 227204 | 222964 | 227028 | 223224 | 227256 | 60 | 1164 | 2244 |
| SE0 | CU3 | 261540 | 265616 | 261688 | 265780 | 261512 | 265572 | 261496 | 265556 | 264 | 1354 | 2436 |
| SE0 | CU4 | 300008 | 304068 | 300136 | 304216 | 299968 | 304028 | 300204 | 304256 | 56 | 1182 | 2312 |
| SE0 | CU5 | 338724 | 342792 | 338608 | 342680 | 338424 | 342480 | 338676 | 342716 | 60 | 1148 | 2228 |
| SE0 | CU6 | 375760 | 379816 | 375656 | 379712 | 375476 | 379532 | 375984 | 379996 | 1196 | 2318 | 3456 |
| SE0 | CU7 | 415236 | 419300 | 415376 | 419452 | 415380 | 419448 | 415492 | 419560 | 56 | 1160 | 2272 |
| SE1 | CU1 | 241900 | 245960 | 242032 | 246092 | 241856 | 245932 | 241820 | 245876 | 64 | 1150 | 2240 |
| SE1 | CU2 | 281084 | 285136 | 281180 | 285228 | 281112 | 285176 | 281284 | 285092 | 64 | 1076 | 2208 |
| SE1 | CU3 | 318576 | 322376 | 318880 | 322824 | 318464 | 322624 | 318636 | 322680 | 60 | 1168 | 2332 |
| SE1 | CU4 | 356588 | 360652 | 356856 | 360912 | 356636 | 360700 | 356856 | 360900 | 64 | 1180 | 2312 |
| SE1 | CU5 | 395208 | 399012 | 395524 | 399460 | 395112 | 399248 | 395268 | 399324 | 60 | 1166 | 2328 |
| SE1 | CU6 | 432972 | 437040 | 433372 | 437416 | 433196 | 437256 | 433164 | 437224 | 60 | 1184 | 2332 |
| SE1 | CU7 | 471768 | 475832 | 471992 | 476060 | 471920 | 475972 | 471964 | 476036 | 60 | 1172 | 2276 |
| SE1 | CU8 | 204800 | 208536 | 204692 | 208788 | 204768 | 208836 | 204728 | 208788 | 52 | 1114 | 2212 |
| SE2 | CU0 | 376772 | 380328 | 376420 | 380476 | 376864 | 380780 | 376580 | 380664 | 56 | 1142 | 2220 |
| SE2 | CU1 | 414632 | 418700 | 414648 | 418716 | 414592 | 418660 | 414456 | 418516 | 56 | 1170 | 2280 |
| SE2 | CU2 | 452652 | 456720 | 452680 | 456736 | 452608 | 456672 | 452748 | 456800 | 60 | 1144 | 2232 |
| SE2 | CU3 | 490964 | 494532 | 491264 | 495044 | 490652 | 494868 | 490796 | 494860 | 56 | 1110 | 2224 |
| SE2 | CU4 | 223436 | 227492 | 223356 | 227388 | 223176 | 227240 | 223316 | 227384 | 56 | 1168 | 2288 |
| SE2 | CU5 | 261432 | 265500 | 261444 | 265512 | 261568 | 265624 | 261552 | 265608 | 68 | 1186 | 2320 |
| SE2 | CU6 | 299536 | 303588 | 299428 | 303500 | 299276 | 303348 | 299676 | 303712 | 68 | 1172 | 2316 |
| SE2 | CU7 | 337944 | 342016 | 338144 | 342212 | 338072 | 342132 | 338120 | 342196 | 68 | 1162 | 2264 |
| SE3 | CU1 | 471580 | 475640 | 471804 | 475848 | 471604 | 475600 | 471740 | 475856 | 264 | 1350 | 2440 |
| SE3 | CU2 | 204444 | 208500 | 204716 | 208756 | 204724 | 208788 | 204644 | 208460 | 60 | 1168 | 2196 |
| SE3 | CU3 | 242828 | 246864 | 242692 | 246756 | 242584 | 246664 | 242636 | 246712 | 56 | 1166 | 2312 |
| SE3 | CU4 | 280976 | 285040 | 280976 | 285056 | 281296 | 285328 | 280936 | 284992 | 64 | 1184 | 2328 |
| SE3 | CU5 | 318504 | 322564 | 318644 | 322696 | 318756 | 322824 | 318668 | 322728 | 52 | 1170 | 2292 |
| SE3 | CU6 | 356912 | 360972 | 357256 | 361312 | 357008 | 361076 | 357152 | 361200 | 60 | 1188 | 2316 |
| SE3 | CU7 | 395092 | 399156 | 395324 | 399360 | 395116 | 399188 | 395252 | 399328 | 348 | 1432 | 2520 |
| SE3 | CU8 | 433544 | 437612 | 433520 | 437576 | 433336 | 437388 | 433720 | 437768 | 388 | 1476 | 2556 |
| MIN | all | 204444 | 208500 | 204692 | 208756 | 204724 | 208788 | 204644 | 208460 | 52 | 1076 | 2184 |
| MED | all | 347656 | 351722 | 347732 | 351796 | 347530 | 351590 | 347766 | 351808 | 60 | 1169 | 2302 |
| MAX | all | 490964 | 494532 | 491264 | 495044 | 490652 | 494868 | 490796 | 494860 | 1196 | 2318 | 3456 |

## Full Trigger Slot Active-Cycle Table

| SE | CU | s0.sl0 | s0.sl1 | s1.sl0 | s1.sl1 | s2.sl0 | s2.sl1 | s3.sl0 | s3.sl1 | gap_min | gap_med | gap_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SE0 | CU0 | 264816 | 268880 | 265060 | 269140 | 264744 | 268804 | 265140 | 269200 | 56 | 1160 | 2252 |
| SE0 | CU1 | 302632 | 306380 | 302768 | 306688 | 302432 | 306320 | 302440 | 306488 | 112 | 1212 | 2344 |
| SE0 | CU2 | 340468 | 344528 | 340468 | 344532 | 340264 | 344320 | 340652 | 344704 | 448 | 1546 | 2632 |
| SE0 | CU3 | 379728 | 383796 | 380160 | 384204 | 379848 | 383912 | 379916 | 383976 | 56 | 1164 | 2276 |
| SE0 | CU4 | 418228 | 422288 | 418612 | 422648 | 418348 | 422408 | 418472 | 422536 | 60 | 1148 | 2232 |
| SE0 | CU5 | 458424 | 462500 | 458424 | 462504 | 458212 | 462284 | 458604 | 462660 | 56 | 1154 | 2248 |
| SE0 | CU6 | 495128 | 499188 | 495580 | 499628 | 495364 | 499428 | 495448 | 499520 | 1592 | 2720 | 3836 |
| SE0 | CU7 | 226224 | 230300 | 226104 | 230164 | 226208 | 230280 | 226356 | 230436 | 64 | 1196 | 2308 |
| SE1 | CU1 | 361368 | 365432 | 361288 | 365352 | 361140 | 365200 | 361532 | 365572 | 60 | 1186 | 2332 |
| SE1 | CU2 | 398732 | 402796 | 398904 | 402972 | 398748 | 402816 | 398884 | 402932 | 60 | 1188 | 2332 |
| SE1 | CU3 | 439168 | 443228 | 439040 | 443112 | 439012 | 443080 | 439188 | 443248 | 56 | 1146 | 2240 |
| SE1 | CU4 | 476248 | 480328 | 476640 | 480696 | 476540 | 480616 | 476332 | 480396 | 64 | 1146 | 2240 |
| SE1 | CU5 | 207408 | 211196 | 207632 | 211592 | 207284 | 211432 | 207336 | 211428 | 52 | 1122 | 2300 |
| SE1 | CU6 | 246144 | 250240 | 246220 | 250276 | 246140 | 250192 | 246252 | 250308 | 60 | 1192 | 2336 |
| SE1 | CU7 | 284264 | 288320 | 284052 | 288108 | 284148 | 288212 | 284404 | 288456 | 60 | 1182 | 2320 |
| SE1 | CU8 | 322336 | 326376 | 322244 | 326316 | 322088 | 326156 | 322464 | 326516 | 60 | 1190 | 2328 |
| SE2 | CU0 | 226012 | 229572 | 226068 | 230056 | 226060 | 230044 | 225888 | 229944 | 52 | 1180 | 2252 |
| SE2 | CU1 | 264380 | 268444 | 264252 | 268328 | 264240 | 268312 | 264376 | 268424 | 60 | 1138 | 2224 |
| SE2 | CU2 | 302552 | 306608 | 302448 | 306512 | 302256 | 306320 | 302764 | 306796 | 56 | 1184 | 2324 |
| SE2 | CU3 | 341128 | 345192 | 340880 | 344944 | 340896 | 344964 | 341152 | 345196 | 56 | 1148 | 2240 |
| SE2 | CU4 | 379424 | 383492 | 379572 | 383620 | 379368 | 383436 | 379620 | 383668 | 60 | 1178 | 2316 |
| SE2 | CU5 | 418604 | 422664 | 418488 | 422560 | 418484 | 422548 | 418856 | 422904 | 52 | 1158 | 2268 |
| SE2 | CU6 | 456352 | 460416 | 456360 | 460428 | 456112 | 460180 | 456632 | 460664 | 52 | 1180 | 2328 |
| SE2 | CU7 | 496864 | 500936 | 496764 | 500836 | 496564 | 500628 | 497076 | 501116 | 56 | 1192 | 2336 |
| SE3 | CU1 | 322296 | 326356 | 322080 | 326148 | 322188 | 326252 | 322440 | 326488 | 216 | 1336 | 2476 |
| SE3 | CU2 | 360068 | 364124 | 360368 | 364428 | 360284 | 364348 | 360156 | 364220 | 472 | 1562 | 2652 |
| SE3 | CU3 | 399128 | 403192 | 399404 | 403452 | 399216 | 403280 | 399260 | 403328 | 52 | 1146 | 2240 |
| SE3 | CU4 | 438172 | 442244 | 438368 | 442432 | 438288 | 442344 | 438148 | 442212 | 344 | 1430 | 2520 |
| SE3 | CU5 | 475224 | 479292 | 475612 | 479660 | 475508 | 479568 | 475296 | 479356 | 64 | 1146 | 2232 |
| SE3 | CU6 | 207244 | 211308 | 207224 | 211280 | 207132 | 211212 | 207488 | 211496 | 56 | 1134 | 2264 |
| SE3 | CU7 | 245080 | 249144 | 245124 | 249200 | 245044 | 249100 | 245184 | 249248 | 1092 | 2194 | 3280 |
| SE3 | CU8 | 283824 | 287912 | 284012 | 288080 | 283844 | 287908 | 283972 | 288016 | 368 | 1490 | 2636 |
| MIN | all | 207244 | 211196 | 207224 | 211280 | 207132 | 211212 | 207336 | 211428 | 52 | 1122 | 2224 |
| MED | all | 350598 | 354658 | 350624 | 354686 | 350590 | 354656 | 350654 | 354708 | 60 | 1181 | 2318 |
| MAX | all | 496864 | 500936 | 496764 | 500836 | 496564 | 500628 | 497076 | 501116 | 1592 | 2720 | 3836 |

## Detailed Reasoning

### 1. Timing proves the effect is data-driven, not FlyDSL-specific

All three implementations are faster under `--trigger-lazy-else`:

- FlyDSL: `270.6 us -> 214.4 us`
- `aiter_ck`: `338.5 us -> 287.4 us`
- `aiter_asm`: `243.9 us -> 186.1 us`

Because `aiter_ck` and `aiter_asm` do not use FlyDSL's lazy-rescale branch implementation, the common factor is the data distribution produced by `--trigger-lazy-else`.

### 2. Lazy-rescale skip count does not explain the speedup

Base random input has:

```text
all_below_true_count = 253952
all_below_false_count = 0
```

Trigger input has:

```text
all_below_true_count = 249984
all_below_false_count = 3968
```

Therefore the base input already takes the lazy skip path everywhere in that debug run. Trigger mode actually takes the else branch in 3968 cases. The speedup cannot come from "more lazy skips."

### 3. Occupancy and load balance do not explain the speedup

The two occupancy traces show:

- Same busy CU count: `26`
- Same max slots/SIMD: `2`
- Same over-subscribed CU count: `26`
- Same imbalance: `0.61`
- Trigger trace has slightly larger maximum active tail: `501116` vs `495044` summed active cycles

If occupancy/load balance were the cause, we would expect trigger mode to have lower tail cycles or better imbalance. It does not.

### 4. Most likely mechanism: lower switching activity and power pressure

The trigger input produces many zero or near-zero operands:

- QK MFMA: most K tiles are zero.
- Softmax: after the high-score tile, most later probabilities are effectively zero.
- PV MFMA: many P operands are zero.

MFMA instructions still issue, but the datapath can toggle fewer bits. This is a data-activity explanation, not a claim that dense BF16 MFMA has a hidden zero-skip path. On MI355X, normal benchmark wall time can be affected by effective clock and power management. A zero-heavy workload can run faster in wall-clock time even when shader-clock-domain occupancy cycles are not lower.

This explains the apparent contradiction:

- ATT occupancy active cycles are slightly higher for trigger mode.
- Normal benchmark wall time is much lower for trigger mode.

They measure different things. The first is a profiled shader-clock-domain wave lifetime; the second is real wall-clock kernel duration.

## Recommended Validation

To confirm the power/DVFS explanation:

1. Fix SCLK and MCLK, then rerun base vs trigger timing.
2. Collect actual SCLK, power, and activity counters for both inputs.
3. Compare instruction counts and per-category trace stats. The expected result is similar instruction mix and occupancy, but lower power or higher effective frequency for trigger mode.

Expected outcome:

- If fixed clocks greatly reduce the trigger speedup, the main cause is DVFS/power.
- If fixed clocks preserve most of the speedup, investigate a data-dependent pipeline effect beyond occupancy, such as lower stall on particular instruction classes. Treat any hardware zero-operand behavior as a separate hypothesis that needs direct counter or microbenchmark evidence.
