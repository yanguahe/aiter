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
- `gemm_batch_isa_runner.py`, `gemm_isa_runner.py`: isolated copies of the
  validated runner used for testing older remote checkouts.
- `test_optimized.sh`: quick correctness and formal benchmark commands.
- `run_att_const0.sh`: collect and analyze one const0 ATT capture with
  `get_isa_runner_att.sh --ana-att`; trace output stays under this directory.
- `att_launch_opt.py`: minimal one-launch ATT target that loads the precompiled
  `act1_opt.co`, avoiding clang/COMGR execution inside rocprof.
- `HARDWARE_CYCLE_LIMIT.md`: derives the MXFP4 compute, memory/TDM/LDS, and
  exact-kernel instruction/occupancy cycle ceilings for the const0 workload.
- `att_const0_analyze.log`: preserved `--ana-att` output used by the cycle-limit
  analysis.

## Regenerate

```bash
python my_code/moe_gemm1_act1_optimized/build_optimized.py
```

## Test

```bash
bash my_code/moe_gemm1_act1_optimized/test_optimized.sh quick-random
bash my_code/moe_gemm1_act1_optimized/test_optimized.sh perf-random
bash my_code/moe_gemm1_act1_optimized/test_optimized.sh perf-const0
bash my_code/moe_gemm1_act1_optimized/run_att_const0.sh
```

The optimized epilogue swaps each `G0,U0,G1,U1` accumulator group into
contiguous gate/up pairs, uses `v_pk_mul_f32` and `v_dual_*`, and pipelines the
next eight-output batch through the current batch's `EXP/RCP` latency slots.
The first `tensor_store_from_lds` remains followed by `s_wait_tensorcnt 0x0`
before LDS reuse.
