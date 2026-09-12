# Persistent-overlap output-pad8 kernel

This document describes the retained output-LDS padding optimization for the
standalone MoE GEMM1 assembly path.  Experimental exactopt, WPT, XDL-arbitration,
and two-address-store candidates were removed after benchmarking; their measured
results remain in `PERSISTENT_OVERLAP_PAD8_PORT_RESULTS.md`.

Retained baseline:

```text
moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent_overlap.s
SHA256=ec3af906acebfdef77db5f734d01fbef45035dfcfdd799f557693aa73dd0d601
```

Retained optimized kernel:

```text
persistent_overlap_output_pad8.s
SHA256=039ed787b1f136ee402b76e3bd0b7c9bf439148a0156d0fcca856ed6ab25ccad
```

The optimized kernel keeps the compute core, ABI, persistent grid, cluster
protocol, and output values unchanged.  It changes each wave's output LDS row
pitch from 128 to 144 bytes.  Output TDM keeps a 128-byte global bound, so the
extra 16 bytes per row are discarded as out-of-bounds columns.

Generate the retained kernel from the fixed baseline:

```bash
python my_code/moe_gemm1_act1_optimized/build_persistent_overlap_pad8.py
```

Run the static contract and LDS-layout audit:

```bash
python my_code/moe_gemm1_act1_optimized/audit_persistent_overlap_pad8.py
```

Run random MoE e2e correctness:

```bash
AITER_HISTORY_CASE_LIST=persistent_overlap,persistent_overlap_pad8 \
ROUNDS=3 RUN_VERIFY=0 RUN_ATT=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-random
```

Run the canonical const0 MoE e2e benchmark:

```bash
AITER_HISTORY_CASE_LIST=persistent_overlap,persistent_overlap_pad8 \
ROUNDS=5 RUN_VERIFY=0 RUN_ATT=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-const0
```

Capture and analyze ATT:

```bash
AITER_HISTORY_CASE_LIST=persistent_overlap,persistent_overlap_pad8 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh att

python3 my_code/moe_gemm1_act1_optimized/trace_segment_cycles.py \
  my_code/moe_gemm1_act1_optimized/pad8_trace_full_task.json

python3 my_code/moe_gemm1_act1_optimized/analyze_pad8_thread_trace.py
```

Supporting reports:

- `PERSISTENT_OVERLAP_PAD8_PORT_RESULTS.md`
- `PERSISTENT_OVERLAP_PAD8_THREAD_TRACE_ANALYSIS.md`
- `PERSISTENT_NEXT_TASK_INPUT_PREFETCH_ANALYSIS.md`
