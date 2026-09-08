#!/usr/bin/env bash
set -euo pipefail

cd /data/yanguahe/code/wk_sp1/aiter

kernel=moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1
output_name=opt_const0_att
test_cmd="python my_code/moe_gemm1_act1_optimized/att_launch_opt.py"

TRACE_ROOT=my_code/moe_gemm1_act1_optimized/att \
HIP_VISIBLE_DEVICES=0 \
bash my_code/get_isa_runner_att.sh \
  "$kernel" \
  "$output_name" \
  "$test_cmd" \
  --ana-att
