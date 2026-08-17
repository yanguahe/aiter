export ENABLE_CK=0
export AITER_MOE_EXPERT_BALANCE=true
export AITER_LOG_MORE=1
export AITER_USE_GROUPED_GEMM=1
export AITER_GROUPED_DEBUG=0
export AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1

# Overridable from the caller: HIP_VISIBLE_DEVICES=1 FLYDSL_DUMP_IR=0 bash my_code/run_gemm.sh
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export FLYDSL_DUMP_IR="${FLYDSL_DUMP_IR:-0}"
export FLYDSL_DUMP_DIR="${FLYDSL_DUMP_DIR:-./my_code/flydsl_dump}"


python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
--scenario bench \
--data-format a4w4 \
--experts 96 \
--tokens 512 \
--topk 6 \
--iters 100 \
--model-dim 7168 \
--inter-dim 3072 \
--act silu \
--no-bias \
--no-check-aot-cache
