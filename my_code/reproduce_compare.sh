#!/usr/bin/env bash
set -Eeuo pipefail

# Reproduce same-machine comparisons for the gfx1250 E96/T16384 GEMM1 target.
# Run this script directly inside the existing ROCm container.

REPO_ROOT="${REPO_ROOT:-/data/yanguahe/code/wk_sp1/aiter}"
ROUNDS="${ROUNDS:-2}"
RUN_VERIFY="${RUN_VERIFY:-1}"
RUN_ATT="${RUN_ATT:-0}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-hyg_gfx1250_gemm_a4w4}"

if [[ ! -f /.dockerenv ]]; then
    echo "This script must be run inside the ROCm container." >&2
    exit 2
fi

cd "${REPO_ROOT}"

read_git_head_without_git() {
    local head ref ref_file
    if [[ ! -f "${REPO_ROOT}/.git/HEAD" ]]; then
        printf '%s\n' unknown
        return
    fi
    IFS= read -r head <"${REPO_ROOT}/.git/HEAD"
    if [[ "${head}" != "ref: "* ]]; then
        printf '%s\n' "${head}"
        return
    fi
    ref="${head#ref: }"
    ref_file="${REPO_ROOT}/.git/${ref}"
    if [[ -f "${ref_file}" ]]; then
        IFS= read -r head <"${ref_file}"
        printf '%s\n' "${head}"
        return
    fi
    awk -v ref="${ref}" '$2 == ref { print $1; found = 1; exit } END { if (!found) print "unknown" }' \
        "${REPO_ROOT}/.git/packed-refs" 2>/dev/null || printf '%s\n' unknown
}

read_git_branch_without_git() {
    local head
    if [[ ! -f "${REPO_ROOT}/.git/HEAD" ]]; then
        printf '%s\n' unknown
        return
    fi
    IFS= read -r head <"${REPO_ROOT}/.git/HEAD"
    if [[ "${head}" == "ref: refs/heads/"* ]]; then
        printf '%s\n' "${head#ref: refs/heads/}"
    else
        printf '%s\n' detached
    fi
}

branch="$(read_git_branch_without_git)"
head="$(read_git_head_without_git)"
if [[ "${branch}" != "${EXPECTED_BRANCH}" ]]; then
    echo "Refusing to run on branch '${branch}'; expected '${EXPECTED_BRANCH}'." >&2
    exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
host="$(hostname -s)"
out_rel="my_code/gemm1_cycle_105pct_20260909/runs/${host}_${timestamp}"
out_dir="${REPO_ROOT}/${out_rel}"
mkdir -p "${out_dir}"

declare -a COMMON_ENV=(
    AITER_USE_GROUPED_GEMM=1
    AITER_GROUPED_DEBUG=0
    ENABLE_CK=0
    FLYDSL_DUMP_IR=0
    AITER_LOG_MORE=1
    AITER_MOE_EXPERT_BALANCE=true
    AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1
)

declare -a TEST_SHAPE=(
    --data-format a4w4
    --experts 96
    --tokens 16384
    --topk 6
    --model-dim 7168
    --inter-dim 3072
    --act silu
    --no-bias
    --no-check-aot-cache
)

declare -a CASES=(
    baseline
    sync_mg4_fc8
    sync_mg2_fc12
    sync_mg4_fc28
    sync_mg4_fc28_hard
    sync_mg4_fc28_relu
)

# Optional comma-separated subset, preserving the caller's requested order.
# Example: CASE_LIST=baseline,sync_mg4_fc28,sync_mg4_fc28_hard
if [[ -n "${CASE_LIST:-}" ]]; then
    IFS=',' read -r -a CASES <<<"${CASE_LIST}"
fi

declare -a CASE_ENV=()
case_env() {
    case "$1" in
        baseline)
            CASE_ENV=(
                AITER_FLYDSL_GEMM1_MMA_GROUP=4
                AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=8
                AITER_FLYDSL_GEMM1_SILU_HARD=0
                AITER_FLYDSL_GEMM1_SILU_RELU=0
            )
            ;;
        sync_mg4_fc8)
            CASE_ENV=(
                AITER_FLYDSL_GEMM1_MMA_GROUP=4
                AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=8
                AITER_FLYDSL_GEMM1_SILU_HARD=0
                AITER_FLYDSL_GEMM1_SILU_RELU=0
            )
            ;;
        sync_mg2_fc12)
            CASE_ENV=(
                AITER_FLYDSL_GEMM1_MMA_GROUP=2
                AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=12
                AITER_FLYDSL_GEMM1_SILU_HARD=0
                AITER_FLYDSL_GEMM1_SILU_RELU=0
            )
            ;;
        sync_mg4_fc28)
            CASE_ENV=(
                AITER_FLYDSL_GEMM1_MMA_GROUP=4
                AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=28
                AITER_FLYDSL_GEMM1_SILU_HARD=0
                AITER_FLYDSL_GEMM1_SILU_RELU=0
            )
            ;;
        sync_mg4_fc28_hard)
            CASE_ENV=(
                AITER_FLYDSL_GEMM1_MMA_GROUP=4
                AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=28
                AITER_FLYDSL_GEMM1_SILU_HARD=1
                AITER_FLYDSL_GEMM1_SILU_RELU=0
            )
            ;;
        sync_mg4_fc28_relu)
            CASE_ENV=(
                AITER_FLYDSL_GEMM1_MMA_GROUP=4
                AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=28
                AITER_FLYDSL_GEMM1_SILU_HARD=0
                AITER_FLYDSL_GEMM1_SILU_RELU=1
            )
            ;;
        *)
            echo "Unknown case: $1" >&2
            return 2
            ;;
    esac
}

case_kernel() {
    local suffix=""
    case "$1" in
        baseline) ;;
        sync_mg4_fc8) ;;
        sync_mg2_fc12) suffix="_mg2_fc12" ;;
        sync_mg4_fc28) suffix="_mg4_fc28" ;;
        sync_mg4_fc28_hard) suffix="_mg4_fc28_silu_hard" ;;
        sync_mg4_fc28_relu) suffix="_mg4_fc28_silu_relu" ;;
        *) return 2 ;;
    esac
    printf '%s%s\n' \
        'a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_wpt1_eb8_sh_rcw' \
        "${suffix}"
}

{
    echo "date_utc=${timestamp}"
    echo "host=${host}"
    echo "branch=${branch}"
    echo "head=${head}"
    echo "rounds=${ROUNDS}"
    echo "run_verify=${RUN_VERIFY}"
    echo "run_att=${RUN_ATT}"
    echo "execution=inside-container"
    echo "moe_e2e_metric=fused_moe end-to-end us"
    echo "command=python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py --scenario bench ${TEST_SHAPE[*]} --iters 20 --const-init 0"
    echo
    echo "task source hashes:"
    sha256sum \
        my_code/reproduce_compare.sh \
        aiter/ops/flydsl/grouped_gemm_mxfp4.py \
        aiter/ops/flydsl/grouped_moe_gfx1250.py \
        aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py
} | tee "${out_dir}/environment.log"

verify_tsv="${out_dir}/verify.tsv"
bench_tsv="${out_dir}/bench.tsv"
printf 'case\treturn_code\tlogits_diff\trel_l2\tpass\toutput_sha256\n' >"${verify_tsv}"
printf 'round\torder\tcase\treturn_code\tgemm1_us\tgemm2_us\tmoe_e2e_us\toutput_sha256\n' >"${bench_tsv}"

run_verify_case() {
    local name="$1"
    local log="${out_dir}/verify_${name}.log"
    local rc logits rel pass hash
    case_env "${name}"
    set +e
    env "${COMMON_ENV[@]}" "${CASE_ENV[@]}" \
        python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
        --scenario verify "${TEST_SHAPE[@]}" --iters 1 \
        2>&1 | tee "${log}"
    rc=${PIPESTATUS[0]}
    set -e
    logits="$(sed -n 's/.*logits_diff=\([^ ]*\).*/\1/p' "${log}" | tail -1)"
    rel="$(sed -n 's/.*rel_l2=\([^ ]*\).*/\1/p' "${log}" | tail -1)"
    pass="$(awk -F'|' '/^\| a4w4/{gsub(/[[:space:]]/,"",$12); print $12}' "${log}" | tail -1)"
    hash="$(sed -n 's/.*output_sha256=//p' "${log}" | tail -1)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${name}" "${rc}" "${logits:-NA}" "${rel:-NA}" "${pass:-NA}" "${hash:-NA}" \
        >>"${verify_tsv}"
    if [[ "${rc}" -ne 0 || "${pass}" != "True" ]]; then
        echo "Correctness failed for ${name}; stopping comparison." >&2
        exit 3
    fi
}

run_bench_case() {
    local round="$1"
    local order="$2"
    local name="$3"
    local log="${out_dir}/bench_r${round}_${order}_${name}.log"
    local rc gemm1 gemm2 moe_e2e hash
    case_env "${name}"
    set +e
    env "${COMMON_ENV[@]}" "${CASE_ENV[@]}" \
        python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
        --scenario bench "${TEST_SHAPE[@]}" --iters 20 --const-init 0 \
        2>&1 | tee "${log}"
    rc=${PIPESTATUS[0]}
    set -e
    gemm1="$(sed -n 's/.*gemm1: device_time_avg=\([0-9.]*\) us.*/\1/p' "${log}" | tail -1)"
    gemm2="$(sed -n 's/.*gemm2: device_time_avg=\([0-9.]*\) us.*/\1/p' "${log}" | tail -1)"
    moe_e2e="$(sed -n 's/.*fused_moe end-to-end us = \([0-9.]*\).*/\1/p' "${log}" | tail -1)"
    hash="$(sed -n 's/.*output_sha256=//p' "${log}" | tail -1)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${round}" "${order}" "${name}" "${rc}" \
        "${gemm1:-NA}" "${gemm2:-NA}" "${moe_e2e:-NA}" "${hash:-NA}" \
        >>"${bench_tsv}"
    if [[ "${rc}" -ne 0 || -z "${gemm1}" || -z "${moe_e2e}" ]]; then
        echo "Benchmark or timing extraction failed for ${name}; stopping comparison." >&2
        exit 4
    fi
}

if [[ "${RUN_VERIFY}" == "1" ]]; then
    for name in "${CASES[@]}"; do
        run_verify_case "${name}"
    done
fi

for ((round = 1; round <= ROUNDS; ++round)); do
    if ((round % 2 == 1)); then
        order_cases=("${CASES[@]}")
    else
        order_cases=()
        for ((i = ${#CASES[@]} - 1; i >= 0; --i)); do
            order_cases+=("${CASES[i]}")
        done
    fi
    order=0
    for name in "${order_cases[@]}"; do
        order=$((order + 1))
        run_bench_case "${round}" "${order}" "${name}"
    done
done

python3 - "${bench_tsv}" "${verify_tsv}" <<'PY' | tee "${out_dir}/summary.md"
import csv
import statistics
import sys
from collections import defaultdict

bench_path, verify_path = sys.argv[1:]
gemm1_samples = defaultdict(list)
moe_e2e_samples = defaultdict(list)
with open(bench_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        gemm1_samples[row["case"]].append(float(row["gemm1_us"]))
        moe_e2e_samples[row["case"]].append(float(row["moe_e2e_us"]))

verify = {}
with open(verify_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        verify[row["case"]] = row

gemm1_baseline = statistics.median(gemm1_samples["baseline"])
moe_e2e_baseline = statistics.median(moe_e2e_samples["baseline"])
print(
    "| case | GEMM1 samples (us) | GEMM1 median us | GEMM1 vs baseline | "
    "MOE e2e samples (us) | MOE e2e median us | MOE e2e vs baseline | random pass | hash |"
)
print("|---|---|---:|---:|---|---:|---:|:---:|---|")
for name, gemm1_values in gemm1_samples.items():
    moe_e2e_values = moe_e2e_samples[name]
    gemm1_med = statistics.median(gemm1_values)
    moe_e2e_med = statistics.median(moe_e2e_values)
    gemm1_gain = (gemm1_baseline - gemm1_med) / gemm1_baseline * 100.0
    moe_e2e_gain = (moe_e2e_baseline - moe_e2e_med) / moe_e2e_baseline * 100.0
    vr = verify.get(name, {})
    gemm1_vals = ", ".join(f"{x:.3f}" for x in gemm1_values)
    moe_e2e_vals = ", ".join(f"{x:.2f}" for x in moe_e2e_values)
    print(
        f"| {name} | {gemm1_vals} | {gemm1_med:.3f} | {gemm1_gain:+.2f}% | "
        f"{moe_e2e_vals} | {moe_e2e_med:.2f} | {moe_e2e_gain:+.2f}% | "
        f"{vr.get('pass', 'not-run')} | "
        f"{vr.get('output_sha256', 'not-run')} |"
    )
PY

if [[ "${RUN_ATT}" == "1" ]]; then
    mkdir -p "${out_dir}/att"
    for name in "${CASES[@]}"; do
        case_env "${name}"
        kernel="$(case_kernel "${name}")"
        test_cmd="${COMMON_ENV[*]} ${CASE_ENV[*]} python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py --scenario verify ${TEST_SHAPE[*]} --iters 2 --const-init 0"
        env TRACE_ROOT="${out_rel}/att" HIP_VISIBLE_DEVICES=0 \
            bash my_code/get_isa_runner_att.sh \
            "${kernel}" "${host}_${name}" "${test_cmd}" --ana-att \
            2>&1 | tee "${out_dir}/att_${name}.log"
    done
fi

chmod -R a+rwX "${out_dir}"
echo "Results: ${out_dir}"
