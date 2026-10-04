#!/usr/bin/env bash
set -euo pipefail

# Compare the retained fused-GEMM1 checkpoints under one machine state.
# Cases are interleaved by round to reduce time-order bias:
#   even round: CASE_LIST order
#   odd round:  reversed CASE_LIST order
# Run from any directory inside the repository, for example:
#   ROUNDS=3 bash my_code/run_gemm1_fused_variant_compare.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ROUNDS="${ROUNDS:-3}"
GPU_IDLE_POLL_SEC="${GPU_IDLE_POLL_SEC:-6}"
GPU_IDLE_TIMEOUT_SEC="${GPU_IDLE_TIMEOUT_SEC:-1800}"
GPU_HELPER="${GPU_HELPER:-/data/yanguahe/code/gpu_users.sh}"
VARIANT_DIR="${SCRIPT_DIR}"
KERNEL_REL="aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_fused_persistent.py"
KERNEL_PATH="${REPO_ROOT}/${KERNEL_REL}"
PYC_GLOB="${REPO_ROOT}/aiter/ops/flydsl/kernels/__pycache__/mxfp4_preshuffle_gfx1250_tdm_fused_persistent"'*.pyc'
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${SCRIPT_DIR}/gemm1_fused_variant_compare_runs/${RUN_STAMP}"
RESULTS_TSV="${OUT_DIR}/results.tsv"
SUMMARY_MD="${OUT_DIR}/summary.md"

declare -A EXPECTED_SHA=(
  [v123]="c54a38e5d4bab82453deb9e3a59c3558fece84a04f7602237b2cedd3110eb6b4"
  [v140]="2bdf5c5c78f6e08db8b083636921e0b6f90e3ba3f9f1a8da6f2188f7e892dca4"
  [v146]="0fc8b4a980f697b335bf5c53d2ee847c67e860be00aa61ce3204ee4ae11014f1"
  [v153]="e7ce682ff76bb64336cc357cbce6b8b65e06b41e2a2a818e441131b81bd7a858"
)

case_text="${CASE_LIST:-v123 v140 v146 v153}"
case_text="${case_text//,/ }"
read -r -a CASES <<<"${case_text}"

[[ "${ROUNDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ROUNDS must be a positive integer: ${ROUNDS}" >&2
  exit 2
}
(( ${#CASES[@]} > 0 )) || {
  echo "CASE_LIST must contain at least one case" >&2
  exit 2
}

mkdir -p "${OUT_DIR}"
original_kernel="$(mktemp /tmp/gemm1_fused_variant_compare.XXXXXX.py)"
cp "${KERNEL_PATH}" "${original_kernel}"

restore_kernel() {
  cp "${original_kernel}" "${KERNEL_PATH}"
  rm -f ${PYC_GLOB}
  rm -f "${original_kernel}"
}
trap restore_kernel EXIT INT TERM

gpu_status() {
  if [[ ! -x "${GPU_HELPER}" ]]; then
    echo "GPU idle helper is missing or not executable: ${GPU_HELPER}" >&2
    return 2
  fi
  # gpu_users.sh returns 1 when the machine is idle, so classify its text rather
  # than treating its process status as success/failure.
  "${GPU_HELPER}" 2>&1 || true
}

wait_for_idle_gpu() {
  local waited=0
  local status
  while true; do
    status="$(gpu_status)"
    printf '%s\n' "${status}"
    if grep -Fq '当前没有进程在使用 GPU。' <<<"${status}"; then
      return 0
    fi
    if (( waited >= GPU_IDLE_TIMEOUT_SEC )); then
      echo "GPU did not become idle within ${GPU_IDLE_TIMEOUT_SEC}s" >&2
      return 1
    fi
    sleep "${GPU_IDLE_POLL_SEC}"
    waited=$((waited + GPU_IDLE_POLL_SEC))
  done
}

printf 'round_id\tround_position\tcase\tsha256\tgemm1_us\tgemm2_us\tmoe_us\tpass\trun_summary\n' >"${RESULTS_TSV}"

# Validate every requested checkpoint before starting any GPU work.
for case_name in "${CASES[@]}"; do
  [[ -n "${EXPECTED_SHA[${case_name}]:-}" ]] || {
    echo "Unknown case: ${case_name}" >&2
    exit 2
  }
  variant_source="${VARIANT_DIR}/${case_name}.py"
  [[ -f "${variant_source}" ]] || {
    echo "Missing variant source: ${variant_source}" >&2
    exit 2
  }

  actual_sha="$(sha256sum "${variant_source}" | awk '{print $1}')"
  [[ "${actual_sha}" == "${EXPECTED_SHA[${case_name}]}" ]] || {
    echo "SHA256 mismatch for ${case_name}: expected=${EXPECTED_SHA[${case_name}]} actual=${actual_sha}" >&2
    exit 2
  }
done

for ((round_id = 0; round_id < ROUNDS; round_id++)); do
  if (( round_id % 2 == 0 )); then
    ROUND_CASES=("${CASES[@]}")
  else
    ROUND_CASES=()
    for ((case_index = ${#CASES[@]} - 1; case_index >= 0; case_index--)); do
      ROUND_CASES+=("${CASES[case_index]}")
    done
  fi

  printf '===== round id %d: %s =====\n' "${round_id}" "${ROUND_CASES[*]}"
  for round_position in "${!ROUND_CASES[@]}"; do
    case_name="${ROUND_CASES[round_position]}"
    variant_source="${VARIANT_DIR}/${case_name}.py"
    actual_sha="$(sha256sum "${variant_source}" | awk '{print $1}')"

    wait_for_idle_gpu
    cp "${variant_source}" "${KERNEL_PATH}"
    rm -f ${PYC_GLOB}
    active_sha="$(sha256sum "${KERNEL_PATH}" | awk '{print $1}')"
    [[ "${active_sha}" == "${actual_sha}" ]] || {
      echo "Active kernel SHA256 mismatch for ${case_name}" >&2
      exit 2
    }

    case_log="${OUT_DIR}/round${round_id}_pos${round_position}_${case_name}.log"
    echo "===== round id ${round_id}, position ${round_position}: ${case_name} (${actual_sha}) =====" | tee "${case_log}"
    ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh \
      e2e-const0 \
      --experts 64 --tokens 1536 --topk 8 \
      --model-dim 7168 --inter-dim 2048 2>&1 | tee -a "${case_log}"

    post_status="$(gpu_status)"
    printf '%s\n' "${post_status}"
    if ! grep -Fq '当前没有进程在使用 GPU。' <<<"${post_status}"; then
      echo "GPU became busy during or immediately after ${case_name}; refusing to record its timing" >&2
      exit 3
    fi

    run_summary="$(sed -n 's/^Summary: //p' "${case_log}" | tail -n 1 | tr -d '\r')"
    [[ -f "${run_summary}" ]] || {
      echo "Could not locate generated summary for round ${round_id}, case ${case_name}" >&2
      exit 2
    }

    python3 - "${round_id}" "${round_position}" "${case_name}" "${actual_sha}" "${run_summary}" "${RESULTS_TSV}" <<'PY'
import pathlib
import sys

round_id, round_position, case_name, sha256, summary_name, output_name = sys.argv[1:]
summary = pathlib.Path(summary_name)
row = next(
    line for line in reversed(summary.read_text().splitlines())
    if line.startswith("| const0 |")
)
cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
selected = [
    round_id,
    round_position,
    case_name,
    sha256,
    cells[5],
    cells[14],
    cells[21],
    cells[23],
    str(summary.resolve()),
]
with open(output_name, "a", encoding="utf-8", newline="") as output:
    output.write("\t".join(selected) + "\n")
PY
  done
done

python3 - "${RESULTS_TSV}" "${SUMMARY_MD}" "${ROUNDS}" "${case_text}" <<'PY'
import csv
import pathlib
import statistics
import sys

results_name, summary_name, rounds, case_text = sys.argv[1:]
with open(results_name, encoding="utf-8", newline="") as src:
    rows = list(csv.DictReader(src, delimiter="\t"))

case_order = case_text.replace(",", " ").split()
grouped = {case_name: [] for case_name in case_order}
for row in rows:
    grouped[row["case"]].append(row)

aggregated = []
for case_name in case_order:
    case_rows = sorted(grouped[case_name], key=lambda row: int(row["round_id"]))
    if len(case_rows) != int(rounds):
        raise RuntimeError(
            f"expected {rounds} samples for {case_name}, got {len(case_rows)}"
        )
    gemm1_samples = [float(row["gemm1_us"]) for row in case_rows]
    gemm2_samples = [float(row["gemm2_us"]) for row in case_rows]
    moe_samples = [float(row["moe_us"]) for row in case_rows]
    aggregated.append({
        "case": case_name,
        "sha256": case_rows[0]["sha256"],
        "gemm1_samples": gemm1_samples,
        "gemm1_median": statistics.median(gemm1_samples),
        "gemm2_samples": gemm2_samples,
        "gemm2_median": statistics.median(gemm2_samples),
        "moe_samples": moe_samples,
        "moe_median": statistics.median(moe_samples),
        "pass": all(row["pass"] == "True" for row in case_rows),
    })

baseline_case = "v123" if "v123" in grouped else case_order[0]
baseline = next(row for row in aggregated if row["case"] == baseline_case)
base_g1 = baseline["gemm1_median"]
base_g2 = baseline["gemm2_median"]
base_moe = baseline["moe_median"]

execution_order = []
for round_id in range(int(rounds)):
    round_cases = case_order if round_id % 2 == 0 else list(reversed(case_order))
    execution_order.append(f"round id {round_id}: {', '.join(round_cases)}")

lines = [
    "# Fused GEMM1 variant comparison",
    "",
    f"- ROUNDS: `{rounds}`",
    "- Shape: `E64/T1536/topk8/M7168/I2048`",
    "- Data: `const0`",
    "- Execution order:",
    *[f"  - `{order}`" for order in execution_order],
    "",
    f"| case | GEMM1 samples (us) | GEMM1 median (us) | vs {baseline_case} | GEMM2 samples (us) | GEMM2 median (us) | vs {baseline_case} | MoE e2e samples (us) | MoE median (us) | vs {baseline_case} | pass | SHA256 |",
    "|---|---|---:|---:|---|---:|---:|---|---:|---:|:---:|---|",
]
for row in aggregated:
    g1 = row["gemm1_median"]
    g2 = row["gemm2_median"]
    moe = row["moe_median"]
    g1_delta = 100.0 * (base_g1 - g1) / base_g1
    g2_delta = 100.0 * (base_g2 - g2) / base_g2
    moe_delta = 100.0 * (base_moe - moe) / base_moe
    gemm1_samples = ", ".join(f"{value:.3f}" for value in row["gemm1_samples"])
    gemm2_samples = ", ".join(f"{value:.3f}" for value in row["gemm2_samples"])
    moe_samples = ", ".join(f"{value:.2f}" for value in row["moe_samples"])
    lines.append(
        f'| {row["case"]} | {gemm1_samples} | {g1:.3f} | {g1_delta:+.2f}% | '
        f'{gemm2_samples} | {g2:.3f} | {g2_delta:+.2f}% | '
        f'{moe_samples} | {moe:.2f} | {moe_delta:+.2f}% | '
        f'{row["pass"]} | `{row["sha256"]}` |'
    )

path = pathlib.Path(summary_name)
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(path.read_text(), end="")
print(f"Summary: {path.resolve()}")
PY
