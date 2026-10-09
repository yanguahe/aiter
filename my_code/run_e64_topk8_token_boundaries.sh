#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROUNDS="${ROUNDS:-3}"
MODE="e2e-const0"

# Union of every interval endpoint in:
#   my_code/e64_topk8_token_range_kernel_dispatch_head_vs_worktree.md
TOKEN_BOUNDARIES=(
  1
  1024
  1025
  1505
  1506
  1535
  1536
  1537
  1538
  2048
  2049
  4096
  4097
  4098
)

usage() {
  cat <<'EOF'
usage: bash my_code/run_e64_topk8_token_boundaries.sh [MODE]

Modes:
  e2e-const0   Run every token boundary with --const-init 0 (default)
  random       Run every token boundary with random initialization
  both         Run random followed by const0 for every token boundary

Aliases:
  e2e-random   Alias for random
  e2e-both     Alias for both

Fixed shape:
  --experts 64 --topk 8 --model-dim 7168 --inter-dim 2048

Token boundaries:
  1 1024 1025 1505 1506 1535 1536
  1537 1538 2048 2049 4096 4097 4098

Environment:
  ROUNDS=N     Number of complete token sweeps (default: 3).
               Zero-based even rounds run forward; odd rounds run in reverse.
  PYTHON_BIN   Python executable passed to the underlying script
               (default: python3)
  LOG_DIR      Output directory override

Outputs:
  <LOG_DIR>/R<round>_O<order>_T<tokens>/results.tsv
  <LOG_DIR>/R<round>_O<order>_T<tokens>/summary.md
  <LOG_DIR>/results.tsv
  <LOG_DIR>/summary.md
EOF
}

if (($# > 1)); then
  printf 'Expected at most one MODE argument.\n' >&2
  usage >&2
  exit 2
fi

if (($# == 1)); then
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
    e2e-const0)
      MODE="e2e-const0"
      ;;
    random|e2e-random)
      MODE="e2e-random"
      ;;
    both|e2e-both)
      MODE="e2e-both"
      ;;
    *)
      printf 'Unknown mode: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
fi

if [[ ! "$ROUNDS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ROUNDS must be a positive integer, got %q.\n' "$ROUNDS" >&2
  exit 2
fi

if [[ ! -f /.dockerenv ]]; then
  printf 'This script must be run inside the ROCm container.\n' >&2
  exit 2
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/e64_topk8_token_boundary_runs/$RUN_ID}"
mkdir -p "$LOG_DIR"

RESULTS_TSV="$LOG_DIR/results.tsv"
SUMMARY_MD="$LOG_DIR/summary.md"
DRIVER_SCRIPT="$SCRIPT_DIR/run_moe_prefill_switch_ab.sh"

printf 'tokens\tdata\tround\torder\tcase\tshape\tmode\tgit_commit\treturn_code\tgemm1_us\tquant_us\tgemm1_tflops\tgemm1_rw_tbps\tgemm1_ref_output_hash128\tgemm1_output_hash128\tgemm2_us\tgemm2_tflops\tgemm2_rw_tbps\tgemm2_ref_output_hash128\tgemm2_output_hash128\tmoe_e2e_us\tlogits_diff\trel_l2\tpass\tgemm1_symbol\tgemm2_symbol\tlog_file\n' \
  >"$RESULTS_TSV"

run_token_case() {
  local round="$1"
  local order="$2"
  local tokens="$3"
  local round_tag order_tag case_dir driver_log rc
  local child_results child_summary

  printf -v round_tag '%03d' "$round"
  printf -v order_tag '%02d' "$order"
  case_dir="$LOG_DIR/R${round_tag}_O${order_tag}_T${tokens}"
  driver_log="$case_dir/driver.log"
  mkdir -p "$case_dir"

  printf '\n[%s] round=%s/%s order=%s tokens=%s mode=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$round" "$ROUNDS" "$order" "$tokens" "$MODE"

  set +e
  ROUNDS=1 \
  PYTHON_BIN="$PYTHON_BIN" \
  LOG_DIR="$case_dir" \
    bash "$DRIVER_SCRIPT" "$MODE" \
      --experts 64 \
      --tokens "$tokens" \
      --topk 8 \
      --model-dim 7168 \
      --inter-dim 2048 \
      >"$driver_log" 2>&1
  rc=$?
  set -e

  if ((rc != 0)); then
    printf 'round=%s order=%s tokens=%s failed with rc=%s; tail of %s follows:\n' \
      "$round" "$order" "$tokens" "$rc" "$driver_log" >&2
    tail -120 "$driver_log" >&2 || true
    exit "$rc"
  fi

  child_results="$case_dir/results.tsv"
  child_summary="$case_dir/summary.md"
  if [[ ! -s "$child_results" || ! -s "$child_summary" ]]; then
    printf 'round=%s order=%s tokens=%s did not produce results.tsv and summary.md.\n' \
      "$round" "$order" "$tokens" >&2
    exit 4
  fi

  awk -v tokens="$tokens" -v round="$round" -v order="$order" \
    'BEGIN {FS=OFS="\t"} NR > 1 {$2=round; $3=order; print tokens, $0}' \
    "$child_results" >>"$RESULTS_TSV"

  printf 'round=%s order=%s tokens=%s completed; summary=%s\n' \
    "$round" "$order" "$tokens" "$child_summary"
}

token_count=${#TOKEN_BOUNDARIES[@]}
for ((round = 0; round < ROUNDS; round++)); do
  if ((round % 2 == 0)); then
    for ((order = 0; order < token_count; order++)); do
      run_token_case "$round" "$order" "${TOKEN_BOUNDARIES[$order]}"
    done
  else
    for ((order = 0; order < token_count; order++)); do
      index=$((token_count - 1 - order))
      run_token_case "$round" "$order" "${TOKEN_BOUNDARIES[$index]}"
    done
  fi
done

"$PYTHON_BIN" - "$RESULTS_TSV" <<'PY' >"$SUMMARY_MD"
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
data_order: list[str] = []
for row in rows:
    token = int(row["tokens"])
    data = row["data"]
    groups[(token, data)].append(row)
    if data not in data_order:
        data_order.append(data)


def floats(case_rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in case_rows]


def samples(values: list[float], digits: int = 3) -> str:
    return ", ".join(f"{value:.{digits}f}" for value in values)


def symbols(case_rows: list[dict[str, str]], key: str) -> str:
    ordered: list[str] = []
    for row in case_rows:
        value = row[key]
        if value not in ordered:
            ordered.append(value)
    return "<br>".join(f"`{value}`" for value in ordered)


print("# E64/topk8 token-boundary sweep\n")
print("Fixed shape: `E64/topk8/M7168/I2048`, A4W4, SiLU, no bias.\n")
print(
    "Token boundaries: `1, 1024, 1025, 1505, 1506, 1535, 1536, "
    "1537, 1538, 2048, 2049, 4096, 4097, 4098`.\n"
)
print(
    "Traversal: zero-based even rounds run in ascending token order; "
    "odd rounds run in descending token order.\n"
)

print(
    "| data | tokens | GEMM1 samples (us) | GEMM1 median (us) | "
    "GEMM2 samples (us) | GEMM2 median (us) | MoE samples (us) | "
    "MoE median (us) | pass | max logits_diff | max rel_l2 | "
    "GEMM1 symbol | GEMM2 symbol |"
)
print(
    "|---|---:|---|---:|---|---:|---|---:|:---:|---:|---:|---|---|"
)

for data in data_order:
    for token in sorted({key[0] for key in groups if key[1] == data}):
        case_rows = sorted(
            groups[(token, data)],
            key=lambda row: (int(row["round"]), int(row["order"])),
        )
        gemm1 = floats(case_rows, "gemm1_us")
        gemm2 = floats(case_rows, "gemm2_us")
        moe = floats(case_rows, "moe_e2e_us")
        logits = floats(case_rows, "logits_diff")
        rel_l2 = floats(case_rows, "rel_l2")
        passed = all(row["pass"] == "True" for row in case_rows)
        print(
            f"| {data} | {token} | {samples(gemm1)} | "
            f"{statistics.median(gemm1):.3f} | {samples(gemm2)} | "
            f"{statistics.median(gemm2):.3f} | {samples(moe, 2)} | "
            f"{statistics.median(moe):.2f} | {passed} | "
            f"{max(logits):.6g} | {max(rel_l2):.6g} | "
            f"{symbols(case_rows, 'gemm1_symbol')} | "
            f"{symbols(case_rows, 'gemm2_symbol')} |"
        )
PY

cat "$SUMMARY_MD"
printf '\nAll token boundaries completed.\nLogs: %s\nSummary: %s\n' \
  "$LOG_DIR" "$SUMMARY_MD"
