#!/usr/bin/env bash
set -Eeuo pipefail

# Reproduce same-machine comparisons for the gfx1250 E96/T16384 MoE GEMM1
# history set. Run this script directly inside the existing ROCm container.

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$HERE/../.." && pwd)}"
SNAPSHOT_ROOT="$HERE/repo_snapshot"
COMPARE="$HERE/compare_asm_variants.py"
E2E="$HERE/run_e2e_candidate.py"
ATT_LAUNCH="$HERE/att_launch_opt.py"
SNAPSHOT_TOOL="$HERE/sync_head_repo_snapshot.py"

MODE="${1:-e2e-const0}"
SEED="${AITER_HISTORY_SEED:-0}"
WARMUP="${AITER_HISTORY_WARMUP:-5}"
ROUNDS="${AITER_HISTORY_ROUNDS:-9}"
LAUNCHES="${AITER_HISTORY_LAUNCHES_PER_SAMPLE:-5}"
VALIDATION_REPEATS="${AITER_HISTORY_VALIDATION_REPEATS:-3}"
E2E_ITERS="${AITER_HISTORY_E2E_ITERS:-20}"
E2E_ROUNDS="${AITER_HISTORY_E2E_ROUNDS:-1}"
RUN_VERIFY="${AITER_HISTORY_RUN_VERIFY:-1}"
RUN_REFERENCE_COMMAND="${AITER_HISTORY_RUN_REFERENCE_COMMAND:-1}"
RUN_ATT="${AITER_HISTORY_RUN_ATT:-0}"
ATT_VALIDATE_ONLY="${AITER_ATT_VALIDATE_ONLY:-0}"
VERIFY_SNAPSHOT="${AITER_HISTORY_VERIFY_SNAPSHOT:-1}"
CLANG="${AITER_GFX1250_CLANG:-/data/yanguahe/code/wk_sp1/llvm-project/mlir_install/bin/clang}"
CLANG_RUNTIME_LIB="${AITER_GFX1250_CLANG_RUNTIME_LIB:-/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/rocm_sysdeps/lib}"
SYMBOL=moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1

usage() {
  cat <<'EOF'
usage: benchmark_history.sh [MODE]

Modes:
  quick-random   One-launch standalone random smoke test
  perf-random    Interleaved standalone random benchmark
  perf-const0    Historical const0 command plus interleaved benchmark
  perf-both      Standalone random and const0 benchmarks
  e2e-random     Full MoE random correctness and profiler benchmark
  e2e-const0     Full MoE const0 correctness and profiler benchmark (default)
  e2e-both       Full MoE random and const0 benchmarks
  att            Compile, capture, and analyze ATT for every selected case
  att-validate   Compile and launch every ATT code object without tracing
  list           Print the selected case table without running a GPU workload

Selection and control:
  AITER_HISTORY_CASE_LIST=baseline,optimized_v1,...
  AITER_HISTORY_CANDIDATE=/path/to/candidate.s
  AITER_HISTORY_CANDIDATE_GRID_X=16
  AITER_HISTORY_CANDIDATE_GRID_Y=16
  AITER_HISTORY_RUN_ATT=1             Append ATT to another mode
  AITER_ATT_VALIDATE_ONLY=1           Validate ATT launches without tracing

Stable cases:
  baseline, optimized_v1, double_lds, persistent, persistent_overlap
EOF
}

if (($# > 1)); then
  usage >&2
  exit 2
fi
case "$MODE" in
  -h|--help|help)
    usage
    exit 0
    ;;
  quick-random|perf-random|perf-const0|perf-both|e2e-random|e2e-const0|e2e-both|att|att-validate|list)
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer, got '$value'" >&2
    exit 2
  fi
}

require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be a non-negative integer, got '$value'" >&2
    exit 2
  fi
}

require_flag() {
  local name="$1"
  local value="$2"
  if [[ "$value" != 0 && "$value" != 1 ]]; then
    echo "$name must be 0 or 1, got '$value'" >&2
    exit 2
  fi
}

require_nonnegative_integer AITER_HISTORY_WARMUP "$WARMUP"
require_positive_integer AITER_HISTORY_ROUNDS "$ROUNDS"
require_positive_integer AITER_HISTORY_LAUNCHES_PER_SAMPLE "$LAUNCHES"
require_positive_integer AITER_HISTORY_VALIDATION_REPEATS "$VALIDATION_REPEATS"
require_positive_integer AITER_HISTORY_E2E_ITERS "$E2E_ITERS"
require_positive_integer AITER_HISTORY_E2E_ROUNDS "$E2E_ROUNDS"
require_flag AITER_HISTORY_RUN_VERIFY "$RUN_VERIFY"
require_flag AITER_HISTORY_RUN_REFERENCE_COMMAND "$RUN_REFERENCE_COMMAND"
require_flag AITER_HISTORY_RUN_ATT "$RUN_ATT"
require_flag AITER_ATT_VALIDATE_ONLY "$ATT_VALIDATE_ONLY"
require_flag AITER_HISTORY_VERIFY_SNAPSHOT "$VERIFY_SNAPSHOT"

BASELINE="$HERE/baseline_act1_independent.s"
OPT_V1="$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_opt.s"
OPT_DOUBLE_LDS="$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_double_lds.s"
OPT_PERSISTENT="$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent.s"
OPT_PERSISTENT_OVERLAP="$HERE/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent_overlap.s"
CANDIDATE=""

if [[ -n "${AITER_HISTORY_CANDIDATE:-}" ]]; then
  candidate_input="$AITER_HISTORY_CANDIDATE"
  if [[ "$candidate_input" != /* ]]; then
    candidate_input="$REPO_ROOT/$candidate_input"
  fi
  if [[ ! -f "$candidate_input" ]]; then
    echo "missing candidate ISA: $candidate_input" >&2
    exit 2
  fi
  CANDIDATE="$(realpath -- "$candidate_input")"
  if [[ -n "${AITER_HISTORY_CANDIDATE_GRID_X:-}" || -n "${AITER_HISTORY_CANDIDATE_GRID_Y:-}" ]]; then
    if [[ -z "${AITER_HISTORY_CANDIDATE_GRID_X:-}" || -z "${AITER_HISTORY_CANDIDATE_GRID_Y:-}" ]]; then
      echo "candidate grid requires both AITER_HISTORY_CANDIDATE_GRID_X and AITER_HISTORY_CANDIDATE_GRID_Y" >&2
      exit 2
    fi
    require_positive_integer AITER_HISTORY_CANDIDATE_GRID_X "$AITER_HISTORY_CANDIDATE_GRID_X"
    require_positive_integer AITER_HISTORY_CANDIDATE_GRID_Y "$AITER_HISTORY_CANDIDATE_GRID_Y"
  fi
fi

declare -a CASES=(baseline optimized_v1 double_lds persistent persistent_overlap)
requested_cases="${AITER_HISTORY_CASE_LIST:-${CASE_LIST:-}}"
if [[ -n "$requested_cases" ]]; then
  IFS=',' read -r -a CASES <<<"$requested_cases"
elif [[ -n "$CANDIDATE" ]]; then
  CASES+=(candidate)
fi

case_source() {
  case "$1" in
    baseline) printf '%s\n' "$BASELINE" ;;
    optimized_v1) printf '%s\n' "$OPT_V1" ;;
    double_lds) printf '%s\n' "$OPT_DOUBLE_LDS" ;;
    persistent) printf '%s\n' "$OPT_PERSISTENT" ;;
    persistent_overlap) printf '%s\n' "$OPT_PERSISTENT_OVERLAP" ;;
    candidate)
      if [[ -z "$CANDIDATE" ]]; then
        echo "case 'candidate' requires AITER_HISTORY_CANDIDATE" >&2
        return 2
      fi
      printf '%s\n' "$CANDIDATE"
      ;;
    *)
      echo "unknown case: $1" >&2
      return 2
      ;;
  esac
}

case_label() {
  case "$1" in
    baseline) printf '%s\n' 'safe baseline' ;;
    optimized_v1) printf '%s\n' 'optimized v1' ;;
    double_lds) printf '%s\n' 'double-output-LDS' ;;
    persistent) printf '%s\n' 'persistent/full-drain' ;;
    persistent_overlap) printf '%s\n' 'persistent/output-drain-overlap' ;;
    candidate) printf '%s\n' 'candidate' ;;
    *) return 2 ;;
  esac
}

case_grid_x() {
  case "$1" in
    baseline|optimized_v1|double_lds) printf '%s\n' '' ;;
    persistent|persistent_overlap) printf '%s\n' 16 ;;
    candidate) printf '%s\n' "${AITER_HISTORY_CANDIDATE_GRID_X:-}" ;;
    *) return 2 ;;
  esac
}

case_grid_y() {
  case "$1" in
    baseline|optimized_v1|double_lds) printf '%s\n' '' ;;
    persistent|persistent_overlap) printf '%s\n' 16 ;;
    candidate) printf '%s\n' "${AITER_HISTORY_CANDIDATE_GRID_Y:-}" ;;
    *) return 2 ;;
  esac
}

case_grid_label() {
  local grid_x grid_y
  grid_x="$(case_grid_x "$1")"
  grid_y="$(case_grid_y "$1")"
  if [[ -z "$grid_x" ]]; then
    printf '%s\n' standard
  else
    printf '%sx%s\n' "$grid_x" "$grid_y"
  fi
}

declare -A seen_cases=()
for name in "${CASES[@]}"; do
  if [[ -z "$name" ]]; then
    echo "case list contains an empty case" >&2
    exit 2
  fi
  case_source "$name" >/dev/null
  if [[ -n "${seen_cases[$name]:-}" ]]; then
    echo "duplicate case: $name" >&2
    exit 2
  fi
  seen_cases[$name]=1
done
if ((${#CASES[@]} == 0)); then
  echo "no benchmark cases selected" >&2
  exit 2
fi

list_cases() {
  printf '%-20s %-34s %-10s %s\n' case label grid source
  for name in "${CASES[@]}"; do
    printf '%-20s %-34s %-10s %s\n' \
      "$name" "$(case_label "$name")" "$(case_grid_label "$name")" "$(case_source "$name")"
  done
}

if [[ "$MODE" == list ]]; then
  list_cases
  exit 0
fi

if [[ ! -f /.dockerenv ]]; then
  echo "This script must be run inside the ROCm container." >&2
  exit 2
fi

cd "$REPO_ROOT"
if [[ ! -f "$SNAPSHOT_ROOT/aiter/__init__.py" || \
      ! -f "$SNAPSHOT_ROOT/op_tests/test_flydsl_grouped_gemm_gfx1250.py" || \
      ! -f "$SNAPSHOT_ROOT/SOURCE_COMMIT" || \
      ! -f "$SNAPSHOT_ROOT/SNAPSHOT_MANIFEST.json" ]]; then
  echo "self-contained pinned snapshot is incomplete: $SNAPSHOT_ROOT" >&2
  echo "run on the host/local checkout: python $SNAPSHOT_TOOL" >&2
  exit 2
fi
if [[ "$VERIFY_SNAPSHOT" == 1 ]]; then
  python3 "$SNAPSHOT_TOOL" --verify
fi
export PYTHONPATH="$SNAPSHOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export AITER_META_DIR="$SNAPSHOT_ROOT"

for name in "${CASES[@]}"; do
  source_path="$(case_source "$name")"
  if [[ ! -f "$source_path" ]]; then
    echo "missing ISA for $name: $source_path" >&2
    exit 2
  fi
done
for required in "$COMPARE" "$E2E" "$ATT_LAUNCH" "$REPO_ROOT/my_code/get_isa_runner_att.sh"; do
  if [[ ! -f "$required" ]]; then
    echo "missing benchmark dependency: $required" >&2
    exit 2
  fi
done

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

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
host="$(hostname -s)"
out_rel="my_code/moe_gemm1_act1_optimized/history_runs/${host}_${timestamp}_${MODE}"
out_dir="$REPO_ROOT/$out_rel"
mkdir -p "$out_dir"
run_log="$out_dir/run.log"
standalone_tsv="$out_dir/standalone.tsv"
e2e_tsv="$out_dir/e2e.tsv"
att_tsv="$out_dir/att.tsv"
printf 'data\tgrid\tcase\treturn_code\tmedian_us\tmean_us\tmin_us\tmax_us\tsamples\n' >"$standalone_tsv"
printf 'data\tround\torder\tcase\tgrid\treturn_code\tgemm1_us\tgemm2_us\tmoe_e2e_us\tlogits_diff\trel_l2\tpass\toutput_sha256\n' >"$e2e_tsv"
printf 'case\tgrid\treturn_code\tcode_object\tisa_sha256\n' >"$att_tsv"

snapshot_commit() {
  if [[ -n "${AITER_HISTORY_GIT_HEAD:-}" ]]; then
    printf '%s\n' "$AITER_HISTORY_GIT_HEAD"
  else
    cat "$SNAPSHOT_ROOT/SOURCE_COMMIT"
  fi
}

print_context() {
  echo "date_utc=$timestamp"
  echo "host=$host"
  echo "mode=$MODE"
  echo "snapshot_commit=$(snapshot_commit)"
  echo "repo_snapshot=$SNAPSHOT_ROOT"
  echo "canonical_metric=e2e const0 profiler gemm1 device_time_avg"
  echo "seed=$SEED"
  echo "standalone_warmup=$WARMUP"
  echo "standalone_rounds=$ROUNDS"
  echo "standalone_launches_per_sample=$LAUNCHES"
  echo "validation_repeats=$VALIDATION_REPEATS"
  echo "e2e_iters=$E2E_ITERS"
  echo "e2e_rounds=$E2E_ROUNDS"
  echo "run_verify=$RUN_VERIFY"
  echo "run_reference_command=$RUN_REFERENCE_COMMAND"
  echo "run_att=$RUN_ATT"
  echo "att_validate_only=$ATT_VALIDATE_ONLY"
  echo "execution=inside-container"
  echo
  echo "case table:"
  list_cases
  echo
  echo "task source hashes:"
  sha256sum \
    "$HERE/benchmark_history.sh" \
    "$COMPARE" \
    "$E2E" \
    "$ATT_LAUNCH" \
    "$SNAPSHOT_ROOT/SOURCE_COMMIT" \
    "$SNAPSHOT_ROOT/SNAPSHOT_MANIFEST.json" \
    "$REPO_ROOT/my_code/get_isa_runner_att.sh"
  for name in "${CASES[@]}"; do
    sha256sum "$(case_source "$name")"
  done
  if command -v rocm-smi >/dev/null 2>&1; then
    timeout 20 rocm-smi --showproductname --showclocks --showuse --showmemuse || true
  fi
}

run_exact_historical_const0() {
  local log="$out_dir/exact_historical_perf_const0.log"
  echo "===== exact historical command: test_optimized.sh perf-const0 ====="
  set +e
  AITER_OPT_ISA="$OPT_V1" bash "$HERE/test_optimized.sh" perf-const0 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "exact historical const0 command failed with return code $rc" >&2
    exit 3
  fi
}

run_standalone_group() {
  local data="$1"
  local grid_key="$2"
  local warmup="$3"
  local rounds="$4"
  local launches="$5"
  shift 5
  local -a group_cases=("$@")
  local -a sources=()
  local -a const_args=()
  local -a grid_args=()
  local -a validation_args=(--validation-repeats "$VALIDATION_REPEATS")
  local name source_path grid_x grid_y log rc line median mean min max samples stem
  local expected_validations actual_validations

  for name in "${group_cases[@]}"; do
    sources+=("$(case_source "$name")")
  done
  if [[ "$data" == const0 ]]; then
    const_args=(--const-init 0)
  fi
  if [[ "$grid_key" != standard ]]; then
    grid_x="${grid_key%x*}"
    grid_y="${grid_key#*x}"
    grid_args=(--grid-x "$grid_x" --grid-y "$grid_y")
  fi
  if [[ "$RUN_VERIFY" == 0 ]]; then
    validation_args=(--skip-validation)
  fi

  log="$out_dir/standalone_${data}_${grid_key}.log"
  echo "===== standalone data=$data grid=$grid_key cases=${group_cases[*]} ====="
  set +e
  python3 "$COMPARE" "${sources[@]}" \
    --seed "$SEED" \
    --warmup "$warmup" \
    --rounds "$rounds" \
    --launches-per-sample "$launches" \
    "${validation_args[@]}" \
    "${grid_args[@]}" \
    "${const_args[@]}" \
    2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "standalone benchmark failed for data=$data grid=$grid_key" >&2
    exit 3
  fi
  if [[ "$RUN_VERIFY" == 1 ]]; then
    expected_validations=$((${#group_cases[@]} * VALIDATION_REPEATS))
    actual_validations="$(grep -c '^validation ' "$log" || true)"
    if [[ "$actual_validations" -ne "$expected_validations" ]]; then
      echo "expected $expected_validations validation records, got $actual_validations for data=$data grid=$grid_key" >&2
      exit 3
    fi
    if grep '^validation ' "$log" | grep -Ev ': err=0 ' >/dev/null; then
      echo "standalone correctness failed for data=$data grid=$grid_key" >&2
      exit 3
    fi
  fi

  for name in "${group_cases[@]}"; do
    source_path="$(case_source "$name")"
    stem="$(basename "${source_path%.s}")"
    line="$(grep -F "RESULT $stem: " "$log" | tail -1)"
    median="$(sed -n 's/.* median=\([0-9.]*\) us.*/\1/p' <<<"$line")"
    mean="$(sed -n 's/.* mean=\([0-9.]*\) us.*/\1/p' <<<"$line")"
    min="$(sed -n 's/.* min=\([0-9.]*\) us.*/\1/p' <<<"$line")"
    max="$(sed -n 's/.* max=\([0-9.]*\) us.*/\1/p' <<<"$line")"
    samples="${line##* samples=}"
    if [[ -z "$median" || -z "$mean" || -z "$min" || -z "$max" || "$samples" == "$line" ]]; then
      echo "failed to extract standalone timing for $name" >&2
      exit 4
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$data" "$grid_key" "$name" "$rc" "$median" "$mean" "$min" "$max" "$samples" \
      >>"$standalone_tsv"
  done
}

run_standalone() {
  local data="$1"
  local warmup="${2:-$WARMUP}"
  local rounds="${3:-$ROUNDS}"
  local launches="${4:-$LAUNCHES}"
  local -a grid_keys=()
  local -a group_cases=()
  local name key known existing

  for name in "${CASES[@]}"; do
    key="$(case_grid_label "$name")"
    known=0
    for existing in "${grid_keys[@]}"; do
      if [[ "$existing" == "$key" ]]; then
        known=1
        break
      fi
    done
    if [[ "$known" == 0 ]]; then
      grid_keys+=("$key")
    fi
  done

  for key in "${grid_keys[@]}"; do
    group_cases=()
    for name in "${CASES[@]}"; do
      if [[ "$(case_grid_label "$name")" == "$key" ]]; then
        group_cases+=("$name")
      fi
    done
    run_standalone_group "$data" "$key" "$warmup" "$rounds" "$launches" "${group_cases[@]}"
  done
}

run_e2e_case() {
  local data="$1"
  local round="$2"
  local order="$3"
  local name="$4"
  local isa grid_x grid_y grid_label log rc
  local -a const_args=()
  local -a grid_args=()
  local gemm1 gemm2 moe_e2e logits rel pass hash

  isa="$(case_source "$name")"
  grid_x="$(case_grid_x "$name")"
  grid_y="$(case_grid_y "$name")"
  grid_label="$(case_grid_label "$name")"
  if [[ "$data" == const0 ]]; then
    const_args=(--const-init 0)
  fi
  if [[ -n "$grid_x" ]]; then
    grid_args=(--grid-x "$grid_x" --grid-y "$grid_y")
  fi
  log="$out_dir/e2e_${data}_r${round}_${order}_${name}.log"
  echo "===== e2e case=$name data=$data round=$round order=$order grid=$grid_label ====="
  set +e
  env "${COMMON_ENV[@]}" \
    python3 -u "$E2E" --isa "$isa" "${grid_args[@]}" -- \
      --scenario bench \
      "${TEST_SHAPE[@]}" \
      --iters "$E2E_ITERS" \
      "${const_args[@]}" \
      2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e

  gemm1="$(sed -n 's/.*gemm1: device_time_avg=\([0-9.]*\) us.*/\1/p' "$log" | tail -1)"
  gemm2="$(sed -n 's/.*gemm2: device_time_avg=\([0-9.]*\) us.*/\1/p' "$log" | tail -1)"
  moe_e2e="$(sed -n 's/.*fused_moe end-to-end us = \([0-9.]*\).*/\1/p' "$log" | tail -1)"
  logits="$(sed -n 's/.*logits_diff=\([^ ]*\).*/\1/p' "$log" | tail -1)"
  rel="$(sed -n 's/.*rel_l2=\([^ ]*\).*/\1/p' "$log" | tail -1)"
  pass="$(awk -F'|' '/^\|[[:space:]]*a4w4/{gsub(/[[:space:]]/,"",$12); print $12}' "$log" | tail -1)"
  hash="$(sed -n 's/.*output_sha256=//p' "$log" | tail -1)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$data" "$round" "$order" "$name" "$grid_label" "$rc" \
    "${gemm1:-NA}" "${gemm2:-NA}" "${moe_e2e:-NA}" \
    "${logits:-NA}" "${rel:-NA}" "${pass:-NA}" "${hash:-NA}" \
    >>"$e2e_tsv"
  if [[ "$rc" -ne 0 || -z "$gemm1" || -z "$gemm2" || -z "$moe_e2e" || "$pass" != True ]]; then
    echo "e2e correctness, benchmark, or timing extraction failed for $name" >&2
    exit 4
  fi
}

run_e2e() {
  local data="$1"
  local round order name i
  local -a order_cases=()
  for ((round = 1; round <= E2E_ROUNDS; ++round)); do
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
      run_e2e_case "$data" "$round" "$order" "$name"
    done
  done
}

compile_code_object() {
  local source_path="$1"
  local tag="$2"
  local object="$att_root/code_objects/${tag}.o"
  local code_object="$att_root/code_objects/${tag}.co"
  LD_LIBRARY_PATH="$CLANG_RUNTIME_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$CLANG" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 \
      -mcode-object-version=6 -c "$source_path" -o "$object" >&2
  LD_LIBRARY_PATH="$CLANG_RUNTIME_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$CLANG" -target amdgcn-amd-amdhsa -mcpu=gfx1250 \
      -mcode-object-version=6 -nostdlib -Wl,--no-undefined -shared \
      "$object" -o "$code_object" >&2
  [[ -s "$code_object" ]] || return 1
  printf '%s\n' "$code_object"
}

run_att_case() {
  local name="$1"
  local source_path grid_x grid_y grid_label code_object isa_sha log rc
  source_path="$(case_source "$name")"
  grid_x="$(case_grid_x "$name")"
  grid_y="$(case_grid_y "$name")"
  grid_label="$(case_grid_label "$name")"
  isa_sha="$(sha256sum "$source_path" | awk '{print $1}')"
  echo "===== ATT case=$name grid=$grid_label ====="
  echo "$isa_sha  $source_path"
  code_object="$(compile_code_object "$source_path" "$name")"
  log="$out_dir/att_${name}.log"
  set +e
  if [[ "$ATT_VALIDATE_ONLY" == 1 ]]; then
    AITER_ATT_CODE_OBJECT="$code_object" \
    AITER_ATT_GRID_X="$grid_x" \
    AITER_ATT_GRID_Y="$grid_y" \
    HIP_VISIBLE_DEVICES=0 \
      python3 "$ATT_LAUNCH" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
  else
    AITER_ATT_CODE_OBJECT="$code_object" \
    AITER_ATT_GRID_X="$grid_x" \
    AITER_ATT_GRID_Y="$grid_y" \
    TRACE_ROOT="$att_root_rel" \
    HIP_VISIBLE_DEVICES=0 \
      bash "$REPO_ROOT/my_code/get_isa_runner_att.sh" \
        "$SYMBOL" \
        "$name" \
        "python3 $ATT_LAUNCH" \
        --ana-att \
        2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
  fi
  set -e
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$grid_label" "$rc" "$code_object" "$isa_sha" >>"$att_tsv"
  if [[ "$rc" -ne 0 ]]; then
    echo "ATT failed for $name with return code $rc" >&2
    exit 5
  fi
}

run_att() {
  local name
  att_root_rel="$out_rel/att"
  att_root="$REPO_ROOT/$att_root_rel"
  mkdir -p "$att_root/code_objects"
  for name in "${CASES[@]}"; do
    run_att_case "$name"
  done
}

write_summaries() {
  python3 - "$standalone_tsv" "$e2e_tsv" "${CASES[@]}" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

standalone_path = Path(sys.argv[1])
e2e_path = Path(sys.argv[2])
case_order = sys.argv[3:]
out_dir = standalone_path.parent

with standalone_path.open(newline="", encoding="utf-8") as handle:
    standalone_rows = list(csv.DictReader(handle, delimiter="\t"))
if standalone_rows:
    with (out_dir / "standalone_summary.md").open("w", encoding="utf-8") as out:
        out.write("| data | grid | case | median us | mean us | min us | max us | samples |\n")
        out.write("|---|---|---|---:|---:|---:|---:|---|\n")
        for row in standalone_rows:
            out.write(
                f"| {row['data']} | {row['grid']} | {row['case']} | "
                f"{float(row['median_us']):.3f} | {float(row['mean_us']):.3f} | "
                f"{float(row['min_us']):.3f} | {float(row['max_us']):.3f} | "
                f"{row['samples']} |\n"
            )

with e2e_path.open(newline="", encoding="utf-8") as handle:
    e2e_rows = list(csv.DictReader(handle, delimiter="\t"))
if e2e_rows:
    grouped = defaultdict(list)
    for row in e2e_rows:
        grouped[(row["data"], row["case"])].append(row)
    data_order = list(dict.fromkeys(row["data"] for row in e2e_rows))
    with (out_dir / "e2e_summary.md").open("w", encoding="utf-8") as out:
        out.write(
            "| data | case | grid | GEMM1 samples (us) | GEMM1 median us | "
            "GEMM1 vs first case | MoE e2e samples (us) | MoE e2e median us | "
            "MoE e2e vs first case | pass | logits_diff | rel_l2 | hash |\n"
        )
        out.write("|---|---|---|---|---:|---:|---|---:|---:|:---:|---:|---:|---|\n")
        for data in data_order:
            present = [name for name in case_order if (data, name) in grouped]
            first = present[0]
            first_gemm1 = statistics.median(
                float(row["gemm1_us"]) for row in grouped[(data, first)]
            )
            first_e2e = statistics.median(
                float(row["moe_e2e_us"]) for row in grouped[(data, first)]
            )
            for name in present:
                rows = grouped[(data, name)]
                gemm1 = [float(row["gemm1_us"]) for row in rows]
                moe_e2e = [float(row["moe_e2e_us"]) for row in rows]
                gemm1_med = statistics.median(gemm1)
                e2e_med = statistics.median(moe_e2e)
                gemm1_gain = (first_gemm1 - gemm1_med) / first_gemm1 * 100.0
                e2e_gain = (first_e2e - e2e_med) / first_e2e * 100.0
                last = rows[-1]
                out.write(
                    f"| {data} | {name} | {last['grid']} | "
                    f"{', '.join(f'{value:.3f}' for value in gemm1)} | "
                    f"{gemm1_med:.3f} | {gemm1_gain:+.2f}% | "
                    f"{', '.join(f'{value:.2f}' for value in moe_e2e)} | "
                    f"{e2e_med:.2f} | {e2e_gain:+.2f}% | {last['pass']} | "
                    f"{last['logits_diff']} | {last['rel_l2']} | "
                    f"{last['output_sha256']} |\n"
                )

summary_parts = [
    ("Standalone", out_dir / "standalone_summary.md"),
    ("MoE e2e", out_dir / "e2e_summary.md"),
]
with (out_dir / "summary.md").open("w", encoding="utf-8") as out:
    for title, path in summary_parts:
        if path.is_file():
            out.write(f"## {title}\n\n")
            out.write(path.read_text(encoding="utf-8"))
            out.write("\n")
PY
  if [[ -s "$out_dir/summary.md" ]]; then
    cat "$out_dir/summary.md"
  fi
}

run_all() {
  print_context | tee "$out_dir/environment.log"
  case "$MODE" in
    quick-random)
      run_standalone random 1 1 1
      ;;
    perf-random)
      run_standalone random
      ;;
    perf-const0)
      if [[ "$RUN_REFERENCE_COMMAND" == 1 ]]; then
        run_exact_historical_const0
      fi
      run_standalone const0
      ;;
    perf-both)
      run_standalone random
      run_standalone const0
      ;;
    e2e-random)
      run_e2e random
      ;;
    e2e-const0)
      run_e2e const0
      ;;
    e2e-both)
      run_e2e random
      run_e2e const0
      ;;
    att)
      run_att
      ;;
    att-validate)
      ATT_VALIDATE_ONLY=1
      run_att
      ;;
  esac
  if [[ "$RUN_ATT" == 1 && "$MODE" != att && "$MODE" != att-validate ]]; then
    run_att
  fi
  write_summaries
}

run_all 2>&1 | tee "$run_log"
chmod -R a+rwX "$out_dir"
echo "Results: $out_dir"
