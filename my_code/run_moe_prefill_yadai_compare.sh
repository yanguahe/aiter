#!/usr/bin/env bash

set -euo pipefail

# Run inside hyg_fyd_e2e from /app/aiter:
#   ROUNDS=3 bash my_code/run_moe_prefill_yadai_compare.sh
#
# /app/aiter is the baseline tree. The yadai branch is kept in a separate host
# worktree visible through /data, so switching revisions is only a directory
# change and /app/aiter remains untouched.
if [[ ! -f /.dockerenv ]]; then
  printf 'Run this script inside the hyg_fyd_e2e container.\n' >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_REPO="${BASELINE_REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
YADAI_REPO="${YADAI_REPO:-/data/yanguahe/code/wk_sp1/aiter_a4w4_prefill_v2_yadai}"
YADAI_REPO="$(readlink -f "$YADAI_REPO")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROUNDS="${ROUNDS:-3}"
BASELINE_COMMIT_LABEL="${BASELINE_COMMIT_LABEL:-04cc526b8f06f1e54f836718964e8f2e2c844fa8}"
BASELINE_BRANCH_LABEL="${BASELINE_BRANCH_LABEL:-hyg/moe_a4w4_pr}"
YADAI_COMMIT_LABEL="${YADAI_COMMIT_LABEL:-pending}"
GPU_USERS_SCRIPT="${GPU_USERS_SCRIPT:-/data/yanguahe/code/gpu_users.sh}"
GIT_ENV_FILE="${GIT_ENV_FILE:-/data/yanguahe/code/git_env}"
TARGET_BRANCH="${TARGET_BRANCH:-dev/a4w4_prefill_v2_yadai}"
TARGET_URL="${TARGET_URL:-git@github.com:ROCm/aiter.git}"
TARGET_REF="refs/remotes/rocm/$TARGET_BRANCH"
IGNORE_GPU_BUSY="${IGNORE_GPU_BUSY:-0}"

EXPERTS=64
TOKENS=1536
TOPK=8
MODEL_DIM=7168
INTER_DIM=2048

if [[ ! "$ROUNDS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ROUNDS must be a positive integer, got %q\n' "$ROUNDS" >&2
  exit 2
fi

if [[ ! -f "$BASELINE_REPO/my_code/run_moe_prefill_switch_ab.sh" ]]; then
  printf 'Missing baseline script: %s\n' \
    "$BASELINE_REPO/my_code/run_moe_prefill_switch_ab.sh" >&2
  exit 2
fi

YADAI_TEST="$YADAI_REPO/op_tests/flydsl_tests/test_flydsl_grouped_gemm.py"
if [[ ! -d "$YADAI_REPO" ]]; then
  printf 'Missing yadai repository: %s\n' "$YADAI_REPO" >&2
  exit 2
fi

update_yadai_checkout() {
  local tracked_changes current_commit latest_commit ignore_gpu_busy_saved
  local -a git_cmd

  if [[ ! -f "$GIT_ENV_FILE" ]]; then
    printf 'Missing Git environment file: %s\n' "$GIT_ENV_FILE" >&2
    exit 2
  fi

  # Required on these gfx1250 development machines before any container-side
  # Git command (SSH key, identity and related environment setup).
  # shellcheck disable=SC1090
  ignore_gpu_busy_saved="$IGNORE_GPU_BUSY"
  source "$GIT_ENV_FILE"
  IGNORE_GPU_BUSY="$ignore_gpu_busy_saved"

  if ! command -v git >/dev/null 2>&1; then
    printf 'Git is unavailable after sourcing %s.\n' "$GIT_ENV_FILE" >&2
    exit 2
  fi

  git_cmd=(git -c safe.directory="$YADAI_REPO" -C "$YADAI_REPO")
  tracked_changes="$("${git_cmd[@]}" status --porcelain --untracked-files=no)"
  if [[ -n "$tracked_changes" ]]; then
    printf 'Yadai worktree has tracked changes; refusing to overwrite them:\n%s\n' \
      "$tracked_changes" >&2
    exit 2
  fi

  printf '\nFetching latest ROCm/%s inside the container...\n' "$TARGET_BRANCH"
  "${git_cmd[@]}" fetch "$TARGET_URL" "$TARGET_BRANCH:$TARGET_REF"
  latest_commit="$("${git_cmd[@]}" rev-parse "$TARGET_REF")"
  current_commit="$("${git_cmd[@]}" rev-parse HEAD)"
  if [[ "$current_commit" != "$latest_commit" ]]; then
    "${git_cmd[@]}" switch --detach "$latest_commit"
  fi

  tracked_changes="$("${git_cmd[@]}" status --porcelain --untracked-files=no)"
  if [[ -n "$tracked_changes" ]]; then
    printf 'Yadai worktree is not clean after update:\n%s\n' \
      "$tracked_changes" >&2
    exit 2
  fi

  YADAI_COMMIT_LABEL="$("${git_cmd[@]}" rev-parse HEAD)"
  printf 'Yadai updated commit: %s\n' "$YADAI_COMMIT_LABEL"
}

require_gpu_idle() {
  local label="$1"
  local users_output smi_output attempt busy

  if [[ "$IGNORE_GPU_BUSY" == 1 ]]; then
    printf '\n===== GPU idle check skipped by IGNORE_GPU_BUSY=1: %s =====\n' \
      "$label"
    return
  fi

  for attempt in 1 2 3 4 5; do
    busy=0
    printf '\n===== GPU idle check: %s (attempt %s/5) =====\n' \
      "$label" "$attempt"

    if [[ -f "$GPU_USERS_SCRIPT" ]]; then
      users_output="$(bash "$GPU_USERS_SCRIPT" 2>&1 || true)"
      printf '%s\n' "$users_output"
      if ! grep -Fq '当前没有进程在使用 GPU' <<<"$users_output"; then
        busy=1
      fi
    else
      users_output="$(fuser -v /dev/kfd 2>&1 || true)"
      printf '%s\n' "$users_output"
      if [[ -n "$users_output" ]]; then
        busy=1
      fi
    fi

    smi_output="$(rocm-smi --showuse --showmemuse 2>&1)"
    printf '%s\n' "$smi_output"
    if grep -Eq 'GPU use \(%\):[[:space:]]*[1-9][0-9]*' <<<"$smi_output"; then
      busy=1
    fi
    if grep -Eq 'VRAM%\):[[:space:]]*[1-9][0-9]*' <<<"$smi_output"; then
      busy=1
    fi
    if ((busy == 0)); then
      return
    fi
    if ((attempt < 5)); then
      sleep 2
    fi
  done

  printf 'GPU/KFD remained busy; performance data would be invalid.\n' >&2
  exit 3
}

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/moe_prefill_yadai_compare_runs/$RUN_ID}"
BASELINE_LOG_DIR="$LOG_ROOT/baseline"
YADAI_LOG_DIR="$LOG_ROOT/a4w4_prefill_v2_yadai"
FAKE_GIT_DIR="$LOG_ROOT/fake_git"
mkdir -p "$BASELINE_LOG_DIR" "$YADAI_LOG_DIR" "$FAKE_GIT_DIR"

# The baseline custom-shape path only asks Git for HEAD and the branch name. A
# tiny shim supplies those labels without running Git inside the container.
cat >"$FAKE_GIT_DIR/git" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-c" ]]; then
  shift 2
fi

case "${1:-}" in
  rev-parse)
    if [[ "${2:-}" == HEAD ]]; then
      printf '%s\n' "$CODEX_TEST_COMMIT"
      exit 0
    fi
    ;;
  symbolic-ref)
    printf '%s\n' "$CODEX_TEST_BRANCH"
    exit 0
    ;;
esac

printf 'Unexpected test-harness git invocation:' >&2
printf ' %q' "$@" >&2
printf '\n' >&2
exit 2
EOF
chmod +x "$FAKE_GIT_DIR/git"

export ENABLE_CK=0
export AITER_MOE_EXPERT_BALANCE=true
export AITER_LOG_MORE=1
export AITER_USE_GROUPED_GEMM=1
export AITER_GROUPED_DEBUG=0
export AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1

printf 'Baseline repository: %s\n' "$BASELINE_REPO"
printf 'Baseline commit label: %s\n' "$BASELINE_COMMIT_LABEL"
printf 'Yadai repository: %s\n' "$YADAI_REPO"
printf 'Yadai target branch: ROCm/%s\n' "$TARGET_BRANCH"
printf 'Rounds: %s\n' "$ROUNDS"
printf 'Logs: %s\n' "$LOG_ROOT"

require_gpu_idle 'before baseline'
printf '\n===== baseline: run_moe_prefill_switch_ab.sh =====\n'
(
  cd "$BASELINE_REPO"
  PATH="$FAKE_GIT_DIR:$PATH" \
  CODEX_TEST_COMMIT="$BASELINE_COMMIT_LABEL" \
  CODEX_TEST_BRANCH="$BASELINE_BRANCH_LABEL" \
  LOG_DIR="$BASELINE_LOG_DIR" \
  ROUNDS="$ROUNDS" \
    bash ./my_code/run_moe_prefill_switch_ab.sh \
      --experts "$EXPERTS" \
      --tokens "$TOKENS" \
      --topk "$TOPK" \
      --model-dim "$MODEL_DIM" \
      --inter-dim "$INTER_DIM"
) 2>&1 | tee "$LOG_ROOT/baseline_console.log"
require_gpu_idle 'after baseline'

update_yadai_checkout
YADAI_TEST="$YADAI_REPO/op_tests/flydsl_tests/test_flydsl_grouped_gemm.py"
if [[ ! -f "$YADAI_TEST" ]]; then
  printf 'Missing yadai Python test after update: %s\n' "$YADAI_TEST" >&2
  exit 2
fi
YADAI_AITER_IMPORT="$(
  cd "$YADAI_REPO"
  PYTHONPATH="$YADAI_REPO${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -c \
      'from pathlib import Path; import aiter; print(Path(aiter.__file__).resolve())'
)"
case "$YADAI_AITER_IMPORT" in
  "$YADAI_REPO"/*) ;;
  *)
    printf 'Yadai test would import AITER from the wrong tree: %s\n' \
      "$YADAI_AITER_IMPORT" >&2
    exit 2
    ;;
esac
printf 'Yadai Python import: %s\n' "$YADAI_AITER_IMPORT"

COMMON_YADAI_ARGS=(
  --data-format a4w4
  --experts "$EXPERTS"
  --tokens "$TOKENS"
  --topk "$TOPK"
  --model-dim "$MODEL_DIM"
  --inter-dim "$INTER_DIM"
  --act silu
  --no-bias
  --no-check-aot-cache
  --warmup 5
  --iters 20
  --data-init zero
  --scale-init zero
)

# Baseline e2e timing is run_perftest(testGraph=False, use_cuda_event=False,
# num_warmup=5, num_iters=20), returning torch.profiler get_trace_perf device
# time. Force and assert exactly the same contract for yadai.
YADAI_E2E_CODE="$(cat <<'PY'
import aiter.test_common as test_common
from aiter import ActivationType
from op_tests.flydsl_tests import test_flydsl_grouped_gemm as grouped_test


original_run_perftest = test_common.run_perftest


def run_perftest_baseline_compatible(*args, **kwargs):
    if kwargs.get("num_warmup") != 5 or kwargs.get("num_iters") != 20:
        raise RuntimeError(
            "yadai e2e timing must match baseline: num_warmup=5, num_iters=20"
        )
    kwargs["testGraph"] = False
    kwargs["use_cuda_event"] = False
    return original_run_perftest(*args, **kwargs)


test_common.run_perftest = run_perftest_baseline_compatible
grouped_test.set_data_format("a4w4")
metrics = grouped_test.run_moe(
    "a4w4",
    experts=64,
    tokens=1536,
    topk=8,
    model_dim=7168,
    inter_dim=2048,
    activation=ActivationType.Silu,
    use_bias=False,
    bench=True,
    kernel_bench=False,
    warmup=5,
    iters=20,
    seed=0,
    data_init="zero",
    scale_init="zero",
    check_aot_cache=False,
)
print(
    "[yadai-e2e graph=False] fused_moe end-to-end us = "
    f"{metrics['us']:.3f}",
    flush=True,
)
print(
    "[yadai-e2e graph=False] "
    f"logits_diff={metrics['logits_diff']:.4e} "
    f"rel_l2={metrics['rel_l2']:.4e} "
    f"pass={metrics['passed']}",
    flush=True,
)
print(
    "[yadai-e2e timing] backend=torch.profiler "
    "metric=get_trace_perf.device_time_sum "
    "num_warmup=5 num_iters=20 testGraph=False use_cuda_event=False",
    flush=True,
)
if not metrics["passed"]:
    raise SystemExit(4)
PY
)"

printf '\n===== a4w4_prefill_v2_yadai: direct Python tests =====\n'
for ((round = 1; round <= ROUNDS; ++round)); do
  require_gpu_idle "before yadai kernel round $round"
  printf '\n===== yadai round %s/%s: GEMM1 and GEMM2 =====\n' "$round" "$ROUNDS"
  (
    cd "$YADAI_REPO"
    PYTHONPATH="$YADAI_REPO${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON_BIN" -u op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
      --scenario kernel \
      "${COMMON_YADAI_ARGS[@]}"
  ) 2>&1 | tee "$YADAI_LOG_DIR/kernel_r${round}.log"
  require_gpu_idle "after yadai kernel round $round"

  require_gpu_idle "before yadai e2e round $round"
  printf '\n===== yadai round %s/%s: MoE e2e =====\n' "$round" "$ROUNDS"
  (
    cd "$YADAI_REPO"
    PYTHONPATH="$YADAI_REPO${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON_BIN" -u -c "$YADAI_E2E_CODE"
  ) 2>&1 | tee "$YADAI_LOG_DIR/e2e_r${round}.log"
  require_gpu_idle "after yadai e2e round $round"
done

YADAI_SUMMARY="$YADAI_LOG_DIR/summary.md"
"$PYTHON_BIN" - "$YADAI_LOG_DIR" "$YADAI_COMMIT_LABEL" <<'PY' >"$YADAI_SUMMARY"
import re
import statistics
import sys
from pathlib import Path


log_dir = Path(sys.argv[1])
commit = sys.argv[2]


def extract(path: Path, pattern: str) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(pattern, text)
    if match is None:
        raise SystemExit(f"missing metric in {path}: {pattern}")
    return float(match.group(1))


def extract_accuracy(path: Path) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"\[yadai-e2e graph=False\] logits_diff=([0-9.eE+-]+) "
        r"rel_l2=([0-9.eE+-]+) pass=True",
        text,
    )
    if match is None:
        raise SystemExit(f"missing graph=False accuracy result in {path}")
    return float(match.group(1)), float(match.group(2))


kernel_logs = sorted(log_dir.glob("kernel_r*.log"))
e2e_logs = sorted(log_dir.glob("e2e_r*.log"))
gemm1 = [
    extract(path, r"\[kernel-bench a4w4 silu\] gemm1: us = ([0-9.]+)")
    for path in kernel_logs
]
gemm2 = [
    extract(path, r"\[kernel-bench a4w4 silu\] gemm2: us = ([0-9.]+)")
    for path in kernel_logs
]
e2e = [
    extract(
        path,
        r"\[yadai-e2e graph=False\] fused_moe end-to-end us = ([0-9.]+)",
    )
    for path in e2e_logs
]
accuracy = [extract_accuracy(path) for path in e2e_logs]


def samples(values: list[float]) -> str:
    return ", ".join(f"{value:.3f}" for value in values)


print("# a4w4_prefill_v2_yadai E64/T1536/topk8 benchmark")
print()
print(f"- commit: `{commit}`")
print("- input: const0 (`--data-init zero --scale-init zero`)")
print(
    "- MoE e2e timing: `testGraph=False`, `use_cuda_event=False`, "
    "`num_warmup=5`, `num_iters=20`, torch profiler "
    "`get_trace_perf(...).device_time_sum`"
)
print()
print("| Metric | Samples (us) | Median (us) |")
print("|---|---|---:|")
print(f"| GEMM1 | {samples(gemm1)} | {statistics.median(gemm1):.3f} |")
print(f"| GEMM2 | {samples(gemm2)} | {statistics.median(gemm2):.3f} |")
print(f"| MoE e2e | {samples(e2e)} | {statistics.median(e2e):.3f} |")
print()
print("| Round | logits_diff | rel_l2 | pass |")
print("|---:|---:|---:|:---:|")
for round_index, (logits_diff, rel_l2) in enumerate(accuracy, 1):
    print(
        f"| {round_index} | {logits_diff:.4e} | {rel_l2:.4e} | True |"
    )
PY

printf '\n================ yadai summary ================\n'
cat "$YADAI_SUMMARY"
printf '================================================\n'
printf 'Baseline summary: %s\n' "$BASELINE_LOG_DIR/summary.md"
printf 'Yadai summary: %s\n' "$YADAI_SUMMARY"
printf 'All logs: %s\n' "$LOG_ROOT"
