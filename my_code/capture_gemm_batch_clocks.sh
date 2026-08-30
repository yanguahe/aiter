#!/usr/bin/env bash
set -Eeuo pipefail

PROGRAM_NAME="${0##*/}"
RUNNER_REL="my_code/isa_runner/gemm_batch_isa_runner.py"
TIMING_SOURCE_REL="my_code/isa_runner/gemm_isa_runner.py"
DEFAULT_AGT="/opt/amd-apps/agt_internal/agt_internal"
DEFAULT_IDLE="/tmp/yanguahe_mi450_idle_pmlog_20260829T1908_CST.csv"

usage() {
    cat <<EOF
Usage:
  $PROGRAM_NAME [capture options] -- <gemm_batch_isa_runner.py arguments>

Run this script on the Linux host.  AGT runs on the host; the fixed
$RUNNER_REL workload runs in the selected container.

Capture options:
  --container NAME       Container name (default: hyg_fyd1)
  --samples N            AGT sample count (default: 120)
  --period-ms N          AGT sample period in milliseconds (default: 50)
  --workdir PATH         Shared host/container aiter repo root
                         (default: parent of this script's my_code directory)
  --output-dir PATH      New output directory
                         (default: WORKDIR/my_code/clock_logs/<UTC timestamp>)
  --agt PATH             Host agt_internal path (default: $DEFAULT_AGT)
  --idle-baseline PATH   Idle AGT CSV.  The known MI450 baseline is used when
                         present; otherwise idle comparison is skipped.
                         Pass "none" to disable comparison explicitly.
  --sync-timeout SEC     Marker/PID/timed-loop timeout (default: 300)
  -h, --help             Show this help

Everything after -- is passed unchanged as arguments to the fixed runner.
An --isa argument is required.

Example:
  ./$PROGRAM_NAME \\
    --container hyg_fyd1 \\
    --samples 120 \\
    -- \\
    --iters 60000 \\
    --isa ./my_code/path/to/kernel.s \\
    --shape 64,6144,7168 \\
    --batch 96
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

need_value() {
    (($# >= 2)) || die "$1 requires a value"
    [[ -n $2 ]] || die "$1 requires a non-empty value"
}

is_positive_integer() {
    [[ $1 =~ ^[1-9][0-9]*$ ]]
}

utc_ms() {
    date -u '+%Y-%m-%dT%H:%M:%S.%3NZ'
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_WORKDIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

CONTAINER="hyg_fyd1"
SAMPLES=120
PERIOD_MS=50
WORKDIR="$DEFAULT_WORKDIR"
OUTPUT_DIR=""
OUTPUT_DIR_EXPLICIT=0
AGT="$DEFAULT_AGT"
IDLE_BASELINE=""
IDLE_EXPLICIT=0
SYNC_TIMEOUT=300
RUNNER_ARGS=()
SAW_SEPARATOR=0

while (($#)); do
    case "$1" in
        --container)
            need_value "$@"
            CONTAINER=$2
            shift 2
            ;;
        --container=*)
            CONTAINER=${1#*=}
            [[ -n $CONTAINER ]] || die "--container requires a non-empty value"
            shift
            ;;
        --samples)
            need_value "$@"
            SAMPLES=$2
            shift 2
            ;;
        --samples=*)
            SAMPLES=${1#*=}
            shift
            ;;
        --period-ms)
            need_value "$@"
            PERIOD_MS=$2
            shift 2
            ;;
        --period-ms=*)
            PERIOD_MS=${1#*=}
            shift
            ;;
        --workdir)
            need_value "$@"
            WORKDIR=$2
            shift 2
            ;;
        --workdir=*)
            WORKDIR=${1#*=}
            [[ -n $WORKDIR ]] || die "--workdir requires a non-empty value"
            shift
            ;;
        --output-dir)
            need_value "$@"
            OUTPUT_DIR=$2
            OUTPUT_DIR_EXPLICIT=1
            shift 2
            ;;
        --output-dir=*)
            OUTPUT_DIR=${1#*=}
            [[ -n $OUTPUT_DIR ]] || die "--output-dir requires a non-empty value"
            OUTPUT_DIR_EXPLICIT=1
            shift
            ;;
        --agt)
            need_value "$@"
            AGT=$2
            shift 2
            ;;
        --agt=*)
            AGT=${1#*=}
            [[ -n $AGT ]] || die "--agt requires a non-empty value"
            shift
            ;;
        --idle-baseline)
            need_value "$@"
            IDLE_BASELINE=$2
            IDLE_EXPLICIT=1
            shift 2
            ;;
        --idle-baseline=*)
            IDLE_BASELINE=${1#*=}
            [[ -n $IDLE_BASELINE ]] || die \
                "--idle-baseline requires a path or 'none'"
            IDLE_EXPLICIT=1
            shift
            ;;
        --sync-timeout)
            need_value "$@"
            SYNC_TIMEOUT=$2
            shift 2
            ;;
        --sync-timeout=*)
            SYNC_TIMEOUT=${1#*=}
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            SAW_SEPARATOR=1
            shift
            RUNNER_ARGS=("$@")
            break
            ;;
        *)
            die "unknown capture option '$1' (runner arguments must follow --)"
            ;;
    esac
done

((SAW_SEPARATOR)) || die "missing -- separator and runner arguments"
((${#RUNNER_ARGS[@]})) || die "-- must be followed by runner arguments"
is_positive_integer "$SAMPLES" || die "--samples must be a positive integer"
is_positive_integer "$PERIOD_MS" || die "--period-ms must be a positive integer"
is_positive_integer "$SYNC_TIMEOUT" || die \
    "--sync-timeout must be a positive integer"
[[ -n $CONTAINER ]] || die "--container must not be empty"
[[ -n $WORKDIR ]] || die "--workdir must not be empty"
[[ -n $AGT ]] || die "--agt must not be empty"

if ((IDLE_EXPLICIT)); then
    if [[ $IDLE_BASELINE == none ]]; then
        IDLE_BASELINE=""
    elif [[ ! -r $IDLE_BASELINE ]]; then
        die "idle baseline is not readable: $IDLE_BASELINE"
    fi
elif [[ -r $DEFAULT_IDLE ]]; then
    IDLE_BASELINE=$DEFAULT_IDLE
fi

[[ -d $WORKDIR ]] || die "workdir does not exist: $WORKDIR"
WORKDIR="$(cd -- "$WORKDIR" && pwd -P)"
RUNNER_HOST="$WORKDIR/$RUNNER_REL"
TIMING_SOURCE="$WORKDIR/$TIMING_SOURCE_REL"
[[ -r $RUNNER_HOST ]] || die "fixed runner is not readable: $RUNNER_HOST"
[[ -r $TIMING_SOURCE ]] || die \
    "timing helper source is not readable: $TIMING_SOURCE"

ISA_ARG=""
for ((i = 0; i < ${#RUNNER_ARGS[@]}; i++)); do
    case "${RUNNER_ARGS[i]}" in
        --isa)
            ((i + 1 < ${#RUNNER_ARGS[@]})) ||
                die "runner --isa requires a value"
            ISA_ARG=${RUNNER_ARGS[i + 1]}
            ;;
        --isa=*)
            ISA_ARG=${RUNNER_ARGS[i]#*=}
            ;;
    esac
done
[[ -n $ISA_ARG ]] || die "runner arguments must include --isa PATH"

if [[ $ISA_ARG = /* ]]; then
    ISA_HOST=$ISA_ARG
else
    ISA_HOST="$WORKDIR/$ISA_ARG"
fi
[[ -r $ISA_HOST ]] || die "runner ISA is not readable on the host: $ISA_HOST"

for command_name in bash date docker fuser pgrep python3 sudo; do
    command -v "$command_name" >/dev/null 2>&1 ||
        die "required host command is missing: $command_name"
done
[[ -x $AGT ]] || die "AGT is not executable on the host: $AGT"

TIMED_LINE="$(
    python3 - "$TIMING_SOURCE" <<'PY'
import ast
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
matches = []
for node in tree.body:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        continue
    if node.name != "_run_batched_cuda_event_timing":
        continue
    for child in ast.walk(node):
        if not isinstance(child, ast.For):
            continue
        call = child.iter
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "range"
            and len(call.args) == 1
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "num_iters"
        ):
            continue
        for statement in child.body:
            if not isinstance(statement, ast.Assign):
                continue
            if not (
                len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "output"
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == "launch"
            ):
                continue
            matches.append(statement.lineno)
if len(matches) != 1:
    raise SystemExit(
        f"expected one timed 'output = launch()' assignment, found {matches}"
    )
print(matches[0])
PY
)" || die "failed to locate the timed launch line"
is_positive_integer "$TIMED_LINE" || die \
    "invalid dynamically detected timed line: $TIMED_LINE"

docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null |
    python3 -c 'import sys; raise SystemExit(0 if sys.stdin.read().strip() == "true" else 1)' ||
    die "container does not exist or is not running: $CONTAINER"
docker exec "$CONTAINER" test -d "$WORKDIR" ||
    die "container does not expose workdir at the same path: $WORKDIR"
docker exec "$CONTAINER" test -r "$WORKDIR/$RUNNER_REL" ||
    die "fixed runner is not readable in the container: $WORKDIR/$RUNNER_REL"
docker exec "$CONTAINER" test -r "$ISA_HOST" ||
    die "runner ISA is not readable at the same path in the container: $ISA_HOST"

sudo -n true >/dev/null 2>&1 ||
    die "passwordless sudo is required for AGT, py-spy, fuser, and cleanup"

if pgrep -x agt_internal >/dev/null 2>&1; then
    die "another host agt_internal process is already running"
fi

set +e
KFD_USERS="$(sudo -n fuser -v /dev/kfd 2>&1)"
KFD_RC=$?
set -e
if ((KFD_RC == 0)); then
    die "GPU/KFD is already in use; refusing concurrent capture: $KFD_USERS"
elif ((KFD_RC > 1)); then
    die "could not inspect /dev/kfd users: $KFD_USERS"
fi

STAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
if ((OUTPUT_DIR_EXPLICIT == 0)); then
    OUTPUT_DIR="$WORKDIR/my_code/clock_logs/$STAMP"
fi
[[ ! -e $OUTPUT_DIR ]] || die "output directory already exists: $OUTPUT_DIR"
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd -P)"

RAW_LOG="$OUTPUT_DIR/workload.raw.log"
TIMESTAMPED_LOG="$OUTPUT_DIR/workload.timestamped.log"
AGT_LOG="$OUTPUT_DIR/agt.log"
CSV_OUT="$OUTPUT_DIR/agt.csv"
SUMMARY="$OUTPUT_DIR/summary.txt"
METADATA="$OUTPUT_DIR/metadata.env"
WORKLOAD_META="$OUTPUT_DIR/workload.meta"
HOST_TMP_PREFIX="/tmp/${USER:-capture}_capture_gemm_batch_${STAMP}_$$"
HOST_TMP_CSV="${HOST_TMP_PREFIX}.csv"
TEMP_PYSPY=""
CALLING_USER=${SUDO_USER:-${USER:-$(id -un)}}
CALLING_GROUP="$(id -gn "$CALLING_USER")"

WORKLOAD_JOB_PID=""
RUNNER_HOST_PID=""
AGT_JOB_PID=""
WORKLOAD_FINISHED=0
AGT_FINISHED=0
CAPTURE_COMPLETE=0

pid_is_our_runner() {
    local wanted_pid=$1 line
    while IFS= read -r line; do
        if [[ $line =~ ^[[:space:]]*([0-9]+)[[:space:]]+(.*)$ ]] &&
            [[ ${BASH_REMATCH[1]} == "$wanted_pid" ]] &&
            [[ ${BASH_REMATCH[2]} == *"$RUNNER_REL"* ]]; then
            return 0
        fi
    done < <(docker top "$CONTAINER" -eo pid,args 2>/dev/null || true)
    return 1
}

terminate_owned_processes() {
    local _
    if [[ -n $AGT_JOB_PID ]] && ((AGT_FINISHED == 0)) &&
        kill -0 "$AGT_JOB_PID" 2>/dev/null; then
        kill -TERM "$AGT_JOB_PID" 2>/dev/null || true
    fi

    if [[ -n $RUNNER_HOST_PID ]] && ((WORKLOAD_FINISHED == 0)) &&
        pid_is_our_runner "$RUNNER_HOST_PID"; then
        sudo -n kill -TERM "$RUNNER_HOST_PID" 2>/dev/null || true
    fi

    if [[ -n $WORKLOAD_JOB_PID ]] && ((WORKLOAD_FINISHED == 0)) &&
        kill -0 "$WORKLOAD_JOB_PID" 2>/dev/null; then
        kill -TERM "$WORKLOAD_JOB_PID" 2>/dev/null || true
    fi

    for _ in {1..30}; do
        local any=0
        [[ -n $AGT_JOB_PID ]] && kill -0 "$AGT_JOB_PID" 2>/dev/null && any=1
        [[ -n $WORKLOAD_JOB_PID ]] &&
            kill -0 "$WORKLOAD_JOB_PID" 2>/dev/null && any=1
        [[ -n $RUNNER_HOST_PID ]] &&
            pid_is_our_runner "$RUNNER_HOST_PID" && any=1
        ((any == 0)) && break
        sleep 0.1
    done

    if [[ -n $RUNNER_HOST_PID ]] && pid_is_our_runner "$RUNNER_HOST_PID"; then
        sudo -n kill -KILL "$RUNNER_HOST_PID" 2>/dev/null || true
    fi
    if [[ -n $AGT_JOB_PID ]] && kill -0 "$AGT_JOB_PID" 2>/dev/null; then
        kill -KILL "$AGT_JOB_PID" 2>/dev/null || true
    fi
    if [[ -n $WORKLOAD_JOB_PID ]] &&
        kill -0 "$WORKLOAD_JOB_PID" 2>/dev/null; then
        kill -KILL "$WORKLOAD_JOB_PID" 2>/dev/null || true
    fi
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM HUP
    if ((CAPTURE_COMPLETE == 0)); then
        terminate_owned_processes
    fi
    if [[ -n $TEMP_PYSPY ]]; then
        rm -f -- "$TEMP_PYSPY" 2>/dev/null || true
    fi
    if [[ -e $HOST_TMP_CSV ]]; then
        sudo -n rm -f -- "$HOST_TMP_CSV" 2>/dev/null || true
    fi
    if [[ -e $WORKLOAD_META ]]; then
        sudo -n chown "$CALLING_USER:$CALLING_GROUP" \
            "$WORKLOAD_META" 2>/dev/null || true
        chmod u+rw,go-rwx "$WORKLOAD_META" 2>/dev/null || true
    fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

detect_pyspy() {
    local candidate container_path
    candidate="$(command -v py-spy 2>/dev/null || true)"
    for candidate in \
        "$candidate" \
        /usr/local/bin/py-spy \
        /usr/bin/py-spy \
        /opt/venv/bin/py-spy; do
        [[ -n $candidate && -x $candidate ]] || continue
        if sudo -n "$candidate" --version >/dev/null 2>&1; then
            PYSPY=$candidate
            return 0
        fi
    done

    container_path="$(
        docker exec "$CONTAINER" sh -c \
            'command -v py-spy 2>/dev/null || true' |
            python3 -c 'import sys; print(sys.stdin.read().strip())'
    )"
    [[ -n $container_path ]] || return 1
    container_path="$(
        docker exec "$CONTAINER" python3 -c \
            'import os,sys; print(os.path.realpath(sys.argv[1]))' \
            "$container_path"
    )"
    TEMP_PYSPY="${HOST_TMP_PREFIX}.py-spy"
    docker cp "$CONTAINER:$container_path" "$TEMP_PYSPY" >/dev/null
    chmod 0700 "$TEMP_PYSPY"
    sudo -n "$TEMP_PYSPY" --version >/dev/null 2>&1 || return 1
    PYSPY=$TEMP_PYSPY
}

PYSPY=""
detect_pyspy || die \
    "py-spy is unavailable or cannot run through sudo; no fixed-sleep fallback"
[[ -x $PYSPY ]] || die "detected py-spy is not executable: $PYSPY"

{
    printf 'capture_version=1\n'
    printf 'stamp=%s\n' "$STAMP"
    printf 'host=%s\n' "$(hostname -f)"
    printf 'container=%s\n' "$CONTAINER"
    printf 'workdir=%s\n' "$WORKDIR"
    printf 'runner=%s\n' "$RUNNER_REL"
    printf 'isa=%s\n' "$ISA_ARG"
    printf 'samples=%s\n' "$SAMPLES"
    printf 'period_ms=%s\n' "$PERIOD_MS"
    printf 'sync_timeout_sec=%s\n' "$SYNC_TIMEOUT"
    printf 'timed_launch_source=%s\n' "$TIMING_SOURCE"
    printf 'timed_launch_line=%s\n' "$TIMED_LINE"
    printf 'py_spy=%s\n' "$PYSPY"
    printf 'host_tmp_csv=%s\n' "$HOST_TMP_CSV"
    printf 'idle_baseline=%s\n' "$IDLE_BASELINE"
    printf 'requested_sample_span_ms=%s\n' "$(((SAMPLES - 1) * PERIOD_MS))"
    printf 'workload_launch_utc=%s\n' "$(utc_ms)"
    printf 'runner_arg_count=%s\n' "${#RUNNER_ARGS[@]}"
    for ((i = 0; i < ${#RUNNER_ARGS[@]}; i++)); do
        printf 'runner_arg_%d=%q\n' "$i" "${RUNNER_ARGS[i]}"
    done
} >"$METADATA"

timestamp_stream() {
    local line timestamp
    while IFS= read -r line || [[ -n $line ]]; do
        printf '%s\n' "$line" >>"$RAW_LOG"
        timestamp="$(utc_ms)"
        printf '%s\t%s\n' "$timestamp" "$line" >>"$TIMESTAMPED_LOG"
    done
}

run_workload() {
    set +e
    docker exec -w "$WORKDIR" "$CONTAINER" bash -c '
        printf "process_start_utc=%s\n" \
            "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" >"$1"
        PYTHONUNBUFFERED=1 AITER_LOG_MORE=1 \
            python -u "$2" "${@:3}"
        rc=$?
        {
            printf "process_end_utc=%s\n" \
                "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
            printf "workload_rc=%s\n" "$rc"
        } >>"$1"
        exit "$rc"
    ' _ "$WORKLOAD_META" "$RUNNER_REL" "${RUNNER_ARGS[@]}" 2>&1 |
        timestamp_stream
    local rc=${PIPESTATUS[0]}
    return "$rc"
}

file_contains_literal() {
    local file=$1 needle=$2 line
    [[ -r $file ]] || return 1
    while IFS= read -r line || [[ -n $line ]]; do
        [[ $line == *"$needle"* ]] && return 0
    done <"$file"
    return 1
}

get_runner_host_pid() {
    local line args executable executable_name
    local -a matches=()
    while IFS= read -r line; do
        [[ $line =~ ^[[:space:]]*([0-9]+)[[:space:]]+(.*)$ ]] || continue
        args=${BASH_REMATCH[2]}
        [[ $args == *"$RUNNER_REL"* ]] || continue
        executable=${args%%[[:space:]]*}
        executable_name=${executable##*/}
        [[ $executable_name == python* ]] || continue
        matches+=("${BASH_REMATCH[1]}")
    done < <(docker top "$CONTAINER" -eo pid,args 2>/dev/null || true)
    ((${#matches[@]} == 1)) || return 1
    printf '%s\n' "${matches[0]}"
}

: >"$RAW_LOG"
: >"$TIMESTAMPED_LOG"
: >"$AGT_LOG"
run_workload &
WORKLOAD_JOB_PID=$!

deadline=$((SECONDS + SYNC_TIMEOUT))
while ! file_contains_literal \
    "$RAW_LOG" "[gemm_batch_isa_runner] loading kernel symbol:"; do
    kill -0 "$WORKLOAD_JOB_PID" 2>/dev/null ||
        die "workload exited before the loading-kernel marker; see $RAW_LOG"
    ((SECONDS < deadline)) ||
        die "timed out waiting for the loading-kernel marker"
    sleep 0.05
done
printf 'loading_kernel_observed_utc=%s\n' "$(utc_ms)" >>"$METADATA"

while [[ -z $RUNNER_HOST_PID ]]; do
    RUNNER_HOST_PID="$(get_runner_host_pid || true)"
    kill -0 "$WORKLOAD_JOB_PID" 2>/dev/null ||
        die "workload exited before its host PID was found"
    ((SECONDS < deadline)) || die "timed out locating the container runner host PID"
    [[ -n $RUNNER_HOST_PID ]] || sleep 0.05
done
printf 'runner_host_pid=%s\n' "$RUNNER_HOST_PID" >>"$METADATA"

stack_attempts=0
stack_failures=0
matched_stack=""
last_stack_output=""
while :; do
    if ! kill -0 "$WORKLOAD_JOB_PID" 2>/dev/null; then
        {
            printf 'py_spy_attempts=%s\n' "$stack_attempts"
            printf 'py_spy_failures=%s\n' "$stack_failures"
            printf 'last_stack_output=%q\n' "$last_stack_output"
        } >>"$METADATA"
        die "workload exited before the timed enqueue loop was detected"
    fi
    stack_attempts=$((stack_attempts + 1))
    set +e
    stack_output="$(
        sudo -n "$PYSPY" dump --nonblocking --full-filenames \
            --pid "$RUNNER_HOST_PID" 2>&1
    )"
    stack_rc=$?
    set -e
    last_stack_output=$stack_output
    if ((stack_rc != 0)); then
        stack_failures=$((stack_failures + 1))
    elif [[ $stack_output == *"_run_batched_cuda_event_timing"* ]] &&
        [[ $stack_output == *"gemm_isa_runner.py:$TIMED_LINE"* ]]; then
        matched_stack=$stack_output
        break
    fi
    ((SECONDS < deadline)) ||
        die "timed out waiting for the exact timed launch stack line"
    sleep 0.05
done

TIMED_DETECTED_UTC="$(utc_ms)"
{
    printf 'timed_loop_detected_utc=%s\n' "$TIMED_DETECTED_UTC"
    printf 'py_spy_attempts=%s\n' "$stack_attempts"
    printf 'py_spy_failures=%s\n' "$stack_failures"
    printf 'matched_stack=%q\n' "$matched_stack"
} >>"$METADATA"

AGT_START_UTC="$(utc_ms)"
printf 'agt_start_utc=%s\n' "$AGT_START_UTC" >>"$METADATA"
sudo -n "$AGT" \
    -i=0 -unilog=PM -unilogallgroups -unilogstopcheck \
    -unilognoesckey -unilogperiod="$PERIOD_MS" -unilogcount="$SAMPLES" \
    -unilogoutput="$HOST_TMP_CSV" >"$AGT_LOG" 2>&1 &
AGT_JOB_PID=$!

set +e
wait "$AGT_JOB_PID"
AGT_RC=$?
AGT_FINISHED=1
AGT_END_UTC="$(utc_ms)"
wait "$WORKLOAD_JOB_PID"
WORKLOAD_RC=$?
WORKLOAD_FINISHED=1
set -e

{
    printf 'agt_end_utc=%s\n' "$AGT_END_UTC"
    printf 'agt_rc=%s\n' "$AGT_RC"
    printf 'workload_wait_end_utc=%s\n' "$(utc_ms)"
    printf 'workload_rc=%s\n' "$WORKLOAD_RC"
} >>"$METADATA"

((AGT_RC == 0)) || die "AGT failed with rc=$AGT_RC; see $AGT_LOG"
[[ -s $HOST_TMP_CSV ]] || die "AGT did not produce a non-empty CSV"

sudo -n install -o "$CALLING_USER" -g "$CALLING_GROUP" -m 0600 \
    "$HOST_TMP_CSV" "$CSV_OUT"
for generated in "$WORKLOAD_META"; do
    [[ -e $generated ]] || continue
    sudo -n chown "$CALLING_USER:$CALLING_GROUP" "$generated"
    chmod u+rw,go-rwx "$generated"
done

set +e
python3 - \
    "$CSV_OUT" "$IDLE_BASELINE" "$RAW_LOG" "$TIMESTAMPED_LOG" \
    "$METADATA" "$WORKLOAD_META" "$PERIOD_MS" "$SAMPLES" \
    "$AGT_RC" "$WORKLOAD_RC" <<'PY' | tee "$SUMMARY"
import csv
import datetime as dt
import math
import pathlib
import re
import statistics
import sys

(
    csv_path,
    idle_path,
    raw_path,
    timestamped_path,
    metadata_path,
    workload_meta_path,
    period_ms_text,
    samples_text,
    agt_rc_text,
    workload_rc_text,
) = sys.argv[1:]
period_ms = int(period_ms_text)
requested_samples = int(samples_text)
agt_rc = int(agt_rc_text)
workload_rc = int(workload_rc_text)


def read_kv(path):
    result = {}
    p = pathlib.Path(path)
    if not p.is_file():
        return result
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def parse_iso(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def fmt_time(value):
    return value.astimezone(dt.timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def parse_csv_timestamp(value):
    match = re.search(
        r"(\d{2})\.(\d{2})\.(\d{4}) "
        r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})",
        value,
    )
    if not match:
        raise ValueError(f"unrecognized AGT timestamp: {value!r}")
    day, month, year, hour, minute, second, millis = map(int, match.groups())
    return dt.datetime(
        year,
        month,
        day,
        hour,
        minute,
        second,
        millis * 1000,
        tzinfo=dt.timezone.utc,
    )


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not fields or not rows:
        raise RuntimeError(f"CSV has no data rows: {path}")
    return fields, rows


def numeric(rows, column):
    values = []
    for row in rows:
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def stat(rows, column):
    values = numeric(rows, column)
    if not values:
        raise RuntimeError(f"no numeric values for required field: {column}")
    return min(values), statistics.fmean(values), max(values)


def combine(rows, columns):
    values = [value for column in columns for value in numeric(rows, column)]
    if not values:
        raise RuntimeError(f"no numeric values for fields: {columns}")
    return min(values), statistics.fmean(values), max(values)


def triple(values):
    return " / ".join(f"{value:.3f}" for value in values)


def locate(fields, pattern, expected=None):
    regex = re.compile(pattern)
    matches = [field for field in fields if regex.search(field)]
    if expected is not None and len(matches) != expected:
        raise RuntimeError(
            f"header pattern {pattern!r}: expected {expected}, found {matches}"
        )
    return matches


metadata = read_kv(metadata_path)
workload_meta = read_kv(workload_meta_path)
raw_lines = pathlib.Path(raw_path).read_text(
    encoding="utf-8", errors="replace"
).splitlines()
timestamped_lines = pathlib.Path(timestamped_path).read_text(
    encoding="utf-8", errors="replace"
).splitlines()

timing_row = None
for index, line in enumerate(raw_lines):
    if "device_time_sum" not in line or "|" not in line:
        continue
    headers = [cell.strip() for cell in line.strip().strip("|").split("|")]
    for candidate in raw_lines[index + 1 :]:
        if "|" not in candidate:
            continue
        cells = [cell.strip() for cell in candidate.strip().strip("|").split("|")]
        if cells and all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if len(cells) == len(headers):
            timing_row = dict(zip(headers, cells))
            break
    if timing_row:
        break
if not timing_row:
    raise RuntimeError("could not parse the runner CUDA-event timing table")

device_sum_us = float(timing_row["device_time_sum"])
device_avg_us = float(timing_row["device_time_avg"])
timed_start = parse_iso(metadata["timed_loop_detected_utc"])
estimated_timed_end = timed_start + dt.timedelta(microseconds=device_sum_us)

validation_start = None
for line in timestamped_lines:
    if "checkAllclose" not in line:
        continue
    timestamp = line.split("\t", 1)[0]
    validation_start = parse_iso(timestamp)
    break

fields, rows = read_csv(csv_path)
time_column = fields[0]
row_times = [parse_csv_timestamp(row[time_column]) for row in rows]
filter_end = (
    min(estimated_timed_end, validation_start)
    if validation_start is not None
    else estimated_timed_end
)
overlap = [
    (row, timestamp)
    for row, timestamp in zip(rows, row_times)
    if timed_start <= timestamp <= filter_end
]

df_read_columns = locate(
    fields, r"DF Transactions GFX Reads AID[01]$", expected=None
)
if df_read_columns:
    active_pairs = [
        (row, timestamp)
        for row, timestamp in overlap
        if all(float(row[column]) > 0.0 for column in df_read_columns)
    ]
    activity_basis = (
        "timed-window overlap + all available DF Transactions GFX Reads "
        "AID fields > 0"
    )
else:
    active_pairs = overlap
    activity_basis = (
        "timed-window overlap; DF GFX read counters unavailable, power/DPM "
        "fields are reported only as auxiliary evidence"
    )
active = [row for row, _ in active_pairs]
active_times = [timestamp for _, timestamp in active_pairs]
if not active:
    raise SystemExit("no active AGT samples survived timed-window filtering")

actual_span_ms = (
    (row_times[-1] - row_times[0]).total_seconds() * 1000.0
    if len(row_times) > 1
    else 0.0
)
expected_span_ms = (len(rows) - 1) * period_ms

print("采集结果")
print(f"AGT rc={agt_rc}; workload rc={workload_rc}")
if workload_rc == 3:
    print(
        "警告：workload rc=3，runner correctness/allclose 失败；"
        "timing 与完整 CSV 仍继续分析。"
    )
elif workload_rc != 0:
    print(f"警告：workload 非零退出 rc={workload_rc}。")
print(
    f"device_time_sum={device_sum_us:.4f} us "
    f"({device_sum_us / 1e6:.6f} s); "
    f"device_time_avg={device_avg_us:.4f} us"
)
summary_row = None
for index, line in enumerate(raw_lines):
    if "gemm_a4w4 TB/s" not in line or "|" not in line:
        continue
    headers = [cell.strip() for cell in line.strip().strip("|").split("|")]
    for candidate in raw_lines[index + 1 :]:
        if "|" not in candidate:
            continue
        cells = [cell.strip() for cell in candidate.strip().strip("|").split("|")]
        if cells and all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if len(cells) == len(headers):
            summary_row = dict(zip(headers, cells))
            break
    if summary_row:
        break
if summary_row:
    print(
        "runner throughput: "
        f"TFLOPS={summary_row.get('gemm_a4w4 TFLOPS', 'unknown')}; "
        f"TB/s={summary_row.get('gemm_a4w4 TB/s', 'unknown')}"
    )
for line in raw_lines:
    if "[gemm_batch_isa_runner] full batch:" in line:
        print(f"correctness: {line}")
        break

print()
print("Timeline（UTC）")
timeline = [
    (workload_meta.get("process_start_utc"), "Workload process 启动"),
    (metadata.get("loading_kernel_observed_utc"), "loading kernel symbol 首次观察"),
    (
        metadata.get("timed_loop_detected_utc"),
        "start event 后进入 N 次 timed enqueue loop",
    ),
    (metadata.get("agt_start_utc"), "AGT 启动"),
]
for timestamp, label in timeline:
    if timestamp:
        print(f"{fmt_time(parse_iso(timestamp))} {label}")
print(
    f"|<=== GPU timed interval ≈ {device_sum_us / 1e6:.6f} s; "
    f"估算结束 {fmt_time(estimated_timed_end)} ===>|"
)
print(f"{fmt_time(row_times[0])} CSV 首个 sample")
print(f"{fmt_time(row_times[-1])} CSV 最后 sample")
if validation_start:
    print(f"{fmt_time(validation_start)} Validation/checkAllclose 开始")
if metadata.get("agt_end_utc"):
    print(f"{fmt_time(parse_iso(metadata['agt_end_utc']))} AGT 结束")
if workload_meta.get("process_end_utc"):
    print(
        f"{fmt_time(parse_iso(workload_meta['process_end_utc']))} "
        "Workload process 结束"
    )
print(
    f"CSV {len(rows)} samples；首末跨度 {actual_span_ms:.3f} ms；"
    f"按 (samples-1)*period 计算为 {expected_span_ms} ms。"
)
if len(rows) != requested_samples:
    print(
        f"警告：请求 {requested_samples} samples，CSV 实际为 {len(rows)}。"
    )

print()
print("Active sample 筛选")
print(
    f"timed-window overlap={len(overlap)}/{len(rows)}；"
    f"activity 验证后={len(active)}/{len(rows)}"
)
print(
    f"范围 {fmt_time(active_times[0])} - {fmt_time(active_times[-1])} UTC"
)
print(f"依据：{activity_basis}")
if df_read_columns:
    for column in df_read_columns:
        values = stat(active, column)
        print(f"{column}: {triple(values)}")
print(
    f"说明：{period_ms} ms 粒度无法解析短于一个 sample 周期的瞬态；"
    "timed end 是检测时间加 device_time_sum 的近似值；"
    "若 validation 更早开始，则以 validation 起点作为筛选上界。"
)

gfx_target = [
    f"GPU0 XCD XCD{index} Target Freq" for index in range(8)
]
gfx_pre = [
    f"GPU0 XCD XCD{index} Pre Deep Sleep Freq" for index in range(8)
]
gfx_post = [
    f"GPU0 XCD XCD{index} Post Deep Sleep Freq" for index in range(8)
]
for column in gfx_target + gfx_pre + gfx_post:
    if column not in fields:
        raise RuntimeError(f"required AGT field is missing: {column}")

print()
print("Active 时钟统计")
print("单位 MHz，格式均为 min / mean / max。")
print(
    f"{'XCD':<5}{'Target':<32}{'Pre Deep Sleep':<32}"
    f"{'Post Deep Sleep'}"
)
for index in range(8):
    print(
        f"{index:<5}{triple(stat(active, gfx_target[index])):<32}"
        f"{triple(stat(active, gfx_pre[index])):<32}"
        f"{triple(stat(active, gfx_post[index]))}"
    )

clock_rows = []
for clock in ("FCLK", "GL2CLK"):
    for aid in range(2):
        target = f"GPU0 Target VDDX Frequencies {clock} AID{aid}"
        pre = f"GPU0 Pre Deep Sleep VDDX Frequencies {clock} AID{aid}"
        post = f"GPU0 Post Deep Sleep VDDX Frequencies {clock} AID{aid}"
        for column in (target, pre, post):
            if column not in fields:
                raise RuntimeError(f"required AGT field is missing: {column}")
        clock_rows.append((f"{clock} AID{aid}", target, pre, post))
print()
print(
    f"{'Domain':<14}{'Target':<32}{'Pre Deep Sleep':<32}"
    f"{'Post Deep Sleep'}"
)
for label, target, pre, post in clock_rows:
    print(
        f"{label:<14}{triple(stat(active, target)):<32}"
        f"{triple(stat(active, pre)):<32}{triple(stat(active, post))}"
    )

print()
for aid in range(2):
    for suffix in ("a", "b"):
        target = f"GPU0 Target Frequencies MCLK_{suffix} AID{aid}"
        effective = (
            f"GPU0 Effective Frequencies MCLK_{suffix} Eff AID{aid}"
        )
        for column in (target, effective):
            if column not in fields:
                raise RuntimeError(f"required AGT field is missing: {column}")
        print(
            f"MCLK_{suffix} AID{aid} "
            f"target={triple(stat(active, target))}  "
            f"Eff={triple(stat(active, effective))}"
        )

groups = {
    "GFX Target": gfx_target,
    "GFX Pre/Post": gfx_pre + gfx_post,
    "FCLK Target": locate(
        fields, r"Target VDDX Frequencies FCLK AID[01]$", expected=2
    ),
    "FCLK Pre/Post": locate(
        fields,
        r"(Pre|Post) Deep Sleep VDDX Frequencies FCLK AID[01]$",
        expected=4,
    ),
    "GL2CLK Target": locate(
        fields, r"Target VDDX Frequencies GL2CLK AID[01]$", expected=2
    ),
    "GL2CLK Pre/Post": locate(
        fields,
        r"(Pre|Post) Deep Sleep VDDX Frequencies GL2CLK AID[01]$",
        expected=4,
    ),
    "MCLK Target/Eff": locate(
        fields,
        r"(Target Frequencies MCLK_[ab] AID[01]|"
        r"Effective Frequencies MCLK_[ab] Eff AID[01])$",
        expected=8,
    ),
}

idle_rows = None
if idle_path:
    try:
        idle_fields, idle_rows = read_csv(idle_path)
        missing = [
            column
            for columns in groups.values()
            for column in columns
            if column not in idle_fields
        ]
        if missing:
            raise RuntimeError(f"idle baseline missing fields: {missing}")
    except Exception as error:
        print()
        print(f"警告：idle baseline 无法比较：{error}")
        idle_rows = None

if idle_rows is not None:
    print()
    print("全局 active 与 idle baseline 对比：")
    print(
        f"{'':<21}{'active min/mean/max':<32}"
        f"{'idle min/mean/max'}"
    )
    group_stats = {}
    for label, columns in groups.items():
        active_stat = combine(active, columns)
        idle_stat = combine(idle_rows, columns)
        group_stats[label] = (active_stat, idle_stat)
        print(
            f"{label:<21}{triple(active_stat):<32}{triple(idle_stat)}"
        )
    for label in ("GFX Target", "GFX Pre/Post", "GL2CLK Target", "GL2CLK Pre/Post"):
        active_stat, idle_stat = group_stats[label]
        print(
            f"{label} mean 差值 active-idle="
            f"{active_stat[1] - idle_stat[1]:.3f} MHz"
        )
    for label in ("FCLK Target", "FCLK Pre/Post", "MCLK Target/Eff"):
        active_stat, idle_stat = group_stats[label]
        delta = active_stat[1] - idle_stat[1]
        verdict = "下降" if delta < -0.5 else "未下降"
        print(f"{label}: {verdict}（mean 差值 {delta:.3f} MHz）")

print()
print("Power / limiter 摘要")
power_columns = [
    "GPU0 Input Telemetry Telemetry Voltage ",
    "GPU0 Input Telemetry Telemetry Current ",
    "GPU0 Input Telemetry Telemetry Power ",
    "GPU0 INFRASTRUCTURE Limit PPT_0",
    "GPU0 INFRASTRUCTURE Value PPT_0",
    "GPU0 INFRASTRUCTURE Limit PPT_1",
    "GPU0 INFRASTRUCTURE Value PPT_1",
    "GPU0 MISC TDP ",
    "GPU0 MISC DPM TASK Busy ",
]
for column in power_columns:
    if column in fields:
        print(f"{column}: {triple(stat(active, column))}")

ppt0_limiter = "GPU0 Frequency Limiters PPT_0 GFXCLK"
global_limiter = "GPU0 Frequency Limiters GLOBAL GFXCLK"
for column in (ppt0_limiter, global_limiter):
    if column in fields:
        print(f"{column}: {triple(stat(active, column))} MHz")

other_limiters = [
    column
    for column in fields
    if "Frequency Limiters " in column
    and column.endswith(" GFXCLK")
    and column not in (ppt0_limiter, global_limiter)
]
if other_limiters:
    other_values = combine(active, other_limiters)
    print(f"其他 GFXCLK limiter 全局范围: {triple(other_values)} MHz")

ratios = []
for limit_column in fields:
    prefix = "GPU0 INFRASTRUCTURE Limit "
    if not limit_column.startswith(prefix):
        continue
    domain = limit_column[len(prefix) :]
    if not (domain.startswith("TDC_") or domain.startswith("THM_")):
        continue
    value_column = f"GPU0 INFRASTRUCTURE Value {domain}"
    if value_column not in fields:
        continue
    limit_mean = stat(active, limit_column)[1]
    value_stats = stat(active, value_column)
    if limit_mean > 0:
        ratios.append((100.0 * value_stats[1] / limit_mean, domain, value_stats))
if ratios:
    highest = max(ratios)
    print(
        f"TDC/thermal 最高 mean/limit={highest[0]:.3f}% "
        f"({highest[1]}; value {triple(highest[2])})"
    )

for label in (
    "GPU0 MISC Peak XCD Temperature ",
    "GPU0 MISC Peak AID Temperature ",
    "GPU0 MISC Peak HBM Temperature ",
    "GPU0 MISC VR HOT Residency ",
    "GPU0 MISC PROCHOT Residency ",
    "GPU0 MISC DfCstate Residency ",
    "GPU0 MISC System Idle Residency ",
):
    if label in fields:
        print(f"{label}: {triple(stat(active, label))}")

for label, pattern in (
    ("XCD PWRBRK Residency", r"XCD XCD[0-7] PWRBRK Residency$"),
    ("XCD PSM DIDT Residency", r"XCD XCD[0-7] PSM DIDT Residency$"),
    ("XCD PIT Stall Residency", r"XCD XCD[0-7] PIT Stall Residency$"),
    ("EDC OCP_WARN Residency", r"EDC OCP_WARN Residency "),
):
    columns = locate(fields, pattern, expected=None)
    if columns:
        print(f"{label}: {triple(combine(active, columns))}")

xcd_busy = locate(fields, r"XCD XCD[0-7] Busy$", expected=None)
if xcd_busy:
    busy_stats = combine(active, xcd_busy)
    print(f"XCD Busy 全局: {triple(busy_stats)}")
    if busy_stats[2] == 0:
        print(
            "警告：XCD Busy 全部为零，当前遥测不可用；"
            "未将它用于 active 筛选。"
        )

ppt_limit = stat(active, "GPU0 INFRASTRUCTURE Limit PPT_0")[1]
ppt_value = stat(active, "GPU0 INFRASTRUCTURE Value PPT_0")[1]
if (
    ppt_limit > 0
    and ppt_value / ppt_limit >= 0.95
    and ppt0_limiter in fields
    and global_limiter in fields
    and abs(
        stat(active, ppt0_limiter)[1]
        - stat(active, global_limiter)[1]
    )
    < 1.0
):
    print(
        "谨慎判断：PPT_0 value 接近 limit，且 PPT_0 与 GLOBAL "
        "GFXCLK limiter 一致；数据指向 PPT_0 power cap 主导 GFX/GL2 "
        "target 下调。Target 不是 effective。"
    )
else:
    print("谨慎判断：现有 limiter 字段不足以唯一确定降频原因。")

print()
print("输出文件")
for label, path in (
    ("CSV", csv_path),
    ("raw workload log", raw_path),
    ("timestamped workload log", timestamped_path),
    ("metadata", metadata_path),
):
    print(f"{label}: {path}")
PY
ANALYZE_RC=${PIPESTATUS[0]}
set -e
((ANALYZE_RC == 0)) || die "CSV/timeline analysis failed with rc=$ANALYZE_RC"

chmod u+rw,go-rwx \
    "$RAW_LOG" "$TIMESTAMPED_LOG" "$AGT_LOG" "$CSV_OUT" \
    "$SUMMARY" "$METADATA"
sudo -n rm -f -- "$HOST_TMP_CSV"
[[ -z $TEMP_PYSPY ]] || rm -f -- "$TEMP_PYSPY"
TEMP_PYSPY=""

CAPTURE_COMPLETE=1
printf '\nCapture artifacts: %s\n' "$OUTPUT_DIR"
if ((WORKLOAD_RC == 3)); then
    printf 'warning: analysis completed, but workload correctness failed (rc=3)\n' \
        >&2
    exit 3
fi
if ((WORKLOAD_RC != 0)); then
    printf 'warning: analysis completed, but workload exited rc=%s\n' \
        "$WORKLOAD_RC" >&2
    exit "$WORKLOAD_RC"
fi
exit 0
