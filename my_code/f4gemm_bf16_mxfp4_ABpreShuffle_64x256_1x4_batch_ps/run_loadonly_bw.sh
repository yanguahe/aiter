#!/usr/bin/env bash
# Prepared vdi_mc run for the load-only TDM bandwidth variant of
# f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.
#
# The kernel writes no output, so gemm_batch_isa_runner.py always fails its
# correctness check and exits 3.  Exit 3 is the expected success case here;
# only the "gemm_a4w4 us" timing row is meaningful.  For M,N,K=64,6144,7168 and
# batch=96 the grid reads 2,269,446,144 distinct bytes of A/B/sA/sB per dispatch
# and issues 2,807,562,240 bytes of TDM requests, so
#   distinct-byte  TB/s = 2269.446144 / us
#   issued-request TB/s = 2807.562240 / us
# The runner's own "gemm_a4w4 TB/s" column adds the 75,497,472 D bytes this
# kernel never writes, so it reads 3.33% high; "TFLOPS" is meaningless here.
#
# Phases, all runnable on their own:
#   bash run_loadonly_bw.sh sync      # host-side git sync only
#   bash run_loadonly_bw.sh run       # container-side benchmark only
#   bash run_loadonly_bw.sh publish   # host-side commit --amend + push -f
#   bash run_loadonly_bw.sh           # sync, then re-exec the pulled copy for run + publish
#
# ####################################################################
# # REQUIRED: the container name has not been provided yet.  Either  #
# # edit the line below or export CONTAINER=<name> before running.   #
# ####################################################################
CONTAINER="${CONTAINER:-<FILL_ME>}"

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/data/yanguahe/code/wk_sp1/aiter}"
KERNEL_DIR="my_code/f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps"
ISA="./${KERNEL_DIR}/f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps_loadonly.s"
BASELINE_ISA="./my_code/f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.s"
OUT_DIR="${KERNEL_DIR}/logs"
SCRIPT_REL="${KERNEL_DIR}/run_loadonly_bw.sh"

SHAPE="${SHAPE:-64,6144,7168}"
BATCH="${BATCH:-96}"
ITERS="${ITERS:-100}"
# Opt-in: also time the unmodified GEMM so the load-only number has a reference.
RUN_BASELINE="${RUN_BASELINE:-0}"

host_sync() {
    cd "${REPO_ROOT}"
    export GIT_SSH_COMMAND='ssh -i /data/yanguahe/code/id_rsa.hyg -o IdentitiesOnly=yes'
    # Explicit refspec: this clone's remote.origin.fetch does not cover hyg_dev,
    # so a bare "git fetch origin hyg_dev" leaves refs/remotes/origin/hyg_dev unset.
    git fetch origin '+refs/heads/hyg_dev:refs/remotes/origin/hyg_dev'
    git checkout -B hyg_dev origin/hyg_dev
    git reset --hard origin/hyg_dev
    git --no-pager log -1 --oneline
}

container_run() {
    cd "${REPO_ROOT}"
    mkdir -p "${OUT_DIR}"
    local log="${OUT_DIR}/loadonly_bw.log"

    {
        echo "date:      $(date -Is)"
        echo "host:      $(hostname)"
        echo "container: ${CONTAINER}"
        echo "repo_head: ${HOST_GIT_COMMIT:-unknown}"
        echo "shape:     M,N,K=${SHAPE} batch=${BATCH} iters=${ITERS}"
        echo "isa:       ${ISA}"
        python -c 'import torch; print("torch:    ", torch.__version__, "hip:", torch.version.hip)'
        rocm-smi --showproductname 2>/dev/null | sed -n '1,12p' || true
        echo
    } 2>&1 | tee "${log}"

    set +e
    AITER_LOG_MORE=1 python my_code/isa_runner/gemm_batch_isa_runner.py \
        --iters "${ITERS}" \
        --isa "${ISA}" \
        --shape "${SHAPE}" \
        --batch "${BATCH}" \
        2>&1 | tee -a "${log}"
    local status="${PIPESTATUS[0]}"
    set -e
    echo "loadonly_exit_status=${status} (3 = expected: no output is written)" | tee -a "${log}"

    if [[ "${RUN_BASELINE}" == "1" ]]; then
        local baseline_log="${OUT_DIR}/baseline_bw.log"
        set +e
        AITER_LOG_MORE=1 python my_code/isa_runner/gemm_batch_isa_runner.py \
            --iters "${ITERS}" \
            --isa "${BASELINE_ISA}" \
            --shape "${SHAPE}" \
            --batch "${BATCH}" \
            2>&1 | tee "${baseline_log}"
        echo "baseline_exit_status=${PIPESTATUS[0]}" | tee -a "${baseline_log}"
        set -e
    fi

    # Every container-generated path must be host-writable before git touches it.
    chmod -R a+rwX "${OUT_DIR}"

    if [[ "${status}" -ne 0 && "${status}" -ne 3 ]]; then
        return "${status}"
    fi
}

host_run() {
    if [[ "${CONTAINER}" == "<FILL_ME>" || -z "${CONTAINER}" ]]; then
        echo "ERROR: set the container name first, e.g. CONTAINER=<name> bash ${SCRIPT_REL} run" >&2
        exit 2
    fi
    local head
    head="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    docker exec -i \
        -e REPO_ROOT="${REPO_ROOT}" \
        -e CONTAINER="${CONTAINER}" \
        -e HOST_GIT_COMMIT="${head}" \
        -e SHAPE="${SHAPE}" \
        -e BATCH="${BATCH}" \
        -e ITERS="${ITERS}" \
        -e RUN_BASELINE="${RUN_BASELINE}" \
        "${CONTAINER}" \
        bash -lc "cd '${REPO_ROOT}' && bash './${SCRIPT_REL}' --inside-container"
}

host_publish() {
    cd "${REPO_ROOT}"
    export GIT_SSH_COMMAND='ssh -i /data/yanguahe/code/id_rsa.hyg -o IdentitiesOnly=yes'
    git add -f "${SCRIPT_REL}" "${ISA#./}"
    if [[ -d "${OUT_DIR}" ]]; then
        git add -f "${OUT_DIR}"
    else
        echo "WARNING: ${OUT_DIR} is missing; run the benchmark phase first." >&2
    fi
    git status --short
    if ! git diff --cached --quiet; then
        git -c user.name=yanguahe -c user.email=yanguahe@amd.com \
            commit --amend --author='yanguahe <yanguahe@amd.com>' -m Update
    fi
    git push -f origin hyg_dev
}

case "${1:-all}" in
    --inside-container)
        container_run
        ;;
    sync)
        host_sync
        ;;
    run)
        host_run
        ;;
    publish)
        host_publish
        ;;
    all)
        host_sync
        # Re-exec the freshly pulled copy so the reset above cannot run stale logic.
        exec bash "${REPO_ROOT}/${SCRIPT_REL}" run-and-publish
        ;;
    run-and-publish)
        host_run
        host_publish
        ;;
    *)
        echo "usage: bash ${SCRIPT_REL} [all|sync|run|publish|run-and-publish]" >&2
        exit 2
        ;;
esac
