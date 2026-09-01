#!/usr/bin/env bash
# Copy selected files to FlyDSL/temp_file, checkout opus_align, then restore them.
#
# Special case:
#   tests/kernels/test_flash_attn_fwd.py -> tests/kernels/test_flash_attn_fwd_ori.py
#
# Usage (from anywhere):
#   bash scripts/sync_temp_files_to_opus_align.sh
#
# Or from FlyDSL repo root:
#   ./scripts/sync_temp_files_to_opus_align.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
TEMP_DIR="${REPO_ROOT}/temp_file"
# TARGET_BRANCH="opus_align"
TARGET_BRANCH="refine_fmha"

# Files to sync (paths relative to FlyDSL repo root).
FILES=(
    # "CLAUDE.md"
    # "README.md"
    # "docs/architecture_guide.md"
    # "docs/prebuilt_kernels_guide.md"
    # "scripts/run_benchmark.sh"
    "kernels/flash_attn_generic.py"
    "kernels/flash_attn_gfx950.py"
    "kernels/flash_attn_interface.py"
    "kernels/rmsnorm_kernel.py"
    "tests/kernels/test_flash_attn_fwd.py"
    # "tests/kernels/test_flash_attn_fwd_ori.py"
)

SPECIAL_DEST=(
    # "tests/kernels/test_flash_attn_fwd.py:tests/kernels/test_flash_attn_fwd_ori.py"
    # "tests/kernels/test_flash_attn_fwd_ori.py:tests/kernels/test_flash_attn_fwd.py"
)

resolve_dest() {
    local src="$1"
    local mapping
    for mapping in "${SPECIAL_DEST[@]}"; do
        if [[ "${mapping%%:*}" == "${src}" ]]; then
            echo "${mapping#*:}"
            return 0
        fi
    done
    echo "${src}"
}

cd "${REPO_ROOT}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "error: ${REPO_ROOT} is not a git repository" >&2
    exit 1
fi

SOURCE_BRANCH="$(git branch --show-current)"
echo "[1/3] Staging ${#FILES[@]} file(s) into ${TEMP_DIR}"

rm -rf "${TEMP_DIR}"
mkdir -p "${TEMP_DIR}"

for rel in "${FILES[@]}"; do
    src="${REPO_ROOT}/${rel}"
    if [[ ! -f "${src}" ]]; then
        echo "error: missing source file: ${rel}" >&2
        exit 1
    fi
    dest="${TEMP_DIR}/${rel}"
    mkdir -p "$(dirname "${dest}")"
    cp -f "${src}" "${dest}"
    echo "  staged: ${rel}"
done

echo "[2/3] Switching branch: ${SOURCE_BRANCH} -> ${TARGET_BRANCH}"
git checkout "${TARGET_BRANCH}"

echo "[3/3] Restoring files on ${TARGET_BRANCH}"
for rel in "${FILES[@]}"; do
    staged="${TEMP_DIR}/${rel}"
    dest_rel="$(resolve_dest "${rel}")"
    dest="${REPO_ROOT}/${dest_rel}"
    mkdir -p "$(dirname "${dest}")"
    cp -f "${staged}" "${dest}"
    if [[ "${rel}" == "${dest_rel}" ]]; then
        echo "  restored: ${dest_rel}"
    else
        echo "  restored: ${rel} -> ${dest_rel}"
    fi
done

echo "Done. Files kept in ${TEMP_DIR} for reference."
