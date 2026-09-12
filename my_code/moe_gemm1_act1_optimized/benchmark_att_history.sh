#!/usr/bin/env bash
set -Eeuo pipefail

# Compatibility entry point. ATT case metadata and execution now live in the
# unified history benchmark so e2e, standalone, and ATT cannot drift apart.

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mode=att
if [[ "${AITER_ATT_VALIDATE_ONLY:-0}" == 1 ]]; then
  mode=att-validate
fi
exec bash "$HERE/benchmark_history.sh" "$mode"
