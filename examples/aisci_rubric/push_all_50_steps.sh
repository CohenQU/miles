#!/bin/bash
# Sweep through all available aisci_rubric checkpoints for one model and push
# every 50 steps to CMU-POPE/Qwen3.5-{model}-aisci-rubric-v00.00.
#
# miles saves at iter 9, 19, 29, ... (every save_interval=10 steps, off by 1
# because of 0-indexed step counting). The closest saves to multiples of 50
# are iter 49, 99, 149, ... — those are what this script picks.
#
# Usage:
#   ./push_all_50_steps.sh <model> [--dry-run] [--public]
#     model: 4B | 9B
#
# Idempotent: skips iters whose HF output already exists locally and whose
# upload would be a no-op (push_ckpt_to_hub.sh handles the latter via hf's
# built-in dedup).

set -euo pipefail

MODEL=${1:?usage: push_all_50_steps.sh <4B|9B> [--dry-run] [--public]}
shift || true
EXTRA_ARGS=("$@")

if [[ "${MODEL}" != "4B" && "${MODEL}" != "9B" ]]; then
    echo "ERROR: model must be 4B or 9B (got: ${MODEL})" >&2
    exit 1
fi

WORKSPACE=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace
PUSH_ONE=${WORKSPACE}/infras/repos/miles/examples/aisci_rubric/push_ckpt_to_hub.sh
CKPT_PARENT=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/models/Qwen3.5-${MODEL}_aisci_rubric

if [[ ! -d "${CKPT_PARENT}" ]]; then
    echo "ERROR: no checkpoints dir at ${CKPT_PARENT}" >&2
    exit 1
fi

# Collect iter numbers from iter_NNNNNNN/ subdirs, keep only those where
# (iter + 1) is a multiple of 50 — i.e. the saves closest to step boundaries
# 50, 100, 150, ...
iters_to_push=()
for d in $(ls "${CKPT_PARENT}" | grep -E "^iter_[0-9]+$" | sort); do
    iter=$(echo "${d}" | sed -E 's/^iter_0*([0-9]+)$/\1/')
    [[ -z "${iter}" ]] && continue
    if (( (iter + 1) % 50 == 0 )); then
        iters_to_push+=("${iter}")
    fi
done

if [[ ${#iters_to_push[@]} -eq 0 ]]; then
    echo "No iter_NNNNNNN dirs in ${CKPT_PARENT} match the every-50-steps pattern (expected iter 49, 99, 149, ...)."
    echo "Available:" && ls "${CKPT_PARENT}" | grep -E "^iter_" | head -20
    exit 0
fi

echo "================================================================"
echo "Will push ${#iters_to_push[@]} checkpoints to CMU-POPE/Qwen3.5-${MODEL}-aisci-rubric-v00.00"
echo "  iters: ${iters_to_push[*]}"
echo "  extra args forwarded to push_ckpt_to_hub.sh: ${EXTRA_ARGS[*]:-(none)}"
echo "================================================================"

for iter in "${iters_to_push[@]}"; do
    echo ""
    echo "================ iter ${iter} ================"
    if ! "${PUSH_ONE}" "${MODEL}" "${iter}" "${EXTRA_ARGS[@]}"; then
        echo "ERROR: push_ckpt_to_hub.sh failed for iter ${iter}; continuing with the next one" >&2
    fi
done

echo ""
echo "================================================================"
echo "Sweep complete."
echo "================================================================"
