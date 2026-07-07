#!/bin/bash
# Push every STRIDE(=100) steps of the Qwen3.5 rubric runs (4B v03.00/02, 9B
# v03.04/05/06) to CohenQu, each ckpt a branch step-N. Idempotent; re-runnable.
# Usage: ./push_all_100_q35.sh [--dry-run] [--public]
set -uo pipefail
STRIDE=${STRIDE:-100}
WORKSPACE=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace
PUSH_ONE=${WORKSPACE}/infras/repos/miles/examples/aisci_rubric/push_ckpt_to_hub_q35.sh
MODEL_ROOT=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/models
EXTRA=("$@")

# "MODEL|CKPT_TAG|HUB_REPO"
RUNS=(
  "4B|aisci_rubric_v03.00|CohenQu/Qwen3.5-4B-aisci-rubric-v03.00"
  "4B|aisci_rubric_v03.02|CohenQu/Qwen3.5-4B-aisci-rubric-v03.02"
  "9B|aisci_rubric_v03.04|CohenQu/Qwen3.5-9B-aisci-rubric-v03.04"
  "9B|aisci_rubric_v03.05|CohenQu/Qwen3.5-9B-aisci-rubric-v03.05"
  "9B|aisci_rubric_v03.06|CohenQu/Qwen3.5-9B-aisci-rubric-v03.06"
)
for entry in "${RUNS[@]}"; do
  IFS='|' read -r MODEL TAG REPO <<< "$entry"
  parent=${MODEL_ROOT}/Qwen3.5-${MODEL}_${TAG}
  [[ -d "$parent" ]] || { echo "skip $TAG (no dir)"; continue; }
  iters=()
  for d in $(ls "$parent" 2>/dev/null | grep -E '^iter_[0-9]+$' | sort -V); do
    it=$(echo "$d" | sed -E 's/^iter_0*([0-9]+)$/\1/'); [[ -z "$it" ]] && continue
    (( (it+1) % STRIDE == 0 )) && iters+=("$it")
  done
  echo "================ ${MODEL} ${TAG} → ${REPO} : step-${STRIDE} iters = ${iters[*]:-none} ================"
  for it in "${iters[@]}"; do
    CKPT_TAG="$TAG" HUB_REPO="$REPO" "$PUSH_ONE" "$MODEL" "$it" "${EXTRA[@]}" \
      || echo "WARN: push failed ${TAG} iter $it; continuing" >&2
  done
done
echo "=== push_all_100_q35 sweep complete ==="
