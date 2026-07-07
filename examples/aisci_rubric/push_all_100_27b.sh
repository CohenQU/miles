#!/bin/bash
# Push every STRIDE (default 100) steps of ALL FOUR Qwen3.6-27B aisci_rubric runs
# to CohenQu/Qwen3.6-27B-aisci-rubric-v03.{08,09,10,11}, each checkpoint a branch
# step-N. Idempotent (skips already-converted/uploaded) → safe to re-run as runs
# progress to pick up new 100-step milestones.
#
# Usage:  ./push_all_100_27b.sh [--dry-run] [--public] [TAG ...]
#   no TAG  → all four runs;  TAG e.g. aisci_rubric_v03.11 → just that run.
# Env: STRIDE (default 100).
set -uo pipefail

STRIDE=${STRIDE:-100}
WORKSPACE=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace
PUSH_ONE=${WORKSPACE}/infras/repos/miles/examples/aisci_rubric/push_ckpt_to_hub_27b.sh
MODEL_ROOT=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/models

# run-tag -> CohenQu repo name
declare -A REPO=(
  [aisci_rubric_v03.08]=CohenQu/Qwen3.6-27B-aisci-rubric-v03.08
  [aisci_rubric_v03.09]=CohenQu/Qwen3.6-27B-aisci-rubric-v03.09
  [aisci_rubric_v03.10]=CohenQu/Qwen3.6-27B-aisci-rubric-v03.10
  [aisci_rubric_v03.11]=CohenQu/Qwen3.6-27B-aisci-rubric-v03.11
  [aisci_rubric_v03.12]=CohenQu/Qwen3.6-27B-aisci-rubric-v03.12
  [aisci_rubric_v03.13]=CohenQu/Qwen3.6-27B-aisci-rubric-v03.13
  [aisci_rubric_v03.14]=CohenQu/Qwen3.6-27B-aisci-rubric-v03.14
)

EXTRA=(); TAGS=()
for a in "$@"; do case "$a" in --*) EXTRA+=("$a");; *) TAGS+=("$a");; esac; done
[[ ${#TAGS[@]} -eq 0 ]] && TAGS=(aisci_rubric_v03.08 aisci_rubric_v03.09 aisci_rubric_v03.10 aisci_rubric_v03.11 aisci_rubric_v03.12 aisci_rubric_v03.13 aisci_rubric_v03.14)

for tag in "${TAGS[@]}"; do
  repo=${REPO[$tag]:-}
  [[ -z "$repo" ]] && { echo "skip $tag (no repo mapping)"; continue; }
  parent=${MODEL_ROOT}/Qwen3.6-27B_${tag}
  [[ -d "$parent" ]] || { echo "skip $tag (no ckpt dir)"; continue; }
  iters=()
  for d in $(ls "$parent" 2>/dev/null | grep -E '^iter_[0-9]+$' | sort -V); do
    it=$(echo "$d" | sed -E 's/^iter_0*([0-9]+)$/\1/'); [[ -z "$it" ]] && continue
    (( (it + 1) % STRIDE == 0 )) && iters+=("$it")
  done
  echo "================ $tag -> $repo : step-$STRIDE iters = ${iters[*]:-none} ================"
  for it in "${iters[@]}"; do
    CKPT_TAG="$tag" HUB_REPO="$repo" "$PUSH_ONE" "$it" "${EXTRA[@]}" \
      || echo "WARN: push failed for $tag iter $it; continuing" >&2
  done
done
echo "=== push_all_100_27b sweep complete ==="
