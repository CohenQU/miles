#!/bin/bash
# Convert ONE Qwen3.6-27B miles torch_dist checkpoint to HF safetensors and push
# it to ${HUB_REPO} as branch `step-${STEP}` (STEP = iter+1; miles is 0-indexed).
# 27B/CohenQu sibling of push_ckpt_to_hub.sh (which is hardcoded to Qwen3.5/CMU-POPE).
#
# Usage:  ./push_ckpt_to_hub_27b.sh <iter> [--dry-run] [--public]
#   iter:      integer (e.g. 99 for iter_0000099 == step 100)
#   --dry-run: convert only, skip the upload
#   --public:  create the hub repo public if new (default: private)
#
# Required env:
#   CKPT_TAG   ckpt-dir suffix under MODEL_ROOT (e.g. aisci_rubric_v03.08)
#   HUB_REPO   target repo, e.g. CohenQu/Qwen3.6-27B-aisci-rubric-v03.08
#
# Idempotent: skips convert if the HF dir exists; hf dedups the upload.
# Required: HF_TOKEN (write scope) in Research-skills/.env — token is CohenQu's.

set -euo pipefail

ITER=${1:?usage: push_ckpt_to_hub_27b.sh <iter> [--dry-run] [--public]}
shift || true
: "${CKPT_TAG:?CKPT_TAG must be set (e.g. aisci_rubric_v03.08)}"
: "${HUB_REPO:?HUB_REPO must be set (e.g. CohenQu/Qwen3.6-27B-aisci-rubric-v03.08)}"

DRY_RUN=0
PRIVATE_FLAG="--private"
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        --public)  PRIVATE_FLAG="" ;;
        *) echo "ERROR: unknown arg ${arg}" >&2; exit 1 ;;
    esac
done
[[ "${ITER}" =~ ^[0-9]+$ ]] || { echo "ERROR: iter must be an integer (got ${ITER})" >&2; exit 1; }

WORKSPACE=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace
MILES_DIR=${WORKSPACE}/infras/repos/miles
MODEL_ROOT=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/models
WANDB_ENV=${WORKSPACE}/Research-skills/.env
CONTAINER_IMAGE=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/lustre/images/miles.sqsh
CONDA_BASE=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/envs/miniconda3
HF_ENV_NAME=hf

ITER_PADDED=$(printf '%07d' "${ITER}")
STEP=$((ITER + 1))
CKPT_DIR=${MODEL_ROOT}/Qwen3.6-27B_${CKPT_TAG}/iter_${ITER_PADDED}
HF_OUT=${MODEL_ROOT}/Qwen3.6-27B_${CKPT_TAG}_hf/iter_${ITER_PADDED}
ORIGIN_HF_DIR=${MODEL_ROOT}/Qwen3.6-27B
REVISION=step-${STEP}

[[ -d "${CKPT_DIR}" ]] || { echo "ERROR: ckpt dir not found: ${CKPT_DIR}" >&2; ls "${MODEL_ROOT}/Qwen3.6-27B_${CKPT_TAG}" 2>/dev/null | grep -E '^iter_' >&2 || true; exit 1; }
[[ -d "${ORIGIN_HF_DIR}" ]] || { echo "ERROR: origin HF dir not found: ${ORIGIN_HF_DIR}" >&2; exit 1; }

echo "================================================================"
echo "ckpt:      ${CKPT_DIR}"
echo "hf out:    ${HF_OUT}"
echo "hub repo:  ${HUB_REPO}"
echo "revision:  ${REVISION}  (iter ${ITER} = step ${STEP})"
echo "dry run:   ${DRY_RUN}"
echo "================================================================"

# ── Step 1: convert torch_dist → HF safetensors (GPU srun, miles container) ──
if [[ -f "${HF_OUT}/model.safetensors.index.json" ]]; then
    echo "[skip convert] ${HF_OUT} already converted"
else
    mkdir -p "$(dirname "${HF_OUT}")"
    echo "[convert] convert_torch_dist_to_hf.py via srun -p ${SRUN_PARTITION:-interactive} (27B: --mem 256G)"
    srun --account=nvr_lacr_llm --partition="${SRUN_PARTITION:-interactive}" --time=00:40:00 \
        --cpus-per-task=16 --mem=256G --gpus-per-node=1 \
        --container-image="${CONTAINER_IMAGE}" --container-mounts=/lustre:/lustre \
        --no-container-mount-home --no-container-entrypoint --export=ALL \
        bash -c "cd ${MILES_DIR} && PYTHONPATH=${MILES_DIR}:/root/Megatron-LM \
            python3 tools/convert_torch_dist_to_hf.py \
                --input-dir ${CKPT_DIR} --output-dir ${HF_OUT} \
                --origin-hf-dir ${ORIGIN_HF_DIR} --force"
    echo "[convert] done:"; ls -la "${HF_OUT}" | head -8
fi

# ── Step 1b: merge base vision tower + MTP (VL completeness) — idempotent ────
# The convert exports LM-only; Qwen3_5 is a VL arch, so add the frozen
# model.visual.* + mtp.* from the base (see add_base_nonlm_weights.py / ERR-026).
if [[ ! -f "${HF_OUT}/model-90000-base-nonlm.safetensors" ]]; then
    echo "[merge] adding base vision tower + mtp via srun -p cpu_short"
    srun --account=nvr_lacr_llm --partition=cpu_short --time=00:20:00 \
        --cpus-per-task=16 --mem=128G \
        --container-image="${CONTAINER_IMAGE}" --container-mounts=/lustre:/lustre \
        --no-container-mount-home --no-container-entrypoint --export=ALL \
        bash -c "cd ${MILES_DIR} && PYTHONPATH=${MILES_DIR}:/root/Megatron-LM \
            python3 examples/aisci_rubric/add_base_nonlm_weights.py ${HF_OUT} ${ORIGIN_HF_DIR}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] converted+merged HF model at ${HF_OUT}; skipping upload"
    exit 0
fi

# ── Step 2: push to HF Hub (host hf conda env) ──────────────────────────────
[[ -x "${CONDA_BASE}/envs/${HF_ENV_NAME}/bin/hf" ]] || { echo "ERROR: hf CLI not found in ${HF_ENV_NAME} env" >&2; exit 1; }
set +e; source "${WANDB_ENV}" 2>/dev/null; set -e
: "${HF_TOKEN:?HF_TOKEN must be set in ${WANDB_ENV}}"
export HF_TOKEN
# shellcheck disable=SC1091
source "${CONDA_BASE}/bin/activate" "${HF_ENV_NAME}"
export HF_HUB_ENABLE_HF_TRANSFER=1

hf repo create "${HUB_REPO}" --repo-type model ${PRIVATE_FLAG} --exist-ok || \
    echo "WARN: repo create returned non-zero (exists or perms)" >&2
# Ensure the branch exists (upload --revision needs it on some hf versions).
hf repo branch create "${HUB_REPO}" "${REVISION}" --repo-type model 2>/dev/null || true

echo "[upload] ${HF_OUT} → ${HUB_REPO}@${REVISION}"
hf upload "${HUB_REPO}" "${HF_OUT}" . --repo-type model --revision "${REVISION}" \
    --commit-message "Qwen3.6-27B ${CKPT_TAG} @ step ${STEP}"

echo "Done: https://huggingface.co/${HUB_REPO}/tree/${REVISION}"
