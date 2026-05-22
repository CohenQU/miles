#!/bin/bash
# Convert one miles torch_dist checkpoint to HF safetensors and push it to
# CMU-POPE/Qwen3.5-{4B,9B}-aisci-rubric-{HUB_VERSION_TAG} as branch `iter-N`.
#
# Usage:
#   ./push_ckpt_to_hub.sh <model> <iter> [--dry-run] [--public]
#     model:     4B | 9B
#     iter:      integer (e.g. 49 for iter_0000049)
#     --dry-run: convert only, skip the upload
#     --public:  create the hub repo as public if it doesn't exist (default: private)
#
# Env knobs:
#   CKPT_TAG          checkpoint-dir suffix under MODEL_ROOT. Default: aisci_rubric
#                     (v01.00 layout). v03 uses aisci_rubric_v03.{00..03}.
#   HUB_VERSION_TAG   version slug in the hub repo name. Default: v00.00
#                     (v01.00's hub repo). v03 uses v03.{00..03}.
#
# Idempotent: if the converted HF dir already exists it skips conversion;
# if the iter-N branch already has the same content the upload is a no-op.
#
# Required: HF_TOKEN in Research-skills/.env with WRITE scope on CMU-POPE/*.

set -euo pipefail

MODEL=${1:?usage: push_ckpt_to_hub.sh <4B|9B> <iter> [--dry-run] [--public]}
ITER=${2:?usage: push_ckpt_to_hub.sh <4B|9B> <iter> [--dry-run] [--public]}
shift 2 || true

DRY_RUN=0
PRIVATE_FLAG="--private"
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        --public)  PRIVATE_FLAG="" ;;
        *) echo "ERROR: unknown arg ${arg}" >&2; exit 1 ;;
    esac
done

if [[ "${MODEL}" != "4B" && "${MODEL}" != "9B" ]]; then
    echo "ERROR: model must be 4B or 9B (got: ${MODEL})" >&2
    exit 1
fi
if ! [[ "${ITER}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: iter must be a positive integer (got: ${ITER})" >&2
    exit 1
fi

WORKSPACE=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace
MILES_DIR=${WORKSPACE}/infras/repos/miles
MODEL_ROOT=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/models
WANDB_ENV=${WORKSPACE}/Research-skills/.env
CONTAINER_IMAGE=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/lustre/images/miles.sqsh
CONDA_BASE=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/envs/miniconda3
HF_ENV_NAME=hf
HF_ENV=${CONDA_BASE}/envs/${HF_ENV_NAME}

CKPT_TAG=${CKPT_TAG:-aisci_rubric}
HUB_VERSION_TAG=${HUB_VERSION_TAG:-v00.00}

ITER_PADDED=$(printf '%07d' "${ITER}")
CKPT_PARENT_DIR=${MODEL_ROOT}/Qwen3.5-${MODEL}_${CKPT_TAG}
CKPT_DIR=${CKPT_PARENT_DIR}/iter_${ITER_PADDED}
HF_OUT=${MODEL_ROOT}/Qwen3.5-${MODEL}_${CKPT_TAG}_hf/iter_${ITER_PADDED}
ORIGIN_HF_DIR=${MODEL_ROOT}/Qwen3.5-${MODEL}
HUB_REPO=CMU-POPE/Qwen3.5-${MODEL}-aisci-rubric-${HUB_VERSION_TAG}
REVISION=iter-${ITER}

if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "ERROR: checkpoint dir not found: ${CKPT_DIR}" >&2
    echo "Available iterations in ${CKPT_PARENT_DIR}:" >&2
    ls "${CKPT_PARENT_DIR}" 2>&1 | grep -E "^iter_" >&2 || true
    exit 1
fi
if [[ ! -d "${ORIGIN_HF_DIR}" ]]; then
    echo "ERROR: origin HF dir not found: ${ORIGIN_HF_DIR}" >&2
    exit 1
fi

echo "================================================================"
echo "ckpt:        ${CKPT_DIR}"
echo "hf out:      ${HF_OUT}"
echo "origin hf:   ${ORIGIN_HF_DIR}"
echo "hub repo:    ${HUB_REPO}"
echo "revision:    ${REVISION}"
echo "dry run:     ${DRY_RUN}"
echo "================================================================"

# ── Step 1: Convert torch_dist → HF safetensors (CPU, in miles container) ───
if [[ -f "${HF_OUT}/model.safetensors.index.json" ]]; then
    echo "[skip convert] ${HF_OUT}/model.safetensors.index.json already exists"
else
    mkdir -p "$(dirname "${HF_OUT}")"
    echo "[convert] running tools/convert_torch_dist_to_hf.py via srun -p ${SRUN_PARTITION:-interactive}"
    # interactive (priority 40) is the proven default — backfill (priority 10)
    # starves under cluster training load, batch_short fills up, cpu is slow.
    # Override with SRUN_PARTITION=<other> if interactive itself is unavailable.
    srun --account=nvr_lacr_llm --partition="${SRUN_PARTITION:-interactive}" --time=00:30:00 \
        --cpus-per-task=16 --mem=128G --gpus-per-node=1 \
        --container-image="${CONTAINER_IMAGE}" \
        --container-mounts=/lustre:/lustre \
        --no-container-mount-home --no-container-entrypoint \
        --export=ALL \
        bash -c "cd ${MILES_DIR} && \
            PYTHONPATH=${MILES_DIR}:/root/Megatron-LM \
            python3 tools/convert_torch_dist_to_hf.py \
                --input-dir ${CKPT_DIR} \
                --output-dir ${HF_OUT} \
                --origin-hf-dir ${ORIGIN_HF_DIR} \
                --force"
    echo "[convert] done; output:"
    ls -la "${HF_OUT}" | head -10
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] skipping upload — converted HF model is at ${HF_OUT}"
    exit 0
fi

# ── Step 2: Push to HuggingFace Hub (host-side hf env, no container) ────────
if [[ ! -x "${HF_ENV}/bin/hf" ]]; then
    echo "ERROR: hf CLI not found at ${HF_ENV}/bin/hf" >&2
    echo "  install with: ${HF_ENV}/bin/pip install 'huggingface_hub[cli]>=1.13' hf_transfer" >&2
    exit 1
fi

# Source .env for HF_TOKEN. Use the soft-source pattern so the host-only conda
# activate line in .env doesn't crash us under set -e.
set +e; source "${WANDB_ENV}" 2>/dev/null; set -e
: "${HF_TOKEN:?HF_TOKEN must be set in ${WANDB_ENV} (write scope on CMU-POPE/*)}"
export HF_TOKEN

# Conda envs don't have a per-env `bin/activate`; use the base conda's
# activate with the env name as arg (per Research-skills/global/preferences.md).
# shellcheck disable=SC1091
source "${CONDA_BASE}/bin/activate" "${HF_ENV_NAME}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Idempotent repo create. Returns 0 if it already exists.
hf repo create "${HUB_REPO}" --repo-type model ${PRIVATE_FLAG} --exist-ok || {
    echo "WARN: hf repo create returned non-zero (may need write perms on CMU-POPE/*)" >&2
}

echo "[upload] pushing ${HF_OUT} → ${HUB_REPO}@${REVISION}"
hf upload "${HUB_REPO}" "${HF_OUT}" . \
    --repo-type model \
    --revision "${REVISION}" \
    --commit-message "aisci_rubric RL Qwen3.5-${MODEL} (${HUB_VERSION_TAG}), iter ${ITER}"

echo "================================================================"
echo "Done. Browse: https://huggingface.co/${HUB_REPO}/tree/${REVISION}"
echo "================================================================"
