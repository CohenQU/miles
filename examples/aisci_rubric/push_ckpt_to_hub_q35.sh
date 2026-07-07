#!/bin/bash
# Convert ONE Qwen3.5-{4B,9B} miles torch_dist checkpoint to HF and push to
# ${HUB_REPO} as branch step-${STEP} (STEP=iter+1). CohenQu/step-N sibling of
# push_ckpt_to_hub.sh (which is hardcoded to CMU-POPE/iter-N).
#
# Usage:  ./push_ckpt_to_hub_q35.sh <4B|9B> <iter> [--dry-run] [--public]
# Env:    CKPT_TAG (e.g. aisci_rubric_v03.04), HUB_REPO (e.g. CohenQu/Qwen3.5-9B-aisci-rubric-v03.04)
# Idempotent: skips convert if HF dir exists; hf dedups upload. Token=CohenQu (write).

set -euo pipefail

MODEL=${1:?usage: push_ckpt_to_hub_q35.sh <4B|9B> <iter> [--dry-run] [--public]}
ITER=${2:?usage: push_ckpt_to_hub_q35.sh <4B|9B> <iter> [--dry-run] [--public]}
shift 2 || true
: "${CKPT_TAG:?CKPT_TAG must be set}"; : "${HUB_REPO:?HUB_REPO must be set}"
[[ "$MODEL" == "4B" || "$MODEL" == "9B" ]] || { echo "ERROR: model must be 4B|9B" >&2; exit 1; }
[[ "$ITER" =~ ^[0-9]+$ ]] || { echo "ERROR: iter must be integer" >&2; exit 1; }

DRY_RUN=0; PRIVATE_FLAG="--private"
for a in "$@"; do case "$a" in --dry-run) DRY_RUN=1;; --public) PRIVATE_FLAG="";; *) echo "ERROR: bad arg $a">&2; exit 1;; esac; done

WORKSPACE=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace
MILES_DIR=${WORKSPACE}/infras/repos/miles
MODEL_ROOT=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/models
WANDB_ENV=${WORKSPACE}/Research-skills/.env
CONTAINER_IMAGE=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/lustre/images/miles.sqsh
CONDA_BASE=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/envs/miniconda3
HF_ENV_NAME=hf

ITER_PADDED=$(printf '%07d' "$ITER"); STEP=$((ITER+1))
CKPT_DIR=${MODEL_ROOT}/Qwen3.5-${MODEL}_${CKPT_TAG}/iter_${ITER_PADDED}
HF_OUT=${MODEL_ROOT}/Qwen3.5-${MODEL}_${CKPT_TAG}_hf/iter_${ITER_PADDED}
ORIGIN_HF_DIR=${MODEL_ROOT}/Qwen3.5-${MODEL}
REVISION=step-${STEP}

[[ -d "$CKPT_DIR" ]] || { echo "ERROR: ckpt dir not found: $CKPT_DIR" >&2; exit 1; }
[[ -d "$ORIGIN_HF_DIR" ]] || { echo "ERROR: origin HF dir not found: $ORIGIN_HF_DIR" >&2; exit 1; }
echo "=== ${MODEL} ${CKPT_TAG} iter ${ITER} (step ${STEP}) → ${HUB_REPO}@${REVISION} (dry=${DRY_RUN}) ==="

if [[ -f "${HF_OUT}/model.safetensors.index.json" ]]; then
    echo "[skip convert] ${HF_OUT} already converted"
else
    mkdir -p "$(dirname "$HF_OUT")"
    echo "[convert] srun -p ${SRUN_PARTITION:-interactive}"
    srun --account=nvr_lacr_llm --partition="${SRUN_PARTITION:-interactive}" --time=00:30:00 \
        --cpus-per-task=16 --mem=128G --gpus-per-node=1 \
        --container-image="${CONTAINER_IMAGE}" --container-mounts=/lustre:/lustre \
        --no-container-mount-home --no-container-entrypoint --export=ALL \
        bash -c "cd ${MILES_DIR} && PYTHONPATH=${MILES_DIR}:/root/Megatron-LM \
            python3 tools/convert_torch_dist_to_hf.py --input-dir ${CKPT_DIR} \
                --output-dir ${HF_OUT} --origin-hf-dir ${ORIGIN_HF_DIR} --force"
fi
# merge base vision tower + MTP (VL completeness) — idempotent (ERR-026)
if [[ ! -f "${HF_OUT}/model-90000-base-nonlm.safetensors" ]]; then
    echo "[merge] adding base vision tower + mtp via srun -p cpu_short"
    srun --account=nvr_lacr_llm --partition=cpu_short --time=00:20:00 --cpus-per-task=16 --mem=128G \
        --container-image="${CONTAINER_IMAGE}" --container-mounts=/lustre:/lustre \
        --no-container-mount-home --no-container-entrypoint --export=ALL \
        bash -c "cd ${MILES_DIR} && PYTHONPATH=${MILES_DIR}:/root/Megatron-LM \
            python3 examples/aisci_rubric/add_base_nonlm_weights.py ${HF_OUT} ${ORIGIN_HF_DIR}"
fi
[[ "$DRY_RUN" == "1" ]] && { echo "[dry-run] converted+merged at ${HF_OUT}; no upload"; exit 0; }

[[ -x "${CONDA_BASE}/envs/${HF_ENV_NAME}/bin/hf" ]] || { echo "ERROR: hf CLI missing" >&2; exit 1; }
set +e; source "$WANDB_ENV" 2>/dev/null; set -e
: "${HF_TOKEN:?HF_TOKEN must be set}"; export HF_TOKEN
# shellcheck disable=SC1091
source "${CONDA_BASE}/bin/activate" "$HF_ENV_NAME"; export HF_HUB_ENABLE_HF_TRANSFER=1
hf repo create "$HUB_REPO" --repo-type model ${PRIVATE_FLAG} --exist-ok || echo "WARN: repo create rc!=0" >&2
hf repo branch create "$HUB_REPO" "$REVISION" --repo-type model 2>/dev/null || true
echo "[upload] ${HF_OUT} → ${HUB_REPO}@${REVISION}"
hf upload "$HUB_REPO" "$HF_OUT" . --repo-type model --revision "$REVISION" \
    --commit-message "Qwen3.5-${MODEL} ${CKPT_TAG} @ step ${STEP}"
echo "Done: https://huggingface.co/${HUB_REPO}/tree/${REVISION}"
