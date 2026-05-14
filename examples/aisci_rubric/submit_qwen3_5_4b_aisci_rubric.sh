#!/bin/bash
#SBATCH --job-name=miles-qwen3.5-4b-aisci-rubric
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/infras/bash/logs/slurm-%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/infras/bash/logs/slurm-%x-%j.err
#
# 1-node x 8 H100 GRPO training of Qwen3.5-4B on the aisci_rubric task.
#
# Backend selection: set AISCI_JUDGE_CONFIG before sbatch'ing.
#   sbatch --export=ALL,AISCI_JUDGE_CONFIG=$MILES_DIR/examples/aisci_rubric/configs/judge_hf.yaml \
#       submit_qwen3_5_4b_aisci_rubric.sh
#
# For backend=local: sbatch the judge first and pass JUDGE_READY_FILE so this
# job blocks until the server is healthy:
#   JUDGE_JOB=$(sbatch --parsable serve_judge_local.sh)
#   sbatch --export=ALL,AISCI_JUDGE_CONFIG=...,JUDGE_READY_FILE=.../judge_ready_${JUDGE_JOB}.txt,JUDGE_URL_FILE=.../judge_url_${JUDGE_JOB}.txt \
#       submit_qwen3_5_4b_aisci_rubric.sh

set -euo pipefail

WORKSPACE=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace
MILES_DIR=${WORKSPACE}/infras/repos/miles
LAUNCHER=${MILES_DIR}/examples/aisci_rubric/run_qwen3_5_4b_aisci_rubric.sh
LOG_DIR=${WORKSPACE}/infras/bash/logs
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/lustre/images/miles.sqsh}
CONTAINER_MOUNTS=/lustre:/lustre

mkdir -p "${LOG_DIR}"

: "${AISCI_JUDGE_CONFIG:?AISCI_JUDGE_CONFIG must be set (e.g. via sbatch --export=ALL,AISCI_JUDGE_CONFIG=...)}"
export AISCI_DATA_CONFIG=${AISCI_DATA_CONFIG:-${MILES_DIR}/examples/aisci_rubric/configs/data_acsci_v3.yaml}
export AISCI_JUDGE_CONFIG

# If we're using a local judge, wait for serve_judge_local.sh to mark itself
# ready, then pull the URL from the file it wrote.
if [[ -n "${JUDGE_READY_FILE:-}" ]]; then
    echo "[submit] waiting on local judge ready file: ${JUDGE_READY_FILE}"
    for i in $(seq 1 360); do
        if [[ -f "${JUDGE_READY_FILE}" ]]; then
            break
        fi
        sleep 10
    done
    if [[ ! -f "${JUDGE_READY_FILE}" ]]; then
        echo "ERROR: judge ready file never appeared after 1h" >&2
        exit 1
    fi
    if [[ -z "${JUDGE_URL_FILE:-}" || ! -f "${JUDGE_URL_FILE}" ]]; then
        echo "ERROR: JUDGE_URL_FILE missing or unreadable: ${JUDGE_URL_FILE:-<unset>}" >&2
        exit 1
    fi
    export JUDGE_BASE_URL=$(cat "${JUDGE_URL_FILE}")
    echo "[submit] local judge ready at ${JUDGE_BASE_URL}"
fi

if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
    echo "ERROR: container image not found at ${CONTAINER_IMAGE}" >&2
    exit 1
fi
if [[ ! -x "${LAUNCHER}" ]]; then
    echo "ERROR: launcher not executable: ${LAUNCHER}" >&2
    exit 1
fi

echo "===================================================================="
echo "Job:        ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Branch:     $(git -C ${MILES_DIR} branch --show-current)"
echo "Judge cfg:  ${AISCI_JUDGE_CONFIG}"
echo "Data cfg:   ${AISCI_DATA_CONFIG}"
echo "Judge URL:  ${JUDGE_BASE_URL:-<from-config-base_url>}"
echo "Launcher:   ${LAUNCHER}"
echo "===================================================================="

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --no-container-mount-home \
    --no-container-entrypoint \
    --export=ALL \
    --nodes=1 --ntasks=1 \
    bash "${LAUNCHER}"

echo "===================================================================="
echo "Slurm job ${SLURM_JOB_ID} finished."
echo "===================================================================="
