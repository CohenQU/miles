#!/bin/bash
#SBATCH --job-name=aisci-judge-server
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/infras/bash/logs/slurm-%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/infras/bash/logs/slurm-%x-%j.err
#
# Self-hosted SGLang judge server for the aisci_rubric task.
#
# Reads `serve.*` from examples/aisci_rubric/configs/judge_local.yaml:
#   - num_nodes, gpus_per_node, tp, dp, max_model_len, mem_fraction_static, port
# 1-node smoke is the default. To scale up:
#   sbatch --nodes=N serve_judge_local.sh
#   (also bump serve.num_nodes in judge_local.yaml so SGLang is launched with
#    --nnodes N --node-rank R --dist-init-addr <head_ip>:<port>.)
#
# Writes the judge URL (http://<HEAD_IP>:<port>/v1) to the file set by
# JUDGE_URL_FILE (default: ${LOG_DIR}/judge_url_${SLURM_JOB_ID}.txt) so the
# trainer's submit_*_aisci_rubric.sh can read and export JUDGE_BASE_URL.

set -euo pipefail

WORKSPACE=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace
MILES_DIR=${WORKSPACE}/infras/repos/miles
INFRAS_DIR=${WORKSPACE}/infras
LOG_DIR=${INFRAS_DIR}/bash/logs
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/lustre/images/miles.sqsh}
CONTAINER_MOUNTS=/lustre:/lustre

JUDGE_CFG=${JUDGE_CFG:-${MILES_DIR}/examples/aisci_rubric/configs/judge_local.yaml}
JUDGE_URL_FILE=${JUDGE_URL_FILE:-${LOG_DIR}/judge_url_${SLURM_JOB_ID}.txt}
JUDGE_READY_FILE=${JUDGE_READY_FILE:-${LOG_DIR}/judge_ready_${SLURM_JOB_ID}.txt}

mkdir -p "${LOG_DIR}"

# Read serve.* via python+yaml (avoids brittle awk parsing).
read_yaml() {
    python3 - "$1" "$2" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
keys = sys.argv[2].split(".")
v = cfg
for k in keys:
    v = v[k]
print(v)
PY
}

NUM_NODES=$(read_yaml "${JUDGE_CFG}" serve.num_nodes)
GPUS_PER_NODE=$(read_yaml "${JUDGE_CFG}" serve.gpus_per_node)
TP=$(read_yaml "${JUDGE_CFG}" serve.tp)
DP=$(read_yaml "${JUDGE_CFG}" serve.dp)
MAX_MODEL_LEN=$(read_yaml "${JUDGE_CFG}" serve.max_model_len)
MEM_FRAC=$(read_yaml "${JUDGE_CFG}" serve.mem_fraction_static)
PORT=$(read_yaml "${JUDGE_CFG}" serve.port)
MODEL=$(read_yaml "${JUDGE_CFG}" model)

HEAD_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname -I | awk '{print $1}')
JUDGE_URL="http://${HEAD_IP}:${PORT}/v1"

echo "===================================================================="
echo "Judge:      SGLang serving ${MODEL}"
echo "Allocation: ${SLURM_JOB_NUM_NODES} nodes x ${GPUS_PER_NODE} GPUs"
echo "Topology:   tp=${TP} dp=${DP} max_model_len=${MAX_MODEL_LEN}"
echo "Endpoint:   ${JUDGE_URL}"
echo "URL file:   ${JUDGE_URL_FILE}"
echo "===================================================================="

if [[ "${SLURM_JOB_NUM_NODES}" -gt 1 ]]; then
    # Multi-node: launch one SGLang per node with --nnodes / --node-rank wired
    # via SLURM_NODEID. SGLang elects rank 0 as the data-parallel coordinator.
    SGLANG_CMD="python3 -m sglang.launch_server \
        --model-path ${MODEL} \
        --host 0.0.0.0 --port ${PORT} \
        --tp-size ${TP} --dp-size ${DP} \
        --nnodes ${SLURM_JOB_NUM_NODES} \
        --node-rank \${SLURM_NODEID} \
        --dist-init-addr ${HEAD_IP}:29500 \
        --max-total-tokens ${MAX_MODEL_LEN} \
        --mem-fraction-static ${MEM_FRAC}"
else
    SGLANG_CMD="python3 -m sglang.launch_server \
        --model-path ${MODEL} \
        --host 0.0.0.0 --port ${PORT} \
        --tp-size ${TP} --dp-size ${DP} \
        --max-total-tokens ${MAX_MODEL_LEN} \
        --mem-fraction-static ${MEM_FRAC}"
fi

# Background the server, then poll /health from the head node before declaring
# the URL file ready. Trainer launcher reads JUDGE_URL_FILE and waits on
# JUDGE_READY_FILE so it never races a cold judge.
srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --no-container-mount-home \
    --no-container-entrypoint \
    --export=ALL \
    bash -c "${SGLANG_CMD}" &
SRUN_PID=$!

# Probe /health up to 20 minutes (gpt-oss-20b loads in ~3-5min on 8xH100).
echo "${JUDGE_URL}" > "${JUDGE_URL_FILE}"
for i in $(seq 1 240); do
    if curl -sf "http://${HEAD_IP}:${PORT}/health" >/dev/null 2>&1; then
        echo "Judge healthy after $((i*5))s"
        echo "ready" > "${JUDGE_READY_FILE}"
        break
    fi
    sleep 5
done

if [[ ! -f "${JUDGE_READY_FILE}" ]]; then
    echo "ERROR: judge did not become healthy within 20min" >&2
    kill ${SRUN_PID} 2>/dev/null || true
    exit 1
fi

# Stay alive — trainer needs the server up for the duration of training.
wait ${SRUN_PID}
