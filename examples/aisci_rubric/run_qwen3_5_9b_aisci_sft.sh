#!/bin/bash
# Trainer launcher: Qwen3.5-9B SUPERVISED FINE-TUNING (no-thinking) baseline for the
# aisci_rubric task. 1 node x 8 H100, miles SFT (Megatron actor only, NO SGLang/judge).
#
# Supervised baseline for the rubric-RL runs (v03.04/05/06): instead of GRPO on the
# rubric reward, imitate the dataset's `reference_answer`. Reads a `messages` parquet
# built by eval/scripts_v3/p9y_build_sft_dataset.py via miles.rollout.sft_rollout.
#
# Mirrors run_qwen3_5_9b_aisci_rubric.sh (same model/paths/parallelism) but swaps the
# RL wiring (rollout data source + reward fn + GRPO + SGLang) for SFT_ARGS
# (--loss-type sft_loss, sft_rollout, --debug-train-only). NOTE: --loss-mask-type qwen3
# is REQUIRED for Qwen3.5 (the arg default is "qwen", which masks the wrong tokens).
#
# Required env:
#   SFT_DATA       path to the messages parquet (p9y output)
#   CKPT_TAG       e.g. aisci_sft_v03.07 (model dir suffix; do NOT clobber RL ckpts)
# Optional env:
#   NUM_EPOCH (3), SAVE_INTERVAL (340 ~= half-epoch), MAX_TOKENS_PER_GPU (8192),
#   EXP_NAME, WANDB_PROJECT_NAME (miles), WANDB_ENTITY (yuxiao98)

if [[ "${RAY_HEAD_ALREADY_STARTED:-0}" != "1" ]]; then
    pkill -9 sglang
    sleep 3
    ray stop --force
    pkill -9 ray
    pkill -9 python
    sleep 3
    pkill -9 ray
    pkill -9 python
else
    echo "[run] RAY_HEAD_ALREADY_STARTED=1 — skipping pkill/ray-stop preamble"
fi

set -ex
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

MILES_DIR=${MILES_DIR:-/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/infras/repos/miles}
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-/root/Megatron-LM}
MODEL_ROOT=${MODEL_ROOT:-/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/tmp/models}
WANDB_ENV=${WANDB_ENV:-/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/yuxiaoq/workspace/Research-skills/.env}

if [[ -f "${WANDB_ENV}" ]]; then
    set +e
    # shellcheck disable=SC1090
    source "${WANDB_ENV}" 2>/dev/null
    set -e
fi

: "${SFT_DATA:?SFT_DATA must point to the messages parquet (p9y output)}"
if [[ ! -f "${SFT_DATA}" ]]; then
    echo "ERROR: SFT_DATA parquet not found: ${SFT_DATA}" >&2 ; exit 1
fi

WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-miles}
RUN_TAG=${RUN_TAG:-qwen3_5_9b_aisci_sft}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME=${EXP_NAME:-${RUN_TAG}_${TIMESTAMP}}

source "${MILES_DIR}/scripts/models/qwen3.5-9B.sh"

CKPT_TAG=${CKPT_TAG:-aisci_sft}

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3.5-9B
   --ref-load ${MODEL_ROOT}/Qwen3.5-9B_torch_dist
   --load ${MODEL_ROOT}/Qwen3.5-9B_${CKPT_TAG}/
   --save ${MODEL_ROOT}/Qwen3.5-9B_${CKPT_TAG}/
   --save-interval ${SAVE_INTERVAL:-340}
)

# SFT data + loss wiring. --loss-mask-type qwen3 is REQUIRED (default "qwen" is wrong
# for Qwen3.5). sft_rollout reads sample.prompt = the parquet `messages` list and masks
# loss to the assistant tokens via MultiTurnLossMaskGenerator.
SFT_ARGS=(
   --rollout-function-path miles.rollout.sft_rollout.generate_rollout
   --prompt-data ${SFT_DATA}
   --input-key messages
   --loss-mask-type qwen3
   --rollout-shuffle
   --num-epoch ${NUM_EPOCH:-3}
   --rollout-batch-size 128
   --global-batch-size 128

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-8192}
)

# Standard SFT optimizer (from scripts/run-qwen3-4B-base-sft.sh): higher LR + cosine
# decay, unlike the RL launcher's 1e-6 constant.
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_ARGS=(
       --use-wandb
       --wandb-project ${WANDB_PROJECT_NAME}
       --wandb-group ${EXP_NAME}
       --wandb-team ${WANDB_ENTITY:-yuxiao98}
    )
else
    echo "WARN: WANDB_API_KEY unset — running without wandb logging." >&2
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

cd "${MILES_DIR}"

export RAY_HEAD_IP=${RAY_HEAD_IP:-"127.0.0.1"}
export MASTER_ADDR=${MASTER_ADDR:-${RAY_HEAD_IP}}
export no_proxy="127.0.0.1,${MASTER_ADDR}"

if [[ "${RAY_HEAD_ALREADY_STARTED:-0}" != "1" ]]; then
    ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
else
    echo "[run] RAY_HEAD_ALREADY_STARTED=1 — using existing ray cluster at ${RAY_HEAD_IP}:6379"
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MILES_DIR}:${MEGATRON_LM_DIR}/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"WANDB_ENTITY\": \"${WANDB_ENTITY:-}\",
    \"HF_TOKEN\": \"${HF_TOKEN:-}\",
    \"HF_HOME\": \"${HF_HOME:-}\"
  }
}"

ray job submit --address="http://${RAY_HEAD_IP}:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${MILES_DIR}/train_async.py \
   --actor-num-nodes ${ACTOR_NUM_NODES:-1} \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}
