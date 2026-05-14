#!/bin/bash
# Trainer launcher: Qwen3.5-9B GRPO on the aisci_rubric task.
# 1 node x 8 H100, miles colocated (Megatron actor + SGLang rollout).
# Same wiring as the 4B launcher, with --tensor-model-parallel-size 4 and a
# halved --max-tokens-per-gpu to fit the larger model.

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

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
    src_rc=$?
    set -e
    if [[ "${src_rc}" -ne 0 ]]; then
        echo "INFO: ${WANDB_ENV} sourced with rc=${src_rc} (likely a path-on-host line that does not apply inside the container) — continuing." >&2
    fi
else
    echo "WARN: ${WANDB_ENV} not found — wandb logging will fail unless WANDB_API_KEY is already set." >&2
fi
: "${WANDB_API_KEY:?WANDB_API_KEY is unset}"
: "${AISCI_JUDGE_CONFIG:?AISCI_JUDGE_CONFIG must point to a judge_*.yaml}"
: "${AISCI_DATA_CONFIG:?AISCI_DATA_CONFIG must point to a data_*.yaml}"

WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-miles}
RUN_TAG=${RUN_TAG:-qwen3_5_9b_aisci_rubric}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME=${EXP_NAME:-${RUN_TAG}_${TIMESTAMP}}

source "${MILES_DIR}/scripts/models/qwen3.5-9B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3.5-9B
   --ref-load ${MODEL_ROOT}/Qwen3.5-9B_torch_dist
   --load ${MODEL_ROOT}/Qwen3.5-9B_aisci_rubric/
   --save ${MODEL_ROOT}/Qwen3.5-9B_aisci_rubric/
   --save-interval ${SAVE_INTERVAL:-20}
)

ROLLOUT_ARGS=(
   --data-source-path examples.aisci_rubric.hf_data_source.HFRolloutDataSource
   --custom-rm-path examples.aisci_rubric.reward_fn.reward_fn
   --reward-key reward_value
   --rollout-shuffle
   --num-rollout ${NUM_ROLLOUT:-3000}
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=()
if [[ -n "${EVAL_INTERVAL:-}" && -n "${EVAL_PROMPT_DATA:-}" ]]; then
    # shellcheck disable=SC2206
    EVAL_ARGS+=(--eval-interval "${EVAL_INTERVAL}" --eval-prompt-data ${EVAL_PROMPT_DATA})
fi

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
   --max-tokens-per-gpu 4608
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.0001
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project ${WANDB_PROJECT_NAME}
   --wandb-group ${EXP_NAME}
   --wandb-team ${WANDB_ENTITY:-yuxiao98}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.55
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

cd "${MILES_DIR}"

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MILES_DIR}:${MEGATRON_LM_DIR}/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
    \"WANDB_ENTITY\": \"${WANDB_ENTITY:-}\",
    \"HF_TOKEN\": \"${HF_TOKEN:-}\",
    \"HF_HOME\": \"${HF_HOME:-}\",
    \"AISCI_JUDGE_CONFIG\": \"${AISCI_JUDGE_CONFIG}\",
    \"AISCI_DATA_CONFIG\": \"${AISCI_DATA_CONFIG}\",
    \"OPENAI_API_KEY\": \"${OPENAI_API_KEY:-}\",
    \"NVIDIA_API_KEY\": \"${NVIDIA_API_KEY:-}\",
    \"JUDGE_BASE_URL\": \"${JUDGE_BASE_URL:-}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${MILES_DIR}/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
