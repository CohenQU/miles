#!/bin/bash
# Trainer launcher: Qwen3.5-4B GRPO on the aisci_rubric task.
# 1 node x 8 H100, miles colocated (Megatron actor + SGLang rollout).
#
# Mirrors infras/bash/run_qwen3_5_4b_dapo_math_grpo.sh structure, but:
#   - data source streams from HF Hub at runtime (--data-source-path)
#   - reward function calls a pluggable LLM judge (--custom-rm-path)
#   - no JSONL --prompt-data, no --rm-type
#
# Required env vars (typically set by the sbatch wrapper):
#   AISCI_JUDGE_CONFIG  — path to one of examples/aisci_rubric/configs/judge_*.yaml
#   AISCI_DATA_CONFIG   — path to examples/aisci_rubric/configs/data_*.yaml
#   OPENAI_API_KEY  / NVIDIA_API_KEY  / JUDGE_BASE_URL   (depending on backend)
#
# Usage (inside the miles container):
#   bash examples/aisci_rubric/run_qwen3_5_4b_aisci_rubric.sh

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
    # Source the shared .env but don't let aspirational lines (e.g. a host-only
    # conda activation) fail the whole job under `set -e`. Vars exported before
    # any failing line are still active afterwards.
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
: "${WANDB_API_KEY:?WANDB_API_KEY is unset; cannot push logs to wandb}"
: "${AISCI_JUDGE_CONFIG:?AISCI_JUDGE_CONFIG must point to a judge_*.yaml}"
: "${AISCI_DATA_CONFIG:?AISCI_DATA_CONFIG must point to a data_*.yaml}"

WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-miles}
RUN_TAG=${RUN_TAG:-qwen3_5_4b_aisci_rubric}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME=${EXP_NAME:-${RUN_TAG}_${TIMESTAMP}}

source "${MILES_DIR}/scripts/models/qwen3.5-4B.sh"

# CKPT_TAG: persistent checkpoint dir suffix. Each (model, response-len, thinking-mode)
# combo MUST use a distinct tag to avoid clobbering. v01.00 used "aisci_rubric"; v02.00
# variants override (e.g. CKPT_TAG=aisci_rubric_v02_think16k).
CKPT_TAG=${CKPT_TAG:-aisci_rubric}

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3.5-4B
   --ref-load ${MODEL_ROOT}/Qwen3.5-4B_torch_dist
   --load ${MODEL_ROOT}/Qwen3.5-4B_${CKPT_TAG}/
   --save ${MODEL_ROOT}/Qwen3.5-4B_${CKPT_TAG}/
   --save-interval ${SAVE_INTERVAL:-20}
)

# HF-streaming data source + custom RM. Note: --apply-chat-template is NOT set
# (the data source already applied the template); --rm-type is NOT set
# (--custom-rm-path takes precedence).
ROLLOUT_ARGS=(
   --data-source-path examples.aisci_rubric.hf_data_source.HFRolloutDataSource
   --custom-rm-path examples.aisci_rubric.reward_fn.reward_fn
   --reward-key reward_value
   --rollout-shuffle
   --num-rollout ${NUM_ROLLOUT:-3000}
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN:-8192}
   --rollout-temperature 0.8

   --global-batch-size 256
   --balance-data
)

# Eval is disabled until we wire HF eval data through --eval-prompt-data.
# Miles validates: if --eval-interval is set, --eval-datasets MUST be set too,
# so we omit the flag entirely unless the caller explicitly opts in by
# exporting both EVAL_INTERVAL and EVAL_PROMPT_DATA="<name> <path>".
# TODO: dump ACSci/v3-train eliminated_test to a small JSONL once and wire it.
EVAL_ARGS=()
if [[ -n "${EVAL_INTERVAL:-}" && -n "${EVAL_PROMPT_DATA:-}" ]]; then
    # shellcheck disable=SC2206
    EVAL_ARGS+=(--eval-interval "${EVAL_INTERVAL}" --eval-prompt-data ${EVAL_PROMPT_DATA})
fi

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size ${CONTEXT_PARALLEL_SIZE:-1}
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-9216}
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
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

cd "${MILES_DIR}"

# Multi-node knobs (default to single-node behavior — backward compatible).
#   RAY_HEAD_IP                IP of the ray head node. Default: 127.0.0.1 (local).
#   RAY_HEAD_ALREADY_STARTED   When 1, the caller (e.g. the 2-node sbatch wrapper)
#                              has already brought up the ray cluster. Skip the
#                              local `ray start --head`.
#   ACTOR_NUM_NODES            Number of actor nodes for miles. Default: 1.
export RAY_HEAD_IP=${RAY_HEAD_IP:-"127.0.0.1"}
export MASTER_ADDR=${MASTER_ADDR:-${RAY_HEAD_IP}}

if [[ "${RAY_HEAD_ALREADY_STARTED:-0}" != "1" ]]; then
    ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
else
    echo "[run] RAY_HEAD_ALREADY_STARTED=1 — using existing ray cluster at ${RAY_HEAD_IP}:6379"
fi

# Forward AISCI_* + judge creds + (when present) the local-judge URL into
# the ray runtime env so Ray workers can hit the configured backend.
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

ray job submit --address="http://${RAY_HEAD_IP}:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${MILES_DIR}/train.py \
   --actor-num-nodes ${ACTOR_NUM_NODES:-1} \
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
