#!/bin/bash
# Trainer launcher: Qwen3.6-27B GRPO on the aisci_rubric task (v03.08 nothink 16k).
#
# Same wiring as run_qwen3_5_9b_aisci_rubric.sh, scaled to the 27B dense model.
# REQUIRES 2 nodes (ACTOR_NUM_NODES=2): TP=4 × PP=2 × DP=2, no optimizer offload.
#
# History (2026-06-20/21): num_query_groups=4 forbids TP>4 (Megatron asserts
# nqg % TP == 0), so per-GPU layers are cut with PIPELINE parallel. The TP=4/PP=1
# template (scripts/run-qwen3.6-27B.sh) OOMs at init (6.75B params/GPU). With PP=2
# each GPU holds 32 of 64 layers → 3.375B params/GPU. The cpu-offload optimizer
# fit the weights but its checkpoint-RESUME path is BROKEN (KeyError in
# cpu_offloading/hybrid_optimizer.py:_update_fp32_params_by_new_state, ERR-024), so
# the afterany chain couldn't continue. Fix = drop cpu-offload and shard the
# optimizer across DP=2 (2 nodes) — the proven v03.04 resume path, scaled 1.5×.
#   - sources scripts/models/qwen3.6-27B.sh (64 layers, hidden 5120, ffn 17408,
#     num_attention_heads 24, num_query_groups 4).
#   - TP=4 × PP=2 × DP=2 on 2 nodes (16 GPUs), CP=1. ~3.375B params/GPU; the
#     distributed optimizer (miles default use_distributed_optimizer=True) shards
#     master+moments across DP=2 → fits on-GPU without cpu-offload, and resumes
#     cleanly. PP+colocate+SGLang is proven in miles (run-qwen3-4B_4xgpu-radixtree.sh).
#   - NO optimizer CPU-offload (its resume path is broken; not needed at DP=2).
#   - MAX_TOKENS_PER_GPU=8192 (default): nothink responses are short (mean ~1.8k,
#     max ~4.5k) so this never truncates; keeps the loss-time logits tensor small
#     (vocab=248320 made 20480 OOM at step 46 on the cpu-offload single-node run).
#     NOT setting expandable_segments — it conflicts with the colocate TorchMemorySaver.
#   - SGLang rollout TP=4 (--rollout-num-gpus-per-engine 4, divides num_query_groups=4),
#     mem-fraction 0.7 — in colocate the actor offloads during rollout.
# Intended to run under the 2-node DFW sbatch (aicsi-rubric-v03-2node.sh, CP=1).

# See 9B variant — skip cleanup when the caller has already brought up ray.
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
RUN_TAG=${RUN_TAG:-qwen3_6_27b_aisci_rubric}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME=${EXP_NAME:-${RUN_TAG}_${TIMESTAMP}}

source "${MILES_DIR}/scripts/models/qwen3.6-27B.sh"

# CKPT_TAG: see comment in run_qwen3_5_4b_aisci_rubric.sh.
CKPT_TAG=${CKPT_TAG:-aisci_rubric}

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3.6-27B
   --ref-load ${MODEL_ROOT}/Qwen3.6-27B_torch_dist
   --load ${MODEL_ROOT}/Qwen3.6-27B_${CKPT_TAG}/
   --save ${MODEL_ROOT}/Qwen3.6-27B_${CKPT_TAG}/
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
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN:-16384}
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
   --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE:-2}
   --context-parallel-size ${CONTEXT_PARALLEL_SIZE:-1}
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-8192}
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef ${KL_LOSS_COEF:-0.00}
   --kl-loss-type low_var_kl
   --entropy-coef ${ENTROPY_COEF:-0.0001}
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
   # NO --optimizer-cpu-offload: its checkpoint-resume path is broken (ERR-024).
   # The distributed optimizer (miles default) shards Adam state across DP=2, so
   # it fits on-GPU without offload at this 2-node geometry.
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project ${WANDB_PROJECT_NAME}
   --wandb-group ${EXP_NAME}
   --wandb-team ${WANDB_ENTITY:-yuxiao98}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC:-0.7}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

cd "${MILES_DIR}"

# Multi-node knobs — see the 4B launcher for documentation.
export RAY_HEAD_IP=${RAY_HEAD_IP:-"127.0.0.1"}
export MASTER_ADDR=${MASTER_ADDR:-${RAY_HEAD_IP}}

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
