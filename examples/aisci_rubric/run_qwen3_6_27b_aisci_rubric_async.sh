#!/bin/bash
# Trainer launcher: Qwen3.6-27B GRPO on aisci_rubric — NON-COLOCATE FULLY-ASYNC (v04).
#
# Derived from run_qwen3_6_27b_aisci_rubric.sh (the colocate/sync launcher used by
# v03.08-v03.14). This variant runs the disaggregated fully-async path:
#   - entrypoint train_async.py (NOT train.py); it asserts `not args.colocate`.
#   - --rollout-function-path fully_async_rollout.generate_rollout_fully_async : a
#     background SGLang worker streams rollouts into a queue; the trainer drains it and
#     syncs weights every --update-weights-interval steps.
#   - NO --colocate. Rollout gets its own GPU pool via --rollout-num-gpus (default 64 =
#     8 rollout nodes). Actor uses ACTOR_NUM_NODES*8 GPUs. Total = actor + rollout.
#   - Off-policy correction with --use-tis/--tis-clip (async introduces weight staleness);
#     optional hard bound with --max-weight-staleness (recycles too-stale groups).
#   - broadcast weight transfer (P2P is unvalidated for Qwen3_5ForConditionalGeneration).
#
# Actor geometry (8 nodes = 64 GPU): TP4 × PP4 × CP4 × DP1. 64k/CP4 = 16k tokens/CP-rank
# (4x the proven-safe 4k — MUST pass the smoke/memory gate; CP8/PP2 8k-rank is the fallback).
# TP stays 4 (num_query_groups=4 => Megatron requires nqg % TP == 0). PP4 => 16 layers/rank,
# freeing param+optimizer memory for the larger per-rank activation.
#
# The aisci wiring (judge reward_fn, HFRolloutDataSource, reward-key) is IDENTICAL to the
# sync launcher — verified to run unchanged inside the async worker.

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
# The fully-async rollout fn is imported top-level as `fully_async_rollout`, so its dir
# must be on PYTHONPATH (in addition to MILES_DIR for the aisci example modules).
FULLY_ASYNC_DIR=${FULLY_ASYNC_DIR:-${MILES_DIR}/examples/fully_async}

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
RUN_TAG=${RUN_TAG:-qwen3_6_27b_aisci_rubric_async}
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
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
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

# EVAL: the async rollout fn raises on evaluation=True. Default = no eval (EVAL_INTERVAL
# unset). If ever enabled, route eval through the SYNC rollout fn so it doesn't crash.
EVAL_ARGS=()
if [[ -n "${EVAL_INTERVAL:-}" && -n "${EVAL_PROMPT_DATA:-}" ]]; then
    # shellcheck disable=SC2206
    EVAL_ARGS+=(--eval-interval "${EVAL_INTERVAL}" --eval-prompt-data ${EVAL_PROMPT_DATA} \
                --eval-function-path miles.rollout.sglang_rollout.generate_rollout)
fi

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE:-4}
   --context-parallel-size ${CONTEXT_PARALLEL_SIZE:-4}
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-16384}
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

# Fully-async / off-policy knobs. TIS on by default (async lag correction); staleness
# filter off by default (update-weights-interval=1 keeps natural staleness low).
ASYNC_ARGS=(
   --update-weight-transfer-mode ${UPDATE_WEIGHT_TRANSFER_MODE:-broadcast}
   --update-weights-interval ${UPDATE_WEIGHTS_INTERVAL:-1}
)
[[ -n "${MAX_WEIGHT_STALENESS:-}" ]] && ASYNC_ARGS+=(--max-weight-staleness ${MAX_WEIGHT_STALENESS})
if [[ "${USE_TIS:-1}" == "1" ]]; then
    ASYNC_ARGS+=(--use-tis --tis-clip ${TIS_CLIP:-2.0})
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   # NO --optimizer-cpu-offload: its checkpoint-resume path is broken (ERR-024).
   # At DP=1 the distributed optimizer cannot shard across DP, so PP=4 (16 layers/rank)
   # is what keeps the on-GPU optimizer footprint in budget.
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project ${WANDB_PROJECT_NAME}
   --wandb-group ${EXP_NAME}
   --wandb-team ${WANDB_ENTITY:-yuxiao98}
)

# Rollout SGLang engines (non-colocate, own their GPUs). mem-fraction-static=0.7:
# 0.85 OOM'd mid-generation on the v04 smoke (SGLang "Scheduler hit an exception: CUDA out of
# memory, tried 3.65 GiB, 332 MiB free") — a 27B TP4 engine holding a 64k-context KV pool needs
# ample transient/decode headroom, so keep the static pool <=0.7. If it still OOMs, drop to 0.6
# and/or lower SGLANG_MAX_RUNNING_REQUESTS (fewer concurrent 64k sequences = smaller peak KV).
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC:-0.7}
   --sglang-max-running-requests ${SGLANG_MAX_RUNNING_REQUESTS:-48}
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
    \"PYTHONPATH\": \"${FULLY_ASYNC_DIR}:${MILES_DIR}:${MEGATRON_LM_DIR}/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}\",
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
   -- python3 ${MILES_DIR}/train_async.py \
   --actor-num-nodes ${ACTOR_NUM_NODES:-8} \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus ${ROLLOUT_NUM_GPUS:-64} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${ASYNC_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
