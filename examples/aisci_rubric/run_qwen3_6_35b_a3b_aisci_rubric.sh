#!/bin/bash
# Trainer launcher: Qwen3.6-35B-A3B (MoE) GRPO on the aisci_rubric task.
#   v03.15 = nothink 16k (KL=0);  v03.16 = think 64k strict (KL=0.001).
#
# Same wiring as run_qwen3_6_27b_aisci_rubric.sh, adapted to the MoE (qwen3_5_moe,
# 256 experts top-8, ~35B total / ~3B active, 40 layers, hidden 2048).
#
# GEOMETRY (both runs, unified): 8 nodes x 8 H100 = 64 GPUs.
#   TP=8 x CP=4 x PP=2 x EP=8, ETP=1, DP=1, EDP(=CP*DP)=4.
#   - PP=2 splits 40 layers 20+20 (PP=1 OOMs on the MoE dispatch buffer; the
#     torch_dist ckpt saved at PP=1 reshapes cleanly to PP=2 at load).
#   - TP=8 halves vocab/MoE/attention activations per rank (TP=4's vocab-parallel
#     entropy step OOMs). num_query_groups=2 < TP → the qwen3_5 attention plugin
#     replicates KV heads for TP>nqg (proven in the 32k throughput study).
#   - EP=8 is the required floor — EP=4 doubles per-rank expert state and OOMs.
#   - CP=4: nothink 16k/4 = 4k/rank; think 64k/4 = 16k/rank (= the proven 32k/CP2
#     throughput profile). This is standard attention (not the 27B's linear attn),
#     so it tolerates the 16k-per-rank chunk.
#   - STANDARD distributed optimizer, cpu-offload OFF: offload + precision-aware optimizer
#     BREAKS the MoE dist-ckpt save (invalid global plan). Memory fits without it at 8 nodes.
#   - --no-ckpt-fully-parallel-save + --auto-detect-ckpt-format (REQUIRED for MoE at PP>=2):
#     the MoE dist-ckpt save fails torch DCP "validate global plan" otherwise. This matches
#     Megatron's own Mixtral MoE ckpt tests (EP=8, PP=4/8). We resume at the SAME parallelism.
#
# MoE-specific runtime (ported from bash/throughput/run.py, proven for this model):
#   - Sources the -nofuse spec (MODEL_SPEC_SCRIPT default) — --moe-permute-fusion is
#     a store_true flag, so the ONLY way to disable it (required, avoids the per-step
#     MoE all-to-all OOM) is to source a spec that omits it.
#   - SGLang rollout runs EP too (--sglang-ep-size) with EAGLE MTP speculative
#     decoding (draft = the spec's --mtp-num-layers 1) + mamba extra_buffer
#     scheduler (required for radix-cache + spec-decode on Qwen3_5Moe*).
#   - TRITON_AUTOTUNE_RANDOM_SAMPLE=1 avoids the MoE grouped-GEMM autotune OOM.
#
# think vs nothink is set entirely in the data yaml (AISCI_DATA_CONFIG:
# enable_thinking + require_think_close); this launcher only sets the matching
# token budget via ROLLOUT_MAX_RESPONSE_LEN (16k nothink / 64k think, from the
# submitter). Data/judge/reward configs are model-agnostic and shared with the 27B.

# See 9B/27B variant — skip cleanup when the caller has already brought up ray.
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
RUN_TAG=${RUN_TAG:-qwen3_6_35b_a3b_aisci_rubric}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME=${EXP_NAME:-${RUN_TAG}_${TIMESTAMP}}

# MoE arch spec. Default -nofuse (permute-fusion OFF — required, see header).
# Set MODEL_SPEC_SCRIPT=qwen3.6-35B-A3B.sh to try the fused spec.
source "${MILES_DIR}/scripts/models/${MODEL_SPEC_SCRIPT:-qwen3.6-35B-A3B-nofuse.sh}"

# CKPT_TAG: see comment in run_qwen3_5_4b_aisci_rubric.sh.
CKPT_TAG=${CKPT_TAG:-aisci_rubric}

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3.6-35B-A3B
   --ref-load ${MODEL_ROOT}/Qwen3.6-35B-A3B_torch_dist
   --load ${MODEL_ROOT}/Qwen3.6-35B-A3B_${CKPT_TAG}/
   --save ${MODEL_ROOT}/Qwen3.6-35B-A3B_${CKPT_TAG}/
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
   --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE:-8}
   --sequence-parallel
   --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE:-2}
   --context-parallel-size ${CONTEXT_PARALLEL_SIZE:-4}
   --expert-model-parallel-size ${EXPERT_MODEL_PARALLEL_SIZE:-8}
   --expert-tensor-parallel-size ${EXPERT_TENSOR_PARALLEL_SIZE:-1}

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

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
# cpu-offload optimizer OFF by default. The offload + --use-precision-aware-optimizer state layout
# BREAKS the MoE dist-checkpoint save: it produces a sharded_state_dict whose global plan fails torch
# DCP validation ("ValueError: Failed to validate global plan") — even with --no-ckpt-fully-parallel-
# save (v03.15/16 smokes, 2026-07-05). The proven-working MoE save uses the STANDARD distributed
# optimizer: Megatron's own Mixtral MoE ckpt tests (8x7b EP8/PP4, 8x22b EP8/PP8) save+resume with
# --use-distributed-optimizer (no offload) + --no-ckpt-fully-parallel-save + --auto-detect-ckpt-format.
# miles already forces --use-distributed-optimizer. Memory fits without offload at 8 nodes (the smoke
# trained no-OOM at both 4k and 16k/rank). OPTIMIZER_CPU_OFFLOAD=1 re-enables it (breaks MoE save).
if [[ "${OPTIMIZER_CPU_OFFLOAD:-0}" == "1" ]]; then
    OPTIMIZER_ARGS+=(
        --optimizer-cpu-offload
        --overlap-cpu-optimizer-d2h-h2d
        --use-precision-aware-optimizer
    )
fi

# MoE dist-checkpoint SAVE: match Megatron's proven MoE+EP+PP checkpoint config.
#   --no-ckpt-fully-parallel-save: the FullyParallelSaveStrategyWrapper's decentralized global-plan
#     validation fails for MoE at PP>=2; Megatron's own Mixtral MoE tests (EP8, PP4/PP8) all disable it.
#     Safe here — we always resume at the SAME parallelism (Megatron only warns that non-parallel save
#     blocks resuming with DIFFERENT parallelism). Re-enable with CKPT_FULLY_PARALLEL_SAVE=1.
#   --auto-detect-ckpt-format: robust load-time format detection (both Mixtral MoE tests use it).
CKPT_SAVE_ARGS=(--auto-detect-ckpt-format)
if [[ "${CKPT_FULLY_PARALLEL_SAVE:-0}" != "1" ]]; then
    CKPT_SAVE_ARGS+=(--no-ckpt-fully-parallel-save)
fi

WANDB_ARGS=(
   --use-wandb
   --wandb-project ${WANDB_PROJECT_NAME}
   --wandb-group ${EXP_NAME}
   --wandb-team ${WANDB_ENTITY:-yuxiao98}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC:-0.45}
   --sglang-ep-size ${SGLANG_EP_SIZE:-${EXPERT_MODEL_PARALLEL_SIZE:-8}}
)
# EAGLE MTP speculative decoding for the Qwen3_5Moe rollout (draft = the spec's
# --mtp-num-layers 1 head). extra_buffer is required when combining the radix
# cache with spec decoding for Qwen3_5Moe* (sglang raises ValueError otherwise).
# Proven in bash/throughput/run.py. Toggle off for debugging with SGLANG_SPEC_DECODE=0.
SGLANG_ENABLE_SPEC_V2_VAL=""
if [[ "${SGLANG_SPEC_DECODE:-1}" == "1" ]]; then
    SGLANG_ARGS+=(
        --sglang-speculative-algorithm EAGLE
        --sglang-speculative-num-steps 2
        --sglang-speculative-eagle-topk 1
        --sglang-speculative-num-draft-tokens 3
        --sglang-mamba-scheduler-strategy extra_buffer
    )
    SGLANG_ENABLE_SPEC_V2_VAL="1"
fi

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

# MoE dist-optimizer checkpoint padding FIX (always on). Prepends the ckptfix dir whose
# sitecustomize.py clamps overshooting optimizer ShardedTensors before dist_checkpointing
# save/load, so the DCP global plan validates (see ckptfix/sitecustomize.py for the bug).
# AISCI_CKPT_DEBUG=1 additionally logs the DCP validation detail.
CKPT_FIX_PP="${MILES_DIR}/examples/aisci_rubric/ckptfix:"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${CKPT_FIX_PP}${MILES_DIR}:${MEGATRON_LM_DIR}/\",
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
    \"JUDGE_BASE_URL\": \"${JUDGE_BASE_URL:-}\",
    \"SGLANG_ENABLE_SPEC_V2\": \"${SGLANG_ENABLE_SPEC_V2_VAL}\",
    \"TRITON_AUTOTUNE_RANDOM_SAMPLE\": \"1\"
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
   ${CKPT_SAVE_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
