#!/bin/bash
# GRPO against the persona judge, starting from the merged SFT checkpoint.
#
#   bash rlaif/convert_checkpoints.sh to_megatron          # once
#   bash rlaif/run_grpo.sh                                 # gated reward
#   REWARD_MODE=sum bash rlaif/run_grpo.sh                 # plain 5-25 total
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash rlaif/run_grpo.sh    # pick the GPUs
#
# Argument structure follows slime's examples/search-r1/run_qwen2.5_3B.sh, the
# upstream example for this base model.
set -ex

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/rlaif/slime_env.sh"
source "${SLIME_DIR}/scripts/models/qwen2.5-3B.sh"

# Default fits the box this was built on, where GPU 2 reports uncorrectable ECC
# errors and killed a run mid-generation. Check before trusting it anywhere else:
#   nvidia-smi --query-gpu=index,memory.used,ecc.errors.uncorrected.volatile.total --format=csv
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
NUM_GPUS=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")

export REWARD_MODE=${REWARD_MODE:-gated}
export JUDGE_MODEL=${JUDGE_MODEL:-gpt-5.4-mini-2026-03-17}
export JUDGE_SCORES_LOG=${JUDGE_SCORES_LOG:-${PROJECT_DIR}/outputs/rlaif/judge_scores_${REWARD_MODE}.jsonl}
[ -n "$OPENAI_API_KEY" ] || { echo "OPENAI_API_KEY is not set; the reward needs it" >&2; exit 1; }

export PYTHONUNBUFFERED=1
pkill -9 sglang || true; ray stop --force || true; pkill -9 ray || true; sleep 3

CKPT_ARGS=(
   --hf-checkpoint ${PROJECT_DIR}/outputs/sft/merged_hf
   # the SFT policy is also the KL reference, so drift is measured from where
   # checkpoint 1 left off rather than from the base model
   --ref-load      ${PROJECT_DIR}/outputs/rlaif/sft_torch_dist
   --save          ${PROJECT_DIR}/outputs/rlaif/ckpt_${REWARD_MODE}
   --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data ${PROJECT_DIR}/data/rlaif_prompts.jsonl
   --input-key input
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 40
   --rollout-batch-size 32
   --n-samples-per-prompt 8          # the group GRPO takes its advantage over
   --rollout-max-response-len 512    # the cap both eval harnesses generate under
   --rollout-temperature 1.0
   --global-batch-size 256
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 1    # 3B fits on one A100; TP would add only knobs
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # the persona already sits at the human ceiling (EVAL_RESULTS.md section 4),
   # so the risk being managed here is drift away from it, not underfitting
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   # one group per call, so the reward can hand an unscorable rollout its group's
   # mean instead of a floor that would skew its siblings' advantages
   --group-rm
   --custom-rm-path rlaif.reward.batched_custom_rm
   --dump-details ${PROJECT_DIR}/outputs/rlaif/details_${REWARD_MODE}
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats

# PROJECT_DIR on PYTHONPATH is what makes --custom-rm-path resolvable in the
# Ray workers; OPENAI_API_KEY has to be forwarded explicitly for the same reason
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${PROJECT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"OPENAI_API_KEY\": \"${OPENAI_API_KEY}\",
    \"JUDGE_MODEL\": \"${JUDGE_MODEL}\",
    \"REWARD_MODE\": \"${REWARD_MODE}\",
    \"JUDGE_SCORES_LOG\": \"${JUDGE_SCORES_LOG}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${SLIME_DIR}/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --rollout-num-gpus ${NUM_GPUS} \
   --num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} ${PERF_ARGS[@]} ${SGLANG_ARGS[@]} ${MISC_ARGS[@]} ${CUSTOM_ARGS[@]}
