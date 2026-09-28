#!/usr/bin/env bash
# Usage: bash RLVR/eval_stages.sh base sft rlaif attempt_verifier attempt_multi (any other name is an RLVR attempt dir).
# Each stage writes to its own EVALS/results_rlvr/<stage>, so the committed SFT and RLAIF results are never overwritten.
set -euo pipefail

BENCHMARKS="humaneval humanevalplus mbpp livecodebench"
RESULTS_ROOT="EVALS/results_rlvr"
MODEL_ID="Qwen/Qwen2.5-3B-Instruct"

evaluate_stage() {
    local stage="$1" base_model="$2" adapter="$3"
    local results_dir="$RESULTS_ROOT/$stage"
    local model_args=(--base-model "$base_model")
    if [ -n "$adapter" ]; then
        model_args+=(--checkpoints "$adapter")
    else
        model_args+=(--base)
    fi
    echo "=== $stage ==="
    uv run python EVALS/evaluation.py "${model_args[@]}" --benchmarks $BENCHMARKS --results-dir "$results_dir"
    uv run python EVALS/evaluate_persona.py "${model_args[@]}" --no-embeddings --results-dir "$results_dir"
    # the human references are the same for every stage, so only the base stage scores them
    local reference_flag="--no-reference"
    if [ "$stage" = "base" ]; then
        reference_flag=""
    fi
    uv run python EVALS/judge_persona.py --generations "$results_dir/persona_generations.json" --results-dir "$results_dir" $reference_flag
}

for stage in "$@"; do
    case "$stage" in
        base)  evaluate_stage base "$MODEL_ID" "" ;;
        sft)   evaluate_stage sft "$MODEL_ID" SFT/model/final_adapter ;;
        rlaif) evaluate_stage rlaif RLVR/model/sft_merged RLAIF/model/RLAIF_model ;;
        *)     evaluate_stage "$stage" RLVR/model/rlaif_merged "RLVR/model/$stage/RLVR_model" ;;
    esac
done
