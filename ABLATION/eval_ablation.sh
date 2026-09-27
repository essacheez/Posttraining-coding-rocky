#!/usr/bin/env bash
# Usage: bash ABLATION/eval_ablation.sh sft_code rlaif_code
# Same instruments as the RLVR stages, so the ablation rows are comparable with the shipped table.
set -euo pipefail

BENCHMARKS="humaneval humanevalplus mbpp livecodebench"
RESULTS_ROOT="EVALS/results_rlvr"
MODEL_ID="Qwen/Qwen2.5-3B-Instruct"

evaluate_stage() {
    local stage="$1" base_model="$2" adapter="$3"
    local results_dir="$RESULTS_ROOT/$stage"
    echo "=== $stage ==="
    uv run python RLVR/eval_verifier.py --checkpoints "$adapter" --base-model "$base_model" --results-dir "$results_dir"
    uv run python EVALS/evaluation.py --base-model "$base_model" --checkpoints "$adapter" --benchmarks $BENCHMARKS --results-dir "$results_dir"
    uv run python EVALS/evaluate_persona.py --base-model "$base_model" --checkpoints "$adapter" --no-embeddings --results-dir "$results_dir"
    uv run python EVALS/judge_persona.py --generations "$results_dir/persona_generations.json" --results-dir "$results_dir" --no-reference
}

for stage in "$@"; do
    case "$stage" in
        sft_code)   evaluate_stage sft_code "$MODEL_ID" ABLATION/model/sft_code/final_adapter ;;
        rlaif_code) evaluate_stage rlaif_code ABLATION/model/sft_code_merged ABLATION/model/rlaif_code/RLAIF_model ;;
        *)          echo "unknown ablation stage: $stage" >&2; exit 1 ;;
    esac
done
