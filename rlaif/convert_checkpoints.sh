#!/bin/bash
# HF <-> Megatron torch_dist conversion, the two one-off commands around training.
#
#   bash rlaif/convert_checkpoints.sh to_megatron
#   bash rlaif/convert_checkpoints.sh to_hf outputs/rlaif/ckpt/iter_0000040
set -e
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/rlaif/slime_env.sh"
source "${SLIME_DIR}/scripts/models/qwen2.5-3B.sh"   # MODEL_ARGS for Qwen2.5-3B(-Instruct)

MERGED_HF="${PROJECT_DIR}/outputs/sft/merged_hf"
TORCH_DIST="${PROJECT_DIR}/outputs/rlaif/sft_torch_dist"

case "$1" in
  to_megatron)
    # pin to a healthy idle GPU: the converter otherwise grabs device 0, which on
    # this shared box usually has someone else's job on it
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
    PYTHONPATH="${MEGATRON_DIR}" python "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
      "${MODEL_ARGS[@]}" --hf-checkpoint "${MERGED_HF}" --save "${TORCH_DIST}"
    ;;
  to_hf)
    [ -n "$2" ] || { echo "usage: $0 to_hf <iter_dir> [out_dir]" >&2; exit 1; }
    OUT="${3:-${PROJECT_DIR}/outputs/rlaif/rlaif_hf}"
    PYTHONPATH="${MEGATRON_DIR}" python "${SLIME_DIR}/tools/convert_torch_dist_to_hf.py" \
      --input-dir "$2" --output-dir "${OUT}" --origin-hf-dir "${MERGED_HF}"
    # the conversion writes weights only; both eval harnesses load the tokenizer
    # from the model dir, so carry it across
    cp "${MERGED_HF}"/tokenizer* "${MERGED_HF}"/chat_template.jinja "${OUT}/" 2>/dev/null || true
    echo "wrote ${OUT}"
    ;;
  *)
    echo "usage: $0 {to_megatron|to_hf <iter_dir> [out_dir]}" >&2; exit 1 ;;
esac
