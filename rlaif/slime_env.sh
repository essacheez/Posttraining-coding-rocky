# Activates the slime conda environment. Sourced by the other rlaif scripts.
# Built by rlaif/install_slime.sh; override SLIME_BASE to point elsewhere.
export SLIME_BASE=${SLIME_BASE:-/raid/lingo/riddhib/slime-env}
export SLIME_DIR=${SLIME_DIR:-$SLIME_BASE/slime}
export MEGATRON_DIR=${MEGATRON_DIR:-$SLIME_BASE/Megatron-LM}
export MAMBA_ROOT_PREFIX=$SLIME_BASE/micromamba
export PATH="$SLIME_BASE/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate slime
export CUDA_HOME="$CONDA_PREFIX"
