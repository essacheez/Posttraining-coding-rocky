#!/bin/bash
# Builds the slime training environment.
#
#   bash rlaif/install_slime.sh conda    # stage 1: conda env + CUDA 12.9 toolkit
#   bash rlaif/install_slime.sh deps     # stage 2: sglang, megatron, slime, patches
#
# This is slime's own build_conda.sh with three changes, and only these:
#   - paths live under $SLIME_BASE instead of /root, since the box this was built
#     on has no docker access and no sudo
#   - micromamba comes from the release tarball, not the interactive installer,
#     which wants a writable $HOME and a tty
#   - `set -ex`, not `set -eux`: conda's own activate.d scripts reference unset
#     variables and abort the build under -u
#
# Split into two stages so a failure in a 40-minute source build does not re-run
# the multi-GB conda solve. Every version pin is upstream's and must stay in sync
# with slime's build_conda.sh.
set -ex -o pipefail

export SLIME_BASE=${SLIME_BASE:-/raid/lingo/riddhib/slime-env}
export SLIME_DIR=$SLIME_BASE/slime
export MAMBA_ROOT_PREFIX=$SLIME_BASE/micromamba
export PIP_CACHE_DIR=$SLIME_BASE/pip_cache

export SGLANG_COMMIT="0b3bb0cbe31873994c9f989fddfe2f87ca839fdd"
export MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"
export PATCH_VERSION="v0.5.15.post1"
export TMS_COMMIT="8d30c59ca12a68d9deccbc9c6599076a1218cbc5"
export FLASH_QLA_COMMIT="821fd9d37ede18fdc2a4e707fefe3770bfc32e58"

mkdir -p "$SLIME_BASE" "$PIP_CACHE_DIR" "$MAMBA_ROOT_PREFIX"
cd "$SLIME_BASE"

if [ ! -x "$SLIME_BASE/bin/micromamba" ]; then
  mkdir -p "$SLIME_BASE/bin"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$SLIME_BASE" bin/micromamba
fi
export PATH="$SLIME_BASE/bin:$PATH"
eval "$(micromamba shell hook -s bash)"

case "${1:-}" in
conda)
  micromamba env list | grep -q "^  slime " || micromamba create -n slime python=3.12 pip -c conda-forge -y
  micromamba activate slime
  export CUDA_HOME="$CONDA_PREFIX"
  micromamba install -n slime cuda=12.9.1 cuda-nvtx=12.9.79 cuda-nvtx-dev=12.9.79 nccl \
    -c nvidia/label/cuda-12.9.1 -c nvidia -c conda-forge -y
  micromamba install -n slime -c conda-forge cudnn -y
  # sglang's editable install builds a Rust extension via setuptools-rust
  micromamba install -n slime -c conda-forge rust -y
  echo "conda stage done: python=$(python -V), CUDA_HOME=$CUDA_HOME"
  ;;

deps)
  micromamba activate slime
  export CUDA_HOME="$CONDA_PREFIX"

  [ -d "$SLIME_BASE/sglang" ] || git clone https://github.com/sgl-project/sglang.git
  cd $SLIME_BASE/sglang && git checkout ${SGLANG_COMMIT}
  pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu129
  # sglang pulls a cu13 stack; these three steps put cu129 back, in upstream's order
  pip install --force-reinstall "cuda-python==12.9"
  pip install --force-reinstall --no-deps \
    torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129
  pip install --force-reinstall --no-deps sglang-kernel==0.4.4 sgl-deep-gemm==0.1.4 \
    --index-url https://docs.sglang.ai/whl/cu129/
  pip uninstall -y nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
    nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver \
    nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink \
    nvidia-nvshmem-cu13 nvidia-nvtx nvidia-cutlass-dsl-libs-cu13 || true
  pip install --force-reinstall --no-deps \
    nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12==9.16.0.29 nvidia-cufft-cu12 \
    nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
    nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 \
    nvidia-nvtx-cu12 \
    --index-url https://download.pytorch.org/whl/cu129 --extra-index-url https://pypi.org/simple

  pip install cmake ninja
  # the validated TE 2.16 stack wants FA2, while sglang's deps bring FA4; this is
  # the community build upstream pins for py3.12 + torch 2.11 + cu12
  pip uninstall -y flash-attn-4 flash_attn_4 || true
  pip install --no-deps \
    "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=3d0c8e60f820321eedd7166e79c33cb816263d8be6e35c3f5ba8fe2df6fea697"
  pip install flash-linear-attention==0.4.2
  pip install git+https://github.com/QwenLM/FlashQLA.git@${FLASH_QLA_COMMIT} --no-build-isolation
  pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/
  pip install --no-build-isolation "transformer_engine[pytorch]==2.16.1"

  NVCC_APPEND_FLAGS="--threads 4" pip -v install --disable-pip-version-check --no-cache-dir \
    --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
    git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

  # --no-build-isolation: TMS needs nvcc and the installed torch to compile its
  # preload hook; without it the wheel is python-only and sglang trips at runtime
  export TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
  pip install -v git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT} \
    --no-cache-dir --force-reinstall --no-build-isolation
  pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
  pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
  python -c "import sglang_router; assert 'slime' in sglang_router.__version__"

  cd $SLIME_BASE
  [ -d "$SLIME_BASE/Megatron-LM" ] || git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
  # --no-build-isolation: setup.py shells out to pybind11 for a C++ extension that
  # is otherwise marked optional and silently skipped, breaking dataset loading
  pip install "setuptools<80.0.0" pybind11 "packaging>=24.2"
  cd $SLIME_BASE/Megatron-LM && git checkout ${MEGATRON_COMMIT} && pip install -e . --no-build-isolation

  cd $SLIME_BASE
  [ -d "$SLIME_DIR" ] || git clone https://github.com/THUDM/slime.git
  cd $SLIME_DIR
  # requirements first, then slime with --no-deps, so pip cannot re-resolve and
  # stomp the pinned cu129 native libs installed above
  pip install -r requirements.txt
  pip install -e . --no-deps
  cd $SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat && pip install . --no-build-isolation
  pip install nvidia-cudnn-cu12==9.16.0.29
  pip install "numpy==1.26.4" "scipy==1.17.1"
  pip install "kernels<0.15.0"

  patch_dir="$SLIME_DIR/docker/patch/${PATCH_VERSION}"
  [ -d "$patch_dir" ] || { echo "no patch directory at $patch_dir" >&2; exit 1; }
  cd $SLIME_BASE/sglang
  for patch_name in sglang.patch sglang-top_p.patch sglang-release_hicache.patch \
                    sglang-pull_weights.patch sglang-deterministic.patch; do
    patch_path="$patch_dir/${patch_name}"
    [ -f "$patch_path" ] || { echo "missing patch $patch_path" >&2; exit 1; }
    git apply --check "$patch_path" 2>/dev/null \
      && git apply "$patch_path" && echo "applied $patch_name" \
      || echo "skipped (already applied or conflicting): $patch_name"
  done
  echo "deps stage done"
  ;;

*)
  echo "usage: $0 {conda|deps}" >&2; exit 1 ;;
esac
