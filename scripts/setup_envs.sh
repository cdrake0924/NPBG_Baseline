#!/usr/bin/env bash
# Build the conda envs for the baselines and write bench_config.toml. Linux counterpart of
# setup_envs.bat. Run from any shell where `conda` works, in the repo root:
#     bash scripts/setup_envs.sh            (3DGS + SOG)
#     bash scripts/setup_envs.sh sharp      (also SHARP)
# Safe to re-run: existing envs are skipped. Delete one with `conda env remove -n <name>` to rebuild.
#
# Optional overrides:
#     NPBG_PYTHON=/path/to/python   NPBG's env (default: the quasar-npbg env, if found)
#     TORCH_CUDA_ARCH_LIST=8.9      GPU architecture to compile for (default: read from nvidia-smi)
set -euo pipefail
cd "$(dirname "$0")/.."

fail() { echo; echo "*** Setup failed: $1. Copy the error output above and ask for help. ***"; exit 1; }

CONDA_BASE="$(conda info --base)"
ENVS="$CONDA_BASE/envs"

# ---- CUDA 11.8 toolchain for building the rasterizers --------------------------------------
# The gs and sogs envs use PyTorch built for CUDA 11.8, so their CUDA extensions must be
# compiled with nvcc 11.8, which in turn rejects gcc newer than 11 (the Linux version of the
# MSVC 14.38 gotcha on Windows). Use a system CUDA 11.8 if there is one; otherwise install
# CUDA 11.8 + gcc 11 into a separate "cuda118" env that only exists to build with.
gcc_major() { "${1:-gcc}" -dumpversion 2>/dev/null | cut -d. -f1; }

if [ -x /usr/local/cuda-11.8/bin/nvcc ] && [ "$(gcc_major)" -le 11 ] 2>/dev/null; then
    export CUDA_HOME=/usr/local/cuda-11.8
    echo "[cuda] using system CUDA 11.8 and gcc $(gcc_major)"
else
    if [ ! -x "$ENVS/cuda118/bin/nvcc" ]; then
        conda create -n cuda118 -y -c nvidia/label/cuda-11.8.0 -c conda-forge \
            cuda-toolkit gcc=11 gxx=11 || fail "creating the cuda118 toolchain env"
    fi
    export CUDA_HOME="$ENVS/cuda118"
    # torch's extension builder links against $CUDA_HOME/lib64; conda puts the libraries in lib/
    [ -e "$CUDA_HOME/lib64" ] || ln -s lib "$CUDA_HOME/lib64"
    export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
    echo "[cuda] using the cuda118 env (nvcc 11.8, gcc $(gcc_major "$CUDA_HOME/bin/gcc"))"
fi
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$(command -v gcc)" CXX="$(command -v g++)"

if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | paste -sd';')"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;8.9}"
fi
echo "[cuda] compiling for compute capability $TORCH_CUDA_ARCH_LIST"

# ---- 3DGS (+ shared evaluator) -------------------------------------------------------------
if [ -x "$ENVS/gs/bin/python" ]; then
    echo "[gs] exists, skipping"
else
    conda create -n gs python=3.10 -y || fail "creating the gs env"
    py="$ENVS/gs/bin/python"
    "$py" -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118 || fail "installing torch (gs)"
    "$py" -m pip install plyfile tqdm opencv-python joblib ninja "numpy<2" || fail "installing gs packages"
    "$py" -m pip install --no-build-isolation \
        external/gaussian-splatting/submodules/diff-gaussian-rasterization \
        external/gaussian-splatting/submodules/simple-knn \
        external/gaussian-splatting/submodules/fused-ssim || fail "building the 3DGS rasterizer"
fi

# ---- Self-Organizing Gaussians -------------------------------------------------------------
# environment.yml pip-installs its rasterizer submodules, which build with the toolchain above.
if [ -x "$ENVS/sogs/bin/python" ]; then
    echo "[sogs] exists, skipping"
else
    (cd external/Self-Organizing-Gaussians && conda env create --file environment.yml) || fail "creating the sogs env"
fi

# ---- SHARP (optional) ----------------------------------------------------------------------
if [ "${1:-}" = "sharp" ]; then
    if [ -x "$ENVS/sharp/bin/python" ]; then
        echo "[sharp] exists, skipping"
    else
        conda create -n sharp python=3.13 -y || fail "creating the sharp env"
        py="$ENVS/sharp/bin/python"
        "$py" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 || fail "installing torch (sharp)"
        "$py" -m pip install -r external/ml-sharp/requirements.txt || fail "installing SHARP requirements"
        "$py" -m pip install -e external/ml-sharp || fail "installing SHARP"
    fi
fi

# ---- bench_config.toml ---------------------------------------------------------------------
NPBG_PYTHON="${NPBG_PYTHON:-}"
if [ -z "$NPBG_PYTHON" ]; then
    for p in "$ENVS/quasar-npbg/bin/python" "$HOME/miniconda3/envs/quasar-npbg/bin/python"; do
        [ -x "$p" ] && NPBG_PYTHON="$p" && break
    done
fi
cat > bench_config.toml <<EOF
[python]
gs    = "$ENVS/gs/bin/python"
sog   = "$ENVS/sogs/bin/python"
sharp = "$ENVS/sharp/bin/python"
npbg  = "${NPBG_PYTHON:-NOT FOUND - set NPBG_PYTHON and re-run}"

[gpu]
index = 0
sample_interval_ms = 200
EOF
echo "Wrote bench_config.toml"

# ---- smoke test: can each env import its CUDA rasterizer? ----------------------------------
"$ENVS/gs/bin/python" -c "import torch, diff_gaussian_rasterization, simple_knn._C, fused_ssim; print('[gs] OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())" || fail "gs smoke test"
"$ENVS/sogs/bin/python" -c "import torch, diff_gaussian_rasterization, simple_knn._C, plas, imagecodecs; print('[sogs] OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())" || fail "sogs smoke test"
if [ "${1:-}" = "sharp" ]; then
    "$ENVS/sharp/bin/python" -c "import torch, sharp; print('[sharp] OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())" || fail "sharp smoke test"
fi
echo "All done."
