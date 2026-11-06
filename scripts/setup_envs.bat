@echo off
:: Build the conda envs for the baselines and write bench_config.toml.
:: Run from a *Miniforge Prompt* in the repo root:
::     scripts\setup_envs.bat            (3DGS + SOG)
::     scripts\setup_envs.bat sharp      (also SHARP)
:: Safe to re-run: existing envs are skipped. Delete one with `conda env remove -n <name>` to rebuild.

setlocal
cd /d "%~dp0.."

:: CUDA 11.8 rejects MSVC 14.40+, so build with the installed 14.38 toolset.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38 || goto :fail
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.6
set CUDA_HOME=%CUDA_PATH_V11_8%

for /f "delims=" %%i in ('conda info --base') do set "CONDA_BASE=%%i"
set "ENVS=%CONDA_BASE%\envs"

:: ---- 3DGS (+ shared evaluator) -------------------------------------------
if exist "%ENVS%\gs\python.exe" (
    echo [gs] exists, skipping
) else (
    call conda create -n gs python=3.10 -y || goto :fail
    call conda activate gs || goto :fail
    pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118 || goto :fail
    pip install plyfile tqdm opencv-python joblib ninja "numpy<2" || goto :fail
    pip install --no-build-isolation external/gaussian-splatting/submodules/diff-gaussian-rasterization external/gaussian-splatting/submodules/simple-knn external/gaussian-splatting/submodules/fused-ssim || goto :fail
    call conda deactivate
)

:: ---- Self-Organizing Gaussians -------------------------------------------
if exist "%ENVS%\sogs\python.exe" (
    echo [sogs] exists, skipping
) else (
    pushd external\Self-Organizing-Gaussians
    call conda env create --file environment.yml || (popd & goto :fail)
    popd
)

:: ---- SHARP (optional) ----------------------------------------------------
if /i "%1"=="sharp" (
    if exist "%ENVS%\sharp\python.exe" (
        echo [sharp] exists, skipping
    ) else (
        call conda create -n sharp python=3.13 -y || goto :fail
        call conda activate sharp || goto :fail
        pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 || goto :fail
        rem requirements.txt contains "-e .", which pip resolves against the current directory
        pushd external\ml-sharp && pip install -r requirements.txt || goto :fail
        popd
        pip install -e external/ml-sharp || goto :fail
        call conda deactivate
    )
)

:: ---- bench_config.toml ---------------------------------------------------
set "FWD=%ENVS:\=/%"
(
    echo [python]
    echo gs    = "%FWD%/gs/python.exe"
    echo sog   = "%FWD%/sogs/python.exe"
    echo sharp = "%FWD%/sharp/python.exe"
    echo.
    echo [gpu]
    echo index = 0
    echo sample_interval_ms = 200
) > bench_config.toml
echo Wrote bench_config.toml

:: ---- smoke test: can each env import its CUDA rasterizer? ----------------
"%ENVS%\gs\python.exe" -c "import torch, diff_gaussian_rasterization, simple_knn._C, fused_ssim; print('[gs] OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())" || goto :fail
"%ENVS%\sogs\python.exe" -c "import torch, diff_gaussian_rasterization, simple_knn._C, plas, imagecodecs; print('[sogs] OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())" || goto :fail
echo All done.
exit /b 0

:fail
echo.
echo *** Setup failed at the step above. Copy the error output and ask for help. ***
exit /b 1
