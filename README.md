# NPBG Baseline Benchmark

Measures **speed, compute, and file size** (plus image quality, so the other numbers mean
something) for three Gaussian-splatting methods, so they can be compared against our own
pipeline under identical conditions.

| Method | What it does | Input | Output |
|---|---|---|---|
| [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) (3DGS) | Optimizes a scene from scratch, ~30k iterations | ~100 posed photos | `.ply`, often 50–700 MB |
| [Self-Organizing Gaussians](https://github.com/fraunhoferhhi/Self-Organizing-Gaussians) (SOG) | 3DGS + sorts the Gaussians into 2D grids and compresses them as images (JPEG XL / EXR) | ~100 posed photos | Folder of compressed images, ~20–40× smaller |
| [SHARP](https://apple-aiml-research.github.io/ml-sharp/) (Apple) | A neural network that predicts Gaussians in one forward pass, with no per-scene training | **1 photo** | `.ply` per image |

> **SHARP is a different kind of method.** It predicts a scene from a single photo, in its own
> camera frame and at its own depth scale. Following the mentor's suggestion, the harness gives it
> **4 photos**, predicts 4 independent scenes, and merges them into the dataset's world frame
> ([bench/sharp_merge.py](bench/sharp_merge.py)):
> 1. Each scene is placed using that photo's COLMAP camera pose.
> 2. Its depth scale is fitted against the sparse points that photo observed (median depth ratio).
> 3. The scenes are concatenated, with no refinement or de-duplication.
>
> The merged `.ply` is then scored on the same test views as everything else. Even where the
> quality is poor, its timing shows how fast a feed-forward method is, which sets a target
> for our pipeline.

## What gets measured

All three methods write standard 3DGS `.ply` files. **One shared evaluator**
([bench/eval_render.py](bench/eval_render.py)) renders every model with the same rasterizer and
the same metric code, so differences come from the methods and not from evaluation code.

| Category | Metric | How |
|---|---|---|
| **Speed** | **End-to-end time (`end_to_end_s`) — headline number** | From inputs on disk (images + poses + point cloud) to rendered test views: build wall time + loading the final model + one cold render pass over the test views |
| | Time to build the model (`train_wall_time_s`) | Wall clock of the whole build command, including process start and data loading. 3DGS: training; SOG: training + sort + compress; SHARP: model load + predict 4 + align + merge |
| | Per-image prediction time (`per_image_predict_s`) | SHARP only; CUDA-synced forward pass after warm-up |
| | Render speed (`eval_render_fps`) | Rasterization only, all test views × 3, after warm-up |
| **Compute** | Peak GPU memory (`train_gpu_peak_mem_delta_mib`) | `nvidia-smi` sampled every 200 ms, idle baseline subtracted |
| | GPU utilization, energy (`train_gpu_mean_util_pct`, `train_gpu_energy_wh`) | Power draw integrated over the run |
| | Peak system RAM (`train_ram_peak_mib`) | Process tree RSS (needs `pip install psutil`) |
| **Size** | **Scene file size (`model_mb`) — headline number** | 3DGS: `.ply`; SOG: compressed folder (what you would ship); SHARP: merged `.ply` |
| | `n_gaussians`, `bytes_per_gaussian` | Normalizes size by model complexity |
| | Network weights (`checkpoint_mb`) | Kept as a secondary number: SHARP needs trained weights to produce any scene; 3DGS/SOG have none |
| **Quality** | PSNR ↑, SSIM ↑, LPIPS(VGG) ↓ | On held-out test views, 8-bit renders (same as 3DGS `metrics.py`) |

Each run also records GPU name, driver, and the method's git commit.

## Data

### What each method needs

| | Images | Camera poses + intrinsics | Sparse point cloud | Notes |
|---|---|---|---|---|
| **3DGS** | ✓ | ✓ | ✓ initialization (random if missing) | Optimizes per scene |
| **SOG** | ✓ | ✓ | ✓ initialization | Same loader as 3DGS |
| **SHARP** | ✓ (only 1 per scene) | Only to merge views (focal length helps) | Only to fit the depth scale when merging | No per-scene optimization |

The common denominator is the **COLMAP format**: `images/` + `sparse/0/{cameras,images,points3D}.bin`.
Every method can read it, so every dataset in the comparison should be in it. Then all
methods get **exactly the same images, poses, sparse points, and held-out test views**
(every 8th image, which is what 3DGS's `--eval` uses).

### Which data to compare on

1. **Our own captures (the main result).** Real images from our current setup, in COLMAP
   format. This answers "how do these methods do on *our* inputs." If our setup already
   produces poses and a point cloud, export them to COLMAP format rather than re-running
   COLMAP, so every method sees the same sparse cloud our pipeline sees.
2. **One public real scene (validation).** Tanks&Temples `truck` or `train` (650 MB, from the
   3DGS authors). 3DGS and SOG publish numbers on these scenes. Matching those numbers shows
   the installs work, and the results become comparable with the literature.
3. **Synthetic (controlled experiments, optional).** NeRF-Synthetic (8 rendered objects with
   exact poses) is clean, but it ships without a point cloud. Use it for things you can only
   control synthetically, such as sweeping point-cloud density or view count, which matter
   for a sparse-point-cloud method. A COLMAP point cloud has to be triangulated from the
   known poses first (not automated yet). Without a point cloud, SHARP can only be timed
   per image.

```powershell
pip install gdown psutil
python scripts/download_data.py tandt_db
python scripts/download_data.py nerf_synthetic
```

## Setup (Windows, RTX 3080 12 GB)

Each method has conflicting dependencies (different CUDA rasterizer forks, Python 3.10 vs 3.13), so
each gets its own conda env. The harness runs in plain Python ≥ 3.11 and calls into them.

**0. Prerequisites:** [Miniforge](https://github.com/conda-forge/miniforge) (conda), CUDA Toolkit 11.8
(installed), VS 2022 Build Tools (installed).

> **Compiler gotcha:** CUDA 11.8 refuses MSVC 14.40+ (`_MSC_VER >= 1940`). Your machine has
> both 14.44 and 14.38, so load the **14.38** toolset before building any CUDA extension (step 2).

**1. Clone the methods** at pinned commits:
```powershell
powershell -File scripts/clone_methods.ps1
```

**2. Build environments and write `bench_config.toml`.** Open a *Miniforge Prompt* in the repo root:
```bat
scripts\setup_envs.bat          :: 3DGS + SOG  (add "sharp" to also build SHARP)
```
It loads the 14.38 compiler, creates the `gs` / `sogs` (/ `sharp`) envs, writes
`bench_config.toml`, and finishes by importing each env's CUDA rasterizer as a smoke test.
Re-running skips envs that already exist. The individual commands are in
[scripts/setup_envs.bat](scripts/setup_envs.bat) if one step needs debugging.

### Linux (lab machine)

```bash
bash scripts/clone_methods.sh       # same pinned commits as clone_methods.ps1
bash scripts/setup_envs.sh          # 3DGS + SOG  (add "sharp" to also build SHARP)
```

CUDA 11.8 (which the `gs` and `sogs` envs are built against) only accepts gcc 11 or older.
The script uses `/usr/local/cuda-11.8` if it exists and the system gcc is 11 or older.
Otherwise it installs CUDA 11.8 + gcc 11 into a separate `cuda118` conda env and builds
with those. It reads the GPU architecture from `nvidia-smi`, and writes the `quasar-npbg`
env's python into `bench_config.toml` as `npbg` when it finds that env.

## Running

```powershell
# Close Chrome, Wallpaper Engine, games: GPU numbers are device-wide.
python bench/run.py 3dgs  data/tandt_db/tandt/truck
python bench/run.py sog   data/tandt_db/tandt/truck
python bench/run.py sharp data/tandt_db/tandt/truck --views 4     # 4 photos -> 4 scenes -> merged
python bench/run.py summary            # -> results/summary.csv, results/summary.md
```

- Quick smoke test: `--iterations 3000 --name lego_smoke`.
- Extra flags go to the method after `--`, e.g. faster 3DGS:
  `python bench/run.py 3dgs data/nerf_synthetic/lego --name lego_accel -- --optimizer_type sparse_adam`
  (requires the `3dgs_accel` rasterizer branch, see the 3DGS README).
- Re-score an existing run: `python bench/run.py eval sog lego`.

Outputs: `results/<method>/<scene>/` holds `model/`, `train.log`, `eval.json`, and `summary.json`.

## Fairness notes

- **Timed training runs skip intermediate test evaluations** (3DGS `--test_iterations -1`, SOG
  `run.test_iterations=[]`). Without this, each method's built-in evaluation would add time
  unevenly. SOG's compression step is included, because it is part of that method.
- **COLMAP preprocessing is excluded from `end_to_end_s`.** It is shared input that every
  method (including ours) consumes. If our system computes poses and points itself, time that
  separately and report it next to the table.
- **SHARP merge quality:** check `scale_log_mad` in its summary. Above ~0.15 means SHARP's
  depth disagrees with the sparse points, so the views will not line up well.
- **Identical iteration counts and data splits:** 30k iterations, official train/test split
  (`--eval`), white background for synthetic scenes.
- **Repeat timing runs** (at least 2× on one scene). Wall clock varies a few percent between runs.
- **SHARP gets the true focal length** from COLMAP (or `transforms_test.json` for synthetic data).
  Otherwise it guesses from EXIF data.
- SOG logs to Weights & Biases by default; the harness sets `WANDB_MODE=disabled`. Use `--wandb-mode offline` if that errors.

## Layout

```
bench/run.py                  benchmark driver (train/predict -> measure -> evaluate -> summary)
bench/monitor.py              subprocess wrapper: wall time, GPU mem/util/power, RAM
bench/eval_render.py          shared evaluator (runs in the gs env)
bench/sharp_merge.py          SHARP: K photos -> K scenes -> aligned + merged .ply (runs in the sharp env)
bench/sharp_predict_timed.py  SHARP per-image timing when no poses/points exist (runs in the sharp env)
scripts/clone_methods.ps1     clone method repos at pinned commits
scripts/download_data.py      NeRF-Synthetic and T&T/DB downloads
external/                     method repos (git-ignored)
data/, results/               datasets and outputs (git-ignored, except results/summary.*)
```
