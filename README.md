# NPBG Baseline Benchmark

Measures **speed, compute, and file size** (plus image quality, so the other numbers mean
something) for three Gaussian-splatting methods and our NPBG pipeline, under identical
conditions: same images, poses, point cloud, test views, and metric code.

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

The three Gaussian methods write standard 3DGS `.ply` files. **One shared evaluator**
([bench/eval_render.py](bench/eval_render.py)) renders every model with the same rasterizer.
NPBG renders with its own renderer ([bench/npbg_render.py](bench/npbg_render.py)), and
[bench/score_renders.py](bench/score_renders.py) scores those renders. **Every method is scored by
the same code** ([bench/scoring.py](bench/scoring.py)) against the same 8-bit ground truth, so
differences come from the methods and not from evaluation code.

Two sets of quality metrics are reported for every method:

| Set | Columns | Definition |
|---|---|---|
| **NPBG (headline)** | `eval_npbg_psnr_full`, `_ssim_full`, `_lpips_full`, `_psnr_covered`, `_ssim_covered` | NPBG's own `metrics.evaluate_metrics`, loaded from the NPBG checkout: float render, reflect-padded SSIM, **LPIPS AlexNet**. `_covered` = only pixels inside the record's geometry mask (`coverage/`; NPBG scenes only) |
| 3DGS (literature check) | `eval_psnr`, `eval_ssim`, `eval_lpips` | Render rounded to 8 bits, 3DGS's `psnr`/`ssim`, **LPIPS VGG**: what the 3DGS and SOG papers report, so T&T numbers can be checked against them |

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
| **Quality** | PSNR ↑, SSIM ↑, LPIPS ↓ | On held-out test views, both metric sets above |

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
- Re-score an existing run: `python bench/run.py eval sog lego` (also `3dgs`, `sharp`, `npbg`).
- `--gpu N` on any method picks the card (see [Using both GPUs](#using-both-gpus)).
- The harness itself needs Python ≥ 3.11 with `psutil`. On the lab machine: `.venv/bin/python bench/run.py ...`.

Outputs: `results/<method>/<scene>/` holds `model/`, `train.log`, `eval.json`, and `summary.json`.

## NPBG

NPBG scenes live in NPBG's own format (`manifest.json`, `point_cloud.npz`, `records/`). To compare
on them, export the scene to COLMAP once. All four methods then read the same files:

```bash
# 1. NPBG scene -> COLMAP (images, coverage masks, poses, NPBG's point cloud, NPBG's train/test split)
python bench/run.py export-npbg /data/points/npbg_datasets/rig_test_scene_e2e_p10_s1_kept_avg data/npbg/rig_e2e_p10

# 2. Run every method on it (full resolution is automatic for exported scenes)
python bench/run.py npbg  data/npbg/rig_e2e_p10 --gpu 0
python bench/run.py 3dgs  data/npbg/rig_e2e_p10 --gpu 1
python bench/run.py sog   data/npbg/rig_e2e_p10 --gpu 0
python bench/run.py sharp data/npbg/rig_e2e_p10 --gpu 1 --name rig_e2e_p10   # merges the 3 training photos
python bench/run.py summary

# Synthetic QUASAR scene with a per-scene trained NPBG checkpoint; --build-s records its training time
python bench/run.py export-npbg /data/points/npbg_datasets/quasar_robot_lab_p05 data/npbg/robot_lab_p05
python bench/run.py npbg data/npbg/robot_lab_p05 --checkpoint /data/points/npbg_runs/fast0_robot_lab_p05/best_p2M.pt \
    --build-s 480.3 --build-source "tensorboard wall-clock span of the training run"
```

What the exporter writes ([bench/npbg_to_colmap.py](bench/npbg_to_colmap.py)):
- `images/`: each record's `target_rgb`, i.e. the exact image NPBG trains/scores against (the
  rig's records are undistorted, so they differ from the raw capture PNGs).
- `sparse/0/`: one PINHOLE camera from the manifest intrinsics; NPBG's OpenGL view matrices
  converted to COLMAP world-to-camera; **NPBG's own point cloud** as `points3D`, coloured from
  the training photos only. Each point is listed as observed by the training views where it is
  the front surface (SHARP uses these to fit its depth scale). The test view observes nothing,
  so no test information reaches the baselines.
- `coverage/`: the record coverage masks, for the `_covered` metrics.
- **Split:** NPBG's own train/validation split, kept exactly (`split_mode` in `npbg_export.json`):
  - `llff`: 3DGS, SOG and SHARP hold out sorted-name index % 8 == 0. When NPBG's validation records
    can be named to land exactly there (the rig: `bottom_left` sorts first), the scene is used as is.
  - `explicit`: otherwise (QUASAR: 12 validation of 120, every 10th), `train/` holds a second model
    with only the training images (symlinks). The baselines train on it with no hold-out (3DGS
    without `--eval`, SOG `dataset.eval=false`, SHARP `--holdout 0`), and the evaluator scores the
    full scene's `test_images`.
- Point observations are capped at 20,000 random visible points per training image
  (`--max-observations`). Only SHARP's scale fit reads them, and a full list would make
  `images.bin`, which 3DGS/SOG parse inside their timed run, needlessly large.

**NPBG timing** (`npbg` command): load the finished scene (point cloud + descriptors + decoder) to
the GPU, then one cold render pass over the test views. The scene build is measured separately and
passed in with `--build-s` (recorded as `npbg_build_s` and added to `end_to_end_s`):
- rig e2e: `build_rig_oneshot.py` + `colour_calibrate.py` from FoundationStereo depth on disk,
  18.4 s on GPU 0 (FoundationStereo itself is excluded, as COLMAP is for the others);
- QUASAR: the per-scene training run's wall clock.

The `train_*` columns stay empty for NPBG. The 3DGS/SOG columns time optimization from images + poses + points.
That process's GPU/RAM stats are stored as `run_*` in its `summary.json`. Rendering follows NPBG's
`eval.py`: fp32 (the viewer's `--bf16` is faster).

**Size:** `model_mb` = what rendering needs: point positions (float32) + per-point descriptors + decoder
weights. The files on disk can hold more (a training checkpoint keeps optimizer state, and
`point_cloud.npz` keeps capture features the renderer doesn't read); their sizes are in
`summary.json` as `scene_files_bytes`, next to `descriptor_bytes` and `decoder_bytes`. `checkpoint_mb` = the network that builds scenes (`[npbg] network`, default the e2e
model), reported like SHARP's weights.

## Using both GPUs

`--gpu N` (nvidia-smi numbering) pins a run to one card. The method, the evaluator and the GPU
monitor all use that card (`CUDA_DEVICE_ORDER=PCI_BUS_ID` keeps CUDA's numbering the same as
nvidia-smi's). Run two methods at once in two shells, one per GPU. Training a single run across both
cards isn't supported: 3DGS and SOG are single-GPU methods. Parallel runs still share CPU, disk
and memory bandwidth, so wall-clock times can shift by a few percent. For headline timings, run
each method alone, or under the same conditions for every method. GPU numbers are device-wide, so
check `nvidia-smi` for other users' jobs first.

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
bench/scoring.py              both metric sets (NPBG's metrics.py + 3DGS's); used by every evaluator
bench/npbg_to_colmap.py       NPBG scene -> COLMAP scene with NPBG's split, cloud and coverage masks
bench/npbg_render.py          NPBG: load finished scene + render test views, timed (runs in quasar-npbg)
bench/score_renders.py        scores saved renders (NPBG) with bench/scoring.py (runs in the gs env)
bench/npbg_profile.py         NPBG per-frame time split: rasterize vs decoder, fp32 vs bf16, decoder GFLOPs
bench/compare_images.py       every method's test renders + error maps + per-view scores, for a viewer
report/                       comparison page: index.html (open in a browser) + img/; rebuild with
                              `.venv/bin/python report/build_report.py` after new runs
scripts/clone_methods.ps1     clone method repos at pinned commits
scripts/download_data.py      NeRF-Synthetic and T&T/DB downloads
external/                     method repos (git-ignored)
data/, results/               datasets and outputs (git-ignored, except results/summary.*)
```
