# What the baseline results show, and why

Results from the lab machine (2× RTX 4090), November 2026. The interactive comparison is
[index.html](index.html), and the raw numbers are in [../results/summary.md](../results/summary.md).
Every method gets the same images, poses, point cloud and held-out views. Every method is scored by
NPBG's own `metrics.py` (PSNR/SSIM, LPIPS with AlexNet) against the same 8-bit ground truth.

## The numbers

**Rig capture** (real; 3 training cameras, held-out `bottom_left`, 1920×1080; NPBG = e2e one-shot model)

| | PSNR | SSIM | LPIPS | PSNR covered | Inputs → first view | Scene size | Render |
|---|---|---|---|---|---|---|---|
| NPBG e2e | **22.95** | **0.722** | 0.254 | **25.02** | **19.0 s** | 65.5 MB | 20 fps (35 fps viewer settings) |
| SOG | 22.26 | 0.678 | **0.247** | 24.68 | 1011 s | **10.7 MB** | **311 fps** |
| 3DGS | 20.99 | 0.638 | 0.328 | 22.46 | 550 s | 78.2 MB | 253 fps |
| SHARP (3 photos merged) | 19.53 | 0.522 | 0.282 | 20.15 | 56 s | 198 MB | 193 fps |

**Robot lab** (synthetic QUASAR render; 108 training views, 12 test views, 960×540; NPBG = classic, trained on this scene)

| | PSNR | SSIM | LPIPS | Inputs → first view | Scene size | Render |
|---|---|---|---|---|---|---|
| NPBG | 23.40 | 0.719 | 0.168 | **481 s** | 86.0 MB | 54 fps |
| SOG | 25.87 | 0.815 | 0.140 | 1424 s | **32.5 MB** | **244 fps** |
| 3DGS | **27.42** | **0.875** | **0.089** | 1064 s | 628 MB | 122 fps |
| SHARP (4 photos merged) | 14.79 | 0.356 | 0.574 | 60 s | 264 MB | 106 fps |

**Install check, Tanks & Temples truck** (3DGS-style metrics, as the papers report them): 3DGS
25.45 / 0.885 / 0.142 and SOG 25.3 / 0.877 / 0.149. These match the earlier Windows runs
(25.46 / 0.885 / 0.142 and 25.5 / 0.875 / 0.151) and the 3DGS paper's truck numbers
(≈25.19 / 0.879 / 0.148). The baseline installs are working, so the differences below come from
the methods, not from broken setups.

## 1. NPBG e2e reaches a first image ~30–50× faster than 3DGS and SOG

**Expected:** a comparable build time, or NPBG slower because it carries a neural network.
**Seen:** 19 s for NPBG e2e, against 550 s for 3DGS and 1010 s for SOG.

**Why:** the methods do different kinds of work to build a scene.
- **3DGS and SOG optimise each scene from scratch:** 30,000 gradient steps, and SOG then sorts and
  compresses the result. Their time is spent training.
- **NPBG e2e is feed-forward.** Its network was trained once, on other scenes. Building this scene
  is one pass: read the FoundationStereo depth, keep 10% of each camera's pixels, predict a
  descriptor per point, then fit a 3×3 colour matrix on the training cameras. This was timed by
  rebuilding the scene from scratch: `build_rig_oneshot.py` 15.35 s + `colour_calibrate.py`
  3.06 s = 18.4 s. The rebuild reproduced the original scene byte for byte. Loading the scene and
  the first render add 0.6 s.
- SHARP is the other feed-forward method, and it lands in the same range (56 s).

**What this does not show:** FoundationStereo depth is NPBG's input and is not timed, just as COLMAP
is not timed for the others. When NPBG is trained per scene (robot lab), its build is optimisation
too: 480 s, the same order as 3DGS (1064 s) and SOG (1424 s). The speed comes from the one-shot
e2e design, not from point-based rendering itself.

## 2. NPBG renders 5–15× slower than the Gaussian methods

**Expected:** real-time rendering on a 4090, in the range of the splatting methods.
**Seen:** 20 fps for NPBG on the rig, against 190–310 fps for the Gaussian methods; 54 fps on
robot lab, against 106–244 fps.

**Why:** three things, measured with `bench/npbg_profile.py` (`results/profile/`).

1. **NPBG runs a neural network on every pixel of every frame, and the e2e network is large.**
   The Gaussian methods only rasterize, in one fused CUDA kernel, so a 1080p frame takes about 4 ms.
   NPBG rasterizes and then decodes the whole frame with a gated U-Net. The e2e decoder is twice
   as wide as NPBG's earlier one (`feature_scale` 2 vs 4, 16-dim vs 8-dim descriptors): 7.7M
   parameters against 1.9M, about 1.1 trillion operations per 1080p frame. On the rig the
   decoder is 39.6 of the 48.5 ms per frame (82%). Decoder cost grows with output resolution and
   network width, not with scene size.
2. **NPBG's rasterizer is plain PyTorch.** `rasterize.py` loops in Python over 5 pyramid levels ×
   9 neighbour offsets (point radius 1): 45 `scatter_reduce_` passes over every point, each its
   own kernel launch. With few points this is cheap (8.7 ms for 457k points on the rig). On robot
   lab, with 1.78M points, it is 15 of the 19 ms per frame. That scene is rasterizer-bound, so
   decoder speed-ups don't help there.
3. **The benchmark renders the way NPBG's `eval.py` scores:** fp32, default memory layout, decoder
   not compiled. NPBG's viewer and live pipeline use `channels_last`, bf16 and (live)
   `torch.compile`. On the rig those settings take the frame from 48.5 ms to 28.6 ms
   (**35 fps**). On robot lab they don't help (19 ms either way). Compiling the decoder there
   actually made the full frame slower (44 ms), which was not investigated. The scoreboard keeps
   the fp32 speed because the quality scores come from that path; NPBG's own notes say bf16
   gives "a different number from eval".

**What would close the gap:** a smaller decoder, or decoding at a lower render scale (the viewer
already has a scale knob; `RENDER_PERFORMANCE.md` shows the decoder dropping from 14.7 ms at
1080p to 3.3 ms at 540p in bf16). Beyond that, a fused CUDA rasterizer for the index pyramid.
Even at viewer settings, NPBG stays several times slower than splatting, because splatting runs
no network.

## 3. On the rig, NPBG e2e beats the Gaussian methods; on robot lab it loses clearly

**Expected:** one method ranking consistently across scenes.
**Seen:** NPBG first on the rig (22.95 dB), but third on robot lab (23.4 dB vs 27.4 for 3DGS),
and behind 3DGS and SOG on all 12 robot-lab test views.

**Why:** the two scenes test different regimes, and they use different NPBG models.
- **View count.** The rig gives the per-scene methods 3 photos from nearly the same position
  (the cameras are ~35 × 52 cm apart). 3DGS overfits them: it leaves floaters and smears
  across the open centre of the room, where the training views disagree or don't reach. SOG
  regularises more and comes close to NPBG. NPBG e2e does not fit the scene at all; its
  predicted descriptors come from a network trained on many scenes, which makes it robust to
  having only 3 views. Robot lab gives 108 views around the scene, the regime 3DGS was designed
  for.
- **View-dependent appearance.** NPBG's decoder sees only per-point descriptors rasterized into
  the view (`model.py`, `NeuralPointRenderer.forward`), with no viewing direction. The Gaussians
  carry degree-3 spherical harmonics, a colour that changes with view direction. Robot lab has
  glossy floors with reflections. The Gaussian methods reproduce them; NPBG's renders are
  softer and lose them (see the error maps in the viewer).
- **Different NPBG models.** The rig uses the e2e one-shot model. Robot lab uses classic NPBG
  trained for 10,000 steps on 512 px crops (`fast0_robot_lab_p05`, 480 s), against 30,000
  steps for 3DGS and SOG. Longer training was not tried here, so how much of the gap it would
  close is unknown.

**Caveats:** the rig has one test view. Treat its ~0.7 dB lead over SOG as indicative until it is
repeated on more frames (`~/calib_2026-09-29/dev_scene` is a second capture). NPBG's visible
weaknesses on the rig are the near cart and drone, which few points reach, and a pale band along
the left and bottom edges where no point lands.

## 4. SHARP scores lowest even where it looks sharp

**Expected:** a feed-forward method to give a usable fast baseline.
**Seen:** the lowest scores on both scenes, and 14.8 dB on robot lab.

**Why:** SHARP predicts each photo's scene on its own, in its own depth scale. The harness places
each prediction with the photo's pose, fits a depth scale against the sparse points, and
concatenates the results with no refinement (`bench/sharp_merge.py`). On the rig the scale fits
are good (log-MAD 0.04–0.05) and the render looks plausible. But small misalignments between
the 3 merged predictions cost PSNR everywhere. On robot lab the 4 photos are far apart, one fit
is at 0.16 log-MAD (past the 0.15 warning line in the README), and the merged scene falls apart
near the camera. These scores measure this naive merge, not SHARP's single-view quality.

**Open question:** SHARP's prediction took about 4.3 s per image with the GPU mostly idle. That is
slower than expected on a 4090, and it inflates SHARP's time. It was not investigated.

## 5. Sizes

SOG is smallest on both scenes (10.7 MB and 32.5 MB), because it compresses its Gaussians into
JPEG XL image grids. NPBG's 65–86 MB is points (float32 positions) + per-point descriptors
(float32) + decoder weights. Most of it is the descriptors, which NPBG's own compact-stream
format stores in 8 bits (`REALTIME_PIPELINE.md`), so NPBG's size has room to shrink. 3DGS on robot
lab is 628 MB (2.5M Gaussians at 248 bytes each). The e2e network itself (156 MB) and SHARP's
network (2.8 GB) are listed separately as network weights. They build any scene and are not
part of one.

## Measurement caveats

- Some timed runs overlapped:
  - 3DGS truck ran while the SOG env compiled and while 3DGS trained on the rig;
  - SOG on truck and on the rig ran together, one per GPU;
  - SHARP on truck and on the rig ran together;
  - robot lab's 3DGS and SOG ran together.

  Parallel runs share CPU, disk and memory bandwidth. Rerun one at a time before quoting wall
  clock as headline numbers.
- NPBG's robot lab build time (480.3 s) comes from the wall-clock span of the training run's
  TensorBoard log (steps 100–10,000). It slightly undercounts: it misses dataset import and the
  first 100 steps.
- NPBG's quality scores use the fp32 render. A bf16 render (viewer settings) would score slightly
  differently. It hasn't been scored.
- NPBG's training-time validation of the same checkpoint (`fast0_robot_lab_p05/best_p2M.pt`, step
  8,500) reported 23.18 dB, and the benchmark measures 23.40 dB. Rounding the ground truth to 8 bits
  is far too small to explain 0.2 dB, so the gap is probably in how the training-time validation
  renders or averages. It was not investigated. On the rig, the benchmark reproduces NPBG's
  published e2e score exactly (22.95 / 0.722 / 0.254).

## Reproduce

```bash
.venv/bin/python bench/run.py export-npbg <npbg dataset> data/npbg/<name>
.venv/bin/python bench/run.py {npbg,3dgs,sog,sharp} data/npbg/<name> --gpu N   # see the README for NPBG flags
~/miniconda3/envs/quasar-npbg/bin/python bench/npbg_profile.py --compile --scene data/npbg/<name> \
    --npbg-root ~/NPBG --out results/profile/npbg_<key>.json
.venv/bin/python bench/run.py summary
.venv/bin/python report/build_report.py
```
