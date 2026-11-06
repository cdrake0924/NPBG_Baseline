"""Shared evaluator: renders the test views of ANY 3DGS-format .ply with the reference
3DGS rasterizer and reports image quality plus render speed.

Every method is scored by this one script, so differences come from the models and not
from different renderers or metric code. Metrics are computed on 8-bit quantized
renders, matching 3DGS's metrics.py (which reads saved PNGs). NPBG's metrics (npbg_*) are
computed on the same renders by bench/scoring.py; if the scene has coverage/<image>.png masks
(scenes exported by npbg_to_colmap.py), the npbg_*_covered scores use them.

Run inside the `gs` env (bench/run.py does this for you):
    python bench/eval_render.py --ply <model.ply> --source <dataset scene> --out eval.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

GS_ROOT = Path(__file__).resolve().parents[1] / "external" / "gaussian-splatting"
sys.path.insert(0, str(GS_ROOT))

from plyfile import PlyData  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene.dataset_readers import sceneLoadTypeCallbacks  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from utils.camera_utils import cameraList_from_camInfos  # noqa: E402
from scoring import Scorer, load_mask, mean_scores  # noqa: E402

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SEPARATE_SH = True  # accelerated rasterizer branch
except ImportError:
    SEPARATE_SH = False


def sh_degree_of(ply_path: Path) -> int:
    props = PlyData.read(str(ply_path)).elements[0].properties
    n_rest = sum(p.name.startswith("f_rest_") for p in props)
    return int(round(math.sqrt(n_rest / 3 + 1))) - 1


def load_test_cameras(source: Path, white: bool, resolution: int):
    export = source / "npbg_export.json"
    split = json.loads(export.read_text()) if export.exists() else {}
    if split.get("split_mode") == "explicit":
        # NPBG's own test views (npbg_to_colmap.py): load every camera, keep the listed ones
        info = sceneLoadTypeCallbacks["Colmap"](str(source), "images", "", False, False)
        wanted = {Path(n).stem for n in split["test_images"]}
        cams = [c for c in info.train_cameras if Path(c.image_name).stem in wanted]
        if len(cams) != len(wanted):
            raise SystemExit(f"found {len(cams)} of {len(wanted)} test images in {source}")
        info = info._replace(test_cameras=cams)
    elif (source / "sparse").exists():
        info = sceneLoadTypeCallbacks["Colmap"](str(source), "images", "", True, False)
    elif (source / "transforms_train.json").exists():
        info = sceneLoadTypeCallbacks["Blender"](str(source), white, "", True)
    else:
        raise SystemExit(f"Unrecognized dataset layout: {source}")
    cam_args = SimpleNamespace(resolution=resolution, data_device="cpu", train_test_exp=False)
    return cameraList_from_camInfos(info.test_cameras, 1.0, cam_args, info.is_nerf_synthetic, True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--white", action="store_true", help="white background (NeRF-Synthetic)")
    ap.add_argument("-r", "--resolution", type=int, default=-1)
    ap.add_argument("--fps-repeats", type=int, default=3)
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--save-renders", type=Path, default=None)
    ap.add_argument("--npbg-root", type=Path, required=True, help="NPBG checkout (for its metrics.py)")
    args = ap.parse_args()

    cams = load_test_cameras(args.source, args.white, args.resolution)
    sh = sh_degree_of(args.ply)
    t0 = time.perf_counter()
    gaussians = GaussianModel(sh)
    gaussians.load_ply(str(args.ply))
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    bg = torch.tensor([1.0, 1.0, 1.0] if args.white else [0.0, 0.0, 0.0], device="cuda")

    def draw(cam):
        return render(cam, gaussians, pipe, bg, separate_sh=SEPARATE_SH)["render"]

    scorer = Scorer(args.npbg_root, lpips=not args.no_lpips)
    rows = []
    if args.save_renders:
        import torchvision
        args.save_renders.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # End-to-end component: one cold pass over the test views, as a user would first see it
        t0 = time.perf_counter()
        for cam in cams:
            draw(cam)
        torch.cuda.synchronize()
        render_test_set_s = time.perf_counter() - t0

        # Quality
        for i, cam in enumerate(cams):
            img = draw(cam)
            gt = cam.original_image[:3].cuda()
            stem = Path(cam.image_name).stem  # this 3DGS version keeps the extension in image_name
            mask = load_mask(args.source / "coverage" / f"{stem}.png", (cam.image_width, cam.image_height), "cuda")
            rows.append({"image": stem, **scorer.score(img, gt, mask)})
            if args.save_renders:
                torchvision.utils.save_image(img.clamp(0, 1), args.save_renders / f"{stem}.png")

        # Speed: rasterization only (no I/O), after warm-up
        for cam in cams[:10]:
            draw(cam)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(args.fps_repeats):
            for cam in cams:
                draw(cam)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    n_frames = len(cams) * args.fps_repeats
    result = {
        "ply": str(args.ply),
        "source": str(args.source),
        "n_test_views": len(cams),
        "resolution_wh": [cams[0].image_width, cams[0].image_height],
        "n_gaussians": int(gaussians.get_xyz.shape[0]),
        "sh_degree": sh,
        **mean_scores([{k: v for k, v in r.items() if k != "image"} for r in rows]),
        "per_view": rows,
        "load_s": load_s,
        "render_test_set_s": render_test_set_s,
        "render_fps": n_frames / dt,
        "render_ms_per_frame": 1000 * dt / n_frames,
        "render_peak_alloc_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
