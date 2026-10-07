"""SHARP multi-view merge: predict one scene per photo for K photos, place each in the shared
world frame, and concatenate them into one 3DGS .ply that the shared evaluator can score.

SHARP outputs each scene in its own camera frame (OpenCV axes, camera at the origin) with its
own depth scale. For each chosen view we
  1. take the camera pose + focal length from COLMAP (focal is passed to SHARP),
  2. fit a scale s = median(COLMAP depth / SHARP depth) over the sparse points that image
     observed, comparing at their keypoint pixels,
  3. map Gaussians to world, x_w = R_c2w (s * x_c) + t_c2w, covariances included.
No refinement or de-duplication: where views overlap, both copies are kept. That is the
naive baseline; anything smarter belongs in a separate, clearly-labeled variant.

Run inside the `sharp` env (bench/run.py does this for you):
    python bench/sharp_merge.py --scene <COLMAP scene> -o <out dir> --views 4
"""

import argparse
import importlib.util
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch

from sharp.cli.predict import DEFAULT_MODEL_URL, predict_image
from sharp.models import PredictorParams, create_predictor
from sharp.utils.gaussians import Gaussians3D, apply_transform, save_ply
from sharp_predict_timed import load_image

GS_ROOT = Path(__file__).resolve().parents[1] / "external" / "gaussian-splatting"


def colmap_loader():
    # Load 3DGS's pure-numpy COLMAP reader by file; importing the `scene` package would pull in CUDA extensions.
    spec = importlib.util.spec_from_file_location("colmap_loader", GS_ROOT / "scene" / "colmap_loader.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_points3d_with_ids(path: Path) -> dict[int, np.ndarray]:
    """points3D.bin -> {point3D_id: xyz}. (3DGS's reader drops the ids, which we need.)"""
    pts = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            pid, x, y, z, _r, _g, _b, _err = struct.unpack("<QdddBBBd", f.read(43))
            (track_len,) = struct.unpack("<Q", f.read(8))
            f.seek(8 * track_len, 1)
            pts[pid] = np.array([x, y, z])
    return pts


def focal_of(cam) -> float:
    if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        return cam.params[0]
    return 0.5 * (cam.params[0] + cam.params[1])  # PINHOLE, OPENCV, ...


def train_views(images: dict, llffhold: int = 8) -> list:
    """Training images under the same LLFF hold-out the 3DGS loader uses with --eval."""
    names = sorted(img.name for img in images.values())
    test = {n for i, n in enumerate(names) if i % llffhold == 0}
    return sorted((img for img in images.values() if img.name not in test), key=lambda im: im.name)


def farthest_views(views: list, k: int, qvec2rotmat) -> list:
    """Spread k views around the scene: farthest-point sampling on camera centers."""
    centers = np.stack([-qvec2rotmat(v.qvec).T @ v.tvec for v in views])
    chosen = [0]
    dist = np.linalg.norm(centers - centers[0], axis=1)
    while len(chosen) < min(k, len(views)):
        chosen.append(int(dist.argmax()))
        dist = np.minimum(dist, np.linalg.norm(centers - centers[chosen[-1]], axis=1))
    return [views[i] for i in chosen]


def fit_scale(means_c: np.ndarray, opac: np.ndarray, f: float, w: int, h: int,
              kp_xy: np.ndarray, kp_depth: np.ndarray, cell: int = 4) -> tuple[float, int, float]:
    """Median ratio of COLMAP depth to SHARP's front-surface depth at the keypoint pixels."""
    z = means_c[:, 2]
    keep = (z > 1e-6) & (opac > 0.3)
    u = f * means_c[keep, 0] / z[keep] + w / 2
    v = f * means_c[keep, 1] / z[keep] + h / 2
    gw, gh = math.ceil(w / cell), math.ceil(h / cell)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx = (v[inside] // cell).astype(np.int64) * gw + (u[inside] // cell).astype(np.int64)
    zbuf = np.full(gw * gh, np.inf)
    np.minimum.at(zbuf, idx, z[keep][inside])

    kx, ky = kp_xy[:, 0], kp_xy[:, 1]
    ok = (kx >= 0) & (kx < w) & (ky >= 0) & (ky < h) & (kp_depth > 0)
    z_sharp = zbuf[(ky[ok] // cell).astype(np.int64) * gw + (kx[ok] // cell).astype(np.int64)]
    ratio = kp_depth[ok] / z_sharp
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if len(ratio) < 10:
        raise RuntimeError(f"Only {len(ratio)} sparse points matched SHARP depth; cannot fit scale.")
    s = float(np.median(ratio))
    spread = float(np.median(np.abs(np.log(ratio / s))))  # ~relative depth disagreement after scaling
    return s, len(ratio), spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True, help="COLMAP scene (images/ + sparse/0/)")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--names", nargs="*", help="explicit image names instead of automatic selection")
    ap.add_argument("-c", "--checkpoint", type=Path, default=None)
    args = ap.parse_args()
    t_start = time.perf_counter()

    cl = colmap_loader()
    sparse = args.scene / "sparse" / "0"
    images = cl.read_extrinsics_binary(str(sparse / "images.bin"))
    cameras = cl.read_intrinsics_binary(str(sparse / "cameras.bin"))
    points = read_points3d_with_ids(sparse / "points3D.bin")
    by_name = {img.name: img for img in images.values()}
    views = [by_name[n] for n in args.names] if args.names else \
        farthest_views(train_views(images), args.views, cl.qvec2rotmat)

    device = torch.device("cuda")
    t0 = time.perf_counter()
    ckpt = args.checkpoint or Path(torch.hub.get_dir()) / "checkpoints" / DEFAULT_MODEL_URL.rsplit("/", 1)[-1]
    if not ckpt.exists():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(DEFAULT_MODEL_URL, str(ckpt))
    model = create_predictor(PredictorParams())
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval().to(device)
    predict_image(model, np.zeros((512, 512, 3), np.uint8), 512.0, device)  # warm-up
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()

    args.output.mkdir(parents=True, exist_ok=True)
    placed, per_view = [], []
    for view in views:
        cam = cameras[view.camera_id]
        image, _ = load_image(args.scene / "images" / view.name, white_bg=False)
        h, w = image.shape[:2]
        sx = w / cam.width  # images/ may be downsampled relative to the COLMAP camera
        f_px = focal_of(cam) * sx

        torch.cuda.synchronize()
        t = time.perf_counter()
        g = predict_image(model, image, f_px, device)
        torch.cuda.synchronize()
        predict_s = time.perf_counter() - t

        t = time.perf_counter()
        R = cl.qvec2rotmat(view.qvec)  # world -> camera (COLMAP = OpenCV axes, same as SHARP)
        seen = [(xy, points[pid]) for xy, pid in zip(view.xys, view.point3D_ids) if pid != -1 and pid in points]
        kp_xy = np.array([xy for xy, _ in seen]) * sx
        kp_depth = np.array([(R @ X + view.tvec)[2] for _, X in seen])
        means_c = g.mean_vectors[0].float().cpu().numpy()
        opac = g.opacities.reshape(-1).float().cpu().numpy()
        s, n_match, spread = fit_scale(means_c, opac, f_px, w, h, kp_xy, kp_depth)

        to_world = torch.zeros(3, 4)
        to_world[:, :3] = torch.from_numpy(R.T * s)
        to_world[:, 3] = torch.from_numpy(-R.T @ view.tvec)
        placed.append(apply_transform(g, to_world.float().to(device)).to("cpu"))
        align_s = time.perf_counter() - t

        per_view.append({"image": view.name, "focal_px": f_px, "predict_s": predict_s, "align_s": align_s,
                         "scale": s, "n_scale_matches": n_match, "scale_log_mad": spread,
                         "n_gaussians": int(g.mean_vectors.shape[1])})
        print(f"{view.name}: predict {predict_s * 1000:.0f} ms, scale {s:.4f} from {n_match} pts (log-MAD {spread:.3f})")
        del g

    t = time.perf_counter()
    merged = Gaussians3D(**{k: torch.cat([getattr(p, k) for p in placed], dim=1) for k in Gaussians3D._fields})
    ply = args.output / "merged.ply"
    save_ply(merged, per_view[0]["focal_px"], image.shape[:2], ply)
    merge_save_s = time.perf_counter() - t

    summary = {
        "views": [v.name for v in views],
        "model_load_s": load_s,
        "checkpoint_bytes": ckpt.stat().st_size,
        "model_params": sum(p.numel() for p in model.parameters()),
        "predict_total_s": sum(v["predict_s"] for v in per_view),
        "align_total_s": sum(v["align_s"] for v in per_view),
        "merge_save_s": merge_save_s,
        "script_total_s": time.perf_counter() - t_start,
        "peak_alloc_mib": torch.cuda.max_memory_allocated() / 2**20,
        "n_gaussians": int(merged.mean_vectors.shape[1]),
        "ply_bytes": ply.stat().st_size,
        "per_view": per_view,
    }
    (args.output / "sharp_merge.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_view"}, indent=2))


if __name__ == "__main__":
    main()
