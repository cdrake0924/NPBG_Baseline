"""Timed wrapper around Apple SHARP prediction (single image -> 3D Gaussians).

`sharp predict` times nothing and its wall time is dominated by model loading, so this
loads the model once, warms up, then times each image's forward pass separately
(CUDA-synchronized) and records peak VRAM and .ply size per image.

Run inside the `sharp` env (bench/run.py does this for you):
    python bench/sharp_predict_timed.py -i <image or dir> -o <out dir> [--transforms transforms_test.json]
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sharp.cli.predict import DEFAULT_MODEL_URL, predict_image
from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import save_ply


def load_image(path: Path, white_bg: bool) -> tuple[np.ndarray, float]:
    """RGB uint8 image, compositing RGBA onto white/black like the 3DGS loaders do."""
    _, _, f_px_exif = io.load_rgb(path)
    img = Image.open(path)
    if img.mode == "RGBA":
        rgba = np.asarray(img).astype(np.float32) / 255.0
        bg = 1.0 if white_bg else 0.0
        rgb = rgba[..., :3] * rgba[..., 3:4] + bg * (1 - rgba[..., 3:4])
        return (rgb * 255).round().astype(np.uint8), f_px_exif
    return np.asarray(img.convert("RGB")), f_px_exif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("-c", "--checkpoint", type=Path, default=None)
    ap.add_argument("--transforms", type=Path, default=None,
                    help="NeRF-style transforms json; its camera_angle_x gives the true focal length")
    ap.add_argument("--white", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="max number of images")
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    exts = {".png", ".jpg", ".jpeg"}
    images = [args.input] if args.input.is_file() else sorted(p for p in args.input.iterdir() if p.suffix.lower() in exts)
    images = [p for p in images if "_depth" not in p.stem and "_normal" not in p.stem][: args.limit]
    camera_angle_x = json.loads(args.transforms.read_text())["camera_angle_x"] if args.transforms else None

    device = torch.device("cuda")
    t0 = time.perf_counter()
    if args.checkpoint is None:
        args.checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / DEFAULT_MODEL_URL.rsplit("/", 1)[-1]
        if not args.checkpoint.exists():
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(DEFAULT_MODEL_URL, str(args.checkpoint))
    state_dict = torch.load(args.checkpoint, weights_only=True)
    model = create_predictor(PredictorParams())
    model.load_state_dict(state_dict)
    model.eval().to(device)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())

    args.output.mkdir(parents=True, exist_ok=True)
    per_image = []
    for k, path in enumerate(images):
        image, f_px = load_image(path, args.white)
        h, w = image.shape[:2]
        if camera_angle_x is not None:
            f_px = 0.5 * w / math.tan(0.5 * camera_angle_x)

        if k == 0:
            for _ in range(args.warmup):
                predict_image(model, image, f_px, device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        gaussians = predict_image(model, image, f_px, device)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t

        ply = args.output / f"{path.stem}.ply"
        save_ply(gaussians, f_px, (h, w), ply)
        per_image.append({
            "image": path.name,
            "predict_s": dt,
            "peak_alloc_mib": torch.cuda.max_memory_allocated() / 2**20,
            "n_gaussians": int(gaussians.mean_vectors.flatten(0, 1).shape[0]),
            "ply_bytes": ply.stat().st_size,
        })
        print(f"[{k + 1}/{len(images)}] {path.name}: {dt * 1000:.1f} ms")

    def mean(key):
        return sum(r[key] for r in per_image) / len(per_image)

    summary = {
        "n_images": len(per_image),
        "model_load_s": load_s,
        "model_params": n_params,
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "mean_predict_s": mean("predict_s"),
        "mean_peak_alloc_mib": mean("peak_alloc_mib"),
        "max_peak_alloc_mib": max(r["peak_alloc_mib"] for r in per_image),
        "mean_n_gaussians": mean("n_gaussians"),
        "mean_ply_bytes": mean("ply_bytes"),
        "per_image": per_image,
    }
    (args.output / "sharp_timing.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
