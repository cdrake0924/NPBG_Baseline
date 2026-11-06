"""Collect every method's test renders for one scene into web-sized JPEGs + error maps + a JSON
of per-view metrics, for a side-by-side viewer.

    python bench/compare_images.py --scene data/npbg/rig_e2e_p10 --runs npbg=results/npbg/rig_e2e_p10 \
        sog=results/sog/rig_e2e_p10 ... --out <dir> --width 1280

Renders: results/<m>/<run>/test_renders/<stem>.png (shared evaluator) or model/renders/<stem>.npy (NPBG).
Error map: per-pixel mean absolute RGB error, x4, through a magma-like ramp (black = exact).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_render(run: Path, stem: str) -> np.ndarray:
    png = run / "test_renders" / f"{stem}.png"
    if png.exists():
        return cv2.imread(str(png))[:, :, ::-1].astype(np.float32) / 255
    npy = run / "model" / "renders" / f"{stem}.npy"
    return np.load(npy).transpose(1, 2, 0).astype(np.float32)


def save_jpg(path: Path, rgb: np.ndarray, width: int):
    h, w = rgb.shape[:2]
    if w > width:
        rgb = cv2.resize(rgb, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)[:, :, ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 86])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="method=results/<method>/<run>")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--key", required=True, help="short scene id used in file names")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--views", nargs="*", help="test image stems to include (default: all)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    export = json.loads((args.scene / "npbg_export.json").read_text())
    stems = args.views or [Path(n).stem for n in export["test_images"]] if "test_images" in export else args.views
    if not stems:
        stems = [Path(r["image"]).stem for r in export["records"] if r["split"] == "validation"]

    runs = dict(r.split("=", 1) for r in args.runs)
    meta = {"scene": args.key, "views": [], "methods": list(runs), "summary": {}}
    for m, run in runs.items():
        s = json.loads((Path(run) / "summary.json").read_text())
        meta["summary"][m] = {k: s.get(k) for k in (
            "end_to_end_s", "train_wall_time_s", "npbg_build_s", "model_mb", "checkpoint_mb", "n_gaussians",
            "eval_npbg_psnr_full", "eval_npbg_ssim_full", "eval_npbg_lpips_full", "eval_npbg_psnr_covered",
            "eval_npbg_ssim_covered", "eval_render_fps", "eval_resolution_wh")}
    for stem in stems:
        gt = cv2.imread(str(args.scene / "images" / f"{stem}.png"))[:, :, ::-1].astype(np.float32) / 255
        save_jpg(args.out / f"{args.key}_{stem}_gt.jpg", gt, args.width)
        view = {"stem": stem, "methods": {}}
        for m, run in runs.items():
            img = load_render(Path(run), stem)
            save_jpg(args.out / f"{args.key}_{stem}_{m}.jpg", img, args.width)
            err = np.abs(np.clip(img, 0, 1) - gt).mean(2)
            heat = cv2.applyColorMap((np.clip(err * 4, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
            save_jpg(args.out / f"{args.key}_{stem}_{m}_err.jpg", heat[:, :, ::-1].astype(np.float32) / 255, args.width)
            ev = json.loads((Path(run) / "eval.json").read_text())
            row = next((r for r in ev.get("per_view", []) if r["image"] == stem), {})
            view["methods"][m] = {k: row.get(k) for k in ("npbg_psnr_full", "npbg_ssim_full", "npbg_lpips_full",
                                                         "npbg_psnr_covered")}
        meta["views"].append(view)
    (args.out / f"{args.key}.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(stems)} views x {len(runs)} methods -> {args.out}")


if __name__ == "__main__":
    main()
