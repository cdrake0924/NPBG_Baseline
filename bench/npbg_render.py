"""NPBG: load a finished scene, render its held-out views, time it. Scoring happens afterwards in
the gs env (bench/score_renders.py) with the same code that scores the other methods.

Mirrors the shared evaluator's timing so the columns mean the same thing:
    load_s              checkpoint (descriptors + decoder) + point cloud -> GPU, model ready
    render_test_set_s   one cold pass over the test views (rasterize index pyramid + decoder)
    render_fps          after warm-up, all test views x --fps-repeats
The test poses are read before the clock starts, as eval_render.py reads its cameras first.
The scene build (FoundationStereo depth -> sampled points -> predicted descriptors) is not timed.

Rendering follows NPBG's eval.py (fp32, padded extent, then cropped back).

Run inside the NPBG env (bench/run.py does this for you):
    python bench/npbg_render.py --scene <exported COLMAP scene> --checkpoint <scene .pt> \
        --npbg-root ~/NPBG --out <run dir>/model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True, help="export from npbg_to_colmap.py")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="scene checkpoint (default: <npbg dataset>/predicted_colour.pt)")
    ap.add_argument("--npbg-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps-repeats", type=int, default=3)
    args = ap.parse_args()

    sys.path.insert(0, str(args.npbg_root.resolve()))
    from dataset import FrameDataset, PointCloud
    from model import NeuralPointRenderer, padded_extent, renderer_options
    from rasterize import index_pyramid
    from schema import resolve_raster_settings

    export = json.loads((args.scene / "npbg_export.json").read_text())
    root = Path(export["npbg_dataset"])
    ckpt_path = (args.checkpoint or root / "predicted_colour.pt").resolve()
    device = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)

    # Test views, read before timing (as the shared evaluator loads its cameras first)
    data = FrameDataset(root, "validation")
    image_of = {r["record"]: r["image"] for r in export["records"]}
    names = [image_of[str(p.relative_to(root))] for p in data.paths]
    frames = [data.full_frame(i).to(device) for i in range(len(data))]
    height, width = frames[0].target.shape[-2:]
    padded_h, padded_w = padded_extent(height, width)
    torch.cuda.synchronize()

    # ---- load: everything needed to render this scene, onto the GPU
    t0 = time.perf_counter()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    config = ckpt["config"]
    near, radius = resolve_raster_settings(None, None, config)
    cloud = PointCloud.from_manifest(root).to(device)
    if ckpt["metadata"]["point_cloud_sha256"] != cloud.sha256:
        raise SystemExit("checkpoint was trained against a different point cloud")
    model = NeuralPointRenderer(cloud.positions.shape[0], int(config["descriptor_size"]),
                                int(config["feature_scale"]), int(config["levels"]),
                                **renderer_options(config)).to(device)
    model.load_state_dict(ckpt["model"])
    model.set_colour_correction(ckpt.get("colour_correction"))
    model.eval()
    levels = int(config["levels"])
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    def draw(f):
        pyramid = index_pyramid(cloud.positions, f.view, f.intr, padded_h, padded_w, levels, near, radius)
        return model(pyramid)[:, :, :height, :width]

    with torch.inference_mode():
        t0 = time.perf_counter()
        for f in frames:
            draw(f)
        torch.cuda.synchronize()
        render_test_set_s = time.perf_counter() - t0

        renders = args.out / "renders"
        renders.mkdir(exist_ok=True)
        for f, name in zip(frames, names):
            img = draw(f).clamp(0, 1)[0].float().cpu().numpy()
            np.save(renders / f"{Path(name).stem}.npy", img)

        for f in frames[:10]:
            draw(f)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(args.fps_repeats):
            for f in frames:
                draw(f)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    state = ckpt["model"]
    desc_bytes = sum(v.numel() * v.element_size() for k, v in state.items() if k.startswith("descriptors."))
    n_frames = len(frames) * args.fps_repeats
    result = {
        "npbg_dataset": str(root),
        "checkpoint": str(ckpt_path),
        "test_images": names,
        "n_test_views": len(frames),
        "resolution_wh": [int(width), int(height)],
        "n_points": int(cloud.positions.shape[0]),
        "load_s": load_s,
        "render_test_set_s": render_test_set_s,
        "render_fps": n_frames / dt,
        "render_ms_per_frame": 1000 * dt / n_frames,
        "render_peak_alloc_mib": torch.cuda.max_memory_allocated() / 2**20,
        "scene_files_bytes": {"point_cloud.npz": (root / "point_cloud.npz").stat().st_size,
                              ckpt_path.name: ckpt_path.stat().st_size},
        # What rendering needs: positions (float32) + per-point descriptors + decoder weights. The
        # files can hold more (a training checkpoint keeps optimizer state; point_cloud.npz keeps
        # capture features the renderer does not read), so they are reported separately.
        "render_bytes": int(cloud.positions.numel() * 4) + desc_bytes
                        + sum(v.numel() * v.element_size() for k, v in state.items() if not k.startswith("descriptors.")),
        "descriptor_bytes": desc_bytes,
        "decoder_bytes": sum(v.numel() * v.element_size() for v in state.values()) - desc_bytes,
        "decoder_params": sum(v.numel() for k, v in state.items() if not k.startswith("descriptors.")),
    }
    (args.out / "npbg_render.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
