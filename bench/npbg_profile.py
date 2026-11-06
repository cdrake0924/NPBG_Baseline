"""Where NPBG's render time goes: point rasterization (index pyramid) vs the neural decoder,
in fp32 (what the benchmark times, as eval.py renders), bf16 autocast, and the viewer / live
pipeline's settings: channels_last + bf16, optionally torch.compile (--compile).

Run inside the NPBG env, on an idle GPU:
    python bench/npbg_profile.py --scene <exported scene> [--checkpoint <scene .pt>] --npbg-root ~/NPBG --out profile.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def timed(fn, repeats: int) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / repeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--npbg-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--compile", action="store_true", help="also time a torch.compile'd decoder (about a minute of warm-up)")
    args = ap.parse_args()
    sys.path.insert(0, str(args.npbg_root.resolve()))
    from dataset import FrameDataset, PointCloud
    from model import NeuralPointRenderer, padded_extent, renderer_options
    from rasterize import index_pyramid
    from schema import resolve_raster_settings

    export = json.loads((args.scene / "npbg_export.json").read_text())
    root = Path(export["npbg_dataset"])
    ckpt = torch.load(args.checkpoint or root / "predicted_colour.pt", map_location="cpu", weights_only=True)
    config = ckpt["config"]
    near, radius = resolve_raster_settings(None, None, config)
    device = torch.device("cuda")
    cloud = PointCloud.from_manifest(root).to(device)
    model = NeuralPointRenderer(cloud.positions.shape[0], int(config["descriptor_size"]),
                                int(config["feature_scale"]), int(config["levels"]),
                                **renderer_options(config)).to(device)
    model.load_state_dict(ckpt["model"])
    model.set_colour_correction(ckpt.get("colour_correction"))
    model.eval()
    levels = int(config["levels"])
    f = FrameDataset(root, "validation").full_frame(0).to(device)
    h, w = f.target.shape[-2:]
    ph, pw = padded_extent(h, w)

    def raster():
        return index_pyramid(cloud.positions, f.view, f.intr, ph, pw, levels, near, radius)

    out = {"scene": str(args.scene), "n_points": int(cloud.positions.shape[0]), "frame_wh": [int(w), int(h)],
           "padded_wh": [int(pw), int(ph)], "levels": levels,
           "decoder_params": sum(p.numel() for n, p in model.named_parameters() if not n.startswith("descriptors."))}
    with torch.inference_mode():
        pyramid = raster()
        out["raster_ms"] = timed(raster, args.repeats)
        out["decoder_fp32_ms"] = timed(lambda: model(pyramid), args.repeats)
        out["total_fp32_ms"] = timed(lambda: model(raster()), args.repeats)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out["decoder_bf16_ms"] = timed(lambda: model(pyramid), args.repeats)
            out["total_bf16_ms"] = timed(lambda: model(raster()), args.repeats)
        try:
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as fc:
                model(pyramid)
            out["decoder_gflops_per_frame"] = fc.get_total_flops() / 1e9
        except Exception as e:  # noqa: BLE001
            out["decoder_gflops_per_frame"] = None
            out["flop_error"] = repr(e)
        # What viewer.py / live_oneshot.py do: channels_last decoder, bf16 autocast, torch.compile
        model.to(memory_format=torch.channels_last)
        out["decoder_cl_fp32_ms"] = timed(lambda: model(pyramid), args.repeats)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out["decoder_cl_bf16_ms"] = timed(lambda: model(pyramid), args.repeats)
            out["total_cl_bf16_ms"] = timed(lambda: model(raster()), args.repeats)
        if args.compile:
            model.net = torch.compile(model.net, dynamic=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t0 = time.perf_counter(); model(pyramid); torch.cuda.synchronize()
                out["compile_warmup_s"] = time.perf_counter() - t0
                out["decoder_compiled_cl_bf16_ms"] = timed(lambda: model(pyramid), args.repeats)
                out["total_compiled_cl_bf16_ms"] = timed(lambda: model(raster()), args.repeats)
    out["fps_fp32"] = 1000 / out["total_fp32_ms"]
    out["fps_cl_bf16"] = 1000 / out["total_cl_bf16_ms"]
    if "total_compiled_cl_bf16_ms" in out:
        out["fps_compiled_cl_bf16"] = 1000 / out["total_compiled_cl_bf16_ms"]
    out["fps_bf16"] = 1000 / out["total_bf16_ms"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
