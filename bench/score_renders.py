"""Score renders saved by a method that is not a 3DGS .ply (NPBG) with the same code, ground
truth and masks the shared evaluator uses, and write eval.json in the same format.

Run inside the gs env (bench/run.py does this for you):
    python bench/score_renders.py --renders <dir of <image stem>.npy> --source <exported scene> \
        --timing <npbg_render.json> --npbg-root ~/NPBG --out eval.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scoring import Scorer, load_mask, mean_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--timing", type=Path, required=True, help="the renderer's json (load / render timings)")
    ap.add_argument("--npbg-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    timing = json.loads(args.timing.read_text())
    scorer = Scorer(args.npbg_root, lpips=not args.no_lpips)
    rows = []
    for name in timing["test_images"]:
        stem = Path(name).stem
        render = torch.from_numpy(np.load(args.renders / f"{stem}.npy")).cuda()
        # Ground truth exactly as 3DGS's loader reads it: 8-bit PNG / 255
        gt = torch.from_numpy(np.asarray(Image.open(args.source / "images" / name).convert("RGB"),
                                         dtype=np.float32) / 255).permute(2, 0, 1).cuda()
        if render.shape != gt.shape:
            raise SystemExit(f"{stem}: render {tuple(render.shape)} vs ground truth {tuple(gt.shape)}")
        mask = load_mask(args.source / "coverage" / name, (gt.shape[2], gt.shape[1]), "cuda")
        rows.append({"image": stem, **scorer.score(render, gt, mask)})

    keep = ("n_test_views", "resolution_wh", "load_s", "render_test_set_s", "render_fps",
            "render_ms_per_frame", "render_peak_alloc_mib")
    result = {"renders": str(args.renders), "source": str(args.source),
              **{k: timing[k] for k in keep},
              **mean_scores([{k: v for k, v in r.items() if k != "image"} for r in rows]),
              "per_view": rows}
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_view"}, indent=2))


if __name__ == "__main__":
    main()
