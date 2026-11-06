"""Image-quality metrics shared by every method, in two sets.

npbg_*   NPBG's own metrics.evaluate_metrics, loaded by file from the NPBG checkout, so every
         method is scored by the exact function NPBG's eval.py uses:
         PSNR / SSIM (reflect-padded, 11x11 gaussian) / LPIPS (AlexNet) on the float render,
         over the full frame and over the covered pixels (the record's geometry mask).
3DGS     The metrics 3DGS and SOG publish: render rounded to 8 bits, 3DGS's psnr / ssim and
         LPIPS (VGG). Kept so the T&T numbers can be checked against the papers.

Runs in the gs env (needs torch, lpips==0.1.4 and the gaussian-splatting checkout).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

GS_ROOT = Path(__file__).resolve().parents[1] / "external" / "gaussian-splatting"
if str(GS_ROOT) not in sys.path:
    sys.path.insert(0, str(GS_ROOT))

from lpipsPyTorch import lpips as gs_lpips  # noqa: E402
from utils.image_utils import psnr as gs_psnr  # noqa: E402
from utils.loss_utils import ssim as gs_ssim  # noqa: E402


def load_npbg_metrics(npbg_root: Path):
    """NPBG's metrics.py by path: putting the NPBG checkout on sys.path would shadow 3DGS modules."""
    path = Path(npbg_root) / "metrics.py"
    if not path.is_file():
        raise SystemExit(f"NPBG metrics not found at {path}; set [npbg] root in bench_config.toml")
    spec = importlib.util.spec_from_file_location("npbg_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Scorer:
    def __init__(self, npbg_root: Path, device: str = "cuda", lpips: bool = True):
        import lpips as lpips_package

        self.npbg = load_npbg_metrics(npbg_root)
        self.device = device
        self.use_lpips = lpips
        # Built once, as NPBG's eval.py does
        self.alex = lpips_package.LPIPS(net="alex", verbose=False).eval().to(device) if lpips else None

    @torch.no_grad()
    def score(self, render: torch.Tensor, gt: torch.Tensor, covered: torch.Tensor | None) -> dict:
        """render, gt: (3, H, W) in [0, 1] on device. covered: (1, H, W) in {0, 1} or None."""
        pred = render.clamp(0, 1).unsqueeze(0).float()
        gt = gt.unsqueeze(0).float()
        has_mask = covered is not None
        mask = covered.unsqueeze(0).float() if has_mask else torch.ones_like(pred[:, :1])

        out = {}
        if self.use_lpips:
            n = self.npbg.evaluate_metrics(pred, gt, mask, self.alex)
        else:  # evaluate_metrics always runs LPIPS; fall back to its parts
            s = self.npbg.ssim_map(pred, gt)
            n = {"psnr_full": self.npbg.psnr(pred, gt), "psnr_covered": self.npbg.psnr(pred, gt, mask),
                 "ssim_full": float(s.mean()), "ssim_covered": self.npbg._masked_mean(s, mask),
                 "lpips_full": None}
        for k, v in n.items():
            if not has_mask and k.endswith("_covered"):
                v = None
            out[f"npbg_{k}"] = v if v is None or math.isfinite(v) else None

        q = (pred * 255).round() / 255  # 3DGS metrics.py reads saved 8-bit PNGs
        out["psnr"] = gs_psnr(q, gt).mean().item()
        out["ssim"] = gs_ssim(q, gt).item()
        out["lpips"] = gs_lpips(q, gt, net_type="vgg").item() if self.use_lpips else None
        return out


def mean_scores(rows: list[dict]) -> dict:
    keys = rows[0].keys() if rows else []
    result = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        result[k] = sum(vals) / len(vals) if vals else None
    return result


def load_mask(path: Path, size_wh: tuple[int, int], device: str) -> torch.Tensor | None:
    """coverage/<image stem>.png written by npbg_to_colmap.py -> (1, H, W) {0, 1}."""
    if not path.is_file():
        return None
    from PIL import Image
    import numpy as np

    m = Image.open(path).convert("L")
    if m.size != tuple(size_wh):
        m = m.resize(size_wh, Image.NEAREST)
    return torch.from_numpy((np.asarray(m) > 127).astype("float32"))[None].to(device)
