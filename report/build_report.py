"""Build the comparison page (report/index.html + report/img/) from benchmark results.

    python report/build_report.py            # harness Python (>= 3.11); calls the npbg env for images

Needs, per scene below: results/<method>/<run>/{summary.json, eval.json, test_renders/ or model/renders/}
and results/profile/npbg_<key>.json (bench/npbg_profile.py --compile). Open report/index.html
in a browser, or publish it as a page; the images are referenced relatively from img/.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
METHODS = ("npbg", "sog", "3dgs", "sharp")
# key: (exported scene, run name, image width)
SCENES = {
    "rig": ("data/npbg/rig_e2e_p10", "rig_e2e_p10", 1600),
    "robot": ("data/npbg/robot_lab_p05", "robot_lab_p05", 960),
}
STANDALONE_HEAD = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>
"""
SUMMARY_KEYS = ("end_to_end_s", "train_wall_time_s", "npbg_build_s", "eval_load_s", "eval_render_test_set_s",
                "model_mb", "checkpoint_mb", "n_gaussians", "eval_npbg_psnr_full", "eval_npbg_ssim_full",
                "eval_npbg_lpips_full", "eval_npbg_psnr_covered", "eval_render_fps", "eval_render_ms_per_frame",
                "eval_n_test_views", "per_image_predict_s", "n_input_views")


def main():
    cfg = tomllib.loads((ROOT / "bench_config.toml").read_text())
    img = REPORT / "img"
    img.mkdir(parents=True, exist_ok=True)
    data = {}
    for key, (scene, run, width) in SCENES.items():
        runs = [f"{m}={ROOT / 'results' / m / run}" for m in METHODS]
        subprocess.run([cfg["python"]["npbg"], "-I", ROOT / "bench" / "compare_images.py", "--scene", ROOT / scene,
                        "--key", key, "--width", str(width), "--out", img, "--runs", *runs], check=True)
        views = json.loads((img / f"{key}.json").read_text())["views"]
        (img / f"{key}.json").unlink()
        methods = {}
        for m in METHODS:
            s = json.loads((ROOT / "results" / m / run / "summary.json").read_text())
            methods[m] = {k: s.get(k) for k in SUMMARY_KEYS}
        profile = json.loads((ROOT / "results" / "profile" / f"npbg_{key}.json").read_text())
        data[key] = {"views": views, "methods": methods, "profile": profile}
    page = (REPORT / "template.html").read_text().replace("__DATA__", json.dumps(data))
    # template.html is written for the artifact host, which supplies the document skeleton;
    # index.html gets its own so it also opens straight from disk
    (REPORT / "index.html").write_text(STANDALONE_HEAD + page)
    print(f"wrote {REPORT / 'index.html'} and {len(list(img.glob('*.jpg')))} images")


if __name__ == "__main__":
    sys.exit(main())
