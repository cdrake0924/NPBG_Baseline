"""Benchmark driver. Runs each method in its own env, measures speed / compute / size,
then scores the result with the shared evaluator.

    python bench/run.py 3dgs  data/nerf_synthetic/lego
    python bench/run.py sog   data/nerf_synthetic/lego
    python bench/run.py sharp data/tandt_db/tandt/truck --views 4   # COLMAP scene: 4 photos merged
    python bench/run.py sharp data/nerf_synthetic/lego --limit 20   # no point cloud: per-image timing only
    python bench/run.py eval  3dgs lego          # re-score an existing run
    python bench/run.py summary                  # -> results/summary.csv + summary.md
    python bench/run.py 3dgs <scene> -- --optimizer_type sparse_adam   # extra args to the method

Each run writes results/<method>/<scene>/{model/, train.log, summary.json}.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
import tomllib
from pathlib import Path

from monitor import gpu_info, run_monitored

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external"
RESULTS = ROOT / "results"
REPOS = {
    "3dgs": EXTERNAL / "gaussian-splatting",
    "sog": EXTERNAL / "Self-Organizing-Gaussians",
    "sharp": EXTERNAL / "ml-sharp",
}
# umbrella_sh.yaml runs two compression variants; "jxl_quant_sh" (all attributes JPEG XL) is the
# current published configuration (results/TanksAndTemples/*.csv "Baseline"). The other,
# "exr_jxl_quant_5_norm", is the older EXR-based variant: slightly larger and ~0.25 dB worse.
SOG_EXPERIMENT = "jxl_quant_sh"


def load_config() -> dict:
    path = ROOT / "bench_config.toml"
    if not path.exists():
        sys.exit("Missing bench_config.toml - copy bench_config.example.toml and fill in env python paths.")
    return tomllib.loads(path.read_text())


def is_synthetic(scene: Path) -> bool:
    return (scene / "transforms_train.json").exists()


def posix(p: Path) -> str:
    # Forward slashes: SOG writes paths into a python literal (cfg_args) that is eval'd later.
    return p.resolve().as_posix()


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_dir(method: str, scene: Path, name: str | None) -> Path:
    d = RESULTS / method / (name or scene.name)
    if (d / "model").exists():
        sys.exit(f"{d} already exists - delete it or pass --name to keep both runs.")
    d.mkdir(parents=True, exist_ok=True)
    return d


def monitored(cfg: dict, cmd: list[str], cwd: Path, log: Path, env: dict | None = None) -> dict:
    gpu = cfg.get("gpu", {})
    print("$", " ".join(map(str, cmd)))
    stats = run_monitored([str(c) for c in cmd], cwd=str(cwd), env=env, log_path=str(log),
                          gpu_index=gpu.get("index", 0), interval_ms=gpu.get("sample_interval_ms", 200))
    if stats["returncode"] != 0:
        sys.exit(f"Command failed (exit {stats['returncode']}); see {log}")
    return stats


def evaluate(cfg: dict, ply: Path, scene: Path, out: Path) -> dict:
    cmd = [cfg["python"]["gs"], ROOT / "bench" / "eval_render.py", "--ply", ply, "--source", scene, "--out", out]
    if is_synthetic(scene):
        cmd.append("--white")
    subprocess.run([str(c) for c in cmd], check=True)
    return json.loads(out.read_text())


def end_to_end_s(s: dict) -> float | None:
    """Inputs (images + poses + point cloud) on disk -> test views rendered.
    = build wall time (incl. process start, data loading, training/prediction, SOG compression)
    + loading the final model + one cold render pass over the test views."""
    if s.get("eval_load_s") is None:
        return None
    return s["train_wall_time_s"] + s["eval_load_s"] + s["eval_render_test_set_s"]


def write_summary(d: Path, method: str, scene: Path, iterations, stats: dict, model_path: Path,
                  model_bytes: int, ev: dict | None, extra: dict | None = None):
    n = ev["n_gaussians"] if ev else (extra or {}).get("n_gaussians")
    summary = {
        "method": method,
        "scene": scene.name,
        "dataset": scene.parent.name,
        "source_path": str(scene),
        "iterations": iterations,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "method_commit": git_commit(REPOS[method]),
        **gpu_info(),
        "model_path": str(model_path),
        "model_bytes": model_bytes,
        "model_mb": model_bytes / 1e6,
        "n_gaussians": n,
        "bytes_per_gaussian": model_bytes / n if n else None,
        **{f"train_{k}": v for k, v in stats.items() if k != "cmd"},
        "train_cmd": stats["cmd"],
        **({f"eval_{k}": v for k, v in ev.items() if k not in ("ply", "source")} if ev else {}),
        **(extra or {}),
    }
    summary["end_to_end_s"] = end_to_end_s(summary)
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, (list, dict))}, indent=2))


# ---------------------------------------------------------------- methods

def run_3dgs(cfg, args):
    scene = args.scene.resolve()
    d = run_dir("3dgs", scene, args.name)
    model = d / "model"
    it = args.iterations
    cmd = [cfg["python"]["gs"], "train.py", "-s", scene, "-m", model, "--eval",
           "--iterations", it, "--save_iterations", it, "--test_iterations", -1, "--disable_viewer"]
    if is_synthetic(scene):
        cmd.append("-w")
    cmd += args.extra
    stats = monitored(cfg, cmd, REPOS["3dgs"], d / "train.log")

    ply = model / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
    ev = None if args.no_eval else evaluate(cfg, ply, scene, d / "eval.json")
    write_summary(d, "3dgs", scene, it, stats, ply, ply.stat().st_size, ev)


def run_sog(cfg, args):
    scene = args.scene.resolve()
    d = run_dir("sog", scene, args.name)
    model = d / "model"
    it = args.iterations
    cmd = [cfg["python"]["sog"], "train.py", "--config-name", "ours_q_sh_local_test",
           f"hydra.run.dir='{posix(model)}'",
           f"dataset.source_path='{posix(scene)}'",
           f"dataset.model_path='{posix(model)}'",
           f"dataset.white_background={'true' if is_synthetic(scene) else 'false'}",
           f"optimization.iterations={it}",
           f"run.save_iterations=[{it}]",
           f"run.compress_iterations=[{it}]",
           "run.test_iterations=[]",
           "run.no_progress_bar=true",
           f"run.name={scene.name}"]
    cmd += args.extra
    stats = monitored(cfg, cmd, REPOS["sog"], d / "train.log", env={"WANDB_MODE": args.wandb_mode})

    compressed = model / "compression" / f"iteration_{it}" / SOG_EXPERIMENT
    sog_stats = json.loads((compressed / "stats.json").read_text())
    compressed_bytes = sog_stats["Size [Bytes]"]
    ply = compressed / "decompressed_model" / "point_cloud" / "iteration_1" / "point_cloud.ply"
    uncompressed_ply = model / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
    ev = None if args.no_eval else evaluate(cfg, ply, scene, d / "eval.json")
    write_summary(d, "sog", scene, it, stats, compressed, compressed_bytes, ev,
                  extra={"uncompressed_ply_bytes": uncompressed_ply.stat().st_size,
                         "n_gaussians": sog_stats["#Gaussians"],
                         "sog_experiment": SOG_EXPERIMENT})


def run_sharp(cfg, args):
    scene = args.scene.resolve()
    if (scene / "sparse").exists():
        return run_sharp_merge(cfg, args, scene)
    d = run_dir("sharp", scene, args.name)
    model = d / "model"
    images = scene / "test" if (scene / "test").is_dir() else scene
    cmd = [cfg["python"]["sharp"], ROOT / "bench" / "sharp_predict_timed.py", "-i", images, "-o", model]
    if args.limit:
        cmd += ["--limit", args.limit]
    if (scene / "transforms_test.json").exists():
        cmd += ["--transforms", scene / "transforms_test.json", "--white"]
    if args.checkpoint:
        cmd += ["-c", args.checkpoint]
    stats = monitored(cfg, cmd, ROOT, d / "train.log")

    t = json.loads((model / "sharp_timing.json").read_text())
    # Without poses + sparse points there is no way to place SHARP's per-image scenes in the
    # dataset's world frame, so the shared evaluator does not apply; report per-image numbers.
    write_summary(d, "sharp", scene, None, stats, model, round(t["mean_ply_bytes"]), None,
                  extra={"n_gaussians": t["mean_n_gaussians"],
                         "per_image_predict_s": t["mean_predict_s"],
                         "per_image_peak_alloc_mib": t["max_peak_alloc_mib"],
                         "model_load_s": t["model_load_s"],
                         "model_params": t["model_params"],
                         "checkpoint_mb": t["checkpoint_bytes"] / 1e6,
                         "n_images": t["n_images"]})


def run_sharp_merge(cfg, args, scene: Path):
    """COLMAP scene: K photos -> K SHARP scenes -> placed with COLMAP poses -> one merged .ply."""
    d = run_dir("sharp", scene, args.name or f"{scene.name}_{args.views}v")
    model = d / "model"
    cmd = [cfg["python"]["sharp"], ROOT / "bench" / "sharp_merge.py", "--scene", scene, "-o", model,
           "--views", args.views]
    if args.checkpoint:
        cmd += ["-c", args.checkpoint]
    stats = monitored(cfg, cmd, ROOT, d / "train.log")

    m = json.loads((model / "sharp_merge.json").read_text())
    ply = model / "merged.ply"
    ev = evaluate(cfg, ply, scene, d / "eval.json")
    write_summary(d, "sharp", scene, None, stats, ply, m["ply_bytes"], ev,
                  extra={"n_input_views": len(m["views"]),
                         "per_image_predict_s": m["predict_total_s"] / len(m["views"]),
                         "sharp_predict_total_s": m["predict_total_s"],
                         "sharp_align_total_s": m["align_total_s"],
                         "model_load_s": m["model_load_s"],
                         "model_params": m["model_params"],
                         "checkpoint_mb": m["checkpoint_bytes"] / 1e6,
                         "scale_log_mad": [v["scale_log_mad"] for v in m["per_view"]]})


def run_eval(cfg, args):
    d = RESULTS / args.method / args.run
    s = json.loads((d / "summary.json").read_text())
    scene = args.source.resolve() if args.source else Path(s["source_path"])
    if args.method == "sog":
        # Re-point at SOG_EXPERIMENT (every variant in the config is compressed during training).
        compressed = Path(s["model_path"]).parent / SOG_EXPERIMENT
        sog_stats = json.loads((compressed / "stats.json").read_text())
        s.update(model_path=str(compressed), model_bytes=sog_stats["Size [Bytes]"],
                 model_mb=sog_stats["Size [Bytes]"] / 1e6, n_gaussians=sog_stats["#Gaussians"],
                 bytes_per_gaussian=sog_stats["Size [Bytes]"] / sog_stats["#Gaussians"],
                 sog_experiment=SOG_EXPERIMENT)
    model_path = Path(s["model_path"])
    ply = model_path if model_path.suffix == ".ply" else \
        model_path / "decompressed_model" / "point_cloud" / "iteration_1" / "point_cloud.ply"
    ev = evaluate(cfg, ply, scene, d / "eval.json")
    s.update({f"eval_{k}": v for k, v in ev.items() if k not in ("ply", "source")})
    s["end_to_end_s"] = end_to_end_s(s)
    (d / "summary.json").write_text(json.dumps(s, indent=2))


COLUMNS = ["method", "run", "scene", "n_input_views", "iterations", "end_to_end_s", "train_wall_time_s", "per_image_predict_s",
           "train_gpu_peak_mem_delta_mib", "train_gpu_mean_util_pct", "train_gpu_energy_wh", "train_ram_peak_mib",
           "model_mb", "checkpoint_mb", "n_gaussians", "bytes_per_gaussian",
           "eval_psnr", "eval_ssim", "eval_lpips", "eval_render_fps", "eval_render_peak_alloc_mib"]


def summary(cfg, args):
    rows = [{**json.loads(p.read_text()), "run": p.parent.name} for p in sorted(RESULTS.glob("*/*/summary.json"))]
    if not rows:
        sys.exit("No results yet.")
    with open(RESULTS / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:,.3f}" if abs(v) < 10 else f"{v:,.1f}"
        return str(v)

    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in COLUMNS) + " |" for r in rows]
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("3dgs", "sog"):
        p = sub.add_parser(name)
        p.add_argument("scene", type=Path)
        p.add_argument("--iterations", type=int, default=30_000)
        p.add_argument("--name", help="run name (default: scene name)")
        p.add_argument("--no-eval", action="store_true")
        if name == "sog":
            p.add_argument("--wandb-mode", default="disabled", choices=["disabled", "offline"])

    p = sub.add_parser("sharp")
    p.add_argument("scene", type=Path)
    p.add_argument("--views", type=int, default=4, help="COLMAP scenes: photos to predict and merge")
    p.add_argument("--limit", type=int, default=20, help="Blender scenes: images to time individually")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--name")

    p = sub.add_parser("eval")
    p.add_argument("method", choices=["3dgs", "sog"])
    p.add_argument("run")
    p.add_argument("--source", type=Path)

    sub.add_parser("summary")

    # Anything after a bare `--` is passed straight to the method's train script.
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = ap.parse_args(argv[:split])
    args.extra = argv[split + 1:]
    cfg = load_config() if args.cmd != "summary" else {}
    {"3dgs": run_3dgs, "sog": run_sog, "sharp": run_sharp, "eval": run_eval, "summary": summary}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
