"""NPBG scene -> COLMAP scene, so 3DGS / SOG / SHARP train and test on exactly what NPBG sees.

    python bench/npbg_to_colmap.py --data /data/points/npbg_datasets/<scene> --out data/npbg/<scene>

Writes
    images/<name>.png            each record's target_rgb (the image NPBG is scored against), 8-bit
    coverage/<name>.png          each record's coverage mask (NPBG's psnr_covered / ssim_covered)
    sparse/0/cameras.bin         one PINHOLE camera from the manifest's intrinsics
    sparse/0/images.bin          per record: pose (NPBG's OpenGL view matrix -> COLMAP/OpenCV
                                 world-to-camera) and the points it observes (for SHARP's scale fit)
    sparse/0/points3D.bin/.ply   NPBG's own point cloud (point_cloud.npz positions), coloured by
                                 averaging the TRAINING photos where each point is visible
    npbg_export.json             provenance + the record <-> image mapping bench/run.py uses

Split: NPBG's own train / validation split is kept exactly, one of two ways:
    llff      3DGS, SOG and sharp_merge.py hold out sorted-name index % 8 == 0. When NPBG's
              validation records can be named to land exactly there (the rig: bottom_left sorts
              first), the scene is used as is with the methods' --eval.
    explicit  Otherwise (QUASAR: 12 of 120, every 10th), train/ holds a second COLMAP model with
              only the training images (symlinks) and the same points. The baselines train on
              train/ without a hold-out, and the evaluator scores the full scene's test_images.
The test images get no point observations, so nothing about them leaks into the baselines'
inputs (their colour never touches points3D).

Runs in any env with numpy + opencv (e.g. quasar-npbg). No GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np

GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def intrinsics_from_manifest(pc: dict) -> np.ndarray:
    if "intrinsic_matrix" in pc:
        return np.asarray(pc["intrinsic_matrix"], dtype=np.float64)
    # Same formula as NPBG capture.intrinsics (glm::perspective)
    w, h, fovy = int(pc["width"]), int(pc["height"]), float(pc["fovy"])
    t = math.tan(fovy / 2)
    fy = h / (2 * t)
    aspect = float(pc.get("aspect", w / h))
    fx = w / (2 * aspect * t)
    return np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1.0]])


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """COLMAP's rotmat2qvec (w, x, y, z)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    return -q if q[0] < 0 else q


def assign_names(records: list[dict], hold: int = 8) -> tuple[list[str], str]:
    """Image names, and "llff" if their sorted order puts exactly the validation records at
    index % hold == 0, else "explicit" (natural names; the split goes in train/ instead)."""
    base = [r.get("camera") or f"frame{int(r.get('frame', i)):06d}" for i, r in enumerate(records)]
    if len(set(base)) != len(base):
        base = [f"{b}_{i:04d}" for i, b in enumerate(base)]
    val = [i for i, r in enumerate(records) if r["split"] == "validation"]
    train = [i for i, r in enumerate(records) if r["split"] == "train"]
    n = len(val) + len(train)
    slots = [k for k in range(n) if k % hold == 0]

    def ok(names):
        order = sorted(range(len(records)), key=lambda i: names[i])
        order = [i for i in order if records[i]["split"] in ("train", "validation")]
        return all((records[i]["split"] == "validation") == (k % hold == 0) for k, i in enumerate(order))

    names = [f"{b}.png" for b in base]
    if ok(names):
        return names, "llff"  # natural names already give the right split (e.g. rig: bottom_left first)
    if len(val) != len(slots):
        return names, "explicit"
    order = [None] * n
    for k, i in zip(slots, val):
        order[k] = i
    rest = iter(train)
    for k in range(n):
        if order[k] is None:
            order[k] = next(rest)
    names = list(names)
    for k, i in enumerate(order):
        names[i] = f"{k:04d}_{base[i]}.png"
    assert ok(names)
    return names, "llff"


def visible_points(cam_pts: np.ndarray, K: np.ndarray, w: int, h: int, tol: float):
    """Indices of points that are the front surface at their pixel, and their pixel coords."""
    z = cam_pts[:, 2]
    front = z > 1e-6
    u = K[0, 0] * cam_pts[:, 0] / np.where(front, z, 1) + K[0, 2]
    v = K[1, 1] * cam_pts[:, 1] / np.where(front, z, 1) + K[1, 2]
    inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx = np.nonzero(inside)[0]
    pix = v[idx].astype(np.int64) * w + u[idx].astype(np.int64)
    zbuf = np.full(w * h, np.inf)
    np.minimum.at(zbuf, pix, z[idx])
    vis = z[idx] <= zbuf[pix] * (1 + tol)
    return idx[vis], np.stack([u[idx[vis]], v[idx[vis]]], 1)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    """Same layout as 3DGS's storePly (x y z nx ny nz red green blue)."""
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.zeros(len(xyz), dt)
    arr["x"], arr["y"], arr["z"] = xyz.T.astype(np.float32)
    arr["red"], arr["green"], arr["blue"] = rgb.T
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              + "".join(f"property float {n}\n" for n in ("x", "y", "z", "nx", "ny", "nz"))
              + "".join(f"property uchar {n}\n" for n in ("red", "green", "blue"))
              + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(arr.tobytes())


def write_cameras(path: Path, w: int, h: int, K: np.ndarray):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, w, h))  # camera_id 1, model 1 = PINHOLE
        f.write(struct.pack("<dddd", K[0, 0], K[1, 1], K[0, 2], K[1, 2]))


def write_images(path: Path, images: list):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for image_id, q, t, name, xys, pids in images:
            f.write(struct.pack("<idddddddi", image_id, *q, *t, 1))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", len(pids)))
            obs = np.zeros(len(pids), np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]))
            obs["x"], obs["y"], obs["id"] = xys[:, 0], xys[:, 1], pids + 1  # point3D ids start at 1
            f.write(obs.tobytes())


def write_points(path: Path, xyz: np.ndarray, rgb: np.ndarray, images: list):
    """points3D.bin with each point's track (image_id, point2D_idx), built without per-observation
    Python loops: the QUASAR scenes have ~2M points seen from ~100 views."""
    n = len(xyz)
    p_all = np.concatenate([pids for *_, pids in images] or [np.zeros(0, np.int64)])
    img_all = np.concatenate([np.full(len(pids), iid, np.int32) for iid, *_, pids in images] or [np.zeros(0, np.int32)])
    k_all = np.concatenate([np.arange(len(pids), dtype=np.int32) for *_, pids in images] or [np.zeros(0, np.int32)])
    order = np.argsort(p_all, kind="stable")
    track = np.stack([img_all[order], k_all[order]], 1).astype("<i4")
    counts = np.bincount(p_all, minlength=n)
    starts = np.concatenate([[0], np.cumsum(counts)])
    head = struct.Struct("<QdddBBBdQ")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", n))
        for p in range(n):
            f.write(head.pack(p + 1, *xyz[p], *rgb[p], 0.0, int(counts[p])))
            if counts[p]:
                f.write(track[starts[p]:starts[p + 1]].tobytes())


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            d.update(chunk)
    return d.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="NPBG dataset (manifest.json, point_cloud.npz, records/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--visibility-tol", type=float, default=0.02,
                    help="a point is visible if within this relative depth of the front surface at its pixel")
    ap.add_argument("--max-observations", type=int, default=20000,
                    help="point observations listed per training image (random subset of the visible points). "
                         "Only SHARP's scale fit reads them; a full list makes images.bin, which 3DGS/SOG "
                         "parse inside their timed run, needlessly large. 0 = all")
    args = ap.parse_args()
    data, out = args.data.resolve(), args.out
    if (out / "sparse").exists():
        raise SystemExit(f"{out} already has an export; delete it first.")

    m = json.loads((data / "manifest.json").read_text())
    pc = m["point_cloud"]
    records = [r for r in m["records"] if r["split"] in ("train", "validation")]
    w, h = int(pc["width"]), int(pc["height"])
    K = intrinsics_from_manifest(pc)
    names, split_mode = assign_names(records)

    cloud = np.load(data / "point_cloud.npz")
    xyz = cloud["positions"].astype(np.float64)
    n_pts = len(xyz)

    (out / "images").mkdir(parents=True)
    (out / "coverage").mkdir()
    (out / "sparse" / "0").mkdir(parents=True)

    rgb_sum = np.zeros((n_pts, 3))
    rgb_cnt = np.zeros(n_pts)
    rng = np.random.default_rng(0)
    images = []  # (image_id, qvec, tvec, name, xys, point_ids)
    for image_id, (r, name) in enumerate(zip(records, names), start=1):
        z = np.load(data / r["path"])
        target = np.clip(z["target_rgb"].astype(np.float32), 0, 1).transpose(1, 2, 0)
        img8 = np.round(target * 255).astype(np.uint8)
        cv2.imwrite(str(out / "images" / name), img8[:, :, ::-1])
        cov = (z["coverage"].astype(np.float32)[0] > 0.5).astype(np.uint8) * 255
        cv2.imwrite(str(out / "coverage" / name), cov)

        view = z["pose"].astype(np.float64)  # OpenGL world -> camera
        R = GL_TO_CV @ view[:3, :3]
        t = GL_TO_CV @ view[:3, 3]
        xys, pids = np.zeros((0, 2)), np.zeros(0, np.int64)
        if r["split"] == "train":
            pids, xys = visible_points(xyz @ R.T + t, K, w, h, args.visibility_tol)
            ui = np.clip(xys[:, 0].astype(int), 0, w - 1)
            vi = np.clip(xys[:, 1].astype(int), 0, h - 1)
            np.add.at(rgb_sum, pids, img8[vi, ui])  # colour from every visible point
            rgb_cnt[pids] += 1
            if args.max_observations and len(pids) > args.max_observations:
                keep = np.sort(rng.choice(len(pids), args.max_observations, replace=False))
                pids, xys = pids[keep], xys[keep]
        images.append((image_id, rotmat2qvec(R), t, name, xys, pids))
        print(f"{name}: {r['split']}, {len(pids)} visible points")

    seen = rgb_cnt > 0
    rgb = np.full((n_pts, 3), 128, np.uint8)
    rgb[seen] = np.round(rgb_sum[seen] / rgb_cnt[seen, None]).astype(np.uint8)

    sp = out / "sparse" / "0"
    write_cameras(sp / "cameras.bin", w, h, K)
    write_images(sp / "images.bin", images)
    write_points(sp / "points3D.bin", xyz, rgb, images)
    write_ply(sp / "points3D.ply", xyz, rgb)

    if split_mode == "explicit":
        # Training-only copy of the model: the baselines train on it with no hold-out
        train_imgs = [im for im, r in zip(images, records) if r["split"] == "train"]
        tsp = out / "train" / "sparse" / "0"
        tsp.mkdir(parents=True)
        (out / "train" / "images").mkdir()
        for im in train_imgs:
            (out / "train" / "images" / im[3]).symlink_to(Path("..", "..", "images", im[3]))
        write_cameras(tsp / "cameras.bin", w, h, K)
        write_images(tsp / "images.bin", train_imgs)
        write_points(tsp / "points3D.bin", xyz, rgb, train_imgs)
        write_ply(tsp / "points3D.ply", xyz, rgb)

    info = {
        "npbg_dataset": str(data),
        "manifest_sha256": sha256(data / "manifest.json"),
        "point_cloud_sha256": sha256(data / "point_cloud.npz"),
        "width": w, "height": h, "K": K.tolist(),
        "records": [{"record": r["path"], "camera": r.get("camera"), "frame": r.get("frame"),
                     "split": r["split"], "image": n} for r, n in zip(records, names)],
        "split_mode": split_mode,
        "test_images": [n for r, n in zip(records, names) if r["split"] == "validation"],
        "train_scene": "train" if split_mode == "explicit" else None,
        "max_observations": args.max_observations,
        "n_points": n_pts,
        "n_points_uncoloured": int((~seen).sum()),
        "visibility_tol": args.visibility_tol,
        # Full resolution for every method: NPBG renders and scores at the record size, and
        # 3DGS's default (-r -1) would downscale anything wider than 1600 px.
        "resolution": 1,
    }
    (out / "npbg_export.json").write_text(json.dumps(info, indent=2))
    print(f"{n_pts} points ({info['n_points_uncoloured']} not seen by any training view, coloured grey) -> {out}")


if __name__ == "__main__":
    main()
