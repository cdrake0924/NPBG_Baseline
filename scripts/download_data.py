"""Download benchmark datasets into data/.

    python scripts/download_data.py nerf_synthetic   # synthetic, ~1.1 GB zip (needs: pip install gdown)
    python scripts/download_data.py tandt_db         # real, 650 MB (Truck, Train, DrJohnson, Playroom)

nerf_synthetic: the 8 Blender scenes from the NeRF paper (chair, drums, ficus, hotdog, lego,
materials, mic, ship); 100 train / 200 test views at 800x800 with exact camera poses.
If Google Drive refuses (download quota), grab nerf_synthetic.zip manually from
https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1 and pass --zip <path>.

tandt_db: the COLMAP-processed Tanks&Temples + Deep Blending scenes hosted by the 3DGS
authors; both 3DGS and SOG report results on these, so they validate the setup.
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
NERF_SYNTHETIC_GDRIVE_ID = "18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG"
TANDT_DB_URL = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"


def extract(zip_path: Path, dest: Path):
    print(f"Extracting {zip_path} -> {dest}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["nerf_synthetic", "tandt_db"])
    ap.add_argument("--zip", type=Path, help="use an already-downloaded zip")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)

    zip_path = args.zip or DATA / f"{args.dataset}.zip"
    if not zip_path.exists():
        if args.dataset == "nerf_synthetic":
            import gdown
            gdown.download(id=NERF_SYNTHETIC_GDRIVE_ID, output=str(zip_path), quiet=False)
        else:
            print(f"Downloading {TANDT_DB_URL}")
            with urllib.request.urlopen(TANDT_DB_URL) as r, open(zip_path, "wb") as f:
                shutil.copyfileobj(r, f)

    if args.dataset == "nerf_synthetic":
        extract(zip_path, DATA)  # zip contains nerf_synthetic/<scene>/
        junk = DATA / "__MACOSX"
        if junk.exists():
            shutil.rmtree(junk)
    else:
        extract(zip_path, DATA / "tandt_db")  # -> tandt_db/{tandt,db}/<scene>/

    if not args.zip and not args.keep_zip:
        zip_path.unlink()
    print("Done:", sorted(p.name for p in DATA.iterdir()))


if __name__ == "__main__":
    main()
