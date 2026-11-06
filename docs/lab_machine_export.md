# Lab machine task: export NPBG data for the baseline benchmark

You are running on the lab capture machine (user `points`). Package a small, fixed set of
NPBG datasets and checkpoints into one tar file so it can be copied to another PC. That PC
benchmarks 3DGS and Self-Organizing Gaussians against our NPBG pipeline on the same inputs.

**Your job is read-only, plus writing one output folder.**
- Do not modify, move, or delete anything under `/data/points/`, `/home/points/`, or `~/NPBG`.
- Do not train, rebuild scenes, or start GPU jobs. Other people share this machine.
- Write only to `~/npbg_bench/`.
- Never put passwords, keys, or VPN configs in the tar file or in your report.

## 1. Check that every input exists

These are the paths we expect. Each is referenced in `~/NPBG/BEST_RESULTS.md`,
`E2E_PIPELINE.md`, or `QUASAR_TRAINING.md` on the `FoundationStereo` branch.

| # | Path | What it is | Must contain |
|---|---|---|---|
| 1 | `/data/points/npbg_datasets/rig_test_scene_crops64_p10_s2_interp_win_avg` | Real data: built rig scene, crops64 model, 10% keep | `manifest.json`, `point_cloud.npz`, `records/`, `predicted_colour.pt` |
| 2 | `/data/points/npbg_datasets/rig_test_scene_e2e_p10_s1_kept_avg` | Real data: built rig scene, current best e2e model, 10% keep | `manifest.json`, `point_cloud.npz`, `records/`, `predicted_colour.pt` |
| 3 | `/home/points/calib_2026-09-29/test_scene/ffs_v2` | FoundationStereo depth per stereo pair (the scene build's input) | `pairs.json` and one folder per pair with `reference.npz` |
| 4 | `/data/points/npbg_runs/crops64_sparse_realistic/best.pt` | crops64 model weights | the file |
| 5 | `/data/points/npbg_runs/e2e_crops64/best.pt` | e2e model weights | the file |
| 6 | `/data/points/npbg_datasets/quasar_robot_lab` | Synthetic data: QUASAR `robot_lab` scene, 120 rendered views | `manifest.json`, `point_cloud.npz`, `records/` |
| 7 | `/data/points/npbg_runs/quasar_robot_lab/best_p*.pt` | NPBG model trained on item 6 | exactly one `best_p*M.pt` |

Check them:

```bash
cd ~/NPBG && git rev-parse --abbrev-ref HEAD && git rev-parse HEAD
ls /data/points/npbg_datasets/rig_test_scene_crops64_p10_s2_interp_win_avg \
   /data/points/npbg_datasets/rig_test_scene_e2e_p10_s1_kept_avg \
   /home/points/calib_2026-09-29/test_scene/ffs_v2 \
   /data/points/npbg_datasets/quasar_robot_lab
ls -l /data/points/npbg_runs/crops64_sparse_realistic/best.pt \
      /data/points/npbg_runs/e2e_crops64/best.pt \
      /data/points/npbg_runs/quasar_robot_lab/best_p*.pt
```

**If a path is missing or named differently, find the closest match instead of guessing:**
- Synthetic run directories may carry the dropout rate in the name (`quasar_robot_lab_p05`).
  List both sides with `ls -d /data/points/npbg_datasets/quasar_robot_lab* /data/points/npbg_runs/quasar_robot_lab*`.
  Pick a dataset and a run that belong together: the run's checkpoint should name that
  dataset folder as its `data` path, and its `p<N>M` tag should match the dataset's
  `point_cloud.num_points` in `manifest.json`.
- If an e2e or crops64 file has a different name (for example `last.pt`), check
  `BEST_RESULTS.md` and `E2E_PIPELINE.md` for which file they actually load.
- If something truly doesn't exist, leave it out and say so in the report. Don't
  substitute a different model or scene without saying so.

## 2. Record what each dataset holds

For each dataset folder (items 1, 2, and 6), print the point count, image size, and splits
so the other machine can confirm it received the right thing:

```bash
for d in /data/points/npbg_datasets/rig_test_scene_crops64_p10_s2_interp_win_avg \
         /data/points/npbg_datasets/rig_test_scene_e2e_p10_s1_kept_avg \
         /data/points/npbg_datasets/quasar_robot_lab; do
  python3 - "$d" <<'EOF'
import json, sys, collections
d = sys.argv[1]
m = json.load(open(f"{d}/manifest.json"))
pc = m["point_cloud"]
splits = collections.Counter(r["split"] for r in m["records"])
print(d)
print("  points:", pc.get("num_points"), " size:", pc.get("width"), "x", pc.get("height"))
print("  splits:", dict(splits))
print("  record names:", [r.get("name", r.get("path")) for r in m["records"]][:8])
EOF
done
```

For the rig scenes we expect 4 records: 3 `train` and 1 `validation` (`bottom_left`). If
yours differ, report what you found.

## 3. Check sizes and free space

```bash
du -sh <each path from step 1>
df -h ~
```

If the total is over ~15 GB, or `~` doesn't have room for a second copy, stop and report
the sizes instead of packing.

## 4. Pack

Store paths relative to `/` so the folder layout survives the copy. Use plain `tar`
without compression, because `.npz` and `.pt` files barely compress. Adjust the list if
step 1 found different names.

```bash
mkdir -p ~/npbg_bench
cd /
tar -cf ~/npbg_bench/npbg_bench.tar \
  data/points/npbg_datasets/rig_test_scene_crops64_p10_s2_interp_win_avg \
  data/points/npbg_datasets/rig_test_scene_e2e_p10_s1_kept_avg \
  home/points/calib_2026-09-29/test_scene/ffs_v2 \
  data/points/npbg_runs/crops64_sparse_realistic/best.pt \
  data/points/npbg_runs/e2e_crops64/best.pt \
  data/points/npbg_datasets/quasar_robot_lab \
  data/points/npbg_runs/quasar_robot_lab/best_p*.pt
tar -tf ~/npbg_bench/npbg_bench.tar | awk -F/ '{print $1"/"$2"/"$3"/"$4}' | sort -u
cd ~/npbg_bench && sha256sum npbg_bench.tar > npbg_bench.tar.sha256 && ls -lh
```

## 5. Report back

End with this block, filled in. The user will paste it into the other machine's session.

```
NPBG branch / commit:     <branch> <full sha>
Hostname for scp:         <output of `hostname -f`>
Tar file:                 ~/npbg_bench/npbg_bench.tar  (<size>)
sha256:                   <hash>

Included (path -> size):
  ...

Dataset contents (from step 2):
  rig crops64:  <points>, <W>x<H>, splits {...}
  rig e2e:      <points>, <W>x<H>, splits {...}
  robot_lab:    <points>, <W>x<H>, splits {...}

Substitutions or missing items:  <none, or what and why>
```

The user will then run this on the other PC (PowerShell, in the benchmark repo root):

```powershell
New-Item -ItemType Directory -Force data\npbg | Out-Null
scp points@<hostname>:npbg_bench/npbg_bench.tar data\npbg\
tar -xf data\npbg\npbg_bench.tar -C data\npbg
```
