#!/usr/bin/env bash
# Clone the three baseline repos into external/ at the commits this harness was written against.
# Pinning matters: results are only comparable if everyone runs the same code.
# Linux counterpart of clone_methods.ps1; keep the two commit lists identical.
#
# NPBG is not cloned here: on the lab machine it already lives in ~/NPBG with its data.
set -euo pipefail
ext="$(cd "$(dirname "$0")/.." && pwd)/external"
mkdir -p "$ext"

repos=(
    "https://github.com/graphdeco-inria/gaussian-splatting       gaussian-splatting        54c035f7834b564019656c3e3fcc3646292f727d"
    "https://github.com/fraunhoferhhi/Self-Organizing-Gaussians  Self-Organizing-Gaussians 5f2b360abf496af560ded8ddc13157a601dade32"
    "https://github.com/apple/ml-sharp                           ml-sharp                  aed6527499ef91cba3b54c18d49a870f25947190"
)

for r in "${repos[@]}"; do
    read -r url dir commit <<< "$r"
    path="$ext/$dir"
    [ -d "$path" ] || git clone --recursive "$url" "$path"
    git -C "$path" checkout "$commit"
    git -C "$path" submodule update --init --recursive
done
