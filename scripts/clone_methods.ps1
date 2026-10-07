# Clone the three baseline repos into external/ at the commits this harness was written against.
# Pinning matters: results are only comparable if everyone runs the same code.
$ErrorActionPreference = "Stop"
$ext = Join-Path $PSScriptRoot "..\external"
New-Item -ItemType Directory -Force $ext | Out-Null

$repos = @(
    @{ url = "https://github.com/graphdeco-inria/gaussian-splatting";        dir = "gaussian-splatting";        commit = "54c035f7834b564019656c3e3fcc3646292f727d" },
    @{ url = "https://github.com/fraunhoferhhi/Self-Organizing-Gaussians";   dir = "Self-Organizing-Gaussians"; commit = "5f2b360abf496af560ded8ddc13157a601dade32" },
    @{ url = "https://github.com/apple/ml-sharp";                            dir = "ml-sharp";                  commit = "aed6527499ef91cba3b54c18d49a870f25947190" }
)

foreach ($r in $repos) {
    $path = Join-Path $ext $r.dir
    if (-not (Test-Path $path)) { git clone --recursive $r.url $path }
    git -C $path checkout $r.commit
    git -C $path submodule update --init --recursive
}
