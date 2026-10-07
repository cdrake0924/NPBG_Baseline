"""Run a command while sampling GPU memory / utilization / power and process RAM.

Works across conda envs because it only wraps a subprocess. GPU numbers are
device-wide (Windows WDDM does not report per-process VRAM), so the idle
baseline measured just before launch is subtracted from peak memory.
Close GPU-heavy apps (browsers, Wallpaper Engine, games) before benchmarking.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field

try:
    import psutil  # optional: peak system RAM of the process tree
except ImportError:
    psutil = None

QUERY = "memory.used,utilization.gpu,power.draw"


def _parse_float(s: str) -> float | None:
    try:
        return float(s.strip())
    except ValueError:
        return None


def query_gpu_once(gpu_index: int = 0) -> tuple[float | None, ...]:
    out = subprocess.run(
        ["nvidia-smi", f"--id={gpu_index}", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return tuple(_parse_float(v) for v in out.strip().split(","))


def gpu_info(gpu_index: int = 0) -> dict:
    out = subprocess.run(
        ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    return {"gpu_name": out[0].strip(), "gpu_mem_total_mib": float(out[1]), "driver": out[2].strip()}


@dataclass
class Samples:
    t: list[float] = field(default_factory=list)
    mem_mib: list[float] = field(default_factory=list)
    util_pct: list[float] = field(default_factory=list)
    power_w: list[float] = field(default_factory=list)
    ram_mib: list[float] = field(default_factory=list)


class GpuSampler:
    """Streams `nvidia-smi -lms` in a background thread."""

    def __init__(self, gpu_index: int = 0, interval_ms: int = 200):
        self.gpu_index = gpu_index
        self.interval_ms = interval_ms
        self.samples = Samples()
        self._proc = None
        self._thread = None

    def _reader(self):
        for line in self._proc.stdout:
            vals = [_parse_float(v) for v in line.split(",")]
            if len(vals) != 3 or vals[0] is None:
                continue
            self.samples.t.append(time.perf_counter())
            self.samples.mem_mib.append(vals[0])
            self.samples.util_pct.append(vals[1] if vals[1] is not None else 0.0)
            self.samples.power_w.append(vals[2] if vals[2] is not None else 0.0)

    def start(self):
        self._proc = subprocess.Popen(
            ["nvidia-smi", f"--id={self.gpu_index}", f"--query-gpu={QUERY}",
             "--format=csv,noheader,nounits", f"-lms={self.interval_ms}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self):
        if self._proc:
            self._proc.terminate()
            self._proc.wait()
        if self._thread:
            self._thread.join(timeout=2)


def _tree_rss_mib(pid: int) -> float:
    try:
        p = psutil.Process(pid)
        procs = [p] + p.children(recursive=True)
        return sum(q.memory_info().rss for q in procs if q.is_running()) / 2**20
    except psutil.Error:
        return 0.0


def run_monitored(cmd: list[str], cwd: str | None = None, env: dict | None = None,
                  log_path: str | None = None, gpu_index: int = 0, interval_ms: int = 200) -> dict:
    """Run `cmd` to completion and return timing / resource statistics."""
    # Idle baseline: median of a few samples before launch.
    idle = sorted(query_gpu_once(gpu_index) for _ in range(5))[2]
    idle_mem, idle_power = idle[0], idle[2]

    sampler = GpuSampler(gpu_index, interval_ms)
    sampler.start()
    log = open(log_path, "w", encoding="utf-8") if log_path else None
    full_env = {**os.environ, **(env or {})}

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=cwd, env=full_env, stdout=log or None, stderr=subprocess.STDOUT if log else None)
    ram = []
    while proc.poll() is None:
        if psutil:
            ram.append(_tree_rss_mib(proc.pid))
        time.sleep(0.5)
    wall = time.perf_counter() - t0
    sampler.stop()
    if log:
        log.close()

    s = sampler.samples
    energy_j = sum((s.t[i] - s.t[i - 1]) * (s.power_w[i] + s.power_w[i - 1]) / 2 for i in range(1, len(s.t)))
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "wall_time_s": wall,
        "gpu_idle_mem_mib": idle_mem,
        "gpu_peak_mem_mib": max(s.mem_mib, default=None),
        "gpu_peak_mem_delta_mib": (max(s.mem_mib) - idle_mem) if s.mem_mib and idle_mem is not None else None,
        "gpu_mean_util_pct": sum(s.util_pct) / len(s.util_pct) if s.util_pct else None,
        "gpu_idle_power_w": idle_power,
        "gpu_mean_power_w": sum(s.power_w) / len(s.power_w) if s.power_w else None,
        "gpu_energy_wh": energy_j / 3600.0,
        "ram_peak_mib": max(ram) if ram else None,
        "n_gpu_samples": len(s.t),
    }
