"""Measurement driver for HW1.

Runs CustomCNN over the (S, B) grid and saves latency, peak memory,
and energy measurements to results/measurements.csv.
"""

import contextlib
import gc
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    pynvml = None
    NVML_AVAILABLE = False

from models import CustomCNN

BASE_S = [32, 64, 128, 224, 256, 384, 512]
EXTRA_S = [96, 160, 320, 448]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
EXTRA_B = [12, 48, 100]

CUDA_REQUIRED_MSG = "measure.py requires a CUDA GPU."


def setup_environment() -> None:
    """Disable dynamic kernel selection algorithms and flush allocator."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_grid(image_sizes=None, batch_sizes=None):
    """Generate (S, B, is_validation) grid."""
    all_s = sorted(BASE_S + EXTRA_S) if image_sizes is None else sorted(image_sizes)
    all_b = sorted(BASE_B + EXTRA_B) if batch_sizes is None else sorted(batch_sizes)
    extra_s_set, extra_b_set = set(EXTRA_S), set(EXTRA_B)

    return [
        (s, b, (s in extra_s_set) or (b in extra_b_set))
        for s in all_s
        for b in all_b
    ]


def _nvml_counter_supported(handle) -> bool:
    try:
        pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        return True
    except Exception:
        return False


def _energy_window_power(handle, model, x, n_iters, device, interval=0.01) -> float | None:
    """Fallback energy measurement via background power sampling."""
    samples: list[float] = []
    stop = threading.Event()

    def _sample() -> None:
        while not stop.is_set():
            with contextlib.suppress(Exception):
                samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            stop.wait(interval)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    thread = threading.Thread(target=_sample, daemon=True)
    thread.start()
    try:
        with torch.inference_mode():
            for _ in range(n_iters):
                _ = model(x)
    finally:
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        stop.set()
        thread.join()

    if not samples or t1 <= t0:
        return None
    return float(np.mean(samples)) * (t1 - t0) / n_iters


def measure_point(
    model,
    s: int,
    b: int,
    repeats: int = 30,
    warmup: int = 10,
    *,
    nvml_handle=None,
    energy_method: str = "counter",
):
    """Measure median latency (s), peak memory (B), and energy (J) for (S, B)."""
    if not torch.cuda.is_available():
        raise RuntimeError(CUDA_REQUIRED_MSG)

    device = next(model.parameters()).device
    x = None
    try:
        torch.manual_seed(0)
        x = torch.randn(b, 3, s, s, device=device, dtype=torch.float32)

        # 1. Warmup cuDNN workspace before peak memory stats reset
        with torch.inference_mode():
            for _ in range(2):
                _ = model(x)
        torch.cuda.synchronize(device)

        # 2. Peak memory (reusing x to avoid double allocation spike)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            _ = model(x)
        torch.cuda.synchronize(device)
        peak_mem = torch.cuda.max_memory_allocated(device)

        # 3. Latency warmup and benchmark
        with torch.inference_mode():
            for _ in range(warmup):
                _ = model(x)
        torch.cuda.synchronize(device)

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings = []

        with torch.inference_mode():
            for _ in range(repeats):
                starter.record()
                _ = model(x)
                ender.record()
                ender.synchronize()
                timings.append(starter.elapsed_time(ender) / 1000.0)

        median_latency = float(np.median(timings))

        # 4. Energy window
        energy_j = None
        if NVML_AVAILABLE and nvml_handle is not None:
            try:
                # Cap iterations: fast points target ~0.5s window, heavy points need >=3 passes
                n_iters = max(3, min(100, int(0.5 / (median_latency + 1e-6))))

                with torch.inference_mode():
                    for _ in range(3):
                        _ = model(x)
                torch.cuda.synchronize(device)

                if energy_method == "power":
                    energy_j = _energy_window_power(
                        nvml_handle, model, x, n_iters, device
                    )
                else:
                    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(nvml_handle)
                    with torch.inference_mode():
                        for _ in range(n_iters):
                            _ = model(x)
                    torch.cuda.synchronize(device)
                    e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(nvml_handle)
                    energy_j = ((e1 - e0) / 1000.0) / n_iters
            except Exception:
                energy_j = None

        return median_latency, peak_mem, energy_j

    except torch.cuda.OutOfMemoryError:
        del x
        gc.collect()
        torch.cuda.empty_cache()
        raise


def save_measurements(df: pd.DataFrame, out_path: str | Path = "results/measurements.csv") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def measure_all(
    image_sizes=None,
    batch_sizes=None,
    *,
    model=None,
    device: str = "cuda",
    out_path: str | Path = "results/measurements.csv",
):
    """Run benchmark over all grid points and save CSV."""
    setup_environment()
    target = torch.device(device)
    grid = get_grid(image_sizes, batch_sizes)

    for s, _, _ in grid:
        if s % 16 != 0:
            raise ValueError(f"image_size {s} must be a multiple of 16")

    if model is None:
        model = CustomCNN().to(target)
    model.eval()

    nvml_handle = None
    nvml_initialized = False
    energy_method = "counter"

    if NVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            nvml_initialized = True
            dev_idx = target.index if target.index is not None else 0
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)
            if not _nvml_counter_supported(nvml_handle):
                energy_method = "power"
        except Exception as err:
            print(f"[measure.py] NVML unavailable ({err}); skipping energy.")

    records = []
    try:
        for idx, (s, b, is_val) in enumerate(grid, 1):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            try:
                lat, mem, energy_val = measure_point(
                    model, s, b, nvml_handle=nvml_handle, energy_method=energy_method
                )
                records.append({
                    "S": s, "B": b, "latency": lat, "memory": mem,
                    "energy": energy_val, "oom": False, "is_validation": is_val,
                })
                print(f"[{idx:03d}/{len(grid)}] S={s:3d}, B={b:3d} | "
                      f"Latency: {lat * 1000:.2f} ms | Mem: {mem / 1e6:.1f} MB | Val: {is_val}")
            except torch.cuda.OutOfMemoryError:
                print(f"[{idx:03d}/{len(grid)}] S={s:3d}, B={b:3d} | OOM")
                records.append({
                    "S": s, "B": b, "latency": None, "memory": None,
                    "energy": None, "oom": True, "is_validation": is_val,
                })
    finally:
        if nvml_initialized:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()

    df = pd.DataFrame(records)
    save_measurements(df, out_path)
    print(f"\nResults saved to {out_path}")
    return df


def run_all_measurements(output_csv: str | Path = "results/measurements.csv"):
    if not torch.cuda.is_available():
        print(f"[measure.py] CUDA not found. {CUDA_REQUIRED_MSG}")
        return None
    print(f"Benchmarking {len(get_grid())} points on {torch.cuda.get_device_name(0)}")
    return measure_all(out_path=output_csv)


if __name__ == "__main__":
    run_all_measurements()