"""Visualization of CNN performance model and GPU measurements."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import equations


def plot_memory_vs_s(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot peak memory allocation vs image size S for selected batch sizes."""
    plt.figure(figsize=(8, 6))
    s_dense = np.linspace(32, 512, 100)

    for b in [1, 16, 64, 128]:
        sub = df[df["B"] == b]
        if not sub.empty:
            plt.scatter(sub["S"], sub["memory"] / 1e6, label=f"Measured B={b}")
        plt.plot(s_dense, equations.memory(s_dense, b) / 1e6, "--", label=f"Predicted B={b}")

    plt.xlabel("Image Size S")
    plt.ylabel("Peak Memory (MB)")
    plt.title("Peak Memory Allocation: Predicted vs Measured")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "memory_vs_S.png", dpi=300)
    plt.close()


def plot_surface(df: pd.DataFrame, theta: dict, out_dir: Path, mode: str = "latency") -> None:
    """Plot 3D model surface against train and validation points."""
    is_lat = mode == "latency"
    col = "latency" if is_lat else "energy"
    if col not in df.columns or (not is_lat and "energy" not in theta):
        return

    pts = df.dropna(subset=[col])
    scale = 1e3  # s -> ms or J -> mJ
    z_label = "Latency (ms)" if is_lat else "Energy (mJ)"
    title = f"Forward Pass {z_label.split()[0]}: Model Surface vs Real GPU"
    cmap = "viridis" if is_lat else "plasma"

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    tr = pts[~pts["is_validation"]]
    val = pts[pts["is_validation"]]
    ax.scatter(tr["S"], tr["B"], tr[col] * scale, color="blue", label="Train Points (Base)")
    ax.scatter(val["S"], val["B"], val[col] * scale, color="red", label="Validation Points (Unseen)")

    s_grid, b_grid = np.meshgrid(np.linspace(32, 512, 30), np.linspace(1, 256, 30))
    if is_lat:
        surf = equations.latency(s_grid, b_grid, theta) * scale
    else:
        surf = equations.energy(s_grid, b_grid, theta["energy"], theta_latency=theta) * scale

    ax.plot_surface(s_grid, b_grid, surf, alpha=0.4, cmap=cmap)
    ax.set_xlabel("Image Size S")
    ax.set_ylabel("Batch Size B")
    ax.set_zlabel(z_label)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{mode}_surface.png", dpi=300)
    plt.close()


def plot_roofline(df: pd.DataFrame, theta: dict, out_dir: Path) -> None:
    """Plot Roofline model with measured points and performance ceilings."""
    pts = df.dropna(subset=["latency"])
    if pts.empty:
        return

    s = pts["S"].to_numpy(dtype=float)
    b = pts["B"].to_numpy(dtype=float)
    fl = np.asarray(equations.flops(s, b), dtype=float)
    bm = np.asarray(equations.bytes_moved(s, b), dtype=float)
    lat = pts["latency"].to_numpy(dtype=float)

    intensity = fl / bm
    achieved = fl / lat

    flops_peak = theta.get("flops_peak", 8.1e12)
    bw_peak = theta.get("bandwidth_peak", 300e9)
    i_ridge = flops_peak / bw_peak

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(intensity, achieved, c=pts["B"], cmap="viridis", s=30)
    plt.colorbar(scatter, label="Batch Size B")

    i_line = np.geomspace(
        min(np.min(intensity) * 0.5, i_ridge * 0.05),
        max(np.max(intensity) * 2.0, i_ridge * 20.0),
        300,
    )
    roof_line = np.minimum(bw_peak * i_line, flops_peak)

    plt.plot(i_line, roof_line, "k--", label=f"Roofline (BW={bw_peak:.3g} B/s, Peak={flops_peak:.3g} FLOP/s)")
    plt.axhline(flops_peak, color="r", linestyle="--", label=f"Compute ceiling ({flops_peak:.3g} FLOP/s)")
    plt.scatter([i_ridge], [flops_peak], color="red", marker="*", s=200, label=f"Ridge I*={i_ridge:.3g} FLOP/byte")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Arithmetic Intensity (FLOPs / byte)")
    plt.ylabel("Achieved Performance (FLOP/s)")
    plt.title("Roofline: measured points vs theta ceilings")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roofline.png", dpi=300)
    plt.close()


def plot_parity(df: pd.DataFrame, theta: dict, out_dir: Path) -> None:
    """Plot predicted vs measured parity for latency, memory, and energy."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def _draw(ax, pred, meas, label_unit, title):
        ax.scatter(pred, meas, s=28, alpha=0.8, color="tab:blue")
        lo = min(np.min(pred), np.min(meas))
        hi = max(np.max(pred), np.max(meas))
        ax.plot([lo, hi], [lo, hi], "k--", label="y = x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Predicted {label_unit}")
        ax.set_ylabel(f"Measured {label_unit}")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    # 1. Latency
    lat_pts = df.dropna(subset=["latency"])
    s, b = lat_pts["S"].to_numpy(dtype=float), lat_pts["B"].to_numpy(dtype=float)
    meas_lat = lat_pts["latency"].to_numpy(dtype=float)
    pred_lat = np.asarray(equations.latency(s, b, theta), dtype=float)
    rel_lat = np.abs(pred_lat - meas_lat) / meas_lat
    _draw(axes[0], pred_lat, meas_lat, "latency (s)",
          f"Latency (median rel err={np.median(rel_lat):.3g}, max={np.max(rel_lat):.3g})")

    # 2. Peak Memory
    mem_pts = df.dropna(subset=["memory"])
    s, b = mem_pts["S"].to_numpy(dtype=float), mem_pts["B"].to_numpy(dtype=float)
    meas_mem = mem_pts["memory"].to_numpy(dtype=float) / 1e6
    pred_mem = np.asarray(equations.memory(s, b), dtype=float) / 1e6
    _draw(axes[1], pred_mem, meas_mem, "peak memory (MB)", "Memory parity")

    # 3. Energy
    if "energy" in theta and "energy" in df.columns:
        en_pts = df.dropna(subset=["energy"])
        s, b = en_pts["S"].to_numpy(dtype=float), en_pts["B"].to_numpy(dtype=float)
        meas_en = en_pts["energy"].to_numpy(dtype=float) * 1e3
        pred_en = np.asarray(equations.energy(s, b, theta["energy"], theta_latency=theta), dtype=float) * 1e3
        rel_en = np.abs(pred_en - meas_en) / meas_en
        _draw(axes[2], pred_en, meas_en, "energy (mJ)",
              f"Energy (median rel err={np.median(rel_en):.3g}, max={np.max(rel_en):.3g})")
    else:
        axes[2].axis("off")

    fig.suptitle("Predicted vs measured parity (assignment §8)")
    fig.tight_layout()
    fig.savefig(out_dir / "parity.png", dpi=300)
    plt.close(fig)


def generate_all_plots(
    csv_path: str | Path = "results/measurements.csv",
    theta_path: str | Path = "results/theta.json",
    out_dir: str | Path = "results/figures",
) -> None:
    """Build all figures from measurements CSV and theta JSON."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    valid = df[~df["oom"]]

    with open(theta_path, "r", encoding="utf-8") as f:
        theta = json.load(f)

    plot_memory_vs_s(valid, out_path)
    plot_surface(valid, theta, out_path, mode="latency")
    plot_surface(valid, theta, out_path, mode="energy")
    plot_roofline(valid, theta, out_path)
    plot_parity(valid, theta, out_path)

    print(f"Generated plots saved in {out_path}/")


if __name__ == "__main__":
    generate_all_plots()