"""Calibration of latency and energy model parameters."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, nnls

import equations

# Unit suffixes for readable magnitudes.
MS = 1e-3      # seconds
GBPS = 1e9     # bytes per second
TFLOPS = 1e12  # FLOP per second


def _compute_validation_metrics(theta: dict, df: pd.DataFrame) -> dict:
    """Compute MAE and max relative error on held-out validation points."""
    val = df[(~df["oom"]) & (df["is_validation"])].dropna(subset=["latency"])
    if val.empty:
        return {}

    s = val["S"].to_numpy(dtype=float)
    b = val["B"].to_numpy(dtype=float)
    lat_actual = val["latency"].to_numpy(dtype=float)
    lat_pred = np.asarray(equations.latency(s, b, theta), dtype=float)

    abs_err = np.abs(lat_pred - lat_actual)
    metrics = {
        "latency_mae_s": float(np.mean(abs_err)),
        "latency_max_rel_err": float(np.max(abs_err / np.maximum(lat_actual, 1e-9))),
    }

    if "energy" in val.columns and "energy" in theta:
        en_val = val.dropna(subset=["energy"])
        if not en_val.empty:
            s_en = en_val["S"].to_numpy(dtype=float)
            b_en = en_val["B"].to_numpy(dtype=float)
            en_actual = en_val["energy"].to_numpy(dtype=float)
            en_pred = np.asarray(
                equations.energy(s_en, b_en, theta["energy"], theta_latency=theta),
                dtype=float,
            )
            metrics["energy_mae_j"] = float(np.mean(np.abs(en_pred - en_actual)))

    return metrics

def _func(X, t_launch, flops_peak, bandwidth_peak, p_norm):
    s_arr, b_arr = X
    theta = {
        "t_launch": t_launch,
        "flops_peak": flops_peak,
        "bandwidth_peak": bandwidth_peak,
        "p_norm": p_norm,
    }
    return equations.latency(s_arr, b_arr, theta)

def fit_latency(s: np.ndarray, b: np.ndarray, lat: np.ndarray) -> dict:
    """Fit Roofline parameters (t_launch, flops_peak, bandwidth_peak, p_norm)."""

    # Priors match the measurement GPU (RTX A1000 laptop: ~0.5 ms launch
    # overhead, ~2.5 TFLOPS FP32, ~192 GB/s); bounds have x10 headroom,
    # p_norm capped at 16 (above ~ hard max).
    p0 = [0.5 * MS, 2.5 * TFLOPS, 200 * GBPS, 2.0]
    bounds = ([0.0, 0.1 * TFLOPS, 10 * GBPS, 1.0], [10 * MS, 10 * TFLOPS, 1000 * GBPS, 16.0])
    x_scale = [0.1 * MS, 1 * TFLOPS, 100 * GBPS, 1.0]  # ~ parameter magnitudes

    # Weighted least squares to minimize relative error
    popt, _ = curve_fit(
        _func,
        (s, b),
        lat,
        p0=p0,
        bounds=bounds,
        sigma=np.maximum(lat, np.finfo(float).tiny),
        x_scale=x_scale,
        maxfev=20000,
    )
    return {
        "t_launch": float(popt[0]),
        "flops_peak": float(popt[1]),
        "bandwidth_peak": float(popt[2]),
        "p_norm": float(popt[3]),
    }


def fit_energy(s: np.ndarray, b: np.ndarray, energy: np.ndarray, theta: dict) -> dict:
    """Fit energy coefficients (p_idle, e_flop, e_byte) using NNLS."""
    sigma = np.maximum(energy, np.finfo(float).tiny)
    lat = np.asarray(equations.latency(s, b, theta), dtype=float)
    fl = np.asarray(equations.flops(s, b), dtype=float)
    bm = np.asarray(equations.bytes_moved(s, b), dtype=float)

    A = np.column_stack([lat / sigma, fl / sigma, bm / sigma])
    y = energy / sigma

    weights, _ = nnls(A, y)
    return {
        "p_idle": float(weights[0]),
        "e_flop": float(weights[1]),
        "e_byte": float(weights[2]),
    }


def calibrate(
    measurements: pd.DataFrame | str | Path = "results/measurements.csv",
    *,
    out_path: str | Path = "results/theta.json",
) -> dict:
    """Calibrate model on training subset and save parameters to JSON."""
    df = pd.read_csv(measurements) if isinstance(measurements, (str, Path)) else measurements
    train = df[(~df["oom"]) & (~df["is_validation"])].dropna(subset=["latency"])

    s_tr = train["S"].to_numpy(dtype=float)
    b_tr = train["B"].to_numpy(dtype=float)
    lat_tr = train["latency"].to_numpy(dtype=float)

    theta = fit_latency(s_tr, b_tr, lat_tr)

    if "energy" in train.columns:
        en_train = train.dropna(subset=["energy"])
        en_train = en_train[en_train["energy"] > 0]
        if len(en_train) > 5:
            energy_theta = fit_energy(
                en_train["S"].to_numpy(dtype=float),
                en_train["B"].to_numpy(dtype=float),
                en_train["energy"].to_numpy(dtype=float),
                theta,
            )
            energy_theta["latency"] = dict(theta)
            theta["energy"] = energy_theta

    metrics = _compute_validation_metrics(theta, df)
    if metrics:
        theta["validation"] = metrics
        print("Validation metrics:")
        for k, v in metrics.items():
            print(f"  {k} = {v:.6g}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(theta, f, indent=4)

    print(f"Parameters saved to {out_path}")
    return theta


def fit_parameters(
    csv_path: str | Path = "results/measurements.csv",
    out_json: str | Path = "results/theta.json",
) -> dict:
    """Alias for backwards compatibility."""
    return calibrate(csv_path, out_path=out_json)


if __name__ == "__main__":
    calibrate()