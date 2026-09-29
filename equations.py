"""Analytical performance model of CustomCNN."""

import numpy as np

ArrayLike = float | np.ndarray

# Model constants (FP32)
NUM_WEIGHTS = 1_040_324
WEIGHTS_BYTES = NUM_WEIGHTS * 4.0  # 4 161 296 bytes

FLOP_S2_COEF = 17_714.0            # Convolutions + GAP per image
FLOP_HEAD = 313_344.0              # Linear layers per image
MEM_ACT_BYTES_PER_PIXEL = 52.0     # Peak at MaxPool: 12 (in) + 32 (conv) + 8 (pool)
TRAFFIC_S2_COEF = 104.0            # 26 elements * 4 bytes/pixel
TRAFFIC_HEAD = 3_472.0             # DRAM traffic for head tensors (bytes)


def flops(image_size: ArrayLike, batch: ArrayLike) -> np.ndarray:
    """Analytical FLOPs of one forward pass: B * (17714 * S^2 + 313344)."""
    s = np.asarray(image_size, dtype=np.float64)
    b = np.asarray(batch, dtype=np.float64)
    return b * (FLOP_S2_COEF * (s ** 2) + FLOP_HEAD)


def weights_memory_bytes() -> float:
    """Static weight memory in bytes."""
    return WEIGHTS_BYTES


def memory(image_size: ArrayLike, batch: ArrayLike) -> np.ndarray:
    """Peak GPU memory in bytes: Weights + 52 * B * S^2."""
    s = np.asarray(image_size, dtype=np.float64)
    b = np.asarray(batch, dtype=np.float64)
    return WEIGHTS_BYTES + MEM_ACT_BYTES_PER_PIXEL * b * (s ** 2)


def bytes_moved(image_size: ArrayLike, batch: ArrayLike) -> np.ndarray:
    """DRAM traffic in bytes: Weights + B * (104 * S^2 + 3472)."""
    s = np.asarray(image_size, dtype=np.float64)
    b = np.asarray(batch, dtype=np.float64)
    return WEIGHTS_BYTES + b * (TRAFFIC_S2_COEF * (s ** 2) + TRAFFIC_HEAD)


def latency(image_size: ArrayLike, batch: ArrayLike, theta: dict) -> np.ndarray:
    """Forward pass latency (seconds) using smooth-max Roofline and launch overhead."""
    s = np.asarray(image_size, dtype=np.float64)
    b = np.asarray(batch, dtype=np.float64)

    t_launch = theta.get("t_launch", 0.0)
    t_compute = flops(s, b) / theta.get("flops_peak", 8.1e12)
    t_memory = bytes_moved(s, b) / theta.get("bandwidth_peak", 300e9)
    p = theta.get("p_norm", 2.0)

    return t_launch + (t_compute ** p + t_memory ** p) ** (1.0 / p)


def energy(
    image_size: ArrayLike,
    batch: ArrayLike,
    theta_energy: dict,
    theta_latency: dict | None = None,
) -> np.ndarray:
    """Total GPU energy (Joules): p_idle * Latency + e_flop * FLOPs + e_byte * Bytes."""
    s = np.asarray(image_size, dtype=np.float64)
    b = np.asarray(batch, dtype=np.float64)

    lat_theta = theta_latency or theta_energy.get("latency") or theta_energy.get("theta_latency")
    lat = latency(s, b, lat_theta) if lat_theta else theta_energy.get("approx_latency", 0.0)

    p_idle = theta_energy.get("p_idle", 10.0)
    e_flop = theta_energy.get("e_flop", 1e-12)
    e_byte = theta_energy.get("e_byte", 1e-11)

    return p_idle * lat + e_flop * flops(s, b) + e_byte * bytes_moved(s, b)