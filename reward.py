"""Reward utilities for waveform optimization RL agents.

Implements the reward outlined in the design slide:

    R = w_G * G_tilde - w_B * B_tilde

where
    G = η * B * log2(1 + SINR) * (1 - BER)
    G_tilde ∈ [0, 1] is the normalized goodput term
    B_tilde ∈ [0, 1] is the normalized BER penalty

Latency is intentionally omitted per user request. Each component is
automatically normalized to [0, 1] and the provided weights are renormalized
so they sum to 1.0, ensuring the agent experiences stable reward magnitudes
across different scenarios.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

EPS = 1e-12


def q_function(x: float) -> float:
    """Complementary cumulative Gaussian (tail probability)."""
    if math.isinf(x):
        return 0.0 if x > 0 else 1.0
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def ber_from_sinr(sinr_linear: float, modulation: str) -> float:
    """Bit error rate using the Q-function fits from the reward slide."""
    mod = modulation.upper()
    sinr_linear = max(sinr_linear, EPS)
    if mod == "QPSK":
        argument = math.sqrt(2.0 * sinr_linear)
        ber = q_function(argument)
    elif mod == "16QAM":
        argument = math.sqrt(0.1 * sinr_linear)
        ber = (3.0 / 8.0) * q_function(argument)
    elif mod == "64QAM":
        argument = math.sqrt(0.07 * sinr_linear)
        ber = (7.0 / 24.0) * q_function(argument)
    else:
        raise ValueError(f"Unsupported modulation '{modulation}'. Use QPSK, 16QAM, or 64QAM.")
    return float(min(max(ber, EPS), 1.0))


def compute_goodput(bw_hz: float, sinr_linear: float, ber: float, eta: float) -> float:
    """Return goodput as η * B * log2(1 + SINR) * (1 - BER)."""
    bw_hz = max(bw_hz, 0.0)
    eta = max(min(eta, 1.0), 0.0)
    return float(eta * bw_hz * math.log2(1.0 + max(sinr_linear, EPS)) * (1.0 - ber))


def _normalized_weights(w_g: float, w_b: float) -> Tuple[float, float]:
    total = w_g + w_b
    if total <= 0.0:
        return 0.5, 0.5
    return w_g / total, w_b / total


def compute_reward(
    bw_hz: float,
    sinr_linear: float,
    modulation: str,
    *,
    eta: float = 0.7,
    w_g: float = 0.7,
    w_b: float = 0.3,
    sinr_ref_db: float = 30.0,
    ber_max: float = 0.5,
) -> Dict[str, float]:
    """Compute reward and return a breakdown dict.

    Args:
        bw_hz: Channel bandwidth (Hz)
        sinr_linear: Signal-to-interference-plus-noise ratio (linear)
        modulation: 'QPSK', '16QAM', or '64QAM'
        eta: Efficiency factor from the slide (0.5-0.8 typical)
        w_g: Raw weight for throughput preference
        w_b: Raw weight for reliability preference (penalizes BER)
        sinr_ref_db: Reference SINR (dB) that maps to max normalized goodput
        ber_max: BER value that maps to max penalty (default 0.5)
    """
    ber = ber_from_sinr(sinr_linear, modulation)
    goodput = compute_goodput(bw_hz, sinr_linear, ber, eta)

    sinr_ref_linear = max(10 ** (sinr_ref_db / 10.0), 1.0)
    g_max = compute_goodput(bw_hz, sinr_ref_linear, EPS, eta)
    if g_max <= 0.0:
        goodput_norm = 0.0
    else:
        goodput_norm = min(max(goodput / g_max, 0.0), 1.0)

    ber_max = max(ber_max, EPS)
    ber_norm = min(max(ber / ber_max, 0.0), 1.0)

    w_g_norm, w_b_norm = _normalized_weights(w_g, w_b)
    reward = w_g_norm * goodput_norm - w_b_norm * ber_norm

    return {
        "reward": float(reward),
        "goodput": float(goodput),
        "goodput_norm": float(goodput_norm),
        "ber": float(ber),
        "ber_norm": float(ber_norm),
        "weights": {
            "w_g": float(w_g_norm),
            "w_b": float(w_b_norm),
        },
        "eta": float(eta),
        "sinr_ref_db": float(sinr_ref_db),
        "ber_max": float(ber_max),
    }


__all__ = [
    "compute_reward",
    "compute_goodput",
    "ber_from_sinr",
]
