#!/usr/bin/env python3
"""
run_waveform_in_jammed_env.py

Load background, floor, and malicious jammer CSV PSD/FFT scans, compute baseline/floor/jam
profiles, instantiate a test waveform, compute SINR, BER and reward, save results and plots.

Usage: python run_waveform_in_jammed_env.py [--bg PATH] [--floor PATH] [--malicious PATH]

Defaults point to /mnt/data/ files but can be overridden.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import pandas as pd


def guess_db_or_linear(arr: np.ndarray) -> str:
    # look at central portion of values
    a = arr.flatten()
    a = a[~np.isnan(a)]
    if a.size == 0:
        return "linear"
    q1, q99 = np.percentile(a, [1, 99])
    if q1 >= -200 and q99 <= 60:
        return "db"
    return "linear"


def load_csv_as_array(path: str) -> np.ndarray:
    # Use pandas to robustly handle optional headers and non-numeric columns
    try:
        df = pd.read_csv(path)
    except Exception:
        # fallback to numpy loader for very simple files
        data = np.loadtxt(path, delimiter=",", ndmin=2)
        return data

    # select only numeric columns (this will drop timestamps or string headers)
    num_df = df.select_dtypes(include=["number"]).copy()
    if num_df.shape[1] == 0:
        # try converting all columns to numeric coercing errors to NaN then drop non-numeric
        num_df = df.apply(pd.to_numeric, errors="coerce")
        num_df = num_df.dropna(axis=1, how="all")

    arr = num_df.to_numpy(dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def to_linear_power(arr: np.ndarray) -> np.ndarray:
    # detect dB vs linear
    mode = guess_db_or_linear(arr)
    if mode == "db":
        return np.power(10.0, arr / 10.0)
    return arr


def clamp_linear(arr: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    # ensure no negative or zero values in linear power arrays
    a = np.array(arr, dtype=float)
    a[np.isnan(a)] = 0.0
    a[a < eps] = eps
    return a


def choose_best_center(freqs_mhz: np.ndarray, interference: np.ndarray, bw_khz: float) -> float:
    """Scan possible centers across the frequency axis and pick the center
    (in MHz) that yields the highest mean interference inside the mask.
    """
    bw_mhz = bw_khz / 1000.0
    centers = freqs_mhz
    best_center = float(freqs_mhz[len(freqs_mhz) // 2])
    best_score = -1.0
    half_bw = bw_mhz / 2.0
    for c in centers:
        low = c - half_bw
        high = c + half_bw
        mask = (freqs_mhz >= low) & (freqs_mhz <= high)
        if np.sum(mask) == 0:
            continue
        score = np.mean(interference[mask])
        if score > best_score:
            best_score = float(score)
            best_center = float(c)
    return best_center


def choose_best_center_by_profile(freqs_mhz: np.ndarray, profile: np.ndarray, bw_khz: float) -> float:
    """Pick center that maximizes mean value of `profile` inside the mask.

    `profile` can be the jam_profile array (mean jamming energy per bin) so the
    waveform is placed where jammer energy is largest.
    """
    bw_mhz = bw_khz / 1000.0
    centers = freqs_mhz
    best_center = float(freqs_mhz[len(freqs_mhz) // 2])
    best_score = -1.0
    half_bw = bw_mhz / 2.0
    for c in centers:
        low = c - half_bw
        high = c + half_bw
        mask = (freqs_mhz >= low) & (freqs_mhz <= high)
        if np.sum(mask) == 0:
            continue
        score = np.mean(profile[mask])
        if score > best_score:
            best_score = float(score)
            best_center = float(c)
    return best_center


def infer_frequencies(n_bins: int, file_path: str, center_freq_mhz: float = 2412.0) -> np.ndarray:
    # Heuristic based on file path/name
    fname = file_path.lower()
    if "2.4" in fname or "2412" in fname or "2.4ghz" in fname or "2400" in fname:
        start = 2400.0
        end = 2483.5
    elif "5ghz" in fname or "5.0" in fname or "5170" in fname:
        start = 5170.0
        end = 5835.0
    else:
        # center around center_freq_mhz with +/- 40 MHz
        half = 40.0
        start = center_freq_mhz - half
        end = center_freq_mhz + half
    return np.linspace(start, end, n_bins)


def compute_baseline_background(bg_linear: np.ndarray) -> np.ndarray:
    # 10th percentile across time axis (rows)
    return np.percentile(bg_linear, 10, axis=0)


def compute_floor(floor_linear: np.ndarray) -> np.ndarray:
    # median across time
    return np.median(floor_linear, axis=0)


def compute_jam_profile(mal_linear: np.ndarray, floor: np.ndarray) -> np.ndarray:
    mean_mal = np.mean(mal_linear, axis=0)
    jam = mean_mal - floor
    jam[jam < 0] = 0.0
    return jam


def make_waveform_mask(freqs_mhz: np.ndarray, center_mhz: float, bw_khz: float) -> np.ndarray:
    bw_mhz = bw_khz / 1000.0
    low = center_mhz - bw_mhz / 2.0
    high = center_mhz + bw_mhz / 2.0
    mask = (freqs_mhz >= low) & (freqs_mhz <= high)
    return mask.astype(float)


def dbm_to_mw(dbm: float) -> float:
    # mW
    return 10 ** (dbm / 10.0)


def mw_to_linear_power_per_bin(total_mw: float, n_bins: int) -> float:
    # uniform distribution
    if n_bins <= 0:
        return 0.0
    return total_mw / float(n_bins)


def sinr_and_metrics(signal_per_bin: np.ndarray, interference_per_bin: np.ndarray, eps=1e-12):
    P_sig = np.sum(signal_per_bin)
    P_int = np.sum(interference_per_bin) + eps
    sinr_linear = P_sig / P_int
    sinr_db = 10.0 * np.log10(sinr_linear + eps)
    return sinr_linear, sinr_db, P_sig, P_int


def ber_from_sinr(sinr_linear: float, modulation: str) -> float:
    # approximate AWGN BER formulas
    # QPSK ~ Q(sqrt(2*Eb/N0)) -> BER ~= 0.5*erfc(sqrt(sinr_linear)) for symbol SNR
    # For M-QAM approximate via union bound: BER ~= (4/log2(M))*(1-1/np.sqrt(M))*0.5*erfc(np.sqrt(3*log2(M)/(M-1)*sinr_linear))
    M = 4 if modulation.upper() == "QPSK" else int(modulation.replace("QAM", ""))
    if M == 4:
        # QPSK
        ber = 0.5 * erfc(np.sqrt(sinr_linear))
        return float(np.clip(ber, 1e-12, 1.0))
    else:
        k = np.log2(M)
        argument = np.sqrt((3.0 * k / (M - 1.0)) * sinr_linear)
        ber_approx = (4.0 / k) * (1 - 1.0 / np.sqrt(M)) * 0.5 * erfc(argument / np.sqrt(2.0))
        return float(np.clip(ber_approx, 1e-12, 1.0))


def compute_goodput(bw_hz: float, sinr_linear: float, ber: float) -> float:
    # G = B * log2(1+SINR) * (1-BER)
    return bw_hz * np.log2(1.0 + sinr_linear) * (1.0 - ber)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bg", default="/mnt/data/2.4ghz_passivescan_background_loc1_1.csv")
    parser.add_argument("--floor", default="/mnt/data/2.4ghz_passivescan_floor_294.csv")
    parser.add_argument("--malicious", default="/mnt/data/2412mhz_jamming_3dbm_gaussian_noise.csv")
    parser.add_argument("--center_freq_mhz", type=float, default=2412.0)
    parser.add_argument("--bandwidth_khz", type=float, default=200.0)
    parser.add_argument("--tx_power_dbm", type=float, default=0.0)
    parser.add_argument("--modulation", default="QPSK", choices=["QPSK", "16QAM", "64QAM"])
    parser.add_argument("--output_dir", default="/mnt/data/waveform_run_output")
    args = parser.parse_args(argv)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load files
    bg_path = args.bg
    floor_path = args.floor
    mal_path = args.malicious

    print(f"Loading background from {bg_path}")
    bg = load_csv_as_array(bg_path)
    print(f"Loading floor from {floor_path}")
    floor_arr = load_csv_as_array(floor_path)
    print(f"Loading malicious from {mal_path}")
    mal = load_csv_as_array(mal_path)

    # Convert to linear
    bg_lin = to_linear_power(bg)
    floor_lin = to_linear_power(floor_arr)
    mal_lin = to_linear_power(mal)

    # Clamp to non-negative small floor to avoid negative/zero linear power issues
    bg_lin = clamp_linear(bg_lin)
    floor_lin = clamp_linear(floor_lin)
    mal_lin = clamp_linear(mal_lin)

    # Ensure same bin counts; if different, attempt to resample by interpolation to mal's bin count
    n_bins = max(bg_lin.shape[1], floor_lin.shape[1], mal_lin.shape[1])

    def resample_rows(arr, target_bins):
        if arr.shape[1] == target_bins:
            return arr
        x_old = np.linspace(0.0, 1.0, arr.shape[1])
        x_new = np.linspace(0.0, 1.0, target_bins)
        res = np.array([np.interp(x_new, x_old, row) for row in arr])
        return res

    if bg_lin.shape[1] != n_bins:
        bg_lin = resample_rows(bg_lin, n_bins)
    if floor_lin.shape[1] != n_bins:
        floor_lin = resample_rows(floor_lin, n_bins)
    if mal_lin.shape[1] != n_bins:
        mal_lin = resample_rows(mal_lin, n_bins)

    # Infer frequencies
    freqs = infer_frequencies(n_bins, bg_path, args.center_freq_mhz)

    baseline = compute_baseline_background(bg_lin)
    floor_est = compute_floor(floor_lin)
    jam_profile = compute_jam_profile(mal_lin, floor_est)

    # Save intermediates
    np.save(outdir / "baseline.npy", baseline)
    np.save(outdir / "floor.npy", floor_est)
    np.save(outdir / "jam_profile.npy", jam_profile)

    # Build interference and consider auto-placement of waveform center
    interference = baseline + jam_profile

    center_to_use = args.center_freq_mhz
    # if user passes -1 for center frequency, auto-select best congested center
    if float(center_to_use) < 0:
        # prefer selecting center based on jam profile peaks
        center_to_use = choose_best_center_by_profile(freqs, jam_profile, args.bandwidth_khz)

    # Waveform mask and power allocation
    mask = make_waveform_mask(freqs, center_to_use, args.bandwidth_khz)
    np.save(outdir / "mask.npy", mask)

    # Tx power conversion (dBm -> mW) and distribute across mask bins
    total_tx_mw = dbm_to_mw(args.tx_power_dbm)
    n_on_bins = int(np.sum(mask > 0.5))
    if n_on_bins == 0:
        print("Warning: waveform mask selects 0 frequency bins. Expanding to nearest bin.")
        # pick nearest bin to center
        idx = np.argmin(np.abs(freqs - args.center_freq_mhz))
        mask[idx] = 1.0
        n_on_bins = 1

    per_bin_mw = mw_to_linear_power_per_bin(total_tx_mw, n_on_bins)
    signal_per_bin = mask * per_bin_mw

    # Interference per bin is baseline + jam profile (both already linear)
    interference_per_bin = baseline + jam_profile

    sinr_linear, sinr_db, P_sig, P_int = sinr_and_metrics(signal_per_bin, interference_per_bin)

    ber = ber_from_sinr(sinr_linear, args.modulation)

    bw_hz = args.bandwidth_khz * 1000.0
    G = compute_goodput(bw_hz, sinr_linear, ber)

    # Normalize G by an arbitrary factor: max theoretical rate for M-QAM with same bw: B*log2(1+SNR_max)
    # For reward, penalize transmit power (in mW) with small coefficient
    max_theoretical = bw_hz * np.log2(1.0 + 1e3)  # arbitrary high SNR
    G_norm = G / (max_theoretical + 1e-12)
    power_cost = total_tx_mw * 1e-3
    reward = G_norm - 0.01 * power_cost

    # Save JSON
    out = {
        "params": {
            "bg": bg_path,
            "floor": floor_path,
            "malicious": mal_path,
            "center_freq_mhz": args.center_freq_mhz,
            "bandwidth_khz": args.bandwidth_khz,
            "tx_power_dbm": args.tx_power_dbm,
            "modulation": args.modulation,
            "n_bins": int(n_bins),
        },
        "results": {
            "P_sig_mw": float(P_sig),
            "P_int_mw": float(P_int),
            "sinr_linear": float(sinr_linear),
            "sinr_db": float(sinr_db),
            "ber": float(ber),
            "goodput_bps": float(G),
            "goodput_normalized": float(G_norm),
            "reward": float(reward),
        }
    }

    with open(outdir / "results.json", "w") as f:
        json.dump(out, f, indent=2)

    # Plotting: overlay baseline, floor, jam and mask (in dB)
    plt.figure(figsize=(10, 6))
    eps = 1e-15
    plt.plot(freqs, 10.0 * np.log10(baseline + eps), label="baseline (10th pct) dB")
    plt.plot(freqs, 10.0 * np.log10(floor_est + eps), label="floor (median) dB")
    plt.plot(freqs, 10.0 * np.log10(jam_profile + eps), label="jam profile (mean-floor) dB")
    # mask as -inf for zeros, and show signal power
    sig_db = 10.0 * np.log10(signal_per_bin + eps)
    plt.plot(freqs, sig_db, label="signal per-bin dB (tx) ")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dB or dBm equivalent)")
    plt.legend()
    plt.grid(True)
    plt.title(f"Waveform and environment around {args.center_freq_mhz} MHz")
    plot_path = outdir / "spectra_overlay.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Print summary
    print(f"SINR (linear) = {sinr_linear:.6e}")
    print(f"SINR (dB) = {sinr_db:.3f} dB")
    print(f"BER = {ber:.6e}")
    print(f"Goodput (bps) = {G:.3f}")
    print(f"Reward = {reward:.6e}")
    print(f"Saved outputs to {outdir}")


if __name__ == "__main__":
    main()
