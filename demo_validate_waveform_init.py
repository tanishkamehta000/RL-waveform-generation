#!/usr/bin/env python3
"""
demo_validate_waveform_init.py

This demo initializes a waveform using the same pipeline as
`run_waveform_in_jammed_env.py` and performs a simple validation that the waveform
is placed in a congested area (i.e., the interference within the waveform mask
is significantly higher than the noise floor / baseline).

It writes `validation.json` in the output directory with pass/fail and metrics.

Usage: python demo_validate_waveform_init.py [--bg] [--floor] [--malicious] [--output_dir] ...
"""

import argparse
import json
from pathlib import Path
import numpy as np

# We'll import helper logic from run_waveform_in_jammed_env.py by reading it as a module
import importlib.util
import sys
from types import ModuleType

SCRIPT_PATH = Path(__file__).resolve().parent / 'run_waveform_in_jammed_env.py'


def load_module_from_path(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location('wf_module', str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['wf_module'] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg', default='benign/background/2.4ghz_passivescan_background_loc1_1.csv')
    parser.add_argument('--floor', default='benign/background/2.4ghz_passivescan_floor_1.csv')
    parser.add_argument('--malicious', default='benign/background/2412mhz_jamming_3dbm_gaussiannoise_1.csv')
    parser.add_argument('--center_freq_mhz', type=float, default=2412.0)
    parser.add_argument('--bandwidth_khz', type=float, default=200.0)
    parser.add_argument('--tx_power_dbm', type=float, default=0.0)
    parser.add_argument('--modulation', default='QPSK')
    parser.add_argument('--output_dir', default='waveform_run_output')
    args = parser.parse_args(argv)

    mod = load_module_from_path(SCRIPT_PATH)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load arrays using the same helpers
    bg = mod.load_csv_as_array(args.bg)
    floor = mod.load_csv_as_array(args.floor)
    mal = mod.load_csv_as_array(args.malicious)

    bg_lin = mod.to_linear_power(bg)
    floor_lin = mod.to_linear_power(floor)
    mal_lin = mod.to_linear_power(mal)

    # clamp to avoid negative or zero linear power issues
    bg_lin = mod.clamp_linear(bg_lin)
    floor_lin = mod.clamp_linear(floor_lin)
    mal_lin = mod.clamp_linear(mal_lin)

    # resample to max bin count
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

    freqs = mod.infer_frequencies(n_bins, args.bg, args.center_freq_mhz)

    baseline = mod.compute_baseline_background(bg_lin)
    floor_est = mod.compute_floor(floor_lin)
    jam_profile = mod.compute_jam_profile(mal_lin, floor_est)

    interference = baseline + jam_profile

    # Auto-choose best center if user passed center_freq_mhz <= 0
    center = args.center_freq_mhz
    if center <= 0:
        center = mod.choose_best_center(freqs, interference, args.bandwidth_khz)

    mask = mod.make_waveform_mask(freqs, center, args.bandwidth_khz)

    # Calculate average interference inside and outside mask
    interference = baseline + jam_profile
    inside_interf = interference[mask > 0.5]
    outside_interf = interference[mask <= 0.5]

    mean_inside = float(np.mean(inside_interf)) if inside_interf.size>0 else 0.0
    mean_outside = float(np.mean(outside_interf)) if outside_interf.size>0 else 0.0

    # Simple congestion criteria:
    # 1) mean interference inside mask is at least X dB above outside (we'll use 3 dB default)
    # 2) at least Y% of bins inside mask have jam_profile > threshold (threshold = median floor)
    threshold_db = 3.0
    # convert ratio to linear
    ratio_linear = 10 ** (threshold_db / 10.0)
    criteria1 = mean_inside >= ratio_linear * (mean_outside + 1e-15)

    # criteria2
    # More robust jam-detection: compare jam_profile inside mask against median jam_profile outside mask
    inside_mask_idx = (mask > 0.5)
    outside_mask_idx = (mask <= 0.5)
    if np.sum(inside_mask_idx) == 0:
        frac_jammed = 0.0
    else:
        if np.sum(outside_mask_idx) > 0:
            median_outside_jam = float(np.median(jam_profile[outside_mask_idx]))
        else:
            median_outside_jam = float(np.median(jam_profile))

        # require jam_profile inside to exceed median outside by factor
        factor = 1.5
        jam_bins = np.sum(jam_profile[inside_mask_idx] > (median_outside_jam * factor))
        frac_jammed = float(jam_bins) / float(max(1, np.sum(inside_mask_idx)))

    min_frac = 0.25
    criteria2 = frac_jammed >= min_frac

    passed = bool(criteria1 or criteria2)

    # note which criteria passed for debugging
    passed_reasons = {
        'criteria1_mean_inside_vs_outside': bool(criteria1),
        'criteria2_frac_jammed_bins': bool(criteria2)
    }

    validation = {
        'params': {
            'bg': args.bg,
            'floor': args.floor,
            'malicious': args.malicious,
            'center_freq_mhz': args.center_freq_mhz,
            'bandwidth_khz': args.bandwidth_khz,
            'tx_power_dbm': args.tx_power_dbm,
            'modulation': args.modulation
        },
        'metrics': {
            'mean_inside_interference_mw': mean_inside,
            'mean_outside_interference_mw': mean_outside,
            'mean_inside_over_outside_ratio': mean_inside / (mean_outside + 1e-15),
            'frac_jammed_bins_in_mask': frac_jammed,
            'criteria1_mean_ratio_threshold_linear': ratio_linear,
            'criteria2_min_frac': min_frac
        },
        'passed': passed
    }

    with open(outdir / 'validation.json', 'w') as f:
        json.dump(validation, f, indent=2)

    if passed:
        print('VALIDATION PASS: waveform initialized in congested area')
    else:
        print('VALIDATION FAIL: waveform not sufficiently in congested area')
    print('\nMetrics:\n', json.dumps(validation['metrics'], indent=2))
    print('\nPassed reasons:\n', json.dumps(passed_reasons, indent=2))


if __name__ == '__main__':
    main()
