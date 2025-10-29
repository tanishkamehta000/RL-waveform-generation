#!/usr/bin/env python3
"""Command-line runner for end-to-end analysis.

Usage examples (defaults assume workspace root):
  python code/run_analyze.py
  python code/run_analyze.py --background ./benign/background/location1 --floor ./benign/floor --out outputs

This script will:
- find background and floor CSVs
- compute P_env(f)
- find most jammed frequency
- synthesize a baseline IQ waveform (numpy .npy) and save metadata and a plot
"""
from pathlib import Path
import argparse
try:
    from code.analysis import (
        compute_environmental_power,
        find_most_jammed,
        estimate_bandwidth,
        synthesize_baseline,
        plot_env,
        save_metadata,
            compute_jammed_metrics,
    )
except Exception:
    # Fallback to support running as a script: `python code/run_analyze.py`
    import os
    import sys
    sys.path.append(os.path.dirname(__file__))
    from analysis import (
        compute_environmental_power,
        find_most_jammed,
        estimate_bandwidth,
        synthesize_baseline,
        plot_env,
        save_metadata,
            compute_jammed_metrics,
    )


def find_csv_files(folder: Path):
    return sorted([p for p in folder.rglob('*.csv')])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--background', '-b', default='./benign/background', help='Background scans directory')
    p.add_argument('--floor', '-f', default='./benign/floor', help='Floor scans directory')
    p.add_argument('--out', '-o', default='./outputs', help='Output directory')
    p.add_argument('--sample-rate', type=float, default=1e6, help='Sample rate for baseline IQ (Hz)')
    p.add_argument('--duration', type=float, default=0.1, help='Duration of baseline waveform (s)')
    p.add_argument('--amplitude', type=float, default=0.5, help='Amplitude of baseline IQ')
    p.add_argument('--tone-offset-hz', type=float, default=0.0, help='Tone offset for baseline (Hz)')
    p.add_argument('--baseline-type', choices=['tone', 'chirp'], default='tone', help='Baseline waveform type')
    p.add_argument('--chirp-bw-hz', type=float, default=0.0, help='Chirp bandwidth (Hz) when baseline-type=chirp')
    p.add_argument('--max-bg-files', type=int, default=None, help='Limit number of background CSVs to speed runs')
    p.add_argument('--max-floor-files', type=int, default=None, help='Limit number of floor CSVs to speed runs')
    args = p.parse_args()

    bg_dir = Path(args.background)
    fl_dir = Path(args.floor)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bg_files = find_csv_files(bg_dir)
    fl_files = find_csv_files(fl_dir)

    if args.max_bg_files is not None:
        bg_files = bg_files[: args.max_bg_files]
    if args.max_floor_files is not None:
        fl_files = fl_files[: args.max_floor_files]

    if len(bg_files) == 0:
        raise SystemExit(f"No background CSV files found in {bg_dir}")
    if len(fl_files) == 0:
        raise SystemExit(f"No floor CSV files found in {fl_dir}")

    print(f"Found {len(bg_files)} background files and {len(fl_files)} floor files")

    freqs, p_env = compute_environmental_power(bg_files, fl_files)

    center_freq, score = find_most_jammed(freqs, p_env)
    f_low, f_high, bw = estimate_bandwidth(freqs, p_env)
    print(f"Most jammed frequency: {center_freq} (P_env={score:.2f} dB), bandwidth ~ {bw:.2f}")

    # Save plot (or CSV if matplotlib unavailable) and capture actual artifact
    requested_plot_path = out_dir / 'p_env.png'
    plot_artifact_path = plot_env(freqs, p_env, requested_plot_path, peak_freq=center_freq, band=(f_low, f_high))

    # Synthesize baseline IQ (complex baseband tone). We treat center_freq as metadata; the IQ tone is generated at baseband.
    iq = synthesize_baseline(center_freq_hz=center_freq,
                             sample_rate=args.sample_rate,
                             duration=args.duration,
                             tone_offset_hz=args.tone_offset_hz,
                             amplitude=args.amplitude,
                             baseline_type=args.baseline_type,
                             chirp_bw_hz=args.chirp_bw_hz)
    # Compute jammedness metrics for the generated baseline
    metrics = compute_jammed_metrics(
        freqs, p_env, center_freq, f_low, f_high,
        baseline_type=args.baseline_type,
        tone_offset_hz=args.tone_offset_hz,
        chirp_bw_hz=args.chirp_bw_hz,
    )

    # Save IQ as CSV (i,q) to avoid binary formats and heavy deps
    iq_path = out_dir / 'baseline_iq.csv'
    with iq_path.open('w', newline='') as f:
        f.write('i,q\n')
        for c in iq:
            f.write(f"{c.real},{c.imag}\n")

    metadata = {
        'center_frequency': float(center_freq),
    'p_env_db': float(score),
    'band_low': float(f_low),
    'band_high': float(f_high),
    'bandwidth': float(bw),
        'n_background_files': len(bg_files),
        'n_floor_files': len(fl_files),
        'sample_rate': float(args.sample_rate),
        'duration': float(args.duration),
        'amplitude': float(args.amplitude),
        'plot': str(plot_artifact_path.name),
        'baseline_iq': iq_path.name,
        'baseline_type': args.baseline_type,
        'tone_offset_hz': float(args.tone_offset_hz),
        'chirp_bw_hz': float(args.chirp_bw_hz),
        'jammed_score_db': metrics['jammed_score_db'],
        'jammed_percentile': metrics['jammed_percentile'],
        'overlap_fraction': metrics['overlap_fraction'],
    }
    save_metadata(out_dir, metadata)

    print(f"Outputs written to {out_dir}")


if __name__ == '__main__':
    main()
