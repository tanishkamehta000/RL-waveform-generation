"""analysis.py

Utilities to load background/floor CSV FFT scans, compute environmental
power per frequency P_env(f) = P_background(f) - P_floor(f), find the
most jammed frequency, estimate jammed bandwidth, and synthesize a baseline
complex IQ waveform centered on that frequency.

Assumptions inferred from CSV samples:
- CSV columns: freq1,noise,max_magnitude,total_gain_db,base_pwr_db,rssi,relpwr_db,avgpwr_db
- We'll use `avgpwr_db` as the measured power for each row.
"""
from pathlib import Path
import csv
import json
from typing import Dict, Tuple, List, Optional


def load_file_avg_power(path: Path, power_col: str = "avgpwr_db") -> Dict[float, float]:
    """Load a CSV and return mean power per frequency (freq1) using csv reader.

    Returns dict: {freq: mean_power_db}
    """
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        col_idx = {name: i for i, name in enumerate(header)}
        if "freq1" not in col_idx or power_col not in col_idx:
            raise ValueError(f"Unexpected CSV columns in {path}: {header}")
        freq_i = col_idx["freq1"]
        pow_i = col_idx[power_col]
        accum: Dict[float, list[float]] = {}
        for row in reader:
            try:
                freq = float(row[freq_i])
                pwr = float(row[pow_i])
            except (ValueError, IndexError):
                continue
            accum.setdefault(freq, []).append(pwr)
    return {k: (sum(v) / len(v)) for k, v in accum.items()}


def aggregate_mean_power(files: list[Path]) -> Tuple[List[float], List[float]]:
    """Given a list of CSV files, compute the mean power per frequency across files.

    Returns (freqs, mean_powers) where freqs is sorted ascending array of frequency (MHz or Hz per CSV) values and mean_powers the corresponding mean power (dB).
    """
    accum: Dict[float, list[float]] = {}
    for f in files:
        d = load_file_avg_power(f)
        for freq, p in d.items():
            accum.setdefault(freq, []).append(float(p))
    freqs = sorted(accum.keys())
    means = [sum(accum[f]) / len(accum[f]) for f in freqs]
    return freqs, means


def compute_environmental_power(background_files: list[Path], floor_files: list[Path]) -> Tuple[List[float], List[float]]:
    """Compute P_env(f) = P_background(f) - P_floor(f).

    If a frequency exists in background but not in floor (or vice versa), missing values are handled by aligning frequencies and using nearest available value.
    Returns (freqs, p_env)
    """
    bg_freqs, bg_means = aggregate_mean_power(background_files)
    fl_freqs, fl_means = aggregate_mean_power(floor_files)

    # Align to union of freqs
    union = sorted(set(bg_freqs + fl_freqs))

    def interp(xs: List[float], ys: List[float], xq: float) -> float:
        n = len(xs)
        if n == 0:
            return 0.0
        if xq <= xs[0]:
            return ys[0]
        if xq >= xs[-1]:
            return ys[-1]
        lo, hi = 0, n - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if xs[mid] == xq:
                return ys[mid]
            if xs[mid] < xq:
                lo = mid
            else:
                hi = mid
        x0, x1 = xs[lo], xs[hi]
        y0, y1 = ys[lo], ys[hi]
        if x1 == x0:
            return y0
        t = (xq - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    bg_interp = [interp(bg_freqs, bg_means, x) for x in union]
    fl_interp = [interp(fl_freqs, fl_means, x) for x in union]
    p_env = [b - f for b, f in zip(bg_interp, fl_interp)]
    return union, p_env


def find_most_jammed(freqs: List[float], p_env: List[float]) -> Tuple[float, float]:
    """Return (center_frequency, score) of the most jammed bin.

    center_frequency is the frequency with maximum P_env. Score is the P_env value.
    """
    if not freqs:
        return 0.0, 0.0
    idx = max(range(len(p_env)), key=lambda i: p_env[i])
    return float(freqs[idx]), float(p_env[idx])


def estimate_bandwidth(freqs: List[float], p_env: List[float], drop_db: float = 3.0) -> Tuple[float, float, float]:
    """Estimate jammed bandwidth around the peak using a -drop_db method.

    Returns (f_low, f_high, bw) in the same units as freqs (likely MHz).
    """
    if len(freqs) == 0:
        return 0.0, 0.0, 0.0
    peak_idx = max(range(len(p_env)), key=lambda i: p_env[i])
    peak_val = p_env[peak_idx]
    thresh = peak_val - drop_db

    # Walk left
    i = peak_idx
    while i > 0 and p_env[i] >= thresh:
        i -= 1
    f_low = freqs[max(i, 0)]

    # Walk right
    j = peak_idx
    while j < len(p_env) - 1 and p_env[j] >= thresh:
        j += 1
    f_high = freqs[min(j, len(freqs) - 1)]

    bw = float(f_high - f_low)
    return float(f_low), float(f_high), bw


def compute_jammed_metrics(
    freqs: List[float],
    p_env: List[float],
    center_freq: float,
    band_low: float,
    band_high: float,
    baseline_type: str = "tone",
    tone_offset_hz: float = 0.0,
    chirp_bw_hz: float = 0.0,
) -> Dict[str, float]:
    """Compute how jammed the baseline is using P_env.

    Metrics returned:
    - jammed_score_db: P_env at the center frequency (tone) or mean P_env across chirp band
    - jammed_percentile: percentile rank of jammed_score_db among P_env samples (0-100, higher=more jammed)
    - overlap_fraction: fraction of the baseline bandwidth that overlaps with the estimated -3 dB jam band

    Units: freqs and band_* are in CSV units (likely MHz). tone_offset_hz and chirp_bw_hz are Hz and will be converted.
    """
    # Helper: linear interpolation over monotonic freqs
    def interp(xs: List[float], ys: List[float], xq: float) -> float:
        n = len(xs)
        if n == 0:
            return 0.0
        if xq <= xs[0]:
            return ys[0]
        if xq >= xs[-1]:
            return ys[-1]
        lo, hi = 0, n - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if xs[mid] == xq:
                return ys[mid]
            if xs[mid] < xq:
                lo = mid
            else:
                hi = mid
        x0, x1 = xs[lo], xs[hi]
        y0, y1 = ys[lo], ys[hi]
        if x1 == x0:
            return y0
        t = (xq - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    # Determine baseline spectral coverage in freq units
    # Convert Hz to MHz if freqs look like MHz (e.g., values in the thousands)
    # Heuristic: if max(freqs) > 1000, assume MHz.
    use_mhz = (len(freqs) > 0 and max(freqs) > 1000.0)
    scale = 1e6 if use_mhz else 1.0

    # Effective baseline band
    if baseline_type.lower() == "chirp" and chirp_bw_hz > 0:
        half_bw = (chirp_bw_hz / scale) / 2.0
    else:
        half_bw = 0.0
    baseline_low = center_freq - half_bw
    baseline_high = center_freq + half_bw

    # Jammed score: point value at center (tone) or average across band (chirp)
    if half_bw == 0.0:
        score = interp(freqs, p_env, center_freq)
    else:
        # Average p_env sampled across baseline band
        if baseline_high <= baseline_low:
            score = interp(freqs, p_env, center_freq)
        else:
            # Sample ~200 points across the band for a smooth average
            n_samples = 200
            step = (baseline_high - baseline_low) / (n_samples - 1)
            total = 0.0
            for k in range(n_samples):
                fq = baseline_low + k * step
                total += interp(freqs, p_env, fq)
            score = total / n_samples

    # Percentile rank of score within P_env distribution
    if len(p_env) == 0:
        percentile = 0.0
    else:
        below = sum(1 for v in p_env if v <= score)
        percentile = 100.0 * below / len(p_env)

    # Overlap with -3 dB jam band
    if half_bw == 0.0:
        overlap = 1.0 if (band_low <= center_freq <= band_high) else 0.0
    else:
        # Intersection length / baseline bandwidth
        baseline_bw = (baseline_high - baseline_low)
        inter_low = max(band_low, baseline_low)
        inter_high = min(band_high, baseline_high)
        inter = max(0.0, inter_high - inter_low)
        overlap = (inter / baseline_bw) if baseline_bw > 0 else 0.0

    return {
        "jammed_score_db": float(score),
        "jammed_percentile": float(percentile),
        "overlap_fraction": float(overlap),
    }


def synthesize_baseline(center_freq_hz: float,
                        sample_rate: float = 1e6,
                        duration: float = 0.1,
                        tone_offset_hz: float = 0.0,
                        amplitude: float = 0.5,
                        baseline_type: str = "tone",
                        chirp_bw_hz: float = 0.0) -> List[complex]:
    """Create a complex baseband IQ tone centered at center_freq_hz.

    Implementation detail: we synthesize complex baseband at low sample_rate where the
    center_freq_hz is treated as metadata (carrier). The returned array contains complex64 IQ samples
    representing a tone at `tone_offset_hz` relative to the center frequency.

    Returns a list of complex IQ samples.
    """
    import math
    nsamples = int(round(sample_rate * duration))
    iq: List[complex] = []
    if baseline_type.lower() == "chirp" and chirp_bw_hz > 0:
        # Linear chirp from -bw/2 to +bw/2 over duration at baseband
        f0 = -chirp_bw_hz / 2.0
        k = chirp_bw_hz / duration  # Hz per second
        for n in range(nsamples):
            t = n / sample_rate
            inst_freq = f0 + k * t  # Hz
            angle = 2.0 * math.pi * (tone_offset_hz * t + (f0 * t + 0.5 * k * t * t))
            iq.append(complex(amplitude * math.cos(angle), amplitude * math.sin(angle)))
    else:
        # Simple tone at tone_offset_hz
        for n in range(nsamples):
            t = n / sample_rate
            angle = 2.0 * math.pi * tone_offset_hz * t
            iq.append(complex(amplitude * math.cos(angle), amplitude * math.sin(angle)))
    return iq


def plot_env(freqs: List[float], p_env: List[float], outpath: Path,
             peak_freq: Optional[float] = None,
             band: Optional[Tuple[float, float]] = None) -> Path:
    """Plot P_env if matplotlib is available; otherwise save CSV alongside.

    When matplotlib is unavailable, writes a simple CSV with columns: freq,p_env.
    """
    try:
        import matplotlib.pyplot as plt  # lazy import to keep dependency optional
    except Exception:
        # Fallback: write CSV
        out_csv = outpath.with_suffix('.csv')
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["freq", "p_env_db"])
            for fr, pe in zip(freqs, p_env):
                w.writerow([fr, pe])
        return out_csv

    plt.figure(figsize=(11, 4))
    plt.plot(freqs, p_env, '-k', linewidth=1.0, label='P_env')
    if peak_freq is not None:
        try:
            peak_idx = min(range(len(freqs)), key=lambda i: abs(freqs[i] - float(peak_freq)))
            plt.plot([float(peak_freq)], [p_env[peak_idx]], 'ro', label='Peak')
        except Exception:
            pass
    if band is not None:
        f_low, f_high = band
        plt.axvline(f_low, color='orange', linestyle='--', linewidth=1.0, label='-3 dB edges')
        plt.axvline(f_high, color='orange', linestyle='--', linewidth=1.0)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('P_env (dB)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.close()
    return outpath


def save_metadata(outdir: Path, metadata: dict):
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    print("analysis.py is a library. Use run_analyze.py to run end-to-end.")
