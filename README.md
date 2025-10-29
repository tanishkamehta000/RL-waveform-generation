# Passive Scan Jammed Region Analyzer


This repo analyzes FFT passive scans of a radio environment to:
- Compute environmental power per frequency: P_env(f) = P_background(f) - P_floor(f)
- Identify the most jammed frequency band
- Synthesize a baseline complex IQ waveform (tone or chirp) centered in the jammed region
- Output all results and metrics for RL or further analysis

## What it does
- Loads CSV scans under `benign/background/` and `benign/floor/`
- Computes P_env over the union of observed frequencies (alignment via interpolation)
- Finds the most jammed frequency (max P_env)
- Estimates a bandwidth around the peak using a -3 dB heuristic
- Produces a plot (PNG if matplotlib available, else CSV) and a baseline complex IQ file (CSV: i,q)


## Quick Start


1. (Optional) Create a Python virtual environment. No dependencies are required to run.

```bash
python -m venv .venv
source .venv/bin/activate
# Optional: pip install matplotlib  # for PNG plots
```


2. Run the analysis end-to-end (from repo root):

```bash
python code/run_analyze.py \
  --background ./benign/background \
  --floor ./benign/floor \
  --out ./outputs \
  --max-bg-files 50 \
  --max-floor-files 10
```


Outputs are written to `./outputs/`:
- `p_env.png` – plot of P_env with peak and estimated -3 dB bandwidth (or `p_env.csv` if matplotlib isn't installed)
- `baseline_iq.csv` – complex baseband IQ samples as two columns `i,q` (tone or chirp)
- `metadata.json` – summary with selected center frequency, bandwidth, baseline parameters, and jammedness metrics

## Success Checklist

- [x] Run completes with no errors
- [x] Outputs folder contains `metadata.json`, `baseline_iq.csv`, and `p_env.csv` or `p_env.png`
- [x] `metadata.json` includes:
  - `center_frequency`, `band_low`, `band_high`, `bandwidth`
  - `baseline_type`, `tone_offset_hz`, `chirp_bw_hz`
  - `jammed_score_db`, `jammed_percentile`, `overlap_fraction`
- [x] Baseline IQ is in the most jammed region (see jammed_percentile and overlap_fraction)


## Baseline Waveform Details
- IQ sampling rate: 1.0 MHz (configurable)
- Duration: 0.1 s (configurable)
- Tone is baseband (0 Hz offset). Treat `center_frequency` in the metadata as the RF center where this IQ would be upconverted.

You can load the IQ CSV in Python:

```python
import csv
iq = []
with open('outputs/baseline_iq.csv') as f:
  r = csv.DictReader(f)
  for row in r:
    iq.append(complex(float(row['i']), float(row['q'])))
```


## Baseline Options

You can customize the baseline IQ:

- Tone (default): single complex tone at --tone-offset-hz
- Chirp: linear sweep across --chirp-bw-hz over the duration

Examples:

```bash
/opt/homebrew/bin/python3 code/run_analyze.py \
  --background ./benign/background \
  --floor ./benign/floor \
  --out ./outputs_tone \
  --tone-offset-hz 1000

/opt/homebrew/bin/python3 code/run_analyze.py \
  --background ./benign/background \
  --floor ./benign/floor \
  --out ./outputs_chirp \
  --baseline-type chirp \
  --chirp-bw-hz 20000
```


## Notes
- CSV fields expected: `freq1, noise, max_magnitude, total_gain_db, base_pwr_db, rssi, relpwr_db, avgpwr_db`
- We use `avgpwr_db` as the per-row power measurement and average by `freq1` within each file.
- Frequencies in the sample CSVs appear in MHz (e.g., 5180, 5200...). We preserve those units.


## For RL Integration
- Use `metadata.json` to seed your waveform center frequency.
- The baseline IQ is intentionally simple; your agent can modify offset, bandwidth (e.g., chirps), or spectral shape.
