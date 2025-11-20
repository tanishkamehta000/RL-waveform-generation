# Quick Start Guide - RL Waveform Anti-Jamming

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Run Simple Example (Fastest Way to Test)

```bash
python example_usage.py
```

This runs a small training demo (50 episodes, 30 steps each) and shows before/after comparison.

---

### 2. Train an Agent (Full Training)

```bash
# Basic training with default settings
python train_rl_agent.py

# Custom training
python train_rl_agent.py \
  --episodes 200 \
  --max_steps 50 \
  --learning_rate 0.1 \
  --output_dir my_training
```

**Output**: Agent saved to `rl_output/TIMESTAMP/training/agent_final.json`

---

### 3. Evaluate a Trained Agent

```bash
python train_rl_agent.py \
  --load rl_output/20250120_143000/training/agent_final.json \
  --evaluate \
  --max_steps 50
```

**Output**: Results in `rl_output/TIMESTAMP/evaluation/`

---

### 4. Use Custom Spectrum Data

```bash
python train_rl_agent.py \
  --bg path/to/your/background.csv \
  --floor path/to/your/floor.csv \
  --malicious path/to/your/jammer.csv \
  --episodes 100
```

---

## Understanding the Output

### Directory Structure
```
rl_output/
└── 20250120_143000/  # Timestamp
    ├── training/
    │   ├── agent_final.json         # ← Load this for evaluation
    │   └── training_progress.png    # ← Check learning curves
    └── evaluation/
        ├── evaluation_results.json  # ← Metrics comparison
        ├── waveform_trajectory.png  # ← See parameter evolution
        └── spectrum_comparison.png  # ← Visualize before/after
```

### Key Files to Check

1. **`training_progress.png`**: Are rewards increasing? Is exploration decaying?
2. **`evaluation_results.json`**: What's the SINR/goodput improvement?
3. **`spectrum_comparison.png`**: Did the agent move to a cleaner frequency?
4. **`waveform_trajectory.png`**: How did parameters evolve during optimization?

---

## Interpreting Results

### Good Training Signs ✅
- Episode rewards trend upward
- Epsilon decays from 1.0 → 0.01
- Best reward increases over time
- Q-table grows (agent exploring states)

### Evaluation Metrics
- **SINR improvement > 1 dB**: Agent found less jammed region
- **Goodput improvement > 0.1 Mbps**: Better throughput
- **Frequency shift ≠ 0**: Agent moved away from jammer
- **Modulation change**: Adapted to channel conditions

---

## Common Commands

### Train for longer
```bash
python train_rl_agent.py --episodes 500 --max_steps 100
```

### Train with more exploration
```bash
python train_rl_agent.py --epsilon_decay 0.999
```

### Train with faster learning
```bash
python train_rl_agent.py --learning_rate 0.2
```

### Continue training from checkpoint
```bash
python train_rl_agent.py \
  --load rl_output/20250120_143000/training/agent_checkpoint_ep100.json \
  --episodes 100
```

---

## Troubleshooting

### "No such file or directory" (spectrum data)
**Solution**: Update paths in command:
```bash
python train_rl_agent.py \
  --bg benign/background/YOUR_FILE.csv \
  --floor benign/background/YOUR_FILE.csv \
  --malicious benign/background/YOUR_FILE.csv
```

### Agent not improving
**Solution**: Train longer or adjust hyperparameters:
```bash
python train_rl_agent.py --episodes 300 --learning_rate 0.15
```

### Training is slow
**Solution**: Reduce steps or episodes:
```bash
python train_rl_agent.py --episodes 50 --max_steps 30
```

---

## Next Steps

1. **Read the full documentation**: `README_RL.md`
2. **Understand the formulas**: `FORMULAS_GUIDE.md`
3. **Explore the code**: `rl_waveform_agent.py`
4. **Tune hyperparameters**: See README_RL.md → "Hyperparameter Tuning"

---

## Quick Python API Usage

```python
from rl_waveform_agent import SpectrumEnvironment, RLWaveformAgent, WaveformParams, ActionSpace

# Load environment
env = SpectrumEnvironment('bg.csv', 'floor.csv', 'jammer.csv')

# Create initial waveform (modulation fixed to OFDM)
waveform = WaveformParams(center_freq_mhz=2412.0, bandwidth_khz=200.0, 
                         tx_power_dbm=0.0, modulation='OFDM')

# Evaluate initial performance
metrics = env.evaluate_waveform(waveform)
print(f"Initial SINR: {metrics['sinr_db']:.2f} dB")
print(f"Initial Goodput: {metrics['goodput_bps']/1e6:.3f} Mbps")

# Create and train agent (modulation fixed to OFDM)
action_space = ActionSpace(
    freq_steps_mhz=[-5, -2, 0, +2, +5],
    bw_multipliers=[0.75, 1.0, 1.25],
    power_steps_dbm=[-1, 0, +1],
)
agent = RLWaveformAgent(action_space)

# Train for 100 episodes
for _ in range(100):
    agent.train_episode(env, waveform, max_steps=50)

# Evaluate optimized waveform
optimized, trajectory = agent.evaluate(env, waveform, max_steps=50)
final_metrics = env.evaluate_waveform(optimized)
print(f"Optimized SINR: {final_metrics['sinr_db']:.2f} dB")
print(f"Optimized Goodput: {final_metrics['goodput_bps']/1e6:.3f} Mbps")
```

---

## Help

For detailed documentation, see:
- **README_RL.md** - Complete documentation
- **FORMULAS_GUIDE.md** - Mathematical formulas and code mapping
- **example_usage.py** - Annotated example code

For issues, check the "Troubleshooting" section in README_RL.md.

