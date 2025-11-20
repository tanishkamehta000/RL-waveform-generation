# Reinforcement Learning for Waveform Anti-Jamming

## Overview

This project implements a **Reinforcement Learning (RL) agent** that learns to optimize radio waveform parameters in the presence of jamming. The agent iteratively adjusts waveform characteristics to maximize communication performance (goodput) while avoiding interference.

## Key Features

- ✅ **Q-Learning Agent**: Tabular Q-learning with epsilon-greedy exploration
- ✅ **Multi-Parameter Optimization**: Tunes center frequency, bandwidth, and TX power (OFDM modulation)
- ✅ **Explainable Design**: Clear reward function and documented decision-making
- ✅ **Comprehensive Metrics**: Tracks SINR, BER, goodput, and reward over time
- ✅ **Visualization**: Plots training progress, optimization trajectories, and spectrum analysis

## Reward Function

The agent optimizes a reward function based on **goodput** (without latency):

```
R = w_G × G_normalized - w_B × B_normalized
```

Where:
- **G (Goodput)** = Bandwidth × log₂(1 + SINR) × (1 - BER)
- **SINR** = Signal Power / (Noise + Jamming)
- **BER** depends on modulation scheme (QPSK, 16-QAM, 64-QAM)

### Component Definitions

**SINR (Signal-to-Interference-plus-Noise Ratio)**:
```
SINR = P_signal / (P_noise + P_jamming)
```

**BER (Bit Error Rate)** - for OFDM with QPSK subcarrier modulation:
- OFDM (QPSK subcarriers): BER = 0.5 × erfc(√SINR)

**Goodput** (effective data rate):
```
Goodput = Bandwidth × log₂(1 + SINR) × (1 - BER)
```

## Architecture

### 1. **SpectrumEnvironment** (`rl_waveform_agent.py`)
   - Loads background noise, noise floor, and jammer spectrum data
   - Computes interference profiles across frequency bins
   - Evaluates waveform performance (SINR, BER, goodput)
   - Provides state features for the RL agent

### 2. **RLWaveformAgent** (`rl_waveform_agent.py`)
   - Implements tabular Q-learning with epsilon-greedy exploration
   - Maintains Q-table: state → action values
   - Learns optimal policy through trial and error
   - Actions modify waveform parameters:
     - Adjust center frequency (±1 to ±10 MHz)
     - Scale bandwidth (0.5× to 1.5×)
     - Adjust TX power (±1 to ±3 dBm)
   - **Note**: Modulation is fixed to OFDM

### 3. **Training Loop** (`train_rl_agent.py`)
   - Trains agent over multiple episodes
   - Each episode: agent takes up to 50 steps to optimize waveform
   - Tracks cumulative rewards and best configurations
   - Saves checkpoints and visualizations

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Required packages:
# - numpy
# - pandas
# - matplotlib
# - scipy
```

## Usage

### Quick Start: Train an Agent

```bash
# Basic training (100 episodes, 50 steps each)
python train_rl_agent.py --episodes 100 --max_steps 50

# Training with custom hyperparameters
python train_rl_agent.py \
  --episodes 200 \
  --max_steps 100 \
  --learning_rate 0.15 \
  --epsilon_decay 0.99 \
  --output_dir my_training_run
```

### Evaluate a Trained Agent

```bash
# Load and evaluate a trained agent
python train_rl_agent.py \
  --load rl_output/20250120_143000/training/agent_final.json \
  --evaluate \
  --max_steps 50
```

### Custom Environment Data

```bash
# Train with different spectrum files
python train_rl_agent.py \
  --bg path/to/background.csv \
  --floor path/to/floor.csv \
  --malicious path/to/jammer.csv \
  --episodes 150
```

## Output Structure

After training, the output directory contains:

```
rl_output/
└── 20250120_143000/  # Timestamp
    ├── args.json                    # Command-line arguments
    ├── training/
    │   ├── agent_final.json         # Trained agent (Q-table + params)
    │   ├── agent_checkpoint_ep50.json
    │   ├── agent_checkpoint_ep100.json
    │   └── training_progress.png    # Reward curves
    └── evaluation/
        ├── evaluation_results.json  # Performance metrics
        ├── waveform_trajectory.png  # Parameter evolution
        └── spectrum_comparison.png  # Before/after visualization
```

## Key Results

### Training Progress
- **Episode Rewards**: Cumulative reward per episode (should increase over time)
- **Exploration Decay**: Epsilon decreases from 1.0 → 0.01 as agent learns

### Evaluation Metrics
- **Initial vs. Optimized**: Side-by-side comparison of waveform performance
- **SINR Improvement**: Change in signal-to-interference ratio (dB)
- **Goodput Improvement**: Change in effective data rate (Mbps)
- **Reward Improvement**: Overall optimization objective

### Visualization
1. **Training Progress**: Reward curves and moving averages
2. **Waveform Trajectory**: How parameters evolve during optimization
3. **Spectrum Comparison**: Waveform placement before and after optimization

## Algorithm Details

### Q-Learning Update Rule

```
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
```

Where:
- **s**: current state (discretized spectrum features)
- **a**: action taken (parameter modification)
- **r**: reward received
- **s'**: next state
- **α**: learning rate (default 0.1)
- **γ**: discount factor (default 0.95)

### Epsilon-Greedy Exploration

```python
if random() < epsilon:
    action = random_action()  # Explore
else:
    action = argmax(Q[state])  # Exploit
```

Epsilon decays over time: `epsilon = max(epsilon_min, epsilon × decay_rate)`

### State Features (10-dimensional)

The agent observes:
1. Normalized center frequency
2. Normalized bandwidth
3. Normalized TX power
4. Mean interference inside waveform (log scale)
5. Std interference inside waveform (log scale)
6. Max interference inside waveform (log scale)
7. Mean interference outside waveform (log scale)
8. Min interference outside waveform (log scale)
9. Mean jamming inside waveform (log scale)
10. Relative interference ratio (inside/outside)

**Note**: Modulation is fixed to OFDM, so it's not part of the state.

## Explainability

### Why is the agent explainable?

1. **Clear Reward Function**: Goodput is directly tied to communication performance
2. **Tabular Q-Learning**: No black-box neural networks; Q-values are inspectable
3. **Discrete Actions**: Each action has a clear interpretation (e.g., "shift frequency +5 MHz")
4. **Traceable Decisions**: Training history shows which actions were taken and why (reward)
5. **Visualizations**: Plots clearly show parameter evolution and performance improvements

### Interpreting Results

**High SINR** → Better signal relative to noise/jamming  
**Low BER** → Fewer bit errors, more reliable communication  
**High Goodput** → More effective data throughput  
**Positive Reward** → Agent found better waveform configuration  

## Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Default | Description | Tuning Advice |
|-----------|---------|-------------|---------------|
| `learning_rate` | 0.1 | How fast Q-values update | Lower (0.05) for stable learning, higher (0.2) for faster adaptation |
| `discount_factor` | 0.95 | Weight of future rewards | Higher (0.99) values long-term gains, lower (0.8) for short-term |
| `epsilon` | 1.0 | Initial exploration rate | Start high to explore, decays over time |
| `epsilon_decay` | 0.995 | Exploration decay rate | Slower decay (0.999) for more exploration |
| `episodes` | 100 | Number of training episodes | More episodes (200-500) for complex environments |
| `max_steps` | 50 | Steps per episode | Increase if agent needs more optimization time |

## Extending the Agent

### Adding Deep RL (Future Work)

For continuous action spaces and larger state spaces, consider:

1. **Deep Q-Network (DQN)**: Neural network approximates Q-function
2. **Policy Gradients (PPO, A3C)**: Directly learn policy function
3. **Actor-Critic (SAC, TD3)**: Combine value and policy learning

See commented lines in `requirements.txt` for PyTorch/Gym dependencies.

### Custom Action Spaces

Modify `create_default_action_space()` in `train_rl_agent.py`:

```python
action_space = ActionSpace(
    freq_steps_mhz=[-20.0, -10.0, 0.0, +10.0, +20.0],  # Larger frequency jumps
    bw_multipliers=[0.25, 0.5, 1.0, 2.0, 4.0],         # Wider bandwidth range
    power_steps_dbm=[-5.0, 0.0, +5.0],                 # Coarser power steps
    # Note: Modulation is fixed to OFDM
)
```

## Troubleshooting

### Issue: Agent not improving

**Solution**: 
- Increase number of episodes (200-500)
- Adjust learning rate (try 0.05 or 0.2)
- Check if initial waveform is in a very bad location
- Verify spectrum data files are loaded correctly

### Issue: Training is slow

**Solution**:
- Reduce `max_steps` per episode
- Simplify action space (fewer actions)
- Use fewer frequency/bandwidth options

### Issue: Reward is always negative

**Solution**:
- Check reward function weights in `SpectrumEnvironment.compute_reward()`
- Verify SINR is computed correctly
- Ensure initial waveform parameters are reasonable

## References

- **Q-Learning**: Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
- **SINR & BER**: Proakis, J. G. (2001). Digital communications (4th ed.). McGraw-Hill.
- **Goodput**: ITU-T Recommendation Y.1540 (2019). Internet protocol data communication service.

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please refer to the project repository or documentation.

