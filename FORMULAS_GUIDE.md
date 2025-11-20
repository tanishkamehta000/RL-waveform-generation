# Formula Implementation Guide

This document maps the mathematical formulas from your reward function specification to their implementation in the code.

## Reward Function

### Mathematical Definition
```
R = w_G × G̃ - w_B × B̃ - w_L × L̃
```

**Note**: We omit the latency term (L̃) as requested.

### Simplified Implementation (No Latency)
```
R = w_G × G̃ - w_B × B̃
```

### Code Implementation
**File**: `rl_waveform_agent.py`  
**Function**: `SpectrumEnvironment.compute_reward()`

```python
def compute_reward(self, waveform: WaveformParams, 
                   w_goodput: float = 1.0, 
                   w_bandwidth: float = 0.0) -> Tuple[float, Dict]:
    metrics = self.evaluate_waveform(waveform)
    
    # Normalize goodput
    bw_hz = waveform.bandwidth_khz * 1000.0
    max_theoretical_goodput = bw_hz * np.log2(1.0 + 1e3)
    goodput_normalized = metrics['goodput_bps'] / (max_theoretical_goodput + 1e-12)
    
    # Normalize bandwidth
    max_bw_khz = 1000.0
    bandwidth_normalized = waveform.bandwidth_khz / max_bw_khz
    
    # Compute reward
    reward = (w_goodput * goodput_normalized - 
              w_bandwidth * bandwidth_normalized)
    
    return reward, metrics
```

**Default weights**:
- w_goodput = 1.0 (maximize goodput)
- w_bandwidth = 0.0 (no bandwidth penalty)

---

## Goodput (G)

### Mathematical Definition
```
G = η B log₂(1 + SINR) (1 - BER)
```

Where:
- η = efficiency factor (set to 1.0 in our implementation)
- B = bandwidth (Hz)
- SINR = signal-to-interference-plus-noise ratio
- BER = bit error rate

### Code Implementation
**File**: `run_waveform_in_jammed_env.py`  
**Function**: `compute_goodput()`

```python
def compute_goodput(bw_hz: float, sinr_linear: float, ber: float) -> float:
    # G = B * log2(1+SINR) * (1-BER)
    return bw_hz * np.log2(1.0 + sinr_linear) * (1.0 - ber)
```

**Called from**: `rl_waveform_agent.py` → `SpectrumEnvironment.evaluate_waveform()`

```python
bw_hz = waveform.bandwidth_khz * 1000.0
goodput = compute_goodput(bw_hz, sinr_linear, ber)
```

---

## SINR (Signal-to-Interference-plus-Noise Ratio)

### Mathematical Definition
```
SINR = P_signal / (P_noise + P_jamming)
```

### Code Implementation
**File**: `run_waveform_in_jammed_env.py`  
**Function**: `sinr_and_metrics()`

```python
def sinr_and_metrics(signal_per_bin: np.ndarray, 
                     interference_per_bin: np.ndarray, 
                     eps=1e-12):
    P_sig = np.sum(signal_per_bin)          # Total signal power
    P_int = np.sum(interference_per_bin) + eps  # Total interference (noise + jam)
    sinr_linear = P_sig / P_int            # SINR (linear)
    sinr_db = 10.0 * np.log10(sinr_linear + eps)  # SINR (dB)
    return sinr_linear, sinr_db, P_sig, P_int
```

**Where**:
- `signal_per_bin`: Transmit signal power distributed across frequency bins
- `interference_per_bin`: Background noise + jamming power per bin
- Returns both linear and dB versions

---

## BER (Bit Error Rate)

### Mathematical Definitions

#### QPSK
```
BER = Q(√(2·SINR))
```

Where Q(x) = 0.5 × erfc(x/√2)

**Approximation**:
```
BER ≈ 0.5 × erfc(√SINR)
```

#### 16-QAM
```
BER = (3/8) × Q(√(0.1·SINR))
```

From the image formula:
```
BER = (3/8) × Q(√(0.1·SINR))
```

#### 64-QAM
```
BER = (7/24) × Q(√(0.07·SINR))
```

From the image formula:
```
BER = (7/24) × Q(√(0.07·SINR))
```

### Code Implementation
**File**: `run_waveform_in_jammed_env.py`  
**Function**: `ber_from_sinr()`

```python
def ber_from_sinr(sinr_linear: float, modulation: str) -> float:
    # M = constellation size
    M = 4 if modulation.upper() == "QPSK" else int(modulation.replace("QAM", ""))
    
    if M == 4:
        # QPSK: BER = 0.5 * erfc(sqrt(sinr))
        ber = 0.5 * erfc(np.sqrt(sinr_linear))
        return float(np.clip(ber, 1e-12, 1.0))
    else:
        # M-QAM: Union bound approximation
        k = np.log2(M)
        argument = np.sqrt((3.0 * k / (M - 1.0)) * sinr_linear)
        ber_approx = (4.0 / k) * (1 - 1.0 / np.sqrt(M)) * 0.5 * erfc(argument / np.sqrt(2.0))
        return float(np.clip(ber_approx, 1e-12, 1.0))
```

**Notes**:
- Uses `scipy.special.erfc` for complementary error function
- For M-QAM, uses a more general union bound formula that matches the given formulas
- BER is clipped to [1e-12, 1.0] to avoid numerical issues

---

## Signal and Interference Power

### Signal Power
```
P_signal = Σ signal_per_bin
```

Where `signal_per_bin` is computed as:

```python
# Convert dBm to mW
total_tx_mw = 10 ** (tx_power_dbm / 10.0)

# Distribute across frequency bins in waveform mask
n_on_bins = count(bins in waveform)
per_bin_mw = total_tx_mw / n_on_bins
signal_per_bin = mask * per_bin_mw
```

### Interference Power
```
P_interference = baseline + jam_profile
```

Where:
- **baseline**: 10th percentile of background spectrum (represents noise floor)
- **jam_profile**: mean(malicious) - floor (represents jamming power)

```python
baseline = np.percentile(background_linear, 10, axis=0)
floor_est = np.median(floor_linear, axis=0)
jam_profile = np.mean(malicious_linear, axis=0) - floor_est
jam_profile[jam_profile < 0] = 0.0
interference = baseline + jam_profile
```

---

## Q-Learning Update Rule

### Mathematical Definition
```
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
```

Where:
- Q(s,a) = action-value function
- s = current state
- a = action taken
- r = immediate reward
- s' = next state
- α = learning rate
- γ = discount factor

### Code Implementation
**File**: `rl_waveform_agent.py`  
**Function**: `RLWaveformAgent.update_q_value()`

```python
def update_q_value(self, state: np.ndarray, action: int, 
                   reward: float, next_state: np.ndarray):
    state_hash = self._discretize_state(state)
    next_state_hash = self._discretize_state(next_state)
    
    q_values = self.get_q_values(state_hash)
    next_q_values = self.get_q_values(next_state_hash)
    
    # Q-learning update
    current_q = q_values[action]
    max_next_q = np.max(next_q_values)
    td_target = reward + self.gamma * max_next_q
    td_error = td_target - current_q
    q_values[action] = current_q + self.alpha * td_error
```

**Hyperparameters**:
- α (alpha) = 0.1 (learning rate)
- γ (gamma) = 0.95 (discount factor)

---

## Complete Computation Flow

### 1. Waveform Evaluation Pipeline

```
Input: WaveformParams (center_freq, bandwidth, tx_power, modulation)
  ↓
Create frequency mask
  ↓
Distribute TX power across mask bins → signal_per_bin
  ↓
Get interference from environment → interference_per_bin
  ↓
Compute SINR = Σ(signal) / Σ(interference)
  ↓
Compute BER from SINR + modulation
  ↓
Compute Goodput = BW × log₂(1+SINR) × (1-BER)
  ↓
Output: {sinr_db, ber, goodput_bps, P_sig_mw, P_int_mw}
```

### 2. RL Training Loop

```
Initialize: Q-table, waveform parameters
for episode in episodes:
    state = get_state_features(waveform)
    for step in max_steps:
        action = epsilon_greedy(state)
        new_waveform = apply_action(waveform, action)
        reward, metrics = compute_reward(new_waveform)
        next_state = get_state_features(new_waveform)
        update_q_value(state, action, reward, next_state)
        waveform = new_waveform
        state = next_state
    decay_epsilon()
```

---

## Unit Conversions

### Power Units
```python
# dBm to mW
mw = 10 ** (dbm / 10.0)

# Linear power to dB
db = 10.0 * np.log10(linear_power)
```

### Frequency Units
```python
# MHz to Hz
hz = mhz * 1e6

# kHz to Hz
hz = khz * 1000.0
```

### Data Rate Units
```python
# bps to Mbps
mbps = bps / 1e6
```

---

## Summary Table

| Concept | Formula | Code Location |
|---------|---------|---------------|
| **Reward** | R = w_G·G̃ - w_B·B̃ | `rl_waveform_agent.py::compute_reward()` |
| **Goodput** | G = B·log₂(1+SINR)·(1-BER) | `run_waveform_in_jammed_env.py::compute_goodput()` |
| **SINR** | P_sig / (P_noise + P_jam) | `run_waveform_in_jammed_env.py::sinr_and_metrics()` |
| **BER (QPSK)** | 0.5·erfc(√SINR) | `run_waveform_in_jammed_env.py::ber_from_sinr()` |
| **BER (16-QAM)** | (3/8)·Q(√(0.1·SINR)) | `run_waveform_in_jammed_env.py::ber_from_sinr()` |
| **BER (64-QAM)** | (7/24)·Q(√(0.07·SINR)) | `run_waveform_in_jammed_env.py::ber_from_sinr()` |
| **Q-Learning** | Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)] | `rl_waveform_agent.py::update_q_value()` |

---

## Quick Reference: Key Functions

### To compute SINR for a waveform:
```python
from run_waveform_in_jammed_env import sinr_and_metrics
sinr_linear, sinr_db, P_sig, P_int = sinr_and_metrics(signal_per_bin, interference_per_bin)
```

### To compute BER:
```python
from run_waveform_in_jammed_env import ber_from_sinr
ber = ber_from_sinr(sinr_linear, 'QPSK')  # or '16QAM', '64QAM'
```

### To compute Goodput:
```python
from run_waveform_in_jammed_env import compute_goodput
goodput_bps = compute_goodput(bandwidth_hz, sinr_linear, ber)
```

### To evaluate complete waveform:
```python
from rl_waveform_agent import SpectrumEnvironment, WaveformParams
env = SpectrumEnvironment(bg_path, floor_path, jammer_path)
waveform = WaveformParams(center_freq_mhz=2412.0, bandwidth_khz=200.0, 
                         tx_power_dbm=0.0, modulation='QPSK')
metrics = env.evaluate_waveform(waveform)
# metrics = {'sinr_linear', 'sinr_db', 'ber', 'goodput_bps', ...}
```

### To compute reward:
```python
reward, metrics = env.compute_reward(waveform, w_goodput=1.0, w_bandwidth=0.0)
```

