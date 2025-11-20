#!/usr/bin/env python3
"""
rl_waveform_agent.py

Reinforcement Learning Agent for Waveform Anti-Jamming

This module implements a reinforcement learning agent that learns to tune waveform
parameters (center frequency, bandwidth, transmit power) to maximize
communication performance in a jammed radio environment using OFDM modulation.

The agent observes the spectrum environment and iteratively adjusts waveform
parameters to avoid jamming and maximize goodput. Modulation is fixed to OFDM.

Reward Function (without latency):
    R = w_G * G_normalized - w_B * bandwidth_penalty
    
    where:
    - G = Bandwidth * log2(1 + SINR) * (1 - BER)  [Goodput]
    - SINR = Signal Power / (Noise + Jamming)
    - BER depends on modulation scheme and SINR
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import sys

# Import helper functions from existing waveform script
from run_waveform_in_jammed_env import (
    load_csv_as_array,
    to_linear_power,
    clamp_linear,
    infer_frequencies,
    compute_baseline_background,
    compute_floor,
    compute_jam_profile,
    make_waveform_mask,
    dbm_to_mw,
    mw_to_linear_power_per_bin,
    sinr_and_metrics,
    ber_from_sinr,
    compute_goodput,
)


@dataclass
class WaveformParams:
    """Waveform parameters that the RL agent can tune"""
    center_freq_mhz: float  # Center frequency in MHz
    bandwidth_khz: float     # Bandwidth in kHz
    tx_power_dbm: float      # Transmit power in dBm
    modulation: str = "OFDM"  # Modulation scheme: Fixed to OFDM
    
    def to_dict(self):
        return asdict(self)
    
    def copy(self):
        return WaveformParams(**asdict(self))


@dataclass
class ActionSpace:
    """Defines the discrete action space for the RL agent (modulation fixed to OFDM)"""
    # Frequency actions: move center frequency
    freq_steps_mhz: List[float]  # e.g., [-5, -2, -1, 0, +1, +2, +5]
    
    # Bandwidth actions: adjust bandwidth
    bw_multipliers: List[float]  # e.g., [0.5, 0.75, 1.0, 1.25, 1.5]
    
    # Power actions: adjust transmit power
    power_steps_dbm: List[float]  # e.g., [-3, -1, 0, +1, +3]
    
    def get_total_actions(self) -> int:
        """Total number of discrete actions available"""
        return (len(self.freq_steps_mhz) + 
                len(self.bw_multipliers) + 
                len(self.power_steps_dbm))
    
    def action_index_to_modification(self, action_idx: int) -> Tuple[str, any]:
        """
        Convert discrete action index to a specific parameter modification
        
        Returns:
            (param_name, modification_value)
        """
        idx = 0
        
        # Frequency actions
        if action_idx < len(self.freq_steps_mhz):
            return ("freq_step", self.freq_steps_mhz[action_idx])
        idx += len(self.freq_steps_mhz)
        
        # Bandwidth actions
        if action_idx < idx + len(self.bw_multipliers):
            return ("bw_mult", self.bw_multipliers[action_idx - idx])
        idx += len(self.bw_multipliers)
        
        # Power actions
        if action_idx < idx + len(self.power_steps_dbm):
            return ("power_step", self.power_steps_dbm[action_idx - idx])
        
        raise ValueError(f"Invalid action index {action_idx}")


class SpectrumEnvironment:
    """
    Represents the radio spectrum environment with background noise and jamming.
    
    This class encapsulates the spectrum state and provides methods to compute
    SINR, BER, and goodput for a given waveform configuration.
    """
    
    def __init__(self, bg_path: str, floor_path: str, malicious_path: str):
        """
        Initialize the spectrum environment from CSV files.
        
        Args:
            bg_path: Path to background spectrum CSV
            floor_path: Path to noise floor spectrum CSV
            malicious_path: Path to jammer spectrum CSV
        """
        # Load and process spectrum data
        bg = load_csv_as_array(bg_path)
        floor_arr = load_csv_as_array(floor_path)
        mal = load_csv_as_array(malicious_path)
        
        # Convert to linear power
        bg_lin = clamp_linear(to_linear_power(bg))
        floor_lin = clamp_linear(to_linear_power(floor_arr))
        mal_lin = clamp_linear(to_linear_power(mal))
        
        # Resample to same number of bins
        n_bins = max(bg_lin.shape[1], floor_lin.shape[1], mal_lin.shape[1])
        bg_lin = self._resample_rows(bg_lin, n_bins)
        floor_lin = self._resample_rows(floor_lin, n_bins)
        mal_lin = self._resample_rows(mal_lin, n_bins)
        
        # Store processed spectrum data
        self.n_bins = n_bins
        self.frequencies = infer_frequencies(n_bins, bg_path, 2412.0)
        self.baseline = compute_baseline_background(bg_lin)
        self.floor = compute_floor(floor_lin)
        self.jam_profile = compute_jam_profile(mal_lin, self.floor)
        self.interference = self.baseline + self.jam_profile
        
        # Store frequency bounds for constraints
        self.freq_min = float(np.min(self.frequencies))
        self.freq_max = float(np.max(self.frequencies))
    
    @staticmethod
    def _resample_rows(arr: np.ndarray, target_bins: int) -> np.ndarray:
        """Resample array rows to target number of bins"""
        if arr.shape[1] == target_bins:
            return arr
        x_old = np.linspace(0.0, 1.0, arr.shape[1])
        x_new = np.linspace(0.0, 1.0, target_bins)
        return np.array([np.interp(x_new, x_old, row) for row in arr])
    
    def get_state_features(self, waveform: WaveformParams) -> np.ndarray:
        """
        Extract state features for the RL agent.
        
        Features include:
        - Current waveform parameters (normalized)
        - Spectrum statistics in and around the waveform
        - Jamming profile statistics
        
        Returns:
            Feature vector (numpy array)
        """
        # Create waveform mask
        mask = make_waveform_mask(self.frequencies, waveform.center_freq_mhz, 
                                   waveform.bandwidth_khz)
        
        # Extract spectrum statistics
        mask_bool = mask > 0.5
        if np.sum(mask_bool) == 0:
            # Fallback to nearest bin
            idx = np.argmin(np.abs(self.frequencies - waveform.center_freq_mhz))
            mask_bool = np.zeros_like(mask_bool)
            mask_bool[idx] = True
        
        # Interference statistics inside waveform
        interf_inside = self.interference[mask_bool]
        mean_interf_inside = float(np.mean(interf_inside))
        std_interf_inside = float(np.std(interf_inside))
        max_interf_inside = float(np.max(interf_inside))
        
        # Interference statistics outside waveform (neighboring regions)
        mask_outside = ~mask_bool
        if np.sum(mask_outside) > 0:
            interf_outside = self.interference[mask_outside]
            mean_interf_outside = float(np.mean(interf_outside))
            min_interf_outside = float(np.min(interf_outside))
        else:
            mean_interf_outside = mean_interf_inside
            min_interf_outside = mean_interf_inside
        
        # Jamming statistics
        jam_inside = self.jam_profile[mask_bool]
        mean_jam_inside = float(np.mean(jam_inside))
        
        # Normalize waveform parameters to [0, 1] range
        freq_norm = (waveform.center_freq_mhz - self.freq_min) / (self.freq_max - self.freq_min + 1e-9)
        bw_norm = waveform.bandwidth_khz / 1000.0  # Normalize by 1 MHz
        power_norm = (waveform.tx_power_dbm + 10) / 20.0  # Assume power range [-10, 10] dBm
        
        # Construct feature vector (modulation is fixed to OFDM, no longer a feature)
        features = np.array([
            freq_norm,
            bw_norm,
            power_norm,
            np.log10(mean_interf_inside + 1e-15),
            np.log10(std_interf_inside + 1e-15),
            np.log10(max_interf_inside + 1e-15),
            np.log10(mean_interf_outside + 1e-15),
            np.log10(min_interf_outside + 1e-15),
            np.log10(mean_jam_inside + 1e-15),
            mean_interf_inside / (mean_interf_outside + 1e-15),  # Relative interference
        ], dtype=np.float32)
        
        return features
    
    def evaluate_waveform(self, waveform: WaveformParams) -> Dict:
        """
        Evaluate waveform performance in the current environment.
        
        Computes SINR, BER, and goodput for the given waveform parameters.
        
        Returns:
            Dictionary with performance metrics
        """
        # Create waveform mask
        mask = make_waveform_mask(self.frequencies, waveform.center_freq_mhz,
                                   waveform.bandwidth_khz)
        
        # Handle edge case: no bins selected
        n_on_bins = int(np.sum(mask > 0.5))
        if n_on_bins == 0:
            idx = np.argmin(np.abs(self.frequencies - waveform.center_freq_mhz))
            mask[idx] = 1.0
            n_on_bins = 1
        
        # Distribute transmit power across mask bins
        total_tx_mw = dbm_to_mw(waveform.tx_power_dbm)
        per_bin_mw = mw_to_linear_power_per_bin(total_tx_mw, n_on_bins)
        signal_per_bin = mask * per_bin_mw
        
        # Compute SINR
        interference_per_bin = self.interference
        sinr_linear, sinr_db, P_sig, P_int = sinr_and_metrics(
            signal_per_bin, interference_per_bin
        )
        
        # Compute BER
        ber = ber_from_sinr(sinr_linear, waveform.modulation)
        
        # Compute goodput
        bw_hz = waveform.bandwidth_khz * 1000.0
        goodput = compute_goodput(bw_hz, sinr_linear, ber)
        
        return {
            'sinr_linear': float(sinr_linear),
            'sinr_db': float(sinr_db),
            'ber': float(ber),
            'goodput_bps': float(goodput),
            'P_sig_mw': float(P_sig),
            'P_int_mw': float(P_int),
        }
    
    def compute_reward(self, waveform: WaveformParams, 
                       w_goodput: float = 1.0, 
                       w_bandwidth: float = 0.0,
                       w_power: float = 0.1) -> Tuple[float, Dict]:
        """
        Compute reward for the given waveform configuration.
        
        Reward function (without latency, with power penalty):
            R = w_G * G_normalized - w_B * B_normalized - w_P * P_normalized
        
        Where:
            G = Bandwidth * log2(1 + SINR) * (1 - BER)  [Goodput]
            B = Bandwidth (penalty for using wide bandwidth)
            P = Power usage (penalty to encourage finding clean spectrum instead of using more power)
        
        Args:
            waveform: Waveform parameters to evaluate
            w_goodput: Weight for goodput term (default 1.0)
            w_bandwidth: Weight for bandwidth penalty (default 0.0)
            w_power: Weight for power penalty (default 0.1) - encourages lower power
        
        Returns:
            (reward, metrics_dict)
        """
        metrics = self.evaluate_waveform(waveform)
        
        # Normalize goodput by theoretical maximum
        # Max theoretical: bandwidth * log2(1 + high_SNR)
        bw_hz = waveform.bandwidth_khz * 1000.0
        max_theoretical_goodput = bw_hz * np.log2(1.0 + 1e3)  # Assume SNR=1000 (30 dB)
        goodput_normalized = metrics['goodput_bps'] / (max_theoretical_goodput + 1e-12)
        
        # Normalize bandwidth penalty (0 to 1, where 1 = max bandwidth)
        max_bw_khz = 1000.0  # Assume max bandwidth of 1 MHz
        bandwidth_normalized = waveform.bandwidth_khz / max_bw_khz
        
        # Normalize power penalty (0 to 1, where higher power = higher penalty)
        # Assume power range [-3, +3] dBm, normalize to [0, 1]
        power_normalized = (waveform.tx_power_dbm + 3.0) / 6.0  # Maps [-3, +3] to [0, 1]
        power_normalized = np.clip(power_normalized, 0.0, 1.0)
        
        # Compute reward with power penalty
        # This encourages the agent to find clean spectrum rather than just using more power
        reward = (w_goodput * goodput_normalized - 
                  w_bandwidth * bandwidth_normalized - 
                  w_power * power_normalized)
        
        # Add detailed metrics
        metrics['goodput_normalized'] = float(goodput_normalized)
        metrics['bandwidth_normalized'] = float(bandwidth_normalized)
        metrics['power_normalized'] = float(power_normalized)
        metrics['reward'] = float(reward)
        
        return float(reward), metrics


class RLWaveformAgent:
    """
    Reinforcement Learning Agent for Waveform Anti-Jamming.
    
    This agent uses a simple Q-learning approach with epsilon-greedy exploration
    to learn optimal waveform parameters in a jammed environment.
    
    Algorithm: Tabular Q-Learning with discretized state space
    - State: Discretized spectrum features
    - Action: Waveform parameter modifications
    - Reward: Based on goodput and bandwidth usage
    """
    
    def __init__(self, action_space: ActionSpace, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize the RL agent.
        
        Args:
            action_space: Discrete action space definition
            learning_rate: Learning rate (alpha) for Q-learning
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Initial exploration rate for epsilon-greedy
            epsilon_decay: Decay rate for epsilon after each episode
            epsilon_min: Minimum epsilon value
        """
        self.action_space = action_space
        self.n_actions = action_space.get_total_actions()
        
        # Q-learning hyperparameters
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: maps (state_hash) -> [Q-values for each action]
        self.q_table = {}
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -np.inf
        self.best_waveform = None
        
    def _discretize_state(self, state_features: np.ndarray, n_bins: int = 10) -> int:
        """
        Discretize continuous state features to create a state hash.
        
        This simplifies the state space for tabular Q-learning.
        
        Args:
            state_features: Continuous feature vector
            n_bins: Number of discrete bins per feature
        
        Returns:
            State hash (integer)
        """
        # Clip features to reasonable range
        features_clipped = np.clip(state_features, -10, 10)
        
        # Discretize each feature
        discretized = np.floor(features_clipped * n_bins / 20.0 + n_bins / 2.0).astype(int)
        discretized = np.clip(discretized, 0, n_bins - 1)
        
        # Create hash
        state_hash = hash(tuple(discretized))
        return state_hash
    
    def get_q_values(self, state_hash: int) -> np.ndarray:
        """Get Q-values for a given state (initialize if new state)"""
        if state_hash not in self.q_table:
            self.q_table[state_hash] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q_table[state_hash]
    
    def select_action(self, state_features: np.ndarray, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state_features: Current state feature vector
            explore: Whether to use exploration (epsilon-greedy) or pure exploitation
        
        Returns:
            Action index
        """
        state_hash = self._discretize_state(state_features)
        q_values = self.get_q_values(state_hash)
        
        if explore and np.random.rand() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best action based on Q-values
            return int(np.argmax(q_values))
    
    def update_q_value(self, state: np.ndarray, action: int, 
                       reward: float, next_state: np.ndarray):
        """
        Update Q-value using Q-learning update rule:
        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state features
            action: Action taken
            reward: Reward received
            next_state: Next state features
        """
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
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def apply_action(self, waveform: WaveformParams, action_idx: int,
                     freq_bounds: Tuple[float, float]) -> WaveformParams:
        """
        Apply action to modify waveform parameters.
        Modulation is fixed to OFDM and cannot be changed.
        
        Args:
            waveform: Current waveform parameters
            action_idx: Action index to apply
            freq_bounds: (min_freq, max_freq) constraints
        
        Returns:
            New waveform parameters
        """
        new_waveform = waveform.copy()
        param_name, modification = self.action_space.action_index_to_modification(action_idx)
        
        if param_name == "freq_step":
            # Adjust center frequency
            new_freq = new_waveform.center_freq_mhz + modification
            new_waveform.center_freq_mhz = np.clip(new_freq, freq_bounds[0], freq_bounds[1])
            
        elif param_name == "bw_mult":
            # Adjust bandwidth
            new_bw = new_waveform.bandwidth_khz * modification
            new_waveform.bandwidth_khz = np.clip(new_bw, 50.0, 1000.0)  # 50 kHz to 1 MHz
            
        elif param_name == "power_step":
            # Adjust transmit power (CAPPED to reasonable range)
            # This forces the agent to optimize frequency placement rather than just using more power
            new_power = new_waveform.tx_power_dbm + modification
            new_waveform.tx_power_dbm = np.clip(new_power, -3.0, 3.0)  # -3 to +3 dBm (capped!)
        
        return new_waveform
    
    def train_episode(self, env: SpectrumEnvironment, 
                      initial_waveform: WaveformParams,
                      max_steps: int = 50) -> Tuple[float, List[Dict]]:
        """
        Train for one episode.
        
        Args:
            env: Spectrum environment
            initial_waveform: Starting waveform configuration
            max_steps: Maximum steps per episode
        
        Returns:
            (total_reward, history)
        """
        waveform = initial_waveform.copy()
        state = env.get_state_features(waveform)
        total_reward = 0.0
        history = []
        
        freq_bounds = (env.freq_min, env.freq_max)
        
        for step in range(max_steps):
            # Select and apply action
            action = self.select_action(state, explore=True)
            new_waveform = self.apply_action(waveform, action, freq_bounds)
            
            # Compute reward
            reward, metrics = env.compute_reward(new_waveform)
            next_state = env.get_state_features(new_waveform)
            
            # Update Q-value
            self.update_q_value(state, action, reward, next_state)
            
            # Record step
            history.append({
                'step': step,
                'action': action,
                'waveform': new_waveform.to_dict(),
                'reward': reward,
                'metrics': metrics,
            })
            
            # Update for next iteration
            total_reward += reward
            state = next_state
            waveform = new_waveform
            
            # Track best waveform
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_waveform = new_waveform.copy()
        
        # Decay exploration rate
        self.decay_epsilon()
        
        # Record episode statistics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(max_steps)
        
        return total_reward, history
    
    def evaluate(self, env: SpectrumEnvironment,
                 initial_waveform: WaveformParams,
                 max_steps: int = 50) -> Tuple[WaveformParams, List[Dict]]:
        """
        Evaluate the learned policy (no exploration).
        
        Returns:
            (best_waveform, trajectory_history)
        """
        waveform = initial_waveform.copy()
        state = env.get_state_features(waveform)
        history = []
        
        freq_bounds = (env.freq_min, env.freq_max)
        best_waveform = waveform.copy()
        best_reward = -np.inf
        
        for step in range(max_steps):
            # Select best action (no exploration)
            action = self.select_action(state, explore=False)
            waveform = self.apply_action(waveform, action, freq_bounds)
            
            # Compute reward
            reward, metrics = env.compute_reward(waveform)
            state = env.get_state_features(waveform)
            
            # Record
            history.append({
                'step': step,
                'action': action,
                'waveform': waveform.to_dict(),
                'reward': reward,
                'metrics': metrics,
            })
            
            # Track best
            if reward > best_reward:
                best_reward = reward
                best_waveform = waveform.copy()
        
        return best_waveform, history
    
    def save(self, filepath: str):
        """Save agent (Q-table and hyperparameters)"""
        data = {
            'q_table': {str(k): v.tolist() for k, v in self.q_table.items()},
            'hyperparameters': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
            },
            'training_history': {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'best_reward': self.best_reward,
                'best_waveform': self.best_waveform.to_dict() if self.best_waveform else None,
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load agent from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load Q-table
        self.q_table = {int(k): np.array(v, dtype=np.float32) 
                        for k, v in data['q_table'].items()}
        
        # Load hyperparameters
        hp = data['hyperparameters']
        self.alpha = hp['alpha']
        self.gamma = hp['gamma']
        self.epsilon = hp['epsilon']
        self.epsilon_decay = hp['epsilon_decay']
        self.epsilon_min = hp['epsilon_min']
        
        # Load training history
        hist = data['training_history']
        self.episode_rewards = hist['episode_rewards']
        self.episode_lengths = hist['episode_lengths']
        self.best_reward = hist['best_reward']
        if hist['best_waveform']:
            self.best_waveform = WaveformParams(**hist['best_waveform'])


def plot_training_progress(agent: RLWaveformAgent, output_dir: Path):
    """Plot training progress: rewards and epsilon over episodes"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot episode rewards
    ax1.plot(agent.episode_rewards, alpha=0.6, label='Episode Reward')
    # Moving average
    if len(agent.episode_rewards) > 10:
        window = 10
        moving_avg = np.convolve(agent.episode_rewards, 
                                  np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(agent.episode_rewards)), moving_avg,
                 'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress: Cumulative Reward per Episode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot epsilon decay
    epsilons = [agent.epsilon_min * (agent.epsilon_decay ** (-i)) 
                for i in range(len(agent.episode_rewards))]
    epsilons = [min(1.0, max(agent.epsilon_min, e)) for e in epsilons]
    ax2.plot(epsilons, 'g-', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon (Exploration Rate)')
    ax2.set_title('Exploration Rate Decay')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_progress.png', dpi=150)
    plt.close()


def plot_waveform_trajectory(history: List[Dict], env: SpectrumEnvironment, 
                             output_dir: Path, title: str = "Waveform Optimization"):
    """Plot how waveform parameters and metrics evolve over steps"""
    steps = [h['step'] for h in history]
    freqs = [h['waveform']['center_freq_mhz'] for h in history]
    bws = [h['waveform']['bandwidth_khz'] for h in history]
    powers = [h['waveform']['tx_power_dbm'] for h in history]
    rewards = [h['reward'] for h in history]
    sinrs_db = [h['metrics']['sinr_db'] for h in history]
    goodputs = [h['metrics']['goodput_bps'] / 1e6 for h in history]  # Convert to Mbps
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Center Frequency
    axes[0, 0].plot(steps, freqs, 'b-o', markersize=3)
    axes[0, 0].set_ylabel('Center Freq (MHz)')
    axes[0, 0].set_title('Center Frequency Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Bandwidth
    axes[0, 1].plot(steps, bws, 'g-o', markersize=3)
    axes[0, 1].set_ylabel('Bandwidth (kHz)')
    axes[0, 1].set_title('Bandwidth Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Transmit Power
    axes[1, 0].plot(steps, powers, 'r-o', markersize=3)
    axes[1, 0].set_ylabel('TX Power (dBm)')
    axes[1, 0].set_title('Transmit Power Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reward
    axes[1, 1].plot(steps, rewards, 'm-o', markersize=3)
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].set_title('Reward Evolution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # SINR
    axes[2, 0].plot(steps, sinrs_db, 'c-o', markersize=3)
    axes[2, 0].set_ylabel('SINR (dB)')
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].set_title('SINR Evolution')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Goodput
    axes[2, 1].plot(steps, goodputs, 'orange', marker='o', markersize=3)
    axes[2, 1].set_ylabel('Goodput (Mbps)')
    axes[2, 1].set_xlabel('Step')
    axes[2, 1].set_title('Goodput Evolution')
    axes[2, 1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'waveform_trajectory.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    print("RL Waveform Agent Module")
    print("This module provides RL agents for waveform anti-jamming.")
    print("Use train_rl_agent.py to train an agent.")

