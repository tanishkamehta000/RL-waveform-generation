#!/usr/bin/env python3
"""
train_rl_agent.py

Train a Reinforcement Learning agent to optimize waveform parameters
in a jammed radio environment.

The agent learns to adjust center frequency, bandwidth, and transmit power
to maximize goodput while avoiding jamming. Modulation is fixed to OFDM.

Usage:
    python train_rl_agent.py --episodes 100 --max_steps 50
    python train_rl_agent.py --load agent.json --evaluate
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from rl_waveform_agent import (
    WaveformParams,
    ActionSpace,
    SpectrumEnvironment,
    RLWaveformAgent,
    plot_training_progress,
    plot_waveform_trajectory,
)


def create_default_action_space() -> ActionSpace:
    """
    Create action space emphasizing frequency optimization over power increases.
    Power is capped to reasonable range to force agent to find clean spectrum.
    """
    return ActionSpace(
        # Frequency adjustments: MORE options to encourage frequency exploration
        freq_steps_mhz=[-20.0, -15.0, -10.0, -5.0, -2.0, -1.0, 0.0, +1.0, +2.0, +5.0, +10.0, +15.0, +20.0],
        
        # Bandwidth multipliers: adjust bandwidth by these factors
        bw_multipliers=[0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5],
        
        # Power adjustments: SMALLER steps, capped range (agent must optimize placement!)
        power_steps_dbm=[-1.0, 0.0, +1.0],
    )


def create_initial_waveform() -> WaveformParams:
    """
    Create initial waveform configuration (OFDM modulation).
    Starts with reasonable power so agent must find clean spectrum.
    """
    return WaveformParams(
        center_freq_mhz=2412.0,  # 2.4 GHz WiFi channel 1
        bandwidth_khz=200.0,      # 200 kHz bandwidth
        tx_power_dbm=0.0,         # 0 dBm transmit power (middle of capped range)
        modulation='OFDM',        # Fixed to OFDM
    )


def train_agent(env: SpectrumEnvironment,
                agent: RLWaveformAgent,
                initial_waveform: WaveformParams,
                n_episodes: int = 100,
                max_steps: int = 50,
                output_dir: Path = Path('rl_training_output')):
    """
    Train the RL agent for multiple episodes.
    
    Args:
        env: Spectrum environment
        agent: RL agent to train
        initial_waveform: Starting waveform configuration
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training RL Agent for Waveform Anti-Jamming")
    print(f"{'='*60}")
    print(f"Episodes: {n_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Initial waveform: {initial_waveform}")
    print(f"Action space: {agent.action_space.get_total_actions()} actions")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Training loop
    for episode in range(n_episodes):
        # Run one episode
        total_reward, history = agent.train_episode(env, initial_waveform, max_steps)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(agent.episode_rewards[-10:])
            print(f"Episode {episode+1}/{n_episodes} | "
                  f"Total Reward: {total_reward:.4f} | "
                  f"Avg (last 10): {avg_reward:.4f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Best Reward: {agent.best_reward:.4f}")
            
            # Save intermediate checkpoint
            if (episode + 1) % 50 == 0:
                checkpoint_path = output_dir / f'agent_checkpoint_ep{episode+1}.json'
                agent.save(str(checkpoint_path))
                print(f"  → Saved checkpoint to {checkpoint_path}")
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Total episodes: {n_episodes}")
    print(f"Best reward achieved: {agent.best_reward:.6f}")
    if agent.best_waveform:
        print(f"Best waveform configuration:")
        print(f"  Center Frequency: {agent.best_waveform.center_freq_mhz:.2f} MHz")
        print(f"  Bandwidth: {agent.best_waveform.bandwidth_khz:.2f} kHz")
        print(f"  TX Power: {agent.best_waveform.tx_power_dbm:.2f} dBm")
        print(f"  Modulation: {agent.best_waveform.modulation}")
    print(f"{'='*60}\n")
    
    # Save final agent
    final_path = output_dir / 'agent_final.json'
    agent.save(str(final_path))
    print(f"Saved final agent to {final_path}")
    
    # Plot training progress
    plot_training_progress(agent, output_dir)
    print(f"Saved training progress plot to {output_dir / 'training_progress.png'}")
    
    return agent


def evaluate_agent(env: SpectrumEnvironment,
                   agent: RLWaveformAgent,
                   initial_waveform: WaveformParams,
                   max_steps: int = 50,
                   output_dir: Path = Path('rl_evaluation_output')):
    """
    Evaluate the trained agent (no exploration).
    
    Args:
        env: Spectrum environment
        agent: Trained RL agent
        initial_waveform: Starting waveform configuration
        max_steps: Maximum optimization steps
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Evaluating RL Agent (Greedy Policy)")
    print(f"{'='*60}")
    print(f"Initial waveform: {initial_waveform}")
    print(f"Max optimization steps: {max_steps}")
    print(f"{'='*60}\n")
    
    # Run evaluation
    best_waveform, history = agent.evaluate(env, initial_waveform, max_steps)
    
    # Extract initial and final metrics
    initial_metrics = history[0]['metrics']
    final_metrics = history[-1]['metrics']
    
    print(f"Initial Configuration:")
    print(f"  Center Freq: {initial_waveform.center_freq_mhz:.2f} MHz")
    print(f"  Bandwidth: {initial_waveform.bandwidth_khz:.2f} kHz")
    print(f"  TX Power: {initial_waveform.tx_power_dbm:.2f} dBm")
    print(f"  Modulation: {initial_waveform.modulation}")
    print(f"  SINR: {initial_metrics['sinr_db']:.2f} dB")
    print(f"  BER: {initial_metrics['ber']:.6e}")
    print(f"  Goodput: {initial_metrics['goodput_bps']/1e6:.3f} Mbps")
    print(f"  Reward: {history[0]['reward']:.6f}")
    
    print(f"\nOptimized Configuration:")
    print(f"  Center Freq: {best_waveform.center_freq_mhz:.2f} MHz")
    print(f"  Bandwidth: {best_waveform.bandwidth_khz:.2f} kHz")
    print(f"  TX Power: {best_waveform.tx_power_dbm:.2f} dBm")
    print(f"  Modulation: {best_waveform.modulation}")
    print(f"  SINR: {final_metrics['sinr_db']:.2f} dB")
    print(f"  BER: {final_metrics['ber']:.6e}")
    print(f"  Goodput: {final_metrics['goodput_bps']/1e6:.3f} Mbps")
    print(f"  Reward: {history[-1]['reward']:.6f}")
    
    print(f"\nImprovement:")
    print(f"  SINR: {final_metrics['sinr_db'] - initial_metrics['sinr_db']:+.2f} dB")
    print(f"  Goodput: {(final_metrics['goodput_bps'] - initial_metrics['goodput_bps'])/1e6:+.3f} Mbps")
    print(f"  Reward: {history[-1]['reward'] - history[0]['reward']:+.6f}")
    print(f"{'='*60}\n")
    
    # Save evaluation results
    results = {
        'initial_waveform': initial_waveform.to_dict(),
        'optimized_waveform': best_waveform.to_dict(),
        'initial_metrics': initial_metrics,
        'final_metrics': final_metrics,
        'improvement': {
            'sinr_db': float(final_metrics['sinr_db'] - initial_metrics['sinr_db']),
            'goodput_bps': float(final_metrics['goodput_bps'] - initial_metrics['goodput_bps']),
            'reward': float(history[-1]['reward'] - history[0]['reward']),
        },
        'trajectory': history,
    }
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {results_path}")
    
    # Plot waveform trajectory
    plot_waveform_trajectory(history, env, output_dir, 
                            title="Waveform Optimization Trajectory (Evaluation)")
    print(f"Saved trajectory plot to {output_dir / 'waveform_trajectory.png'}")
    
    # Plot spectrum with initial and optimized waveforms
    plot_spectrum_comparison(env, initial_waveform, best_waveform, output_dir)
    print(f"Saved spectrum comparison to {output_dir / 'spectrum_comparison.png'}")
    
    return best_waveform, results


def plot_spectrum_comparison(env: SpectrumEnvironment,
                             initial_waveform: WaveformParams,
                             optimized_waveform: WaveformParams,
                             output_dir: Path):
    """Plot spectrum with initial and optimized waveform placements"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    freqs = env.frequencies
    eps = 1e-15
    
    # Convert to dB for plotting
    baseline_db = 10.0 * np.log10(env.baseline + eps)
    jam_profile_db = 10.0 * np.log10(env.jam_profile + eps)
    interference_db = 10.0 * np.log10(env.interference + eps)
    
    # Plot 1: Initial waveform
    ax1.plot(freqs, baseline_db, 'b-', alpha=0.6, label='Baseline (Background)')
    ax1.plot(freqs, jam_profile_db, 'r-', alpha=0.6, label='Jamming Profile')
    ax1.plot(freqs, interference_db, 'k-', linewidth=1.5, label='Total Interference')
    
    # Show initial waveform mask
    from run_waveform_in_jammed_env import make_waveform_mask
    mask_initial = make_waveform_mask(freqs, initial_waveform.center_freq_mhz,
                                       initial_waveform.bandwidth_khz)
    # Scale mask for visualization
    mask_scaled = (mask_initial > 0.5) * (np.max(interference_db) - np.min(interference_db)) + np.min(interference_db)
    ax1.fill_between(freqs, np.min(interference_db), mask_scaled,
                     where=(mask_initial > 0.5), alpha=0.3, color='green',
                     label=f'Initial Waveform ({initial_waveform.center_freq_mhz:.1f} MHz)')
    
    ax1.set_ylabel('Power (dB)')
    ax1.set_title('Initial Waveform Placement')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Optimized waveform
    ax2.plot(freqs, baseline_db, 'b-', alpha=0.6, label='Baseline (Background)')
    ax2.plot(freqs, jam_profile_db, 'r-', alpha=0.6, label='Jamming Profile')
    ax2.plot(freqs, interference_db, 'k-', linewidth=1.5, label='Total Interference')
    
    # Show optimized waveform mask
    mask_optimized = make_waveform_mask(freqs, optimized_waveform.center_freq_mhz,
                                        optimized_waveform.bandwidth_khz)
    mask_scaled = (mask_optimized > 0.5) * (np.max(interference_db) - np.min(interference_db)) + np.min(interference_db)
    ax2.fill_between(freqs, np.min(interference_db), mask_scaled,
                     where=(mask_optimized > 0.5), alpha=0.3, color='orange',
                     label=f'Optimized Waveform ({optimized_waveform.center_freq_mhz:.1f} MHz)')
    
    ax2.set_xlabel('Frequency (MHz)')
    ax2.set_ylabel('Power (dB)')
    ax2.set_title('Optimized Waveform Placement (RL Agent)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spectrum_comparison.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Train or evaluate RL agent for waveform anti-jamming'
    )
    
    # Environment arguments
    parser.add_argument('--bg', default='benign/background/2.4ghz_passivescan_background_loc1_1.csv',
                       help='Path to background spectrum CSV')
    parser.add_argument('--floor', default='benign/background/2.4ghz_passivescan_floor_1.csv',
                       help='Path to noise floor spectrum CSV')
    parser.add_argument('--malicious', default='benign/background/2412mhz_jamming_3dbm_gaussiannoise_1.csv',
                       help='Path to jammer spectrum CSV')
    
    # Training arguments
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=50,
                       help='Maximum steps per episode')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                       help='Learning rate (alpha) for Q-learning')
    parser.add_argument('--discount_factor', type=float, default=0.95,
                       help='Discount factor (gamma) for future rewards')
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='Initial exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                       help='Exploration rate decay per episode')
    
    # Mode arguments
    parser.add_argument('--train', action='store_true', default=True,
                       help='Train the agent (default)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate a trained agent')
    parser.add_argument('--load', type=str, default=None,
                       help='Load agent from file (for evaluation or continued training)')
    
    # Output arguments
    parser.add_argument('--output_dir', default='rl_output',
                       help='Directory to save training/evaluation results')
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save command-line arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize environment
    print("Loading spectrum environment...")
    env = SpectrumEnvironment(args.bg, args.floor, args.malicious)
    print(f"  Frequency range: {env.freq_min:.2f} - {env.freq_max:.2f} MHz")
    print(f"  Number of bins: {env.n_bins}")
    
    # Create action space
    action_space = create_default_action_space()
    print(f"  Total actions: {action_space.get_total_actions()}")
    
    # Initialize or load agent
    if args.load:
        print(f"Loading agent from {args.load}...")
        agent = RLWaveformAgent(action_space)
        agent.load(args.load)
        print(f"  Loaded agent with {len(agent.q_table)} states in Q-table")
    else:
        print("Initializing new RL agent...")
        agent = RLWaveformAgent(
            action_space=action_space,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
        )
    
    # Create initial waveform
    initial_waveform = create_initial_waveform()
    
    # Train or evaluate
    if args.evaluate:
        # Evaluation mode
        eval_dir = output_dir / 'evaluation'
        best_waveform, results = evaluate_agent(
            env, agent, initial_waveform, args.max_steps, eval_dir
        )
    else:
        # Training mode
        train_dir = output_dir / 'training'
        agent = train_agent(
            env, agent, initial_waveform, 
            args.episodes, args.max_steps, train_dir
        )
        
        # After training, run evaluation
        print("\nRunning post-training evaluation...")
        eval_dir = output_dir / 'evaluation'
        best_waveform, results = evaluate_agent(
            env, agent, initial_waveform, args.max_steps, eval_dir
        )
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

