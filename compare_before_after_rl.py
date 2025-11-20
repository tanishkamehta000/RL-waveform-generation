#!/usr/bin/env python3
"""
compare_before_after_rl.py

Comprehensive comparison script that:
1. Initializes a waveform in a jammed environment
2. Calculates initial performance (SINR, BER, goodput)
3. Trains an RL agent to optimize the waveform
4. Calculates optimized performance
5. Compares before/after results with visualizations

Usage:
    python compare_before_after_rl.py
    python compare_before_after_rl.py --episodes 100 --max_steps 50
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
)


def print_header(text, char='='):
    """Print a formatted header"""
    print(f"\n{char * 70}")
    print(f"{text}")
    print(f"{char * 70}\n")


def print_metrics(title, metrics, waveform):
    """Print formatted metrics"""
    print(f"{title}")
    print("-" * 70)
    print(f"  Waveform Configuration:")
    print(f"    Center Frequency: {waveform.center_freq_mhz:.2f} MHz")
    print(f"    Bandwidth: {waveform.bandwidth_khz:.2f} kHz")
    print(f"    TX Power: {waveform.tx_power_dbm:.2f} dBm")
    print(f"    Modulation: {waveform.modulation}")
    print(f"\n  Performance Metrics:")
    print(f"    SINR: {metrics['sinr_db']:.2f} dB ({metrics['sinr_linear']:.4f} linear)")
    print(f"    BER: {metrics['ber']:.6e}")
    print(f"    Goodput: {metrics['goodput_bps']/1e6:.4f} Mbps ({metrics['goodput_bps']:.1f} bps)")
    print(f"    Signal Power: {metrics['P_sig_mw']:.6f} mW")
    print(f"    Interference Power: {metrics['P_int_mw']:.6f} mW")
    print()


def plot_comparison(initial_waveform, initial_metrics, optimized_waveform, 
                   optimized_metrics, env, training_history, output_dir):
    """Create comprehensive comparison plots"""
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Spectrum with waveform placements
    ax1 = plt.subplot(3, 3, 1)
    freqs = env.frequencies
    eps = 1e-15
    interference_db = 10.0 * np.log10(env.interference + eps)
    jam_db = 10.0 * np.log10(env.jam_profile + eps)
    
    ax1.plot(freqs, interference_db, 'k-', linewidth=1.5, label='Total Interference')
    ax1.plot(freqs, jam_db, 'r-', alpha=0.6, label='Jammer')
    
    # Show initial waveform location
    from run_waveform_in_jammed_env import make_waveform_mask
    mask_initial = make_waveform_mask(freqs, initial_waveform.center_freq_mhz,
                                       initial_waveform.bandwidth_khz)
    mask_opt = make_waveform_mask(freqs, optimized_waveform.center_freq_mhz,
                                   optimized_waveform.bandwidth_khz)
    
    # Shade initial position
    ax1.axvspan(initial_waveform.center_freq_mhz - initial_waveform.bandwidth_khz/2000,
                initial_waveform.center_freq_mhz + initial_waveform.bandwidth_khz/2000,
                alpha=0.3, color='red', label='Initial Position')
    # Shade optimized position
    ax1.axvspan(optimized_waveform.center_freq_mhz - optimized_waveform.bandwidth_khz/2000,
                optimized_waveform.center_freq_mhz + optimized_waveform.bandwidth_khz/2000,
                alpha=0.3, color='green', label='Optimized Position')
    
    ax1.set_xlabel('Frequency (MHz)')
    ax1.set_ylabel('Power (dB)')
    ax1.set_title('Spectrum: Initial vs Optimized Waveform Placement')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. SINR Comparison
    ax2 = plt.subplot(3, 3, 2)
    categories = ['Initial', 'Optimized']
    sinr_values = [initial_metrics['sinr_db'], optimized_metrics['sinr_db']]
    colors = ['red', 'green']
    bars = ax2.bar(categories, sinr_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('SINR (dB)')
    ax2.set_title('SINR Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, val in zip(bars, sinr_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f} dB', ha='center', va='bottom', fontweight='bold')
    
    # 3. BER Comparison (log scale)
    ax3 = plt.subplot(3, 3, 3)
    ber_values = [initial_metrics['ber'], optimized_metrics['ber']]
    bars = ax3.bar(categories, ber_values, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('BER (log scale)')
    ax3.set_yscale('log')
    ax3.set_title('Bit Error Rate Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, val in zip(bars, ber_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 4. Goodput Comparison
    ax4 = plt.subplot(3, 3, 4)
    goodput_values = [initial_metrics['goodput_bps']/1e6, optimized_metrics['goodput_bps']/1e6]
    bars = ax4.bar(categories, goodput_values, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Goodput (Mbps)')
    ax4.set_title('Goodput Comparison')
    ax4.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, val in zip(bars, goodput_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f} Mbps', ha='center', va='bottom', fontweight='bold')
    
    # 5. Improvement Percentages
    ax5 = plt.subplot(3, 3, 5)
    sinr_improvement = ((optimized_metrics['sinr_linear'] - initial_metrics['sinr_linear']) / 
                        initial_metrics['sinr_linear'] * 100)
    goodput_improvement = ((optimized_metrics['goodput_bps'] - initial_metrics['goodput_bps']) / 
                           initial_metrics['goodput_bps'] * 100)
    ber_improvement = ((initial_metrics['ber'] - optimized_metrics['ber']) / 
                       initial_metrics['ber'] * 100)
    
    improvements = ['SINR', 'Goodput', 'BER\nReduction']
    values = [sinr_improvement, goodput_improvement, ber_improvement]
    colors_imp = ['green' if v > 0 else 'red' for v in values]
    bars = ax5.barh(improvements, values, color=colors_imp, alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Improvement (%)')
    ax5.set_title('Performance Improvements')
    ax5.grid(True, alpha=0.3, axis='x')
    ax5.axvline(0, color='black', linewidth=0.8)
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax5.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:+.1f}%', ha='left' if val > 0 else 'right', 
                va='center', fontweight='bold')
    
    # 6. Training Progress (if available)
    if training_history:
        ax6 = plt.subplot(3, 3, 6)
        episodes = range(1, len(training_history) + 1)
        ax6.plot(episodes, training_history, 'b-', alpha=0.6, linewidth=1)
        # Moving average
        if len(training_history) > 10:
            window = 10
            moving_avg = np.convolve(training_history, np.ones(window)/window, mode='valid')
            ax6.plot(range(window, len(training_history) + 1), moving_avg,
                    'r-', linewidth=2, label=f'{window}-Episode Avg')
            ax6.legend()
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Total Reward')
        ax6.set_title('RL Training Progress')
        ax6.grid(True, alpha=0.3)
    
    # 7. Frequency Shift Visualization
    ax7 = plt.subplot(3, 3, 7)
    freq_shift = optimized_waveform.center_freq_mhz - initial_waveform.center_freq_mhz
    ax7.arrow(0, 0, freq_shift, 0, head_width=0.3, head_length=abs(freq_shift)*0.1,
             fc='blue', ec='blue', linewidth=2)
    ax7.plot([0], [0], 'ro', markersize=10, label='Initial')
    ax7.plot([freq_shift], [0], 'go', markersize=10, label='Optimized')
    ax7.set_xlim([min(0, freq_shift)-5, max(0, freq_shift)+5])
    ax7.set_ylim([-1, 1])
    ax7.set_xlabel('Frequency Shift (MHz)')
    ax7.set_title(f'Frequency Adjustment: {freq_shift:+.2f} MHz')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_yticks([])
    
    # 8. Parameter Changes
    ax8 = plt.subplot(3, 3, 8)
    bw_change = ((optimized_waveform.bandwidth_khz - initial_waveform.bandwidth_khz) / 
                 initial_waveform.bandwidth_khz * 100)
    power_change = optimized_waveform.tx_power_dbm - initial_waveform.tx_power_dbm
    
    params = ['Bandwidth\n(%)', 'Power\n(dBm)']
    changes = [bw_change, power_change]
    colors_param = ['green' if v >= 0 else 'red' for v in changes]
    bars = ax8.barh(params, changes, color=colors_param, alpha=0.7, edgecolor='black')
    ax8.set_xlabel('Change')
    ax8.set_title('Parameter Adjustments')
    ax8.grid(True, alpha=0.3, axis='x')
    ax8.axvline(0, color='black', linewidth=0.8)
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, changes)):
        width = bar.get_width()
        label = f'{val:+.1f}%' if i == 0 else f'{val:+.1f} dBm'
        ax8.text(width, bar.get_y() + bar.get_height()/2.,
                label, ha='left' if val >= 0 else 'right', 
                va='center', fontweight='bold')
    
    # 9. Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
    PERFORMANCE SUMMARY
    {'='*40}
    
    Initial Configuration:
      • Frequency: {initial_waveform.center_freq_mhz:.1f} MHz
      • SINR: {initial_metrics['sinr_db']:.2f} dB
      • BER: {initial_metrics['ber']:.2e}
      • Goodput: {initial_metrics['goodput_bps']/1e6:.4f} Mbps
    
    Optimized Configuration:
      • Frequency: {optimized_waveform.center_freq_mhz:.1f} MHz
      • SINR: {optimized_metrics['sinr_db']:.2f} dB
      • BER: {optimized_metrics['ber']:.2e}
      • Goodput: {optimized_metrics['goodput_bps']/1e6:.4f} Mbps
    
    Improvements:
      • SINR: {sinr_improvement:+.1f}%
      • Goodput: {goodput_improvement:+.1f}%
      • BER Reduction: {ber_improvement:+.1f}%
    
    Agent Actions:
      • Frequency Shift: {freq_shift:+.2f} MHz
      • Bandwidth Change: {bw_change:+.1f}%
      • Power Change: {power_change:+.1f} dBm
    """
    
    ax9.text(0.1, 0.5, summary_text, transform=ax9.transAxes,
            fontsize=9, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('RL Waveform Optimization: Before vs After Comparison', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'before_after_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved comparison plot to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare waveform performance before and after RL optimization'
    )
    
    # Data paths
    parser.add_argument('--bg', default='vip_data/2.4ghz_passivescan_background_loc1_1.csv',
                       help='Background spectrum CSV')
    parser.add_argument('--floor', default='vip_data/2.4ghz_passivescan_floor_1.csv',
                       help='Noise floor spectrum CSV')
    parser.add_argument('--malicious', default='vip_data/2412mhz_jamming_3dbm_gaussiannoise_1.csv',
                       help='Jammer spectrum CSV')
    
    # Initial waveform configuration
    parser.add_argument('--init_freq', type=float, default=2412.0,
                       help='Initial center frequency (MHz)')
    parser.add_argument('--init_bw', type=float, default=200.0,
                       help='Initial bandwidth (kHz)')
    parser.add_argument('--init_power', type=float, default=0.0,
                       help='Initial TX power (dBm)')
    
    # RL training parameters
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=50,
                       help='Maximum steps per episode')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                       help='Exploration decay rate')
    
    # Output
    parser.add_argument('--output_dir', default='comparison_output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_header("RL WAVEFORM OPTIMIZATION: BEFORE vs AFTER COMPARISON")
    
    # Save configuration
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {output_dir / 'config.json'}")
    
    # ========================================================================
    # STEP 1: Load Environment
    # ========================================================================
    print_header("STEP 1: Loading Spectrum Environment", char='-')
    print(f"  Background: {args.bg}")
    print(f"  Floor: {args.floor}")
    print(f"  Jammer: {args.malicious}")
    
    env = SpectrumEnvironment(args.bg, args.floor, args.malicious)
    
    print(f"  ✓ Loaded successfully")
    print(f"  Frequency range: {env.freq_min:.2f} - {env.freq_max:.2f} MHz")
    print(f"  Number of bins: {env.n_bins}")
    
    # ========================================================================
    # STEP 2: Initialize Waveform and Evaluate
    # ========================================================================
    print_header("STEP 2: Initialize Waveform (BEFORE RL)", char='-')
    
    initial_waveform = WaveformParams(
        center_freq_mhz=args.init_freq,
        bandwidth_khz=args.init_bw,
        tx_power_dbm=args.init_power,
        modulation='OFDM'
    )
    
    initial_metrics = env.evaluate_waveform(initial_waveform)
    initial_reward, _ = env.compute_reward(initial_waveform)
    
    print_metrics("Initial Waveform Performance:", initial_metrics, initial_waveform)
    print(f"  Initial Reward: {initial_reward:.6f}\n")
    
    # ========================================================================
    # STEP 3: Train RL Agent
    # ========================================================================
    print_header("STEP 3: Training RL Agent", char='-')
    
    # Create action space - emphasize frequency exploration over power
    # Power is capped to force agent to find clean spectrum
    action_space = ActionSpace(
        freq_steps_mhz=[-20.0, -15.0, -10.0, -5.0, -2.0, -1.0, 0.0, +1.0, +2.0, +5.0, +10.0, +15.0, +20.0],
        bw_multipliers=[0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5],
        power_steps_dbm=[-1.0, 0.0, +1.0],  # Smaller steps, capped range
    )
    
    # Create agent
    agent = RLWaveformAgent(
        action_space=action_space,
        learning_rate=args.learning_rate,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=args.epsilon_decay,
    )
    
    print(f"  Total actions: {action_space.get_total_actions()}")
    print(f"  Training episodes: {args.episodes}")
    print(f"  Max steps per episode: {args.max_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epsilon decay: {args.epsilon_decay}")
    print()
    
    # Training loop with progress
    print("  Training in progress...")
    training_rewards = []
    
    for episode in range(args.episodes):
        total_reward, history = agent.train_episode(env, initial_waveform, args.max_steps)
        training_rewards.append(total_reward)
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(training_rewards[-10:])
            print(f"    Episode {episode+1}/{args.episodes} | "
                  f"Reward: {total_reward:.4f} | "
                  f"Avg (last 10): {avg_reward:.4f} | "
                  f"Best: {agent.best_reward:.4f} | "
                  f"ε: {agent.epsilon:.3f}")
    
    print(f"\n  ✓ Training complete!")
    print(f"  Q-table size: {len(agent.q_table)} states")
    print(f"  Best reward achieved: {agent.best_reward:.6f}")
    
    # Save trained agent
    agent_path = output_dir / 'trained_agent.json'
    agent.save(str(agent_path))
    print(f"  Saved agent to: {agent_path}")
    
    # ========================================================================
    # STEP 4: Evaluate Optimized Waveform
    # ========================================================================
    print_header("STEP 4: Evaluate Optimized Waveform (AFTER RL)", char='-')
    
    # Run evaluation (no exploration)
    optimized_waveform, optimization_trajectory = agent.evaluate(
        env, initial_waveform, max_steps=args.max_steps
    )
    
    optimized_metrics = env.evaluate_waveform(optimized_waveform)
    optimized_reward, _ = env.compute_reward(optimized_waveform)
    
    print_metrics("Optimized Waveform Performance:", optimized_metrics, optimized_waveform)
    print(f"  Optimized Reward: {optimized_reward:.6f}\n")
    
    # ========================================================================
    # STEP 5: Compare Results
    # ========================================================================
    print_header("STEP 5: Comparison Summary", char='-')
    
    # Calculate improvements
    sinr_diff = optimized_metrics['sinr_db'] - initial_metrics['sinr_db']
    sinr_pct = ((optimized_metrics['sinr_linear'] - initial_metrics['sinr_linear']) / 
                initial_metrics['sinr_linear'] * 100)
    
    ber_diff = optimized_metrics['ber'] - initial_metrics['ber']
    ber_pct = ((initial_metrics['ber'] - optimized_metrics['ber']) / 
               initial_metrics['ber'] * 100)
    
    goodput_diff = optimized_metrics['goodput_bps'] - initial_metrics['goodput_bps']
    goodput_pct = (goodput_diff / initial_metrics['goodput_bps'] * 100)
    
    freq_shift = optimized_waveform.center_freq_mhz - initial_waveform.center_freq_mhz
    bw_shift = optimized_waveform.bandwidth_khz - initial_waveform.bandwidth_khz
    power_shift = optimized_waveform.tx_power_dbm - initial_waveform.tx_power_dbm
    
    print("  SINR:")
    print(f"    Initial: {initial_metrics['sinr_db']:.2f} dB")
    print(f"    Optimized: {optimized_metrics['sinr_db']:.2f} dB")
    print(f"    Change: {sinr_diff:+.2f} dB ({sinr_pct:+.1f}%)")
    
    print("\n  BER:")
    print(f"    Initial: {initial_metrics['ber']:.6e}")
    print(f"    Optimized: {optimized_metrics['ber']:.6e}")
    print(f"    Reduction: {ber_pct:+.1f}%")
    
    print("\n  Goodput:")
    print(f"    Initial: {initial_metrics['goodput_bps']/1e6:.4f} Mbps")
    print(f"    Optimized: {optimized_metrics['goodput_bps']/1e6:.4f} Mbps")
    print(f"    Change: {goodput_diff/1e6:+.4f} Mbps ({goodput_pct:+.1f}%)")
    
    print("\n  Waveform Adjustments:")
    print(f"    Frequency shift: {freq_shift:+.2f} MHz")
    print(f"    Bandwidth change: {bw_shift:+.2f} kHz")
    print(f"    Power change: {power_shift:+.2f} dBm")
    
    # Overall assessment
    print("\n  Overall Assessment:")
    if sinr_pct > 10 and goodput_pct > 10:
        print("    ✅ EXCELLENT: RL agent significantly improved performance!")
    elif sinr_pct > 5 and goodput_pct > 5:
        print("    ✅ GOOD: RL agent improved performance notably")
    elif sinr_pct > 0 and goodput_pct > 0:
        print("    ✅ MODEST: RL agent achieved some improvement")
    else:
        print("    ⚠️  LIMITED: RL agent found minimal improvement")
        print("       (Initial waveform may already be near-optimal)")
    
    # ========================================================================
    # STEP 6: Save Results and Visualizations
    # ========================================================================
    print_header("STEP 6: Saving Results", char='-')
    
    # Save comparison results
    results = {
        'initial': {
            'waveform': initial_waveform.to_dict(),
            'metrics': initial_metrics,
            'reward': initial_reward,
        },
        'optimized': {
            'waveform': optimized_waveform.to_dict(),
            'metrics': optimized_metrics,
            'reward': optimized_reward,
        },
        'improvements': {
            'sinr_db_change': float(sinr_diff),
            'sinr_percent_change': float(sinr_pct),
            'ber_percent_reduction': float(ber_pct),
            'goodput_bps_change': float(goodput_diff),
            'goodput_percent_change': float(goodput_pct),
            'frequency_shift_mhz': float(freq_shift),
            'bandwidth_change_khz': float(bw_shift),
            'power_change_dbm': float(power_shift),
        },
        'training': {
            'episodes': args.episodes,
            'final_epsilon': agent.epsilon,
            'q_table_size': len(agent.q_table),
            'episode_rewards': training_rewards,
        }
    }
    
    results_path = output_dir / 'comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved results to: {results_path}")
    
    # Create visualizations
    print("  Creating visualizations...")
    plot_comparison(initial_waveform, initial_metrics, optimized_waveform,
                   optimized_metrics, env, training_rewards, output_dir)
    
    print_header("DONE! All results saved to: " + str(output_dir))
    
    return results


if __name__ == '__main__':
    main()

