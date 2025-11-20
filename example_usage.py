#!/usr/bin/env python3
"""
example_usage.py

Simple example demonstrating how to use the RL waveform agent.

This script shows:
1. How to initialize the environment
2. How to create and train an agent
3. How to evaluate the trained agent
4. How to interpret the results
"""

from pathlib import Path
from rl_waveform_agent import (
    WaveformParams,
    ActionSpace,
    SpectrumEnvironment,
    RLWaveformAgent,
)


def main():
    print("\n" + "="*70)
    print("RL Waveform Agent - Simple Example")
    print("="*70 + "\n")
    
    # Step 1: Initialize the spectrum environment
    print("Step 1: Loading spectrum environment...")
    print("-" * 70)
    
    # Note: Update these paths to your actual data files
    bg_path = "benign/background/2.4ghz_passivescan_background_loc1_1.csv"
    floor_path = "benign/background/2.4ghz_passivescan_floor_1.csv"
    jammer_path = "benign/background/2412mhz_jamming_3dbm_gaussiannoise_1.csv"
    
    env = SpectrumEnvironment(bg_path, floor_path, jammer_path)
    
    print(f"✓ Loaded environment")
    print(f"  Frequency range: {env.freq_min:.2f} - {env.freq_max:.2f} MHz")
    print(f"  Number of bins: {env.n_bins}")
    print()
    
    # Step 2: Define initial waveform
    print("Step 2: Creating initial waveform...")
    print("-" * 70)
    
    initial_waveform = WaveformParams(
        center_freq_mhz=2412.0,  # WiFi channel 1
        bandwidth_khz=200.0,      # 200 kHz
        tx_power_dbm=0.0,         # 0 dBm
        modulation='OFDM',        # OFDM modulation (fixed)
    )
    
    print(f"✓ Initial waveform:")
    print(f"  Center frequency: {initial_waveform.center_freq_mhz} MHz")
    print(f"  Bandwidth: {initial_waveform.bandwidth_khz} kHz")
    print(f"  TX Power: {initial_waveform.tx_power_dbm} dBm")
    print(f"  Modulation: {initial_waveform.modulation}")
    
    # Evaluate initial waveform
    initial_metrics = env.evaluate_waveform(initial_waveform)
    print(f"\n  Initial performance:")
    print(f"    SINR: {initial_metrics['sinr_db']:.2f} dB")
    print(f"    BER: {initial_metrics['ber']:.6e}")
    print(f"    Goodput: {initial_metrics['goodput_bps']/1e6:.3f} Mbps")
    print()
    
    # Step 3: Create action space and agent
    print("Step 3: Initializing RL agent...")
    print("-" * 70)
    
    action_space = ActionSpace(
        freq_steps_mhz=[-20.0, -15.0, -10.0, -5.0, -2.0, 0.0, +2.0, +5.0, +10.0, +15.0, +20.0],
        bw_multipliers=[0.75, 1.0, 1.25],
        power_steps_dbm=[-1.0, 0.0, +1.0],  # Limited power range
    )
    
    agent = RLWaveformAgent(
        action_space=action_space,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
    )
    
    print(f"✓ Created RL agent")
    print(f"  Total actions: {action_space.get_total_actions()}")
    print(f"  Learning rate: {agent.alpha}")
    print(f"  Discount factor: {agent.gamma}")
    print(f"  Initial epsilon: {agent.epsilon}")
    print()
    
    # Step 4: Train the agent
    print("Step 4: Training agent...")
    print("-" * 70)
    
    n_episodes = 50
    max_steps = 30
    
    print(f"Training for {n_episodes} episodes ({max_steps} steps each)...")
    
    for episode in range(n_episodes):
        total_reward, history = agent.train_episode(env, initial_waveform, max_steps)
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = sum(agent.episode_rewards[-10:]) / 10
            print(f"  Episode {episode+1}/{n_episodes} | "
                  f"Reward: {total_reward:.4f} | "
                  f"Avg (last 10): {avg_reward:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print(f"\n✓ Training complete!")
    print(f"  Best reward: {agent.best_reward:.6f}")
    print(f"  Q-table size: {len(agent.q_table)} states")
    print()
    
    # Step 5: Evaluate the trained agent
    print("Step 5: Evaluating trained agent...")
    print("-" * 70)
    
    optimized_waveform, trajectory = agent.evaluate(env, initial_waveform, max_steps=30)
    
    # Get final metrics
    final_metrics = env.evaluate_waveform(optimized_waveform)
    final_reward, _ = env.compute_reward(optimized_waveform)
    
    print(f"✓ Optimization complete!")
    print(f"\n  Optimized waveform:")
    print(f"    Center frequency: {optimized_waveform.center_freq_mhz:.2f} MHz "
          f"(Δ {optimized_waveform.center_freq_mhz - initial_waveform.center_freq_mhz:+.2f})")
    print(f"    Bandwidth: {optimized_waveform.bandwidth_khz:.2f} kHz "
          f"(Δ {optimized_waveform.bandwidth_khz - initial_waveform.bandwidth_khz:+.2f})")
    print(f"    TX Power: {optimized_waveform.tx_power_dbm:.2f} dBm "
          f"(Δ {optimized_waveform.tx_power_dbm - initial_waveform.tx_power_dbm:+.2f})")
    print(f"    Modulation: {optimized_waveform.modulation}")
    
    print(f"\n  Final performance:")
    print(f"    SINR: {final_metrics['sinr_db']:.2f} dB "
          f"({final_metrics['sinr_db'] - initial_metrics['sinr_db']:+.2f})")
    print(f"    BER: {final_metrics['ber']:.6e}")
    print(f"    Goodput: {final_metrics['goodput_bps']/1e6:.3f} Mbps "
          f"({(final_metrics['goodput_bps'] - initial_metrics['goodput_bps'])/1e6:+.3f})")
    print(f"    Reward: {final_reward:.6f}")
    
    # Step 6: Interpret results
    print("\n" + "="*70)
    print("Interpretation:")
    print("-" * 70)
    
    sinr_improvement = final_metrics['sinr_db'] - initial_metrics['sinr_db']
    goodput_improvement = (final_metrics['goodput_bps'] - initial_metrics['goodput_bps']) / 1e6
    
    if sinr_improvement > 1.0:
        print("✓ SIGNIFICANT IMPROVEMENT: Agent found less jammed frequency region")
    elif sinr_improvement > 0:
        print("✓ MODEST IMPROVEMENT: Agent slightly improved SINR")
    else:
        print("✗ NO IMPROVEMENT: Agent did not find better configuration")
        print("  (Try training longer or with different hyperparameters)")
    
    if goodput_improvement > 0.1:
        print(f"✓ GOODPUT BOOST: +{goodput_improvement:.3f} Mbps gain in throughput")
    
    if optimized_waveform.center_freq_mhz != initial_waveform.center_freq_mhz:
        print(f"✓ FREQUENCY SHIFT: Moved from {initial_waveform.center_freq_mhz:.1f} MHz "
              f"to {optimized_waveform.center_freq_mhz:.1f} MHz to avoid jamming")
    
    # Note: Modulation is fixed to OFDM and cannot change
    
    print("\n" + "="*70)
    print("Example complete! For full training/evaluation, use train_rl_agent.py")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

