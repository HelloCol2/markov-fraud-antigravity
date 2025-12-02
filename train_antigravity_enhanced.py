"""
Enhanced Training Script for Antigravity Defender
Uses fraud_antigravity_synth-2.csv with strategic behavior principles
"""
import numpy as np
import pandas as pd
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.fraud_env import FraudAntigravityEnv
from agents.fraudster_agent import FraudsterAgent
from agents.antigravity_enhanced import AntigravityDefenderEnhanced, load_enhanced_dataset
from utils.wrappers import SingleAgentWrapper
from utils.metrics import GameMetrics


def train_antigravity_enhanced(
    dataset_path='fraud_antigravity_synth-2.csv',
    episodes=2000,
    max_steps=100,
    save_dir='checkpoints_enhanced',
    verbose=True
):
    """
    Train Antigravity Defender with enhanced strategic principles
    
    Training Philosophy:
    1. Learn to recognize fraud as strategic opponent
    2. Apply dynamic counter-force based on adversarial profitability
    3. Optimize long-term equilibrium over short-term detection
    4. Collapse fraudster payoff while maintaining efficiency
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load enhanced dataset for reference (optional, used for analysis)
    df = load_enhanced_dataset(dataset_path)
    
    if verbose:
        print("=" * 70)
        print("🧲 ANTIGRAVITY DEFENDER - ENHANCED TRAINING")
        print("=" * 70)
        print(f"\n📊 Dataset: {dataset_path}")
        print(f"   Samples: {len(df):,}")
        print(f"   Episodes: {episodes}")
        print(f"   Max steps: {max_steps}")
        print("\n🎯 Training Objectives:")
        print("   ✓ Collapse fraudster profitability over time")
        print("   ✓ Minimize system loss through adaptive counter-force")
        print("   ✓ Balance security ↔ cost ↔ trust equilibrium")
        print("   ✓ Counter-adapt to strategic fraud evolution\n")
    
    # Initialize environment
    env = FraudAntigravityEnv(max_steps=max_steps, seed=42)
    
    # Initialize enhanced Antigravity Defender
    defender = AntigravityDefenderEnhanced(
        obs_space=env.defender_obs_space,
        action_space=env.defender_action_space,
        learning_rate=2e-4  # Slightly lower for stability
    )
    
    # Initialize strategic fraudster (oracle mode)
    fraudster = FraudsterAgent(
        obs_space=env.fraudster_obs_space,
        action_space=env.fraudster_action_space,
        mode='oracle'  # Strategic opponent
    )
    
    # === PHASE 1: Defender Pre-Training ===
    if verbose:
        print("📈 PHASE 1: Defender Pre-Training Against Strategic Fraudster")
        print("   Learning to counter heuristic fraud patterns...\n")
    
    defender_env = SingleAgentWrapper(env, agent_type='defender', opponent=fraudster)
    defender.initialize_model(defender_env)
    
    phase1_steps = (episodes // 2) * max_steps
    defender.learn(total_timesteps=phase1_steps)
    
    if verbose:
        print("\n✅ Phase 1 Complete - Defender learned basic counter-strategies\n")
    
    # === PHASE 2: Adversarial Co-Training ===
    if verbose:
        print("📈 PHASE 2: Adversarial Co-Training (Fraudster Learns to Adapt)")
        print("   Teaching defender to counter-evolve against adaptive adversary...\n")
    
    # Switch fraudster to learner mode
    fraudster_learner = FraudsterAgent(
        obs_space=env.fraudster_obs_space,
        action_space=env.fraudster_action_space,
        mode='learner'
    )
    fraudster_env = SingleAgentWrapper(env, agent_type='fraudster', opponent=defender)
    fraudster_learner.initialize_model(fraudster_env)
    
    # Alternate training rounds
    num_rounds = 5
    steps_per_round = (episodes // 2) * max_steps // num_rounds
    
    for round_idx in range(num_rounds):
        if verbose:
            print(f"🔄 Round {round_idx + 1}/{num_rounds}")
        
        # Train fraudster to exploit current defender
        fraudster_learner.learn(total_timesteps=steps_per_round // 2)
        
        # Train defender to counter new fraudster strategy
        defender_env = SingleAgentWrapper(env, agent_type='defender', opponent=fraudster_learner)
        defender.learn(total_timesteps=steps_per_round)
        
        # Evaluate current equilibrium
        if verbose and round_idx % 2 == 1:
            eval_results = quick_evaluation(env, fraudster_learner, defender, num_eps=20)
            print(f"   → Fraud Success: {eval_results['fraud_success_rate']:.2%}")
            print(f"   → System Loss: {eval_results['system_loss']:.3f}")
            print(f"   → Antigravity Pressure: {-eval_results['fraudster_reward']:.3f}\n")
    
    if verbose:
        print("✅ Phase 2 Complete - Defender learned to counter-adapt\n")
    
    # === FINAL EVALUATION ===
    if verbose:
        print("🎯 FINAL EVALUATION: Testing Antigravity Effectiveness\n")
    
    final_results = quick_evaluation(env, fraudster_learner, defender, num_eps=100)
    
    if verbose:
        print("=" * 70)
        print("🏆 ANTIGRAVITY DEFENDER - FINAL PERFORMANCE")
        print("=" * 70)
        print(f"📊 Fraud Success Rate: {final_results['fraud_success_rate']:.2%}")
        print(f"📊 System Loss: {final_results['system_loss']:.3f}")
        print(f"📊 Defender Reward: {final_results['defender_reward']:.3f}")
        print(f"📊 Fraudster Payoff (collapsed): {final_results['fraudster_reward']:.3f}")
        print(f"📊 Investigation Efficiency: {final_results['inv_cost']:.3f}")
        print("=" * 70)
    
    # Save models
    defender.save(os.path.join(save_dir, "antigravity_defender_enhanced"))
    fraudster_learner.save(os.path.join(save_dir, "fraudster_adaptive"))
    
    if verbose:
        print(f"\n💾 Models saved to {save_dir}/")
        print("✅ Training Complete!\n")
    
    return defender, fraudster_learner, final_results


def quick_evaluation(env, fraudster, defender, num_eps=20):
    """Quick evaluation during training"""
    fraud_successes = []
    system_losses = []
    defender_rewards = []
    fraudster_rewards = []
    inv_costs = []
    
    for _ in range(num_eps):
        fraudster_obs, defender_obs = env.reset()
        done = False
        
        ep_sys_loss = 0.0
        ep_def_reward = 0.0
        ep_fraud_reward = 0.0
        ep_inv_cost = 0.0
        fraud_attempts = 0
        fraud_successes_ep = 0
        
        while not done:
            f_action = fraudster.predict(fraudster_obs, deterministic=True)
            d_action = defender.predict(defender_obs, deterministic=True)
            
            (fraudster_obs, defender_obs), (f_r, d_r), done, info = env.step(f_action, d_action)
            
            ep_sys_loss += info['system_loss']
            ep_def_reward += d_r
            ep_fraud_reward += f_r
            ep_inv_cost += info['investigation_cost']
            
            if info['fraudster_action'] > 0:
                fraud_attempts += 1
                if not info['detected']:
                    fraud_successes_ep += 1
        
        fraud_successes.append(fraud_successes_ep / max(fraud_attempts, 1))
        system_losses.append(ep_sys_loss)
        defender_rewards.append(ep_def_reward)
        fraudster_rewards.append(ep_fraud_reward)
        inv_costs.append(ep_inv_cost)
    
    return {
        'fraud_success_rate': np.mean(fraud_successes),
        'system_loss': np.mean(system_losses),
        'defender_reward': np.mean(defender_rewards),
        'fraudster_reward': np.mean(fraudster_rewards),
        'inv_cost': np.mean(inv_costs)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Enhanced Antigravity Defender with Strategic Principles"
    )
    parser.add_argument("--dataset", type=str, 
                       default="fraud_antigravity_synth-2.csv",
                       help="Path to enhanced dataset")
    parser.add_argument("--episodes", type=int, default=2000,
                       help="Total training episodes")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Max steps per episode")
    parser.add_argument("--save-dir", type=str, 
                       default="checkpoints_enhanced",
                       help="Directory to save models")
    parser.add_argument("--test-mode", action="store_true",
                       help="Quick test with 10 episodes")
    
    args = parser.parse_args()
    
    if args.test_mode:
        args.episodes = 10
        print("🧪 TEST MODE: Running 10 episodes only\n")
    
    # Train
    defender, fraudster, results = train_antigravity_enhanced(
        dataset_path=args.dataset,
        episodes=args.episodes,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        verbose=True
    )
    
    print("\n🎉 Enhanced Antigravity Defender training complete!")
    print("   The defender has learned to collapse fraudster profitability")
    print("   through adaptive counter-force application.\n")


if __name__ == "__main__":
    main()
