"""
Evaluation script for comparing defender policies
Tests Antigravity Defender vs baselines (random, static, adaptive)
"""
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.fraud_env import FraudAntigravityEnv
from agents.fraudster_agent import FraudsterAgent, RandomFraudster, AggressiveFraudster
from agents.defender_agent import (AntigravityDefender, RandomDefender, 
                                     StaticThresholdDefender, AdaptiveThresholdDefender,
                                     AlwaysStrictDefender)
from utils.metrics import compute_metrics


def evaluate_policy(env, fraudster, defender, num_episodes=100, verbose=False):
    """
    Evaluate a defender policy against a fraudster
    
    Returns:
        results: dict with performance metrics
    """
    fraud_success_rates = []
    system_losses = []
    defender_rewards = []
    fraudster_rewards = []
    investigation_costs = []
    fp_costs = []
    
    # For computing detection metrics
    all_fraud_labels = []
    all_detections = []
    
    for ep in tqdm(range(num_episodes), desc="Evaluating", disable=not verbose):
        fraudster_obs, defender_obs = env.reset()
        done = False
        
        episode_system_loss = 0.0
        episode_defender_reward = 0.0
        episode_fraudster_reward = 0.0
        episode_inv_cost = 0.0
        episode_fp_cost = 0.0
        
        fraud_attempts = 0
        fraud_successes = 0
        
        ep_fraud_labels = []
        ep_detections = []
        
        while not done:
            fraudster_action = fraudster.predict(fraudster_obs, deterministic=True)
            defender_action = defender.predict(defender_obs, deterministic=True)
            
            (fraudster_obs, defender_obs), (f_reward, d_reward), done, info = \
                env.step(fraudster_action, defender_action)
            
            episode_system_loss += info['system_loss']
            episode_defender_reward += d_reward
            episode_fraudster_reward += f_reward
            episode_inv_cost += info['investigation_cost']
            episode_fp_cost += info['fp_cost']
            
            if info['fraudster_action'] > 0:
                fraud_attempts += 1
                if not info['detected']:
                    fraud_successes += 1
            
            # Track for metrics
            ep_fraud_labels.append(1 if info['fraudster_action'] > 0 else 0)
            ep_detections.append(1 if info['detected'] else 0)
        
        fraud_success_rates.append(fraud_successes / max(fraud_attempts, 1))
        system_losses.append(episode_system_loss)
        defender_rewards.append(episode_defender_reward)
        fraudster_rewards.append(episode_fraudster_reward)
        investigation_costs.append(episode_inv_cost)
        fp_costs.append(episode_fp_cost)
        
        all_fraud_labels.extend(ep_fraud_labels)
        all_detections.extend(ep_detections)
    
    # Compute detection metrics
    detection_metrics = compute_metrics(all_fraud_labels, all_detections)
    
    return {
        'fraud_success_rate_mean': np.mean(fraud_success_rates),
        'fraud_success_rate_std': np.std(fraud_success_rates),
        'system_loss_mean': np.mean(system_losses),
        'system_loss_std': np.std(system_losses),
        'defender_reward_mean': np.mean(defender_rewards),
        'defender_reward_std': np.std(defender_rewards),
        'fraudster_reward_mean': np.mean(fraudster_rewards),
        'fraudster_reward_std': np.std(fraudster_rewards),
        'investigation_cost_mean': np.mean(investigation_costs),
        'fp_cost_mean': np.mean(fp_costs),
        'detection_precision': detection_metrics['precision'],
        'detection_recall': detection_metrics['recall'],
        'detection_f1': detection_metrics['f1']
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Antigravity Defender vs Baselines")
    parser.add_argument("--defender-model", type=str, default=None,
                        help="Path to trained defender model (if None, skip Antigravity)")
    parser.add_argument("--fraudster-model", type=str, default=None,
                        help="Path to trained fraudster model (if None, use oracle)")
    parser.add_argument("--episodes", type=int, default=100, help="Evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results/evaluation.json",
                        help="Output path")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "results", 
                exist_ok=True)
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Initialize environment
    env = FraudAntigravityEnv(max_steps=args.max_steps, seed=args.seed)
    
    # Initialize fraudster
    if args.fraudster_model:
        print(f"📂 Loading fraudster model from {args.fraudster_model}")
        fraudster = FraudsterAgent(env.fraudster_obs_space, env.fraudster_action_space, mode='learner')
        fraudster.load(args.fraudster_model, None)
    else:
        print("🎭 Using oracle fraudster")
        fraudster = FraudsterAgent(env.fraudster_obs_space, env.fraudster_action_space, mode='oracle')
    
    # Define defenders to test
    defenders = {}
    
    if args.defender_model:
        print(f"📂 Loading Antigravity Defender from {args.defender_model}")
        antigravity = AntigravityDefender(env.defender_obs_space, env.defender_action_space)
        antigravity.load(args.defender_model, None)
        defenders['Antigravity Defender'] = antigravity
    
    defenders['Random'] = RandomDefender(env.defender_action_space)
    defenders['Static Threshold'] = StaticThresholdDefender(env.defender_action_space)
    defenders['Adaptive Threshold'] = AdaptiveThresholdDefender(env.defender_action_space)
    defenders['Always Strict'] = AlwaysStrictDefender(env.defender_action_space)
    
    # Evaluate each defender
    print("\n" + "="*60)
    print("🧪 EVALUATION: Antigravity Defender vs Baselines")
    print("="*60)
    
    results = {}
    
    for name, defender in defenders.items():
        print(f"\n🔬 Testing: {name}")
        eval_results = evaluate_policy(env, fraudster, defender, 
                                       num_episodes=args.episodes, verbose=True)
        results[name] = eval_results
        
        print(f"\n   📊 Results for {name}:")
        print(f"      Fraud Success Rate: {eval_results['fraud_success_rate_mean']:.2%} ± {eval_results['fraud_success_rate_std']:.2%}")
        print(f"      System Loss: {eval_results['system_loss_mean']:.3f} ± {eval_results['system_loss_std']:.3f}")
        print(f"      Defender Reward: {eval_results['defender_reward_mean']:.3f} ± {eval_results['defender_reward_std']:.3f}")
        print(f"      Detection F1: {eval_results['detection_f1']:.3f}")
        print(f"      Precision: {eval_results['detection_precision']:.3f}")
        print(f"      Recall: {eval_results['detection_recall']:.3f}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {args.output}")
    
    # Create comparison table
    print("\n" + "="*60)
    print("📊 COMPARISON TABLE")
    print("="*60)
    
    df_data = []
    for name, res in results.items():
        df_data.append({
            'Defender': name,
            'Fraud Success ↓': f"{res['fraud_success_rate_mean']:.2%}",
            'System Loss ↓': f"{res['system_loss_mean']:.3f}",
            'Defender Reward ↑': f"{res['defender_reward_mean']:.3f}",
            'F1 Score ↑': f"{res['detection_f1']:.3f}"
        })
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    
    # Determine winner
    if 'Antigravity Defender' in results:
        antigravity_loss = results['Antigravity Defender']['system_loss_mean']
        best_baseline_loss = min([res['system_loss_mean'] for name, res in results.items() 
                                  if name != 'Antigravity Defender'])
        
        improvement = (best_baseline_loss - antigravity_loss) / best_baseline_loss * 100
        
        print(f"\n🏆 RESULT:")
        if antigravity_loss < best_baseline_loss:
            print(f"   ✅ Antigravity Defender WINS with {improvement:.1f}% lower system loss!")
        else:
            print(f"   ⚠️  Best baseline still outperforms Antigravity (need more training)")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
