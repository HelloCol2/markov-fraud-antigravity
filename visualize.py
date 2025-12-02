"""
Visualization utilities for Antigravity Defender
Learning curves, action heatmaps, strategy analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List


# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_learning_curves(metrics_path: str, save_path: str = None):
    """
    Plot learning curves from training metrics
    
    Args:
        metrics_path: Path to training_metrics.json
        save_path: Path to save figure (if None, display)
    """
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    episodes = data['episodes']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🧲 Antigravity Defender: Learning Curves', fontsize=16, fontweight='bold')
    
    # 1. Fraud Success Rate (lower is better for defender)
    ax = axes[0, 0]
    fraud_success = smooth_curve(data['fraud_success_rates'], window=50)
    ax.plot(episodes, fraud_success, color='crimson', linewidth=2, label='Fraud Success Rate')
    ax.fill_between(episodes, 0, fraud_success, alpha=0.3, color='crimson')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Fraud Success Rate')
    ax.set_title('Fraud Success Rate Over Time (↓ Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. System Loss (lower is better)
    ax = axes[0, 1]
    system_loss = smooth_curve(data['system_losses'], window=50)
    ax.plot(episodes, system_loss, color='orange', linewidth=2, label='System Loss')
    ax.fill_between(episodes, 0, system_loss, alpha=0.3, color='orange')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative System Loss')
    ax.set_title('System Loss Over Time (↓ Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Agent Rewards
    ax = axes[1, 0]
    fraudster_reward = smooth_curve(data['fraudster_rewards'], window=50)
    defender_reward = smooth_curve(data['defender_rewards'], window=50)
    ax.plot(episodes, fraudster_reward, color='red', linewidth=2, label='Fraudster Reward', alpha=0.8)
    ax.plot(episodes, defender_reward, color='blue', linewidth=2, label='Defender Reward', alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Agent Rewards Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Detection Rate
    ax = axes[1, 1]
    detection_rate = np.array(data['detections']) / np.maximum(np.array(data['fraud_attempts']), 1)
    detection_rate_smooth = smooth_curve(detection_rate.tolist(), window=50)
    ax.plot(episodes, detection_rate_smooth, color='green', linewidth=2, label='Detection Rate')
    ax.fill_between(episodes, 0, detection_rate_smooth, alpha=0.3, color='green')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Detection Rate')
    ax.set_title('Fraud Detection Rate Over Time (↑ Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Learning curves saved to {save_path}")
    else:
        plt.show()


def plot_action_heatmap(env, agent, agent_type='defender', num_samples=1000, save_path=None):
    """
    Plot action distribution heatmap based on observation features
    
    Args:
        env: Environment instance
        agent: Agent instance
        agent_type: 'defender' or 'fraudster'
        num_samples: Number of observation samples
        save_path: Path to save figure
    """
    obs_list = []
    actions = []
    
    # Sample observations and actions
    for _ in range(num_samples // 100):
        if agent_type == 'defender':
            obs, _ = env.reset()
        else:
            _, obs = env.reset()
        
        for _ in range(100):
            action = agent.predict(obs, deterministic=True)
            obs_list.append(obs.copy())
            actions.append(action)
            
            # Step environment
            fraudster_action = env.fraudster_action_space.sample()
            defender_action = action if agent_type == 'defender' else env.defender_action_space.sample()
            
            if agent_type == 'fraudster':
                fraudster_action = action
            
            (_, obs), _, done, _ = env.step(fraudster_action, defender_action)
            
            if done:
                break
    
    obs_array = np.array(obs_list)
    actions_array = np.array(actions)
    
    # Create heatmap grid
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'🎯 {agent_type.capitalize()} Action Distribution Heatmap', 
                 fontsize=14, fontweight='bold')
    
    if agent_type == 'defender':
        action_names = ['Lenient', 'Normal', 'Strict']
        feature_pairs = [
            (0, 1, 'Customer Risk', 'Transaction Amount'),
            (3, 4, 'Fraud Rate', 'FP Rate'),
            (0, 3, 'Customer Risk', 'Fraud Rate')
        ]
    else:
        action_names = ['No Attack', 'Low Fraud', 'High Fraud']
        feature_pairs = [
            (0, 1, 'Customer Risk', 'Transaction Amount'),
            (5, 9, 'Fraud Budget', 'Defender Entropy'),
            (0, 5, 'Customer Risk', 'Fraud Budget')
        ]
    
    for idx, (ax, (f1_idx, f2_idx, f1_name, f2_name)) in enumerate(zip(axes, feature_pairs)):
        # Create 2D histogram for each action
        for action_idx in range(3):
            mask = actions_array == action_idx
            if np.sum(mask) > 10:
                ax.scatter(obs_array[mask, f1_idx], obs_array[mask, f2_idx],
                          alpha=0.3, s=10, label=action_names[action_idx])
        
        ax.set_xlabel(f1_name)
        ax.set_ylabel(f2_name)
        ax.set_title(f'{f1_name} vs {f2_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Action heatmap saved to {save_path}")
    else:
        plt.show()


def plot_comparison_bar(evaluation_results: Dict, save_path: str = None):
    """
    Create bar chart comparing defender policies
    
    Args:
        evaluation_results: Dict with results from evaluate.py
        save_path: Path to save figure
    """
    if isinstance(evaluation_results, str):
        with open(evaluation_results, 'r') as f:
            evaluation_results = json.load(f)
    
    defenders = list(evaluation_results.keys())
    
    metrics_to_plot = {
        'fraud_success_rate_mean': 'Fraud Success Rate (↓)',
        'system_loss_mean': 'System Loss (↓)',
        'defender_reward_mean': 'Defender Reward (↑)',
        'detection_f1': 'Detection F1 (↑)'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🏆 Defender Policy Comparison', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot.items()):
        ax = axes[idx]
        
        values = [evaluation_results[d][metric_key] for d in defenders]
        colors = ['#2ecc71' if d == 'Antigravity Defender' else '#95a5a6' for d in defenders]
        
        bars = ax.bar(range(len(defenders)), values, color=colors, alpha=0.8, edgecolor='black')
        
        # Highlight best performer
        if '↓' in metric_name:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        ax.set_xticks(range(len(defenders)))
        ax.set_xticklabels(defenders, rotation=45, ha='right')
        ax.set_ylabel(metric_name.split('(')[0].strip())
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison chart saved to {save_path}")
    else:
        plt.show()


def smooth_curve(values: List[float], window: int = 50) -> np.ndarray:
    """Apply moving average smoothing"""
    if len(values) < window:
        return np.array(values)
    
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2)
        smoothed.append(np.mean(values[start:end]))
    
    return np.array(smoothed)


def create_all_visualizations(metrics_path: str, eval_results_path: str, output_dir: str = "figures"):
    """
    Generate all visualizations
    
    Args:
        metrics_path: Path to training_metrics.json
        eval_results_path: Path to evaluation.json
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("📊 Generating visualizations...")
    
    # Learning curves
    plot_learning_curves(metrics_path, save_path=os.path.join(output_dir, "learning_curves.png"))
    
    # Comparison bar chart
    plot_comparison_bar(eval_results_path, save_path=os.path.join(output_dir, "policy_comparison.png"))
    
    print(f"✅ All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument("--metrics", type=str, default="checkpoints/training_metrics.json")
    parser.add_argument("--eval-results", type=str, default="results/evaluation.json")
    parser.add_argument("--output-dir", type=str, default="figures")
    
    args = parser.parse_args()
    
    create_all_visualizations(args.metrics, args.eval_results, args.output_dir)
