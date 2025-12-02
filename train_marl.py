"""
Multi-Agent RL Training Loop
Adversarial co-training of fraudster and defender agents
"""
import numpy as np
import argparse
import os
from tqdm import tqdm
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.fraud_env import FraudAntigravityEnv
from agents.fraudster_agent import FraudsterAgent, RandomFraudster
from agents.defender_agent import AntigravityDefender, RandomDefender
from utils.metrics import compute_metrics, GameMetrics


class MultiAgentTrainer:
    """
    Self-play training for fraudster vs defender Markov game
    Implements adversarial co-evolution with periodic best-response updates
    """
    
    def __init__(self, 
                 env: FraudAntigravityEnv,
                 fraudster: FraudsterAgent,
                 defender: AntigravityDefender,
                 save_dir: str = "checkpoints"):
        self.env = env
        self.fraudster = fraudster
        self.defender = defender
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Metrics tracking
        self.metrics = GameMetrics()
        
    def train_episode(self, render: bool = False) -> dict:
        """
        Run one episode of the Markov game
        
        Returns:
            episode_info: dict with rewards, fraud success, costs, etc.
        """
        fraudster_obs, defender_obs = self.env.reset()
        
        episode_fraudster_reward = 0.0
        episode_defender_reward = 0.0
        episode_system_loss = 0.0
        fraud_attempts = 0
        fraud_successes = 0
        detections = 0
        
        done = False
        step = 0
        
        while not done:
            # Both agents select actions
            fraudster_action = self.fraudster.predict(fraudster_obs, deterministic=False)
            defender_action = self.defender.predict(defender_obs, deterministic=False)
            
            # Environment step
            (fraudster_obs, defender_obs), (f_reward, d_reward), done, info = \
                self.env.step(fraudster_action, defender_action)
            
            episode_fraudster_reward += f_reward
            episode_defender_reward += d_reward
            episode_system_loss += info['system_loss']
            
            if info['fraudster_action'] > 0:
                fraud_attempts += 1
                if not info['detected']:
                    fraud_successes += 1
            
            if info['detected']:
                detections += 1
            
            if render:
                self.env.render()
            
            step += 1
        
        fraud_success_rate = fraud_successes / max(fraud_attempts, 1)
        
        return {
            'fraudster_reward': episode_fraudster_reward,
            'defender_reward': episode_defender_reward,
            'system_loss': episode_system_loss,
            'fraud_success_rate': fraud_success_rate,
            'fraud_attempts': fraud_attempts,
            'detections': detections,
            'steps': step
        }
    
    def train_alternating(self, 
                         total_episodes: int,
                         fraudster_train_interval: int = 10,
                         defender_train_interval: int = 10,
                         eval_interval: int = 50,
                         verbose: bool = True):
        """
        Alternating training: train fraudster, then defender, repeat
        
        Args:
            total_episodes: Total training episodes
            fraudster_train_interval: Episodes between fraudster updates
            defender_train_interval: Episodes between defender updates
            eval_interval: Episodes between evaluation runs
            verbose: Print progress
        """
        
        if verbose:
            print("🚀 Starting Multi-Agent RL Training")
            print(f"Total Episodes: {total_episodes}")
            print("="*60)
        
        for episode in tqdm(range(total_episodes), desc="Training"):
            # Run episode
            episode_info = self.train_episode(render=False)
            
            # Log metrics
            self.metrics.add_episode(episode_info)
            
            # Periodic evaluation
            if episode % eval_interval == 0 and episode > 0:
                eval_results = self.evaluate(num_episodes=10)
                if verbose:
                    print(f"\n📊 Evaluation at Episode {episode}")
                    print(f"   Fraud Success Rate: {eval_results['fraud_success_rate']:.2%}")
                    print(f"   Avg System Loss: {eval_results['avg_system_loss']:.3f}")
                    print(f"   Avg Defender Reward: {eval_results['avg_defender_reward']:.3f}")
                
                # Save checkpoint
                self.save_checkpoint(f"episode_{episode}")
        
        if verbose:
            print("\n✅ Training Complete!")
        
        # Final save
        self.save_checkpoint("final")
        self.save_metrics()
    
    def evaluate(self, num_episodes: int = 100, deterministic: bool = True) -> dict:
        """
        Evaluate current policies
        
        Args:
            num_episodes: Number of evaluation episodes
            deterministic: Use deterministic actions
        
        Returns:
            evaluation_results: dict with aggregated metrics
        """
        fraud_success_rates = []
        system_losses = []
        defender_rewards = []
        fraudster_rewards = []
        
        for _ in range(num_episodes):
            fraudster_obs, defender_obs = self.env.reset()
            done = False
            
            episode_system_loss = 0.0
            episode_defender_reward = 0.0
            episode_fraudster_reward = 0.0
            fraud_attempts = 0
            fraud_successes = 0
            
            while not done:
                fraudster_action = self.fraudster.predict(fraudster_obs, deterministic=deterministic)
                defender_action = self.defender.predict(defender_obs, deterministic=deterministic)
                
                (fraudster_obs, defender_obs), (f_reward, d_reward), done, info = \
                    self.env.step(fraudster_action, defender_action)
                
                episode_system_loss += info['system_loss']
                episode_defender_reward += d_reward
                episode_fraudster_reward += f_reward
                
                if info['fraudster_action'] > 0:
                    fraud_attempts += 1
                    if not info['detected']:
                        fraud_successes += 1
            
            fraud_success_rates.append(fraud_successes / max(fraud_attempts, 1))
            system_losses.append(episode_system_loss)
            defender_rewards.append(episode_defender_reward)
            fraudster_rewards.append(episode_fraudster_reward)
        
        return {
            'fraud_success_rate': np.mean(fraud_success_rates),
            'avg_system_loss': np.mean(system_losses),
            'avg_defender_reward': np.mean(defender_rewards),
            'avg_fraudster_reward': np.mean(fraudster_rewards),
            'std_fraud_success_rate': np.std(fraud_success_rates)
        }
    
    def save_checkpoint(self, name: str):
        """Save agent models"""
        fraudster_path = os.path.join(self.save_dir, f"fraudster_{name}")
        defender_path = os.path.join(self.save_dir, f"defender_{name}")
        
        if hasattr(self.fraudster, 'save'):
            self.fraudster.save(fraudster_path)
        if hasattr(self.defender, 'save'):
            self.defender.save(defender_path)
    
    def save_metrics(self):
        """Save training metrics to JSON"""
        metrics_path = os.path.join(self.save_dir, "training_metrics.json")
        self.metrics.save(metrics_path)


def main():
    parser = argparse.ArgumentParser(description="Train Antigravity Defender MARL")
    parser.add_argument("--episodes", type=int, default=2000, help="Total training episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-mode", action="store_true", help="Quick test with 10 episodes")
    parser.add_argument("--fraudster-mode", type=str, default="learner", choices=["learner", "oracle"],
                        help="Fraudster agent mode")
    
    args = parser.parse_args()
    
    if args.test_mode:
        args.episodes = 10
        print("🧪 Test mode: Running 10 episodes only")
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Initialize environment
    env = FraudAntigravityEnv(max_steps=args.max_steps, seed=args.seed)
    
    # Initialize agents
    fraudster = FraudsterAgent(
        obs_space=env.fraudster_obs_space,
        action_space=env.fraudster_action_space,
        mode=args.fraudster_mode
    )
    
    defender = AntigravityDefender(
        obs_space=env.defender_obs_space,
        action_space=env.defender_action_space
    )
    
    # Create wrapper for training
    from utils.wrappers import SingleAgentWrapper
    
    # Train defender against fraudster
    print("🎯 Initializing Defender Model...")
    defender_env = SingleAgentWrapper(env, agent_type='defender', opponent=fraudster)
    defender.initialize_model(defender_env)
    
    print("⚔️  Initializing Fraudster Model...")
    fraudster_env = SingleAgentWrapper(env, agent_type='fraudster', opponent=defender)
    if args.fraudster_mode == 'learner':
        fraudster.initialize_model(fraudster_env)
    
    # Create trainer
    trainer = MultiAgentTrainer(env, fraudster, defender, save_dir=args.save_dir)
    
    # Train
    print("\n" + "="*60)
    print("🧲 ANTIGRAVITY DEFENDER TRAINING")
    print("="*60)
    
    # Self-play training
    print("\n📈 Phase 1: Defender Training (Fraudster Oracle)")
    defender.learn(total_timesteps=args.episodes * args.max_steps // 2)
    
    if args.fraudster_mode == 'learner':
        print("\n📈 Phase 2: Adversarial Co-Training")
        for round_idx in range(5):
            print(f"\n🔄 Round {round_idx + 1}/5")
            fraudster.learn(total_timesteps=args.episodes * args.max_steps // 10)
            defender.learn(total_timesteps=args.episodes * args.max_steps // 10)
    
    # Final evaluation
    print("\n🎯 Final Evaluation...")
    trainer.train_alternating(
        total_episodes=args.episodes,
        eval_interval=max(50, args.episodes // 10),
        verbose=True
    )
    
    print(f"\n💾 Models saved to: {args.save_dir}")
    print("✅ Training pipeline complete!")


if __name__ == "__main__":
    main()
