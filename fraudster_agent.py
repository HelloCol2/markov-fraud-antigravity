"""
Fraudster Agent Implementation
Strategic fraud agent that learns to maximize payoff while avoiding detection
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Optional


class FraudsterAgent:
    """
    PPO-based fraudster agent that learns optimal attack strategies
    Can be configured as adaptive learner or fixed oracle
    """
    
    def __init__(self, obs_space, action_space, learning_rate=3e-4, mode='learner'):
        """
        Args:
            obs_space: Observation space from environment
            action_space: Action space (Discrete(3))
            learning_rate: PPO learning rate
            mode: 'learner' (adaptive) or 'oracle' (heuristic-based)
        """
        self.obs_space = obs_space
        self.action_space = action_space
        self.mode = mode
        
        if mode == 'learner':
            # Create PPO model
            self.model = None  # Will be initialized in train loop
            self.learning_rate = learning_rate
        
    def initialize_model(self, env):
        """Initialize PPO model with environment"""
        if self.mode == 'learner':
            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=self.learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,  # Encourage exploration
                verbose=0
            )
    
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action based on observation
        
        Args:
            obs: Fraudster observation vector [10]
            deterministic: If True, select best action; if False, sample
        
        Returns:
            action: {0: no_attack, 1: low_fraud, 2: high_fraud}
        """
        if self.mode == 'learner' and self.model is not None:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            return int(action)
        else:
            # Oracle heuristic strategy
            return self._heuristic_action(obs)
    
    def _heuristic_action(self, obs: np.ndarray) -> int:
        """
        Heuristic-based fraud strategy (rule-based oracle)
        
        Strategy: Attack when risk is low and budget is available
        """
        customer_risk = obs[0]
        amount_norm = obs[1]
        prev_detected = obs[4]
        fraud_budget = obs[5]
        defender_entropy = obs[9]
        
        # Don't attack if budget is low
        if fraud_budget < 0.2:
            return 0  # no attack
        
        # Recently detected? Be cautious
        if prev_detected > 0.5:
            return np.random.choice([0, 1], p=[0.8, 0.2])
        
        # Calculate attack probability based on features
        attack_signal = (1 - customer_risk) + 0.5 * amount_norm + 0.3 * defender_entropy
        
        if attack_signal > 1.2:
            return 2  # high fraud
        elif attack_signal > 0.7:
            return 1  # low fraud
        else:
            return 0  # no attack
    
    def learn(self, total_timesteps: int):
        """Train the PPO model"""
        if self.mode == 'learner' and self.model is not None:
            self.model.learn(total_timesteps=total_timesteps)
    
    def save(self, path: str):
        """Save model to disk"""
        if self.model is not None:
            self.model.save(path)
    
    def load(self, path: str, env):
        """Load model from disk"""
        if self.mode == 'learner':
            self.model = PPO.load(path, env=env)


class RandomFraudster:
    """Baseline: Random attack strategy"""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> int:
        return self.action_space.sample()


class AggressiveFraudster:
    """Baseline: Always attack with high intensity"""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> int:
        fraud_budget = obs[5]
        if fraud_budget > 0.1:
            return 2  # high fraud
        else:
            return 0  # no attack
