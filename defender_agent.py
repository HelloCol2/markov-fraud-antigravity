"""
Defender (Antigravity) Agent Implementation
Adaptive defense policy that learns to minimize fraud success and system loss
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Optional


class AntigravityDefender:
    """
    PPO-based adaptive defender that applies "counter-force" against fraud
    Learns to balance detection accuracy with operational costs
    """
    
    def __init__(self, obs_space, action_space, learning_rate=3e-4):
        """
        Args:
            obs_space: Observation space from environment
            action_space: Action space (Discrete(3))
            learning_rate: PPO learning rate
        """
        self.obs_space = obs_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.model = None
    
    def initialize_model(self, env):
        """Initialize PPO model with environment"""
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
            ent_coef=0.01,
            verbose=0,
            policy_kwargs=dict(net_arch=[128, 128])  # Larger network for defender
        )
    
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        Select defense action based on observation
        
        Args:
            obs: Defender observation vector [12]
            deterministic: If True, select best action; if False, sample
        
        Returns:
            action: {0: lenient, 1: normal, 2: strict}
        """
        if self.model is not None:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            return int(action)
        else:
            # Default to normal strictness
            return 1
    
    def learn(self, total_timesteps: int):
        """Train the PPO model"""
        if self.model is not None:
            self.model.learn(total_timesteps=total_timesteps)
    
    def save(self, path: str):
        """Save model to disk"""
        if self.model is not None:
            self.model.save(path)
    
    def load(self, path: str, env):
        """Load model from disk"""
        self.model = PPO.load(path, env=env)


class RandomDefender:
    """Baseline: Random defense strategy"""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> int:
        return self.action_space.sample()


class StaticThresholdDefender:
    """
    Baseline: Rule-based defense using static thresholds
    Adjusts strictness based on risk score and amount
    """
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        Static threshold logic:
        - High risk + high amount -> strict
        - Medium risk/amount -> normal
        - Low risk/amount -> lenient
        """
        customer_risk = obs[0]
        amount_norm = obs[1]
        fraud_rate_recent = obs[3]
        
        # Combine signals
        threat_score = customer_risk + 0.5 * amount_norm + 0.3 * fraud_rate_recent
        
        if threat_score > 1.0:
            return 2  # strict
        elif threat_score > 0.5:
            return 1  # normal
        else:
            return 0  # lenient


class AlwaysStrictDefender:
    """Baseline: Always apply strict detection (high cost)"""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> int:
        return 2  # always strict


class AdaptiveThresholdDefender:
    """
    Baseline: Adaptive threshold that responds to recent fraud rate
    More sophisticated rule-based approach
    """
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        Adaptive logic: Increase strictness when fraud rate is high
        """
        customer_risk = obs[0]
        amount_norm = obs[1]
        fraud_rate_recent = obs[3]
        fp_rate_recent = obs[4]
        
        # Dynamic threshold based on recent fraud activity
        if fraud_rate_recent > 0.6:
            # High fraud -> be strict
            return 2 if customer_risk > 0.3 else 1
        elif fraud_rate_recent > 0.3:
            # Medium fraud -> normal/strict based on risk
            return 1 if customer_risk < 0.6 else 2
        else:
            # Low fraud -> be lenient unless clear signals
            if customer_risk > 0.7 and amount_norm > 0.7:
                return 2
            elif customer_risk > 0.5 or amount_norm > 0.5:
                return 1
            else:
                return 0
