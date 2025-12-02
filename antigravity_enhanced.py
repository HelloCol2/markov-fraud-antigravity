"""
Antigravity Defender Agent - Enhanced Training Configuration
Embodies adaptive counter-force principles against strategic fraud
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd


class AntigravityDefenderEnhanced:
    """
    Enhanced Antigravity Defender with strategic counter-adaptation principles
    
    Core Principles:
    1. Fraud is a strategic, adapting opponent (not random anomaly)
    2. Dynamic counter-force: adjust strictness based on adversarial profitability
    3. High strictness when fraud becomes profitable + risk increases
    4. Low strictness when fraud drops or costs outweigh damage
    5. Long-term optimization over short-term detection
    6. Stable equilibrium: Security ↔ Cost ↔ Trust
    """
    
    def __init__(self, obs_space, action_space, learning_rate=3e-4):
        self.obs_space = obs_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        
        # Enhanced policy kwargs for strategic learning
        self.policy_kwargs = dict(
            net_arch=[256, 256, 128],  # Deeper network for complex patterns
            activation_fn=torch.nn.ReLU,
        )
        
        self.model = None
    
    def initialize_model(self, env):
        """Initialize PPO with antigravity-optimized hyperparameters"""
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate,
            n_steps=2048,
            batch_size=128,  # Larger batch for stability
            n_epochs=15,     # More epochs for deeper learning
            gamma=0.995,     # High discount for long-term optimization
            gae_lambda=0.98,
            clip_range=0.2,
            ent_coef=0.015,  # Higher entropy for exploration
            verbose=1,
            policy_kwargs=self.policy_kwargs,
            tensorboard_log="./tensorboard_logs/"
        )
    
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        Predict defense action with antigravity principles
        
        Decision logic:
        - High strictness (2): When fraud_rate > 0.5 AND (high risk OR high payoff trend)
        - Normal strictness (1): Balanced state, moderate fraud activity
        - Lenient (0): Low fraud AND (high FP cost OR low risk contexts)
        """
        if self.model is not None:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            return int(action)
        else:
            # Fallback heuristic embodying antigravity principles
            return self._antigravity_heuristic(obs)
    
    def _antigravity_heuristic(self, obs: np.ndarray) -> int:
        """
        Heuristic that embodies antigravity counter-force principles
        Applied when model is not yet trained
        """
        # Extract key features (defender obs vector)
        customer_risk = obs[0]
        amount_norm = obs[1]
        fraud_rate_recent = obs[3]
        fp_rate_recent = obs[4]
        defense_budget = obs[5]
        fraudster_aggressiveness = obs[7]
        fraudster_payoff_trend = obs[8]
        sys_loss_cum_norm = obs[9]
        
        # ANTIGRAVITY PRINCIPLE 1: Collapse fraudster profitability
        # If fraudster payoff is trending up, apply counter-force
        if fraudster_payoff_trend > 0.3 and fraud_rate_recent > 0.4:
            return 2  # Strict - disrupt exploitation
        
        # ANTIGRAVITY PRINCIPLE 2: Adapt to strategic behavior
        # If fraud is high AND risk contexts increase
        if fraud_rate_recent > 0.6 or (fraudster_aggressiveness > 0.7 and customer_risk > 0.6):
            return 2  # Strict - counter-adapt
        
        # PRINCIPLE 3: Efficiency over paranoia
        # If fraud is low AND FP cost is high, relax
        if fraud_rate_recent < 0.2 and fp_rate_recent > 0.3:
            return 0  # Lenient - preserve trust & efficiency
        
        # PRINCIPLE 4: Budget-aware sustainable defense
        # If defense budget is low, optimize for efficiency
        if defense_budget < 0.3:
            return 0 if fraud_rate_recent < 0.4 else 1
        
        # PRINCIPLE 5: Long-term equilibrium
        # Default to normal strictness for stable operation
        threat_score = (customer_risk + amount_norm + fraud_rate_recent * 2) / 4
        
        if threat_score > 0.65:
            return 2  # Strict
        elif threat_score > 0.35:
            return 1  # Normal
        else:
            return 0  # Lenient
    
    def learn(self, total_timesteps: int):
        """Train with progress monitoring"""
        if self.model is not None:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=AntigravityCallback()
            )
    
    def save(self, path: str):
        if self.model is not None:
            self.model.save(path)
    
    def load(self, path: str, env=None):
        self.model = PPO.load(path, env=env)


class AntigravityCallback(BaseCallback):
    """
    Custom callback to monitor antigravity effectiveness
    Tracks: fraud_success_rate, system_loss, fraudster_payoff_trend
    """
    
    def __init__(self, verbose=0):
        super(AntigravityCallback, self).__init__(verbose)
        self.fraud_successes = []
        self.system_losses = []
        
    def _on_step(self) -> bool:
        # Extract info from environment
        if len(self.model.ep_info_buffer) > 0:
            info = self.locals.get('infos', [{}])[0]
            
            if 'fraud_success_rate' in info:
                self.fraud_successes.append(info['fraud_success_rate'])
            if 'cumulative_system_loss' in info:
                self.system_losses.append(info['cumulative_system_loss'])
        
        return True
    
    def _on_training_end(self) -> None:
        if len(self.fraud_successes) > 0:
            avg_fraud_success = np.mean(self.fraud_successes[-100:])
            avg_system_loss = np.mean(self.system_losses[-100:])
            
            print(f"\n🧲 Antigravity Performance:")
            print(f"   Fraud Success Rate: {avg_fraud_success:.2%}")
            print(f"   Avg System Loss: {avg_system_loss:.3f}")


def load_enhanced_dataset(csv_path='fraud_antigravity_synth-2.csv'):
    """
    Load the enhanced strategic behavior dataset
    
    Expected columns:
    - risk_score, amount_norm, time_bucket
    - attack, detected, fraud_budget, defense_budget
    - strictness, investigation_cost, fp_cost
    - system_loss_cum, fraudster_payoff_trend
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded enhanced dataset: {len(df)} samples")
        print(f"   Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"⚠️  Enhanced dataset not found at {csv_path}")
        print("   Falling back to original synthetic data")
        return pd.read_csv('env/fraud_antigravity_synth.csv')


# ============================================
# ANTIGRAVITY TRAINING PRINCIPLES (SUMMARY)
# ============================================
"""
The Antigravity Defender embodies these strategic principles:

1. **Strategic Opponent Recognition**
   - Fraud is NOT random noise
   - Fraudster learns and adapts
   - Defense must counter-evolve

2. **Dynamic Counter-Force Application**
   - ↑ Strictness when fraud becomes profitable
   - ↓ Strictness when costs outweigh damage
   - Disrupts adversarial exploitation patterns

3. **Long-Term Optimization**
   - Minimize: fraud_success_rate ↓, system_loss ↓, fraudster_reward ↓
   - Balance: investigation efficiency, false positives
   - Sustain: operational cost, user trust

4. **Adaptive Equilibrium**
   - Not always-strict (destroys efficiency)
   - Not always-lenient (allows exploitation)
   - Mixed strategy Nash equilibrium

5. **Payoff Collapse via Antigravity**
   - Learn to suppress fraudster profitability over time
   - Make fraud unprofitable through adaptive counter-pressure
   - Stabilize system even as adversary adapts

Training Objective:
    R_defender = -(system_loss + investigation_cost + fp_cost)
    
    Maximize long-term cumulative reward by collapsing fraudster payoff
    while maintaining operational efficiency and user trust.
"""
