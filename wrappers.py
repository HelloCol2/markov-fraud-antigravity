"""
Environment wrappers for single-agent training in multi-agent game
"""
import numpy as np
import gym
from gym import spaces


class SingleAgentWrapper(gym.Env):
    """
    Wrapper to convert multi-agent environment to single-agent for stable-baselines3
    Fixes one agent as opponent and exposes the other agent's interface
    """
    
    def __init__(self, env, agent_type='defender', opponent=None):
        """
        Args:
            env: FraudAntigravityEnv instance
            agent_type: 'defender' or 'fraudster' (which agent to train)
            opponent: Agent instance for the fixed opponent
        """
        super(SingleAgentWrapper, self).__init__()
        
        self.env = env
        self.agent_type = agent_type
        self.opponent = opponent
        
        # Set observation and action spaces based on agent type
        if agent_type == 'defender':
            self.observation_space = env.defender_obs_space
            self.action_space = env.defender_action_space
        else:  # fraudster
            self.observation_space = env.fraudster_obs_space
            self.action_space = env.fraudster_action_space
    
    def reset(self):
        """Reset environment and return observation for training agent"""
        fraudster_obs, defender_obs = self.env.reset()
        
        if self.agent_type == 'defender':
            return defender_obs
        else:
            return fraudster_obs
    
    def step(self, action):
        """
        Execute step with action from training agent and action from opponent
        
        Args:
            action: Action from the agent being trained
        
        Returns:
            observation, reward, done, info (standard gym interface)
        """
        # Get opponent action
        if self.agent_type == 'defender':
            # We're training defender, get fraudster action from opponent
            fraudster_obs, _ = self._get_current_obs()
            if self.opponent is not None:
                fraudster_action = self.opponent.predict(fraudster_obs, deterministic=False)
            else:
                fraudster_action = self.env.fraudster_action_space.sample()
            
            defender_action = action
        else:
            # We're training fraudster, get defender action from opponent
            _, defender_obs = self._get_current_obs()
            if self.opponent is not None:
                defender_action = self.opponent.predict(defender_obs, deterministic=False)
            else:
                defender_action = self.env.defender_action_space.sample()
            
            fraudster_action = action
        
        # Execute environment step
        (fraudster_obs, defender_obs), (fraudster_reward, defender_reward), done, info = \
            self.env.step(fraudster_action, defender_action)
        
        # Return observation and reward for training agent
        if self.agent_type == 'defender':
            return defender_obs, defender_reward, done, info
        else:
            return fraudster_obs, fraudster_reward, done, info
    
    def _get_current_obs(self):
        """Get current observations for both agents"""
        # This is a helper to get observations without stepping
        # We'll need to call the environment's observation methods
        fraudster_obs = self.env._get_fraudster_obs()
        defender_obs = self.env._get_defender_obs()
        return fraudster_obs, defender_obs
    
    def render(self, mode='human'):
        """Render the environment"""
        return self.env.render(mode=mode)
