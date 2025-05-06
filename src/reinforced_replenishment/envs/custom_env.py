import gym
import numpy as np
from gym import spaces


class CustomEnv(gym.Env):
    """Custom Environment that follows OpenAI Gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # Example: Discrete action space with 2 actions
        self.action_space = spaces.Discrete(2)
        # Example: Continuous observation space with 3 dimensions
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

    def step(self, action):
        # Execute one time step within the environment
        observation = np.random.random(self.observation_space.shape)
        reward = 1.0  # Example reward
        done = False  # Example termination condition
        info = {}  # Additional info
        return observation, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        return np.random.random(self.observation_space.shape)

    def render(self, mode="human"):
        # Render the environment to the screen
        pass

    def close(self):
        # Clean up resources
        pass
