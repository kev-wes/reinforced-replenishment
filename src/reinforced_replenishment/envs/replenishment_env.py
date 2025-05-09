import logging
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class ReplenishmentEnv(gym.Env):
    """Custom Environment for Replenishment"""

    metadata = {
        "render_modes": ["human"],  # Updated key to 'render_modes'
        "render_fps": 30,  # Define a default FPS for rendering
    }

    def __init__(self, forecast_horizon=10, max_inventory=100, max_order=50):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.max_inventory = max_inventory
        self.max_order = max_order

        # Action space: Order quantity (discrete values from 0 to max_order)
        self.action_space = spaces.Discrete(max_order + 1)

        # Observation space: Forecast and current inventory
        self.observation_space = spaces.Box(
            low=0,
            high=max_inventory,
            shape=(forecast_horizon + 1,),  # Forecast + current inventory
            dtype=np.float32,
        )

        # Initialize state
        self.reset()

    def step(self, action):
        # Apply the action (order quantity)
        self.inventory += action

        # Simulate real demand
        real_demand = self.forecast[0]
        self.inventory -= real_demand

        # Calculate reward
        revenue = min(real_demand, self.inventory) * 10  # Example: $10 per unit sold
        holding_cost = max(self.inventory, 0) * 1  # Example: $1 per unit held
        stockout_cost = max(-self.inventory, 0) * 5  # Example: $5 per unit short
        reward = revenue - holding_cost - stockout_cost

        # Update state
        self.forecast = np.roll(self.forecast, -1)  # Shift forecast
        self.forecast[-1] = self.np_random.integers(
            0, self.max_inventory
        )  # New forecast
        self.state = np.append(self.forecast, self.inventory).astype(
            np.float32
        )  # Cast to float32

        # Check if the episode is terminated or truncated
        terminated = len(self.forecast) == 0  # Example termination condition
        truncated = False  # Example: No truncation logic implemented

        return self.state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        # Set the seed for reproducibility
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        # Reset forecast and inventory
        self.forecast = self.np_random.integers(
            0, self.max_inventory, size=self.forecast_horizon
        )
        self.inventory = self.max_inventory // 2  # Start with half capacity
        self.state = np.append(self.forecast, self.inventory).astype(
            np.float32
        )  # Cast to float32

        return self.state, {}

    def render(self, mode="human"):
        """Render the environment's state as a plot."""
        logger.info(f"Inventory: {self.inventory}, Forecast: {self.forecast}")

    def close(self):
        pass
