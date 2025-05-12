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

    def __init__(
        self,
        max_steps=100,
        forecast_horizon=1,
        max_order=50,
        demand_prob=1,
        avg_demand=10,
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.max_order = max_order
        self.demand_prob = demand_prob  # Probability of demand occurring
        self.avg_demand = avg_demand  # Average demand when it occurs
        self.max_steps = max_steps  # Maximum steps per episode
        self.step_count = 0  # Initialize step counter

        # Initialize backorder variable
        self.backorder = 0

        # Action space: Order quantity (discrete values from 0 to max_order)
        self.action_space = spaces.Discrete(max_order + 1)

        # Observation space: Forecast, current inventory, and backorder
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(forecast_horizon + 2,),  # Forecast + current inventory + backorder
            dtype=np.int32,
        )

        # Initialize state
        self.reset()

    def _generate_intermittent_demand(self):
        """Generate intermittent demand using a Bernoulli-Poisson process."""
        demand = np.zeros(self.forecast_horizon, dtype=np.int32)
        for i in range(self.forecast_horizon):
            if (
                self.np_random.random() < self.demand_prob
            ):  # Demand occurs with probability `demand_prob`
                demand[i] = self.np_random.poisson(
                    self.avg_demand
                )  # Poisson-distributed demand
        return demand

    def step(self, action):
        self.step_count += 1
        holding_cost = self.inventory * 0.5

        # Simulate real demand
        real_demand = self.forecast[0]

        # Apply the action (order quantity)
        self.inventory += action

        # Fulfill demand and track backorders
        if self.inventory >= real_demand:
            self.inventory -= real_demand
            self.backorder = 0
        else:
            self.backorder = real_demand - self.inventory
            self.inventory = 0
        # Calculate reward
        revenue = min(real_demand, self.inventory + action) * 1
        backorder_cost = self.backorder * 1
        reward = revenue - holding_cost - backorder_cost

        # Store action and reward for rendering
        self.last_action = action
        self.last_reward = reward
        self.last_real_demand = real_demand
        self.last_revenue = revenue
        self.last_holding_cost = holding_cost
        self.last_backorder_cost = backorder_cost

        # Update state
        self.forecast = np.roll(self.forecast, -1)  # Shift forecast
        self.forecast[-1] = self._generate_intermittent_demand()[0]  # New forecast
        self.state = np.append(self.forecast, [self.inventory, self.backorder]).astype(
            np.int32
        )

        # Check if the episode is terminated or truncated
        terminated = self.step_count >= self.max_steps
        truncated = False  # Example: No truncation logic implemented

        return self.state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        self.step_count = 0
        # Set the seed for reproducibility
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        # Reset forecast, inventory, and backorder
        self.forecast = self._generate_intermittent_demand()
        self.inventory = 0  # Start w/ zero inventory
        self.backorder = 0  # No backorders at the start
        self.last_action = None
        self.last_reward = None
        self.last_real_demand = None
        self.last_revenue = None
        self.last_holding_cost = None
        self.last_backorder_cost = None
        self.state = np.append(self.forecast, [self.inventory, self.backorder]).astype(
            np.int32
        )

        return self.state, {}

    def render(self, mode="human"):
        """Render the environment's state as a plot."""
        print(
            f"Holding Cost: {getattr(self, 'last_holding_cost', None)}, "
            f"Action: {getattr(self, 'last_action', None)}, "
            f"Real Demand: {getattr(self, 'last_real_demand', None)}, "
            f"Inventory: {self.inventory}, "
            f"Backorder: {self.backorder}, "
            f"Revenue: {getattr(self, 'last_revenue', None)}, "
            f"Backorder Cost: {getattr(self, 'last_backorder_cost', None)}, "
            f"Reward: {getattr(self, 'last_reward', None)}, "
            f"Forecast: {self.forecast}"
        )

    def close(self):
        pass
