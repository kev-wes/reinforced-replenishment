from gym.envs.registration import register

register(
    id="CustomEnv-v0",  # Unique identifier for your environment
    entry_point="reinforced_replenishment.envs.custom_env:CustomEnv",
)
