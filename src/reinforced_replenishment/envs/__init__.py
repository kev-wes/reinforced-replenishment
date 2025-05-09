from gymnasium.envs.registration import register, registry

if "CustomEnv-v0" not in registry:
    register(
        id="CustomEnv-v0",  # Unique identifier for your environment
        entry_point="reinforced_replenishment.envs.custom_env:CustomEnv",
    )
if "ReplenishmentEnv-v0" not in registry:
    register(
        id="ReplenishmentEnv-v0",  # Unique identifier for the environment
        entry_point="reinforced_replenishment.envs.replenishment_env:ReplenishmentEnv",
    )
