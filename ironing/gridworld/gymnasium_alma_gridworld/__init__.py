from gymnasium.envs.registration import register

register(
    id="gymnasium_alma/GridWorld-v0",
    entry_point="gymnasium_alma_gridworld.envs:GridWorldEnv",
)
