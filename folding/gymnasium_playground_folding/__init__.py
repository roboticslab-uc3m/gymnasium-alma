from gymnasium.envs.registration import register

register(
    id="gymnasium_playground/Folding-v0",
    entry_point="gymnasium_playground_folding.envs:FoldingEnv",
)
