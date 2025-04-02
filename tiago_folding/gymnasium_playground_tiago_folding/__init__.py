from gymnasium.envs.registration import register

register(
    id="gymnasium_playground/TiagoFolding-v0",
    entry_point="gymnasium_playground_tiago_folding.envs:TiagoFoldingEnv",
)
