from gymnasium.envs.registration import register

register(
    id="gymnasium_alma/TiagoFolding-v0",
    entry_point="gymnasium_alma_tiago_folding.envs:TiagoFoldingEnv",
)
