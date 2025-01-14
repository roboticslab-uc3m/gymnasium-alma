from gymnasium.envs.registration import register

register(
    id="gymnasium_playground/FakeFolding-v0",
    entry_point="gymnasium_playground_fakefolding.envs:FakeFoldingEnv",
)
