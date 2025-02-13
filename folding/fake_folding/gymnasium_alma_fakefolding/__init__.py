from gymnasium.envs.registration import register

register(
    id="gymnasium_alma/FakeFolding-v0",
    entry_point="gymnasium_alma_fakefolding.envs:FakeFoldingEnv",
)
