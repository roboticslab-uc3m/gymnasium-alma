from gymnasium.envs.registration import register

register(
    id="gymnasium_alma/FakeIroning-v0",
    entry_point="gymnasium_alma_fakeironing.envs:FakeIroningEnv",
)
