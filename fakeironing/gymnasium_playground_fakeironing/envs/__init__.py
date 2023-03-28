from gymnasium.envs.registration import register

register(
    id="gymnasium_playground/FakeIroning-v0",
    entry_point="gymnasium_playground_fakeironing.envs:FakeIroningEnv",
)
