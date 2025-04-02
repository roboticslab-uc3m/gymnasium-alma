from gymnasium.envs.registration import register

register(
    id="gymnasium_playground/TiagoIroning-v0",
    entry_point="gymnasium_playground_tiago_ironing.envs:TiagoIroningEnv",
)
