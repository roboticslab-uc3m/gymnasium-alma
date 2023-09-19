from gymnasium.envs.registration import register

register(
    id="gymnasium_playground/Ironing-v0",
    entry_point="gymnasium_playground_ironing.envs:IroningEnv",
)
