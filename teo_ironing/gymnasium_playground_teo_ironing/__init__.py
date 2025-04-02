from gymnasium.envs.registration import register

register(
    id="gymnasium_playground/TeoIroning-v0",
    entry_point="gymnasium_playground_teo_ironing.envs:TeoIroningEnv",
)
