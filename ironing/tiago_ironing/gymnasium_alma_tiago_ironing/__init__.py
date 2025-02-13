from gymnasium.envs.registration import register

register(
    id="gymnasium_alma/TiagoIroning-v0",
    entry_point="gymnasium_alma_tiago_ironing.envs:TiagoIroningEnv",
)
