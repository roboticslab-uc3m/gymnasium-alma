from gymnasium.envs.registration import register

register(
    id="gymnasium_alma/TeoIroning-v0",
    entry_point="gymnasium_alma_teo_ironing.envs:TeoIroningEnv",
)
