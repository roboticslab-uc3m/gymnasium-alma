from gymnasium.envs.registration import register

register(
    id="gymnasium_alma/Bandit-v0",
    entry_point="gymnasium_alma_bandit.envs:BanditEnv",
)
