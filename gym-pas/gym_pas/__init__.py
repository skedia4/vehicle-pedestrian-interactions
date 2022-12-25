from gym.envs.registration import register

register(
    id='pas-v0',
    entry_point='gym_pas.envs:PaSEnv',
)