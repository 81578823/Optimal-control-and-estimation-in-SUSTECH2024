from gymnasium.envs.registration import register

register(
    id='InvertedPendulumCustom-v0',
    entry_point='gym_custom_envs:InvertedPendulumEnv',
    max_episode_steps=1000,
)
from gym_custom_envs.inverted_pendulum import InvertedPendulumEnv