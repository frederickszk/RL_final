# import gym
# import numpy as np
#
# from ale_py import ALEInterface
#
# ale = ALEInterface()
# env = gym.make('ALE/Breakout-v5')
# space = env.action_space
# valid_actions = np.arange(space.n).tolist()
#
#
# obs_current = env.reset()
# env.render()


from gym_wrappers import MainGymWrapper
import gym

env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
env = MainGymWrapper.wrap(env)

obs_current = env.reset()
action = 1
# action = env.action_space.sample()
next_state, reward, terminal, info = env.step(action)

# for e in range(max_episode):
#     obs_current = env.reset()
#     env.render()