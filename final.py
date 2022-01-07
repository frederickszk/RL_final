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
# import sys
# sys.path.append("./game_models")

from gym_wrappers import MainGymWrapper
import gym
from game_models.backbones import DQN
import torch
import numpy as np
from game_models.dqn_game_model import DQNTrainer
from game_models.ddqn_game_model import DDQNTrainer



FRAMES_IN_OBSERVATION = 4
FRAME_SIZE = 84
INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)

# -------------------------------------------------------------------------------- #
"""
Preamble
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# -------------------------------------------------------------------------------- #
"""
For testing the environment and "State" structure
"""
game_name = 'BreakoutNoFrameskip-v4'
# env = gym.make(game_name, render_mode='human')
env = gym.make(game_name)
env = MainGymWrapper.wrap(env)

current_state = env.reset()
action = 1
# action = env.action_space.sample()
next_state, reward, terminal, info = env.step(action)

# --------------------------------------------------------------------------------- #

"""
For testing the DQN module
"""
#
policy_net = DQN((4, 84, 84), 4).to(device)
test_input = torch.tensor(np.concatenate(next_state._frames))
test_input = torch.unsqueeze(test_input, dim=0).float().to(device)

# test_output = policy_net(test_input)
with torch.no_grad():
    test_output = policy_net(test_input)


# ---------------------------------------------------------------------------------- #
"""
For testing the dqn_game_model
"""
from collections import namedtuple
Transition = namedtuple('Transition',
                        ('current_state', 'action', 'next_state', 'reward'))

# game_model = DQNTrainer(game_name, INPUT_SHAPE, env.action_space.n, device)
game_model = DDQNTrainer(game_name, INPUT_SHAPE, env.action_space.n, device)
for i in range(32):
    game_model.remember(current_state, action, next_state, reward)

loss, max_q = game_model._train()









