import numpy as np
import os
import random
import shutil
from statistics import mean
from game_models.base_game_model import BaseGameModel
from game_models.backbones import DQN
import torch
import torch.nn as nn
import torch.optim as optim

from collections import namedtuple, deque
Transition = namedtuple('Transition',
                        ('current_state', 'action', 'next_state', 'reward'))

GAMMA = 0.99
# MEMORY_SIZE = 900000
MEMORY_SIZE = 250000  # For server
# MEMORY_SIZE = 30000  # For local

BATCH_SIZE = 32
TRAINING_FREQUENCY = 4


# TARGET_NETWORK_UPDATE_FREQUENCY = 5000
TARGET_NETWORK_UPDATE_FREQUENCY = 10000
MODEL_PERSISTENCE_UPDATE_FREQUENCY = 5000

REPLAY_START_SIZE = 50000  # For server
# REPLAY_START_SIZE = 10000  # For Local

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_TEST = 0.02
# EXPLORATION_STEPS = 850000
EXPLORATION_STEPS = 450000
# EXPLORATION_STEPS = 350000  # For server
# EXPLORATION_STEPS = 50000  # For local


EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS


class DQNGameModel(BaseGameModel):
    def __init__(self, game_name, mode_name, input_shape, action_space, logger_path, model_path):
        BaseGameModel.__init__(self, game_name,
                               mode_name,
                               logger_path,
                               input_shape,
                               action_space)
        self.model_path = model_path
        self.policy_net = DQN(self.input_shape, action_space)
        if os.path.isfile(self.model_path):
            print("Loaded the policy_net with ", self.model_path)
            self.policy_net.load_state_dict(torch.load(self.model_path))

    def _save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_path)


class DDQNSolver(DQNGameModel):

    def __init__(self, game_name, input_shape, action_space, device):
        testing_model_path = "./output/weights/" + game_name + "/ddqn/testing/model.pth"
        assert os.path.exists(os.path.dirname(testing_model_path)), "No testing model in: " + str(testing_model_path)
        DQNGameModel.__init__(self,
                              game_name,
                              "DDQN testing",
                              input_shape,
                              action_space,
                              "./output/logs/" + game_name + "/ddqn/testing/" + self._get_date() + "/",
                              testing_model_path)

        self.device = device
        self.policy_net.to(device)

    def move(self, state):
        # e-greedy
        if np.random.rand() < EXPLORATION_TEST:
            return random.randrange(self.action_space)

        # predict by the agent
        input_state = np.expand_dims(np.asarray(state).astype(np.float32), axis=0)
        input_state = torch.tensor(input_state).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(input_state)
        return np.argmax(q_values.cpu().numpy()[0])


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DDQNTrainer(DQNGameModel):

    def __init__(self, game_name, input_shape, action_space, device):
        DQNGameModel.__init__(self,
                              game_name,
                              "DDQN training",
                              input_shape,
                              action_space,
                              "./output/logs/" + game_name + "/ddqn/training/" + self._get_date() + "/",
                              "./output/weights/" + game_name + "/ddqn/" + self._get_date() + "/model.pth")
        self.device = device
        if os.path.exists(os.path.dirname(self.model_path)):
            # If exists, remove the directory.
            shutil.rmtree(os.path.dirname(self.model_path), ignore_errors=True)
        os.makedirs(os.path.dirname(self.model_path))

        self.policy_net.to(device)
        self.target_net = DQN(self.input_shape, action_space)
        self.target_net.eval()
        self.target_net.to(device)

        self._reset_target_network()
        self.epsilon = EXPLORATION_MAX
        self.memory = ReplayMemory(MEMORY_SIZE)

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=0.00025, eps=0.01, weight_decay=0.95)
        # self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=0.00025)

    def show_info(self):
        print("Memory size: ", MEMORY_SIZE)
        print("Replay start: ", REPLAY_START_SIZE)
        print("Exploration steps: ", EXPLORATION_STEPS)
        print("Policy network save frequency: ", MODEL_PERSISTENCE_UPDATE_FREQUENCY)
        print("Target network update frequency: ", TARGET_NETWORK_UPDATE_FREQUENCY)
        print("Sample batch: ", BATCH_SIZE)

    def move(self, state):
        # e-greedy
        if np.random.rand() < self.epsilon or len(self.memory) < REPLAY_START_SIZE:
            return random.randrange(self.action_space)

        # predict by the agent
        input_state = np.expand_dims(np.asarray(state).astype(np.float32), axis=0)
        input_state = torch.tensor(input_state).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(input_state)
        return np.argmax(q_values.cpu().numpy()[0])

    def remember(self, current_state, action, next_state, reward):
        current_state = np.expand_dims(np.asarray(current_state).astype(np.float32), axis=0)
        current_state = torch.tensor(current_state)
        next_state = np.expand_dims(np.asarray(next_state).astype(np.float32), axis=0)
        next_state = torch.tensor(next_state)
        action = torch.tensor([[action]], dtype=torch.long)
        reward = torch.tensor([reward])

        self.memory.push(current_state, action, next_state, reward)

    def step_update(self, total_step):
        if self.memory.__len__() < REPLAY_START_SIZE:
            # print("Not enough buffer, pass. Current memory length: ", self.memory.__len__())
            return
        if total_step % TRAINING_FREQUENCY == 0:
            # print("Total_step:", total_step, "Train here")
            loss, average_max_q = self._train()
            self.logger.add_loss(loss)
            # self.logger.add_accuracy(accuracy)
            self.logger.add_q(average_max_q)

        self._update_epsilon()

        if total_step % MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0:
            self._save_model()

        if total_step % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
            self._reset_target_network()
            print('{{"metric": "epsilon", "value": {}}}'.format(self.epsilon))
            print('{{"metric": "total_step", "value": {}}}'.format(total_step))


    def _train(self):
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).to(self.device)

        current_state_batch = torch.cat(batch.current_state).to(self.device)  # {Tensor: (batch, frame, width, height)}
        action_batch = torch.cat(batch.action).to(self.device)  # {Tensor: (batch, 1)}
        reward_batch = torch.cat(batch.reward).to(self.device)  # {Tensor: (batch, )}

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(current_state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)

        # For DDQN
        # Firstly use policy_net to select the max action: a' = argmax_a Q(s_{t+1}, a)
        # Then use the target_net to evaluate the state_action value: Q'(s_{t+1}, a')
        action_policy_batch = self.policy_net(non_final_next_states).max(1)[1].view(-1, 1)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, action_policy_batch).view(-1,)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        average_max_q = torch.mean(expected_state_action_values)

        # Compute Huber loss
        # criterion = nn.SmoothL1Loss()
        # criterion = nn.HuberLoss()
        criterion = nn.MSELoss()
        # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = criterion(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item(), average_max_q.item()

    def _update_epsilon(self):
        self.epsilon -= EXPLORATION_DECAY
        self.epsilon = max(EXPLORATION_MIN, self.epsilon)

    def _reset_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
