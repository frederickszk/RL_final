import gym
import argparse
import numpy as np
# import atari_py
from game_models.dqn_game_model import DQNTrainer, DQNSolver
from game_models.ddqn_game_model import DDQNTrainer, DDQNSolver
from gym_wrappers import MainGymWrapper
import torch

FRAMES_IN_OBSERVATION = 4
FRAME_SIZE = 84
INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)


class Atari:

    def __init__(self):
        game_name, game_mode, render, total_step_limit, total_run_limit, clip = self._args()
        print(render)
        if render:
            env = gym.make(game_name, render_mode='human')
        else:
            env = gym.make(game_name)
        env = MainGymWrapper.wrap(env)

        # Setting the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[Main Loop] Using device", device)

        game_model = self._game_model(game_mode, game_name, env.action_space.n, device)
        game_model.show_info()

        self._main_loop(game_model, env, total_step_limit, total_run_limit, clip)

    def _main_loop(self, game_model, env, total_step_limit, total_run_limit, clip):
        run = 0
        total_step = 0
        while True:
            if total_run_limit is not None and run >= total_run_limit:
                print("Reached total run limit of: " + str(total_run_limit))
                exit(0)

            run += 1
            current_state = env.reset()
            step = 0
            score = 0
            while True:
                if total_step >= total_step_limit:
                    print("Reached total step limit of: " + str(total_step_limit))
                    exit(0)
                total_step += 1
                step += 1

                # No need to explicitly .render()
                # if render:
                #     env.render()

                action = game_model.move(current_state)
                next_state, reward, terminal, info = env.step(action)
                if clip:
                    reward = np.sign(reward)
                score += reward
                game_model.remember(current_state, action, next_state, reward)
                current_state = next_state

                game_model.step_update(total_step)

                # print("Run:{}, Step:{}, Score:{}, Terminal:{}".format(run, step, score, terminal))
                # print(total_step)
                if terminal:
                    game_model.save_run(score, step, run)
                    break

    def _args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-g", "--game", help="Default is 'breakout'.", default="BreakoutNoFrameskip-v4")
        parser.add_argument("-m", "--mode",
                            help="Choose from available modes: dqn_training/testing, ddqn_training/testing. "
                                 "Default is 'ddqn_training'.",
                            default="ddqn_training")
        parser.add_argument("-r", "--render", help="Choose if the game should be rendered. Default is 'False'.",
                            action='store_true')
        parser.add_argument("-tsl", "--total_step_limit",
                            help="Set how many steps in one episode . Default is '5000000'.", default=5000000, type=int)
        parser.add_argument("-trl", "--total_run_limit",
                            help="Set how many episodes we should sample. Default is None (no limit).",
                            default=None, type=int)
        parser.add_argument("-c", "--clip",
                            help="Choose whether we should clip rewards to (0, 1) range. Default is 'True'",
                            default=True, type=bool)
        args = parser.parse_args()
        game_mode = args.mode
        game_name = args.game
        render = args.render
        total_step_limit = args.total_step_limit
        total_run_limit = args.total_run_limit
        clip = args.clip
        print("Selected game: " + str(game_name))
        print("Selected mode: " + str(game_mode))
        print("Should clip: " + str(clip))
        print("Total steps per episode: " + str(total_step_limit))
        print("Total episodes: " + str(total_run_limit))
        return game_name, game_mode, render, total_step_limit, total_run_limit, clip

    def _game_model(self, game_mode, game_name, action_space, device):
        if game_mode == "dqn_training":
            return DQNTrainer(game_name, INPUT_SHAPE, action_space, device)
        elif game_mode == "dqn_testing":
            return DQNSolver(game_name, INPUT_SHAPE, action_space, device)
        elif game_mode == "ddqn_training":
            return DDQNTrainer(game_name, INPUT_SHAPE, action_space, device)
        elif game_mode == "ddqn_testing":
            return DDQNSolver(game_name, INPUT_SHAPE, action_space, device)
        else:
            print("Unrecognized mode. Use --help")
            exit(1)


if __name__ == "__main__":
    Atari()