import numpy as np
from rlbench.environment import Environment
from rlbench.action_modes import ActionMode, ArmActionMode, GripperActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.exploration import GaussianNoise
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv


def parse_args():
    parser = argparse.ArgumentParser(help='testing for tianshou dqn + rlbench environment')

    parser.add_argument('--step', default=120, type=int)
    parser.add_argument('--task', default='reach_target', type=str, help='robotic task selected')

    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--step_per_epoch', default=5000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--training_num', default=1, type=int, help='training env num')
    parser.add_argument('--eval_num', default=1, type=1, help='evaluating env num')
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--gamma', default=0.1, type=float)

    args = parser.parse_args()
    return args


def test_dqn(args):
    # train_envs = SubprocVectorEnv(
    #     [lambda: gym.make(args.task)],
    #     norm_obs=True
    # )
    pass


if __name__=='__main__':
    import argparse
    args = parse_args()
    test_dqn(args)
