import os
import gym
from tianshou import policy
import torch
import pprint
import datetime
import argparse
import numpy as np
import rlbench.gym
from typing import Union, Optional, Sequence

from torch import nn
from torch import random
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Independent, Normal

from tianshou.policy import DQNPolicy
from tianshou.utils import BasicLogger
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv, DummyVectorEnv


class DQN(nn.Module):
    """
    """
    def __init__(self, c: int, h: int, w: int, 
                 action_shape: Sequence[int], 
                 device: Union[str, int, torch.device] = "cpu", 
                 features_only: bool = False) -> None:
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])
        if not features_only:
            self.net = nn.Sequential(
                self.net,
                nn.Linear(self.output_dim, 512),
                nn.ReLU(),
                nn.Linear(512, np.prod(action_shape))
            )
            self.output_dim = np.prod(action_shape)

    def forward(self, 
                obs: Union[np.ndarray, torch.Tensor], 
                state: Optional[any] = None):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state


def parse_args():
    parser = argparse.ArgumentParser('testing for tianshou dqn + rlbench environment')

    parser.add_argument('--task', default='reach_target', type=str, help='robotic task selected')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--watch', action='store_true', help='watch the play of pre-trained policy only')

    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--step_per_epoch', default=10000, type=int)
    parser.add_argument('--step_per_collect', default=5, type=int)
    parser.add_argument('--update_per_step', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--training_num', default=10, type=int, help='training env num')
    parser.add_argument('--eval_num', default=1, type=int, help='evaluating env num')
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--target_update_freq', default=500, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_decay', action='store_false')
    parser.add_argument('--max_grad_norm', default=0.5, type=float)
    parser.add_argument('--buffer_size', default=100000, type=int)
    parser.add_argument('--frames_stack', default=4, type=int)
    parser.add_argument('--n_step', default=3, type=int, help='estimation step')

    parser.add_argument('--render', action='store_true')
    parser.add_argument('--render_interval', default=0., type=float, help='time inteval between rendering')
    parser.add_argument('--logdir', default='logs', type=str)

    args = parser.parse_args()
    return args


def test_dqn(args):
    task = args.task + '-state-v0'
    env = gym.make(task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    print(args.state_shape, args.action_shape)

    train_envs = SubprocVectorEnv(
        [lambda: gym.make(task) for _ in range(args.training_num)],
        # norm_obs=True
    )
    # train_envs = env
    eval_envs = SubprocVectorEnv(
        [lambda: gym.make(task) for _ in range(args.eval_num)],
        # norm_obs=True, obs_rms=train_envs.obs_rms, update_obs_rms=False
    )
    # eval_envs = env
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    eval_envs.seed(args.seed)
    # make q net
    qnet = DQN(*args.state_shape, args.action_shape, device=args.device)
    optimizor = optim.Adam(qnet.parameters(), lr=args.lr)
    # policy
    dqn_policy = DQNPolicy(qnet, optim=optimizor, discount_factor=args.gamma,
                           estimation_step=args.n_step,
                           target_update_freq=args.target_update_freq)
    buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
        # ignore_obs_next=True, save_only_last_obs=True, stack_num=args.frames_stack)
    train_collector = Collector(dqn_policy, train_envs, buffer, exploration_noise=True)
    eval_collector = Collector(dqn_policy, eval_envs)
    train_collector.collect(n_step=args.start_timesteps, random=True)
    # log
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    desc = f'seed_{args.seed}_dqn_{args.task}_{dt}'
    logdir = os.path.join(args.logdir, desc)
    tb = SummaryWriter(logdir)
    tb.add_text('args', str(args))
    logger = BasicLogger(tb)

    def train_fn():
        """modify eps accordingly"""
        # TODO parameter tuning
        pass

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(logdir, 'DQNpolicy.pt'))

    if not args.watch:
        result = offpolicy_trainer(
            dqn_policy, train_collector, eval_collector, args.epoch,
            args.step_per_epoch, args.step_per_collect, args.eval_num,
            args.batch_size, args.update_per_step, save_fn=save_fn,
            logger=logger, test_in_train=False)
        pprint.pprint(result)

    # evaluate
    policy.eval()
    eval_envs.seed(args.seed)
    eval_collector.reset()
    result = eval_collector.collect(n_episode=args.eval_num, render=args.render_interval)
    pprint.pprint(result)
    print(f'reward: {result["rews"].mean()} length: {result["lens"].mean()}')


if __name__ == '__main__':
    args = parse_args()
    test_dqn(args)
