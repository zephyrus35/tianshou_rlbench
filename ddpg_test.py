import os
import gym
import pprint
import datetime
import argparse
import numpy as np
from torch._C import device
import rlbench.gym
from typing import Union, Optional, Sequence

import torch
from torch import nn
from torch import random
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DDPGPolicy
from tianshou.utils import BasicLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, Batch, ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.exploration import GaussianNoise


def parser_args():
    parser = argparse.ArgumentParser('testing for tianshou ddpg + rlbench environment')
    
    parser.add_argument('--observation', default=False, action='store_true', help='')
    parser.add_argument('--task', default='reach_target', type=str, help='')
    parser.add_argument('--resume_path', default=None, type=str)
    parser.add_argument('--watch', default=False, action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    
    parser.add_argument('--training_num', default=10, type=int, help='')
    parser.add_argument('--eval_num', default=1, type=int, help='')
    parser.add_argument('--hidden-sizes',  default=[256, 256], type=int, nargs='*')
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--gamma', default=0.99, type=float, help='')
    parser.add_argument('--tau', default=0.05, type=float, help='')
    parser.add_argument('--epoch', default=150, type=int)
    parser.add_argument('--step_per_epoch', default=5000, type=int)
    parser.add_argument('--step_per_collect', default=1, type=int)
    parser.add_argument('--update_per_step', default=1, type=int)
    parser.add_argument('--n_step', default=1, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--exploration_noise', default=0.1, type=float)
    parser.add_argument('--start_timesteps', default=20000, type=int)
    parser.add_argument('--buffer_size', default=100000, type=int)
    parser.add_argument('--render', type=float, default=0.)
    
    parser.add_argument('--device', 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        type=str, help='use gpu')
    parser.add_argument('--logdir', default='logs', type=str)
    return parser.parse_args()


def test_ddpg(args=parser_args()):
    # set up envrionments
    state = '-observation-v0' if args.observation else '-state-v0'
    args.task = args.task + state
    # env = gym.make(task)
    train_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)]
    )
    eval_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)]
    )
    print(type(train_envs), type(eval_envs))
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    eval_envs.seed(args.seed)
    # networks & optimizers
    net_actor = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    net_critic = Net(args.state_shape, args.action_shape, 
                     hidden_sizes=args.hidden_sizes, concat=True, device=args.device)
    actor = Actor(net_actor, args.action_shape, max_action=args.max_action, 
                  device=args.device).to(args.device)
    critic = Critic(net_critic, device=args.device).to(args.device)
    actor_optim = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=args.critic_lr)
    # policy
    ddpg = DDPGPolicy(actor, actor_optim, critic, critic_optim, tau=args.tau, gamma=args.gamma,
                      exploration_noise=GaussianNoise(sigma=args.exploration_noise), 
                      estimation_step=args.n_step, action_space=env.action_space)
    
    if args.resume_path:
        ddpg.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print(f'loaded policy from {args.resume_path}')
    
    # buffer & collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, args.training_num)
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(ddpg, train_envs, buffer, exploration_noise=True)
    eval_collector = Collector(ddpg, eval_envs)
    train_collector.collect(n_step=args.start_timesteps, random=True)
    # log
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = f'{dt}_ddpg_{args.task}_seed_{args.seed}'
    print(net_desc)
    logdir = os.path.join(args.logdir, 'tensorboard', 'ddpg', net_desc)
    tb = SummaryWriter(logdir)
    tb.add_text('args', str(args))
    logger = BasicLogger(tb, update_interval=100, train_interval=100)
    
    def save_fn():
        save_dir = os.path.join(args.logdir, 'models', net_desc, 'ddpg.pt')
        torch.save(ddpg.state_dict(), save_dir)
    
    def save_checkpoint_fn(epoch, env_step, iter_step):
        torch.save({
            'epoch': epoch,
            'env_step': env_step,
            'iter_step': iter_step,
            'actor_optimizor': actor_optim.state_dict(),
            'critic_optimizor': critic_optim.state_dict(),
            'policy': ddpg.state_dict()
        }, os.path.join(args.logdir, 'models', 'ddpg', net_desc, f'ddpg_{epoch}.pt'))
    
    if not args.watch:
        result = offpolicy_trainer(
            ddpg, train_collector, eval_collector, args.epoch, args.step_per_epoch,
            args.step_per_collect, args.eval_num, args.batch_size, args.update_per_step,
            save_checkpoint_fn=save_checkpoint_fn, save_fn=save_fn,
            logger=logger, test_in_train=False
        )
        pprint(result)
        
    ddpg.eval()
    eval_envs.seed(args.seed)
    eval_collector.reset()
    result = eval_collector.collect(n_episode=args.eval_num, render=args.render)
    pprint(result)


if __name__=='__main__':
    args = parser_args()
    test_ddpg(args)
