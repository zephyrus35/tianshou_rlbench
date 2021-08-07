import os
import gym
import pprint
import datetime
import argparse
import numpy as np
import rlbench.gym
from typing import Union, Optional, Sequence

import torch
from torch import nn
from torch import random
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import PPOPolicy
from tianshou.utils import BasicLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, Batch, ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv, DummyVectorEnv


def parse_args():
    parser = argparse.ArgumentParser('testing for tianshou ppo + rlbench environment')

    parser.add_argument('--task', default='reach_target', type=str, help='rlbench task')
    parser.add_argument('--observation', default=False, action='store_true', 
                        help='use state or raw observation')
    parser.add_argument('--watch', action='store_true')

    parser.add_argument('--training_num', default=10, type=int)
    parser.add_argument('--eval_num', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--buffer_size', default=4096, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        type=str, help='use gpu')
    parser.add_argument('--hidden_sizes', default=[64, 64], type=int, nargs='*')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--step_per_epoch', default=30000, type=int)
    parser.add_argument('--step_per_collect', default=2048, type=int)
    parser.add_argument('--repeat_per_collect', default=10, type=int)
    parser.add_argument('--render', type=float, default=0.)
    
    parser.add_argument('--rew_norm', default=True, type=int)
    parser.add_argument('--gae_lambda', default= 0.95, type=float)
    parser.add_argument('--vf_coef', default=0.25, type=float)
    parser.add_argument('--ent_coef', default=0., type=float)
    parser.add_argument('--action_bound_method', default='clip', type=str)
    parser.add_argument('--lr_decay', action='store_false')
    parser.add_argument('--max_grad_norm', default=0.5, type=float)
    parser.add_argument('--eps_clip', default=0.2, type=float)
    parser.add_argument('--dual_clip', default=None, type=float)
    parser.add_argument('--value_clip', default=0, type=int)
    parser.add_argument('--norm_adv', default=0, type=int)
    parser.add_argument('--recompute_adv', default=1, type=int)
    
    parser.add_argument('--logdir', default='logs', type=str)
    parser.add_argument('--resume_path', default=None, type=str)

    args = parser.parse_args()
    return args


def test_ppo(args):
    state = 'observation' if args.observation else 'state'
    args.task = args.task + '-' + state + '-v0'
    
    # set up envs
    train_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)],
        # norm_obs=True
    )
    # train_envs = env
    eval_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.eval_num)],
        # norm_obs=True, obs_rms=train_envs.obs_rms, update_obs_rms=False
    )
    # eval_envs = env
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    eval_envs.seed(args.seed)
    # networks
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, activation=nn.Tanh, device=args.device)
    actor = ActorProb(net_a, args.action_shape, max_action=args.max_action, 
                      unbounded=True,device=args.device).to(args.device)
    net_c = Net(args.state_shape, hidden_sizes=args.hidden_sizes, activation=nn.Tanh, device=args.device)
    critic = Critic(net_c, device=args.device).to(args.device)
    nn.init.constant_(actor.sigma_param, -0.5)
    for i in list(actor.parameters()) + list(critic.parameters()):
        if isinstance(i, nn.Linear):
            nn.init.orthogonal_(i.weight, gain=np.sqrt(2))
            nn.init.zeros_(i.bias)
    # last layer scaling
    for i in actor.mu.modules():
        if isinstance(i, nn.Linear):
            i.weight.data.copy_(i.weight.data * 0.01)
            nn.init.zeros_(i.bias)
            
    optimizer = optim.Adam(list(actor.parameters())+list(critic.parameters()), lr=args.lr)
    if args.lr_decay:
        max_update = np.ceil(
            args.step_per_epoch / args.step_per_collect
        ) * args.epoch
        
        lr_scheduler = LambdaLR(
            optimizer, lambda epoch: 1 - epoch/max_update)
    
    def dist(*logits):
        return Independent(Normal(*logits), 1)
    
    policy = PPOPolicy(actor, critic, optim, dist, discount_factor=args.gamma,
                       gae_lambda=args.gae_lambda, max_grad_norm=args.max_grad_norm,
                       vf_coef=args.vf_coef, ent_coef=args.ent_coef, 
                       reward_normalization=args.rew_norm, action_scaling=True,
                       action_bound_method=args.action_bound_method,
                       lr_scheduler=lr_scheduler, action_space=env.action_space, eps_clip=args.eps_clip,
                       value_clip=args.value_clip, dual_clip=args.dual_clip,
                       advantage_normalization=args.norm_adv, recompute_advantage=args.recompute_adv)
    # load previous path
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print(f'loaded policy from {args.resume_path}')
        
    # log
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = f'{dt}_ppo_{args.task}_seed_{args.seed}'
    print(net_desc)
    logdir = os.path.join(args.logdir, 'tensorboard', 'ppo', net_desc)
    tb = SummaryWriter(logdir)
    logger = BasicLogger(tb, update_interval=100, train_interval=100)
    
    # data & collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, args.training_num)
    else:
        buffer = ReplayBuffer(args.buffer_size)
    # TODO termination of envs after max steps
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    eval_collector = Collector(policy, eval_envs)
    
    print('collector')
    def save_fn(policy):
        pass
    
    def save_checkpoint_fn(epoch, env_step, iter_step):
        torch.save({
            'epoch': epoch,
            'env_step': env_step,
            'iter_step': iter_step,
            'optimizor': optimizer.state_dict(),
            'policy': policy.state_dict()
        }, os.path.join(args.logdir, 'models', 'ppo', net_desc, f'ppo_{epoch}.pt'))
    
    if not args.watch:
        result = onpolicy_trainer(
            policy, train_collector, eval_collector, args.epoch, args.step_per_epoch,
            args.repeat_per_collect, args.eval_num, args.batch_size, args.step_per_collect,
            save_checkpoint_fn=save_checkpoint_fn, test_in_train=False
        )
        pprint(result)
        
    # evaluation
    policy.eval()
    eval_envs.seed(args.seed)
    eval_collector.reset()
    result = eval_collector.collect(n_episode=args.eval_num, render=args.render)
    pprint(result)
    

if __name__=='__main__':
    args = parse_args()
    test_ppo(args)
