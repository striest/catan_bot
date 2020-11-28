import torch
from torch import nn
import matplotlib.pyplot as plt
import copy

from catanbot.rl.debug.gridworld import Gridworld
from catanbot.rl.debug.gridworld_qmlp import GridworldQMLP
from catanbot.rl.networks.valuemlp import InitialPlacementsQMLP, InitialPlacementsDoubleQMLP
from catanbot.rl.debug.epsilongreedy_agent import EpsilonGreedyGridworldAgent

from catanbot.rl.collectors.initial_placement_collector import InitialPlacementCollector, DebugCollector
from catanbot.rl.collectors.base import ParallelizeCollector
from catanbot.rl.replaybuffers.simple_replay_buffer import SimpleReplayBuffer

from catanbot.rl.algos.dqn import DQN

from catanbot.rl.experiments.experiment import Experiment

s = Gridworld('agent placeholder')
s.reset()
s.render()

insize = s.observation_space['total']
hiddens = [128, 128]
bias = [True, True, True]

qf1 = GridworldQMLP(s, hiddens = hiddens, scale=1., hidden_activation=nn.Tanh,  bias=bias, gpu=False)
qf2 = GridworldQMLP(s, hiddens = hiddens, scale=1., hidden_activation=nn.Tanh,  bias=bias, gpu=False)
target_qf1 = GridworldQMLP(s, hiddens = hiddens, scale=1., hidden_activation=nn.Tanh,  bias=bias, gpu=False)
target_qf2 = GridworldQMLP(s, hiddens = hiddens, scale=1., hidden_activation=nn.Tanh,  bias=bias, gpu=False)

qf = InitialPlacementsDoubleQMLP(qf1, qf2)
target_qf = InitialPlacementsDoubleQMLP(target_qf1, target_qf2)

eps = 0.1

s.players = [EpsilonGreedyGridworldAgent(qf, lambda e:eps, pidx=0), EpsilonGreedyGridworldAgent(qf, lambda e:eps, pidx=1), EpsilonGreedyGridworldAgent(qf, lambda e:eps, pidx=2), EpsilonGreedyGridworldAgent(qf, lambda e:eps, pidx=3)]

collector = InitialPlacementCollector(s, reset_board=True, reward_scale=1.) #Multiply reward scale so best actions always beat random net output

s_eval = Gridworld('agent placeholder')
s_eval.players = [EpsilonGreedyGridworldAgent(target_qf, lambda e:0, pidx=0), EpsilonGreedyGridworldAgent(target_qf, lambda e:0, pidx=1), EpsilonGreedyGridworldAgent(target_qf, lambda e:0, pidx=2), EpsilonGreedyGridworldAgent(target_qf, lambda e:0, pidx=3)]
eval_collector = InitialPlacementCollector(s, reset_board=True, reward_scale=1.) #Multiply reward scale so best actions always beat random net output

collector = ParallelizeCollector(collector, nthreads=4)
eval_collector = ParallelizeCollector(eval_collector, nthreads=4)
buf = SimpleReplayBuffer(s, capacity = 100000)

algo = DQN(s, qf, target_qf, buf, collector, eval_collector, rollouts_per_epoch=50, qf_itrs=50, qf_batch_size=64, qf_lr=1e-4, discount=0.9, target_update_tau=0.005)

experiment = Experiment(algo, '../../../../experiments/catanbot_debug', save_every=5, save_logs_every=5)
import torch
with torch.autograd.set_detect_anomaly(True):
    experiment.run()

