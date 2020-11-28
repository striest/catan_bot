import torch
from torch import nn
import matplotlib.pyplot as plt
import copy
import argparse

from catanbot.rl.debug.gridworld import Gridworld
from catanbot.rl.debug.gridworld_qmlp import GridworldQMLP
from catanbot.rl.networks.valuemlp import InitialPlacementsQMLP, InitialPlacementsDoubleQMLP
from catanbot.rl.debug.epsilongreedy_agent import EpsilonGreedyGridworldAgent

from catanbot.rl.collectors.initial_placement_collector import InitialPlacementCollector, DebugCollector
from catanbot.rl.collectors.base import ParallelizeCollector
from catanbot.rl.replaybuffers.simple_replay_buffer import SimpleReplayBuffer

from catanbot.rl.algos.dqn import DQN

from catanbot.rl.experiments.experiment import Experiment

parser = argparse.ArgumentParser(description='Parse videomaker params')
parser.add_argument('--qf_fp', type=str, required=True, help='location to the Q network')
args = parser.parse_args()

s = Gridworld('agent placeholder')
s.reset()
s.render()

qf = torch.load(args.qf_fp)

s.players = [EpsilonGreedyGridworldAgent(qf, lambda e:0, pidx=0), EpsilonGreedyGridworldAgent(qf, lambda e:0, pidx=1), EpsilonGreedyGridworldAgent(qf, lambda e:0, pidx=2), EpsilonGreedyGridworldAgent(qf, lambda e:0, pidx=3)]

collector = InitialPlacementCollector(s, reset_board=True, reward_scale=1.) #Multiply reward scale so best actions always beat random net output
s.reset()

while not s.terminal:
    obs = collector.flatten_observation(s.observation)
    obs = torch.tensor(obs).float()
    with torch.no_grad():
        qvals = qf(obs.unsqueeze(0)).squeeze()

    print('Player = {}'.format(s.player_idx + 1))
    print('obs = \n{}'.format(obs))
    print('QF = \n{}'.format(qvals))

    act = s.players[s.player_idx].action(obs, obs)
    s.step(act)
    s.render()
