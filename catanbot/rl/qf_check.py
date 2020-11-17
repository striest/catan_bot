import torch
import numpy as np;np.set_printoptions(precision=6, suppress=True)
import matplotlib.pyplot as plt
import copy
import argparse

from catanbot.core.board import Board
from catanbot.agents.independent_actions_agents import IndependentActionsHeuristicAgent, IndependentActionsAgent
from catanbot.board_placement.agents.base import InitialPlacementAgent, MakeDeterministic
from catanbot.board_placement.agents.mlp_agent import MLPPlacementAgent
from catanbot.board_placement.agents.epsilon_greedy_agent import EpsilonGreedyPlacementAgent
from catanbot.core.independent_action_simulator import IndependentActionsCatanSimulator
from catanbot.board_placement.initial_placement_simulator import InitialPlacementSimulator, InitialPlacementSimulatorWithPenalty

from catanbot.rl.networks.mlp import MLP
from catanbot.rl.networks.valuemlp import VMLP
from catanbot.rl.collectors.initial_placement_collector import InitialPlacementCollector
from catanbot.rl.collectors.base import ParallelizeCollector
from catanbot.rl.replaybuffers.simple_replay_buffer import SimpleReplayBuffer

from catanbot.rl.algos.ma_a2c import MultiagentA2C

from catanbot.rl.experiments.experiment import Experiment

parser = argparse.ArgumentParser(description='Parse videomaker params')
parser.add_argument('--qf_fp', type=str, required=True, help='location to the Q network')
args = parser.parse_args()

b = Board() 
b.reset()
#    agents = [IndependentActionsAgent(b), IndependentActionsAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsAgent(b)]
agents = [IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b)]
#    agents = [HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b)]
s = IndependentActionsCatanSimulator(board=b, players = agents, max_vp=10)

placement_agents = [MLPPlacementAgent(b, MLP(1, 1, [1, ])), InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b)]
qf = torch.load(args.qf_fp)
qf.eval()

#placement_simulator = InitialPlacementSimulator(s, placement_agents)
placement_simulator = InitialPlacementSimulatorWithPenalty(s, placement_agents)
placement_simulator.players = [EpsilonGreedyPlacementAgent(b, qf, lambda e:0.25, pidx=0), EpsilonGreedyPlacementAgent(b, qf, lambda e:0.25, pidx=1), EpsilonGreedyPlacementAgent(b, qf, lambda e:0.25, pidx=2), EpsilonGreedyPlacementAgent(b, qf, lambda e:0.25, pidx=3)]

collector = InitialPlacementCollector(placement_simulator, reset_board=True, reward_scale=1.)

obs = collector.flatten_observation(placement_simulator.observation)
obs = torch.tensor(obs).float()
with torch.no_grad():
    qvals = qf(obs.unsqueeze(0)).squeeze()

for i in range(4):
    qi = qvals[i]
    print('PLAYER {} Q =\n{}'.format(i+1, torch.cat([torch.arange(54).unsqueeze(1), qi.view(54, 3)], dim=1).numpy()))
    print('TOP 5 =\n{}'.format(qi.flatten().topk(5).indices/3))
    print('MAX Q= \n{}'.format(qi.max()))
    break

placement_simulator.render()
