import torch
import numpy as np;np.set_printoptions(precision=6, suppress=True)
import matplotlib.pyplot as plt
import copy
import argparse
import time

from catanbot.core.board import Board
from catanbot.agents.independent_actions_agents import IndependentActionsHeuristicAgent, IndependentActionsAgent
from catanbot.board_placement.agents.base import InitialPlacementAgent, MakeDeterministic
from catanbot.board_placement.agents.mlp_agent import MLPPlacementAgent
from catanbot.board_placement.agents.epsilon_greedy_agent import EpsilonGreedyPlacementAgent, GNNEpsilonGreedyPlacementAgent
from catanbot.core.independent_action_simulator import IndependentActionsCatanSimulator
from catanbot.board_placement.initial_placement_simulator import InitialPlacementSimulator, InitialPlacementSimulatorWithPenalty

from catanbot.rl.networks.mlp import MLP
from catanbot.rl.networks.valuemlp import VMLP
from catanbot.rl.collectors.initial_placement_collector import InitialPlacementCollector
from catanbot.rl.collectors.graph_initial_placement_collector import GraphInitialPlacementCollector
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

placement_simulator = InitialPlacementSimulator(s, placement_agents)
placement_simulator.players = [GNNEpsilonGreedyPlacementAgent(b, qf, lambda e:0., pidx=0), GNNEpsilonGreedyPlacementAgent(b, qf, lambda e:0., pidx=1), GNNEpsilonGreedyPlacementAgent(b, qf, lambda e:0., pidx=2), GNNEpsilonGreedyPlacementAgent(b, qf, lambda e:0., pidx=3)]

collector = GraphInitialPlacementCollector(placement_simulator, reset_board=True, reward_scale=1.)

q_seq = []
timing = []
obses = []

while not placement_simulator.terminal:
    obs = collector.flatten_observation(placement_simulator.graph_observation)
    obs = {k:torch.tensor(v).float().unsqueeze(0) for k,v in obs.items()}
    with torch.no_grad():
        t = time.time()
        qvals = qf(obs).squeeze()
        timing.append(time.time() - t)

    q_seq.append(qvals.max(dim=1)[0])

    act = placement_simulator.players[placement_simulator.player_idx].action(obs, obs)
    placement_simulator.step(act)

    placement_simulator.render()

    obses.append(obs)

obs = collector.flatten_observation(placement_simulator.graph_observation)
obs = {k:torch.tensor(v).float().unsqueeze(0) for k,v in obs.items()}

with torch.no_grad():
    qvals = qf(obs).squeeze()
q_seq.append(qvals.max(dim=1)[0])
q_seq = torch.stack(q_seq, dim=0)

for i in range(4):
    print('Player {} Q = '.format(i+1))
    print(qvals[i])

print('Q sequence')
print(q_seq)

print('Final Qs')
print(q_seq[-1])

print('NN time = {:.4f}'.format(sum(timing) / len(timing)))

print('Simulated wins:')
print(placement_simulator.reward(n=100))

placement_simulator.render()
