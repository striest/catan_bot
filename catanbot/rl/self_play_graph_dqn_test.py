import torch
from torch import nn
import matplotlib.pyplot as plt
import copy

from catanbot.core.board import Board
from catanbot.agents.independent_actions_agents import IndependentActionsHeuristicAgent, IndependentActionsAgent
from catanbot.board_placement.agents.base import InitialPlacementAgent, MakeDeterministic
from catanbot.board_placement.agents.epsilon_greedy_agent import GNNEpsilonGreedyPlacementAgent
from catanbot.core.independent_action_simulator import IndependentActionsCatanSimulator
from catanbot.board_placement.initial_placement_simulator import InitialPlacementSimulator

from catanbot.rl.networks.mlp import MLP
from catanbot.rl.networks.graphsage import GraphSAGENet, GraphSAGEQMLP
from catanbot.rl.networks.valuemlp import InitialPlacementsQMLP, InitialPlacementsDoubleQMLP
from catanbot.rl.collectors.graph_initial_placement_collector import GraphInitialPlacementCollector, GraphInitialPlacementComparisonCollector
from catanbot.rl.collectors.base import ParallelizeCollector
from catanbot.rl.replaybuffers.graph_replay_buffer import GraphReplayBuffer

from catanbot.rl.algos.self_play_dqn import SelfPlayDQN

from catanbot.rl.experiments.experiment import Experiment

b = Board() 
b.reset()
agents = [IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b)]
s = IndependentActionsCatanSimulator(board=b, players = agents, max_vp=10)

placement_agents = [InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b)]
placement_simulator = InitialPlacementSimulator(s, placement_agents)

hiddens = [64] * 4
embeddings = [64] * 5
outsize = 16

qf1 = GraphSAGENet(40, 5, outsize, b.structure_tensor, embedding_sizes = embeddings, hidden_sizes = hiddens)
qf2 = GraphSAGENet(40, 5, outsize, b.structure_tensor, embedding_sizes = embeddings, hidden_sizes = hiddens)
target_qf1 = GraphSAGENet(40, 5, outsize, b.structure_tensor, embedding_sizes = embeddings, hidden_sizes = hiddens)
target_qf2 = GraphSAGENet(40, 5, outsize, b.structure_tensor, embedding_sizes = embeddings, hidden_sizes = hiddens)

qf1 = GraphSAGEQMLP(placement_simulator, qf1)
target_qf1 = GraphSAGEQMLP(placement_simulator, target_qf1)
qf2 = GraphSAGEQMLP(placement_simulator, qf2)
target_qf2 = GraphSAGEQMLP(placement_simulator, target_qf2)

qf = InitialPlacementsDoubleQMLP(qf1, qf2)
target_qf = InitialPlacementsDoubleQMLP(target_qf1, target_qf2)

eps = 0.05

placement_simulator.players = [GNNEpsilonGreedyPlacementAgent(b, qf, lambda e:eps, pidx=0), GNNEpsilonGreedyPlacementAgent(b, qf, lambda e:eps, pidx=1), GNNEpsilonGreedyPlacementAgent(b, qf, lambda e:eps, pidx=2), GNNEpsilonGreedyPlacementAgent(b, qf, lambda e:eps, pidx=3)]

sim2 = InitialPlacementSimulator(s, placement_agents)
#sim2 = InitialPlacementSimulatorWithPenalty(s, placement_agents)
sim2.players = [GNNEpsilonGreedyPlacementAgent(b, target_qf, lambda e:0, pidx=0), GNNEpsilonGreedyPlacementAgent(b, target_qf, lambda e:0, pidx=1), GNNEpsilonGreedyPlacementAgent(b, target_qf, lambda e:0., pidx=2), GNNEpsilonGreedyPlacementAgent(b, target_qf, lambda e:0., pidx=3)]

collector = GraphInitialPlacementCollector(placement_simulator, reset_board=True, reward_scale=1.) #Multiply reward scale so best actions always beat random net output.
eval_collector = GraphInitialPlacementCollector(sim2, reset_board=True, reward_scale=1.)
cmp_collector = GraphInitialPlacementComparisonCollector(sim2, reset_board=True, reward_scale=1.)

collector = ParallelizeCollector(collector, nthreads=8)
eval_collector = ParallelizeCollector(eval_collector, nthreads=4)
#cmp_collector = ParallelizeCollector(cmp_collector, nthreads=4)
buf = GraphReplayBuffer(placement_simulator, capacity = 250000)

algo = SelfPlayDQN(placement_simulator, qf, target_qf, buf, collector, eval_collector, cmp_collector, rollouts_per_epoch=200, cmp_rollouts_per_epoch=100, qf_itrs=200, qf_batch_size=64, qf_lr=1e-4, target_update_tau=0.005, discount=1.0, eval_rollouts_per_epoch=10, eval_every=10)

experiment = Experiment(algo, '../../../experiments/catan_initial_placement_alphago_graph_dqn', save_every=20, save_logs_every=5)
import torch
with torch.autograd.set_detect_anomaly(True):
    experiment.run()

