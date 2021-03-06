import torch
from torch import nn
import matplotlib.pyplot as plt
import copy

from catanbot.core.board import Board
from catanbot.agents.independent_actions_agents import IndependentActionsHeuristicAgent, IndependentActionsAgent
from catanbot.board_placement.agents.base import InitialPlacementAgent, MakeDeterministic
from catanbot.board_placement.agents.epsilon_greedy_agent import EpsilonGreedyPlacementAgent, EpsilonGreedyPlacementAgentWithMask
from catanbot.core.independent_action_simulator import IndependentActionsCatanSimulator
from catanbot.board_placement.initial_placement_simulator import InitialPlacementSimulator, InitialPlacementSimulatorWithPenalty

from catanbot.rl.networks.mlp import MLP
from catanbot.rl.networks.valuemlp import InitialPlacementsQMLP, InitialPlacementsDoubleQMLP
from catanbot.rl.collectors.initial_placement_collector import InitialPlacementCollector, DebugCollector, InitialPlacementComparisonCollector
from catanbot.rl.collectors.base import ParallelizeCollector
from catanbot.rl.replaybuffers.simple_replay_buffer import SimpleReplayBuffer

from catanbot.rl.algos.self_play_dqn import SelfPlayDQN

from catanbot.rl.experiments.experiment import Experiment

b = Board() 
b.reset()
agents = [IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b)]
s = IndependentActionsCatanSimulator(board=b, players = agents, max_vp=10)

insize = s.observation_space['total']
attention_insize = 12
attention_outsize = 256
outsize = 54*3
hiddens = [512, 256]
bias = [True, True, True, True]

placement_agents = [InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b)]
placement_simulator = InitialPlacementSimulator(s, placement_agents)
#placement_simulator = InitialPlacementSimulatorWithPenalty(s, placement_agents)

qf1 = InitialPlacementsQMLP(placement_simulator, hiddens = hiddens, scale=1., hidden_activation=nn.Tanh,  bias=bias, gpu=False)
qf2 = InitialPlacementsQMLP(placement_simulator, hiddens = hiddens, scale=1., hidden_activation=nn.Tanh,  bias=bias, gpu=False)
target_qf1 = InitialPlacementsQMLP(placement_simulator, hiddens = hiddens, scale=1., hidden_activation=nn.Tanh,  bias=bias, gpu=False)
target_qf2 = InitialPlacementsQMLP(placement_simulator, hiddens = hiddens, scale=1., hidden_activation=nn.Tanh,  bias=bias, gpu=False)

qf = InitialPlacementsDoubleQMLP(qf1, qf2)
target_qf = InitialPlacementsDoubleQMLP(target_qf1, target_qf2)

eps = 0.05

placement_simulator.players = [EpsilonGreedyPlacementAgent(b, qf, lambda e:eps, pidx=0), EpsilonGreedyPlacementAgent(b, qf, lambda e:eps, pidx=1), EpsilonGreedyPlacementAgent(b, qf, lambda e:eps, pidx=2), EpsilonGreedyPlacementAgent(b, qf, lambda e:eps, pidx=3)]

#placement_simulator.players = [EpsilonGreedyPlacementAgentWithMask(b, qf, lambda e:0.05, pidx=0), EpsilonGreedyPlacementAgentWithMask(b, qf, lambda e:0.05, pidx=1), EpsilonGreedyPlacementAgentWithMask(b, qf, lambda e:0.05, pidx=2), EpsilonGreedyPlacementAgentWithMask(b, qf, lambda e:0.05, pidx=3)]

#placement_simulator.players[0] = EpsilonGreedyPlacementAgent(b, qf, lambda e:0.1, pidx=0)

#placement_agents_deterministic = [MakeDeterministic(a) if isinstance(a, MLPPlacementAgent) else a for a in placement_agents]
sim2 = InitialPlacementSimulator(s, placement_agents)
#sim2 = InitialPlacementSimulatorWithPenalty(s, placement_agents)
sim2.players = [EpsilonGreedyPlacementAgent(b, target_qf, lambda e:0, pidx=0), EpsilonGreedyPlacementAgent(b, target_qf, lambda e:0, pidx=1), EpsilonGreedyPlacementAgent(b, target_qf, lambda e:0., pidx=2), EpsilonGreedyPlacementAgent(b, target_qf, lambda e:0., pidx=3)]

#collector = DebugCollector(placement_simulator, reset_board=True, reward_scale=1., nboards=2)
#eval_collector = DebugCollector(sim2, reset_board=True, reward_scale=1.)
#eval_collector.stored_boards = collector.stored_boards

collector = InitialPlacementCollector(placement_simulator, reset_board=True, reward_scale=1.) #Multiply reward scale so best actions always beat random net output.
eval_collector = InitialPlacementCollector(sim2, reset_board=True, reward_scale=1.)
cmp_collector = InitialPlacementComparisonCollector(sim2, reset_board=True, reward_scale=1.)

collector = ParallelizeCollector(collector, nthreads=8)
eval_collector = ParallelizeCollector(eval_collector, nthreads=4)
#cmp_collector = ParallelizeCollector(cmp_collector, nthreads=4)
buf = SimpleReplayBuffer(placement_simulator, capacity = 500000)

#plt.ion()
#placement_simulator.render()
#plt.pause(1e-2)

algo = SelfPlayDQN(placement_simulator, qf, target_qf, buf, collector, eval_collector, cmp_collector, rollouts_per_epoch=200, cmp_rollouts_per_epoch=200, qf_itrs=200, qf_batch_size=256, qf_lr=1e-4, target_update_tau=0.005, discount=1.0)

experiment = Experiment(algo, '../../../experiments/catan_initial_placement_alphago_dqn', save_every=20, save_logs_every=5)
import torch
with torch.autograd.set_detect_anomaly(True):
    experiment.run()

