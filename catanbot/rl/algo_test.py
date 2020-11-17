from torch import nn
import matplotlib.pyplot as plt
import copy

from catanbot.core.board import Board
from catanbot.agents.independent_actions_agents import IndependentActionsHeuristicAgent, IndependentActionsAgent
from catanbot.board_placement.agents.base import InitialPlacementAgent, MakeDeterministic
from catanbot.board_placement.agents.mlp_agent import MLPPlacementAgent
from catanbot.core.independent_action_simulator import IndependentActionsCatanSimulator
from catanbot.board_placement.initial_placement_simulator import InitialPlacementSimulator

from catanbot.rl.networks.mlp import MLP
from catanbot.rl.networks.valuemlp import VMLP
from catanbot.rl.networks.multihead_attention import MultiheadAttention
from catanbot.rl.collectors.initial_placement_collector import InitialPlacementCollector, DebugCollector
from catanbot.rl.collectors.base import ParallelizeCollector
from catanbot.rl.replaybuffers.simple_replay_buffer import SimpleReplayBuffer

from catanbot.rl.algos.ma_a2c import MultiagentA2C
from catanbot.rl.algos.ma_ppo import MultiagentPPO

from catanbot.rl.experiments.experiment import Experiment

b = Board() 
b.reset()
#    agents = [IndependentActionsAgent(b), IndependentActionsAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsAgent(b)]
agents = [IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b)]
#    agents = [HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b)]
s = IndependentActionsCatanSimulator(board=b, players = agents, max_vp=10)

insize = s.observation_space['total']
attention_insize = 12
attention_outsize = 256
outsize = 54*3
hiddens = [1024, 512, 512]
hiddens = [8192, 200, ]
bias = [True, True, True, True]
import pdb;pdb.set_trace()

attn = MultiheadAttention(attention_insize, attention_outsize)

shared_mlp = MLP(insize, outsize, hiddens, hidden_activation=nn.Tanh, bias=bias, gpu=False)
#shared_mlp = nn.Sequential(MultiheadAttention(attention_insize, attention_outsize), MLP(attention_outsize, outsize, hiddens, hidden_activation=nn.Tanh, bias=bias, gpu=False))

#placement_agents = [MLPPlacementAgent(b, MLP(insize, outsize, hiddens, bias=bias)), MLPPlacementAgent(b, MLP(insize, outsize, hiddens, bias=bias)), MLPPlacementAgent(b, MLP(insize, outsize, hiddens, bias=bias)), MLPPlacementAgent(b, MLP(insize, outsize, hiddens, bias=bias))]
#placement_agents = [MLPPlacementAgent(b, MLP(insize, outsize, hiddens)), InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b)]
placement_agents = [MLPPlacementAgent(b, shared_mlp), MLPPlacementAgent(b, shared_mlp), MLPPlacementAgent(b, shared_mlp), MLPPlacementAgent(b, shared_mlp)]
placement_simulator = InitialPlacementSimulator(s, placement_agents)

placement_agents_deterministic = [MakeDeterministic(a) if isinstance(a, MLPPlacementAgent) else a for a in placement_agents]
sim2 = InitialPlacementSimulator(s, placement_agents_deterministic)

vf_hiddens = [1024, 1024, 512, 512]
vf_hiddens = [8192, 256, ]
vf = VMLP(placement_simulator, hiddens = vf_hiddens, scale=1., hidden_activation=nn.Tanh,  bias=bias, gpu=False)

#vf = nn.Sequential(MultiheadAttention(attention_insize, attention_outsize), MLP(attention_outsize, 4, hiddens, hidden_activation=nn.Tanh, bias=bias, gpu=False))

import pdb;pdb.set_trace()

#collector = DebugCollector(placement_simulator, reset_board=True, reward_scale=1.)
#eval_collector = DebugCollector(sim2, reset_board=True, reward_scale=1.)
collector = InitialPlacementCollector(placement_simulator, reset_board=True, reward_scale=1.)
eval_collector = InitialPlacementCollector(sim2, reset_board=True, reward_scale=1.)
#eval_collector.stored_boards = collector.stored_boards

collector = ParallelizeCollector(collector, nthreads=4)
eval_collector = ParallelizeCollector(eval_collector, nthreads=4)
#buf = SimpleReplayBuffer(placement_simulator, capacity = int(1e6))

algo = MultiagentA2C(placement_simulator, placement_agents, vf, collector, eval_collector, rollouts_per_epoch = 100, eval_rollouts_per_epoch = 10, epochs=1000, vf_itrs=50, eval_every=1, entropy_coeff = 0.01)
#algo = MultiagentPPO(placement_simulator, placement_agents, vf, collector, eval_collector, rollouts_per_epoch = 10, eval_rollouts_per_epoch = 10, epochs=1000, vf_itrs=100, eval_every=1, entropy_coeff = 0.005)
#plt.ion()
#placement_simulator.render()
#plt.pause(1e-2)

experiment = Experiment(algo, '../../../experiments/catan_initial_placement_a2c', save_every=5, save_logs_every=5)
import torch
with torch.autograd.set_detect_anomaly(True):
    experiment.run()

