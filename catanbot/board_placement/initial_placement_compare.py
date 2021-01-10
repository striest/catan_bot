import time
import copy
import matplotlib.pyplot as plt
import numpy as np;np.set_printoptions(precision=4, linewidth=1e6, suppress=True)
import torch
import ray
import argparse

from catanbot.core.board import Board
from catanbot.agents.independent_actions_agents import IndependentActionsHeuristicAgent, IndependentActionsAgent
from catanbot.board_placement.agents.base import InitialPlacementAgent, MakeDeterministic, HeuristicInitialPlacementAgent
from catanbot.core.independent_action_simulator import IndependentActionsCatanSimulator
from catanbot.board_placement.initial_placement_simulator import InitialPlacementSimulator, InitialPlacementSimulatorWithPenalty
from catanbot.rl.collectors.initial_placement_collector import InitialPlacementCollector
from catanbot.rl.collectors.graph_initial_placement_collector import GraphInitialPlacementCollector
from catanbot.rl.networks.graphsage import GraphSAGEQMLP
from catanbot.board_placement.agents.epsilon_greedy_agent import EpsilonGreedyPlacementAgent, GNNEpsilonGreedyPlacementAgent
from catanbot.board_placement.initial_placement_mcts import RayMCTSQFSearch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse videomaker params')
    parser.add_argument('--qf_fp', type=str, required=True, help='location to the Q network')
    args = parser.parse_args()
    qf = torch.load(args.qf_fp)
    qf.eval()

    b = Board() 
    b.reset()
    agents = [IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b)]
    s = IndependentActionsCatanSimulator(board=b, players = agents, max_vp=10)

    placement_agents = [InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b)]

    placement_simulator = InitialPlacementSimulator(s, placement_agents)
    placement_agents = [GNNEpsilonGreedyPlacementAgent(b, qf, lambda e:0., pidx=0), GNNEpsilonGreedyPlacementAgent(b, qf, lambda e:0., pidx=1), GNNEpsilonGreedyPlacementAgent(b, qf, lambda e:0., pidx=2), GNNEpsilonGreedyPlacementAgent(b, qf, lambda e:0., pidx=3)]
    placement_agents = [HeuristicInitialPlacementAgent(b), HeuristicInitialPlacementAgent(b), HeuristicInitialPlacementAgent(b), HeuristicInitialPlacementAgent(b)]
    placement_agents = [InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b)]


    isgraph = isinstance(qf, GraphSAGEQMLP) or isinstance(qf.qf1, GraphSAGEQMLP)

    if isgraph:
        collector = GraphInitialPlacementCollector(placement_simulator, reset_board=True, reward_scale=1.)
    else:
        collector = InitialPlacementCollector(placement_simulator, reset_board=True, reward_scale=1.)

    mcts1 = RayMCTSQFSearch(placement_simulator, qf, n_threads=12, use_graph=isgraph, lam=0.5)
    mcts2 = RayMCTSQFSearch(placement_simulator, qf, n_threads=12, use_graph=isgraph, lam=0.0)
    mcts2 = qf

    mctss = [mcts1, mcts2]

    total_wins = 0.
    itrs = 50

    for i in range(itrs):
        x = np.random.permutation(np.arange(4))
        players = np.array([0, 0, 1, 1])[x]
        while not placement_simulator.terminal:
            mcts = mctss[players[placement_simulator.player_idx]]
            if isinstance(mcts, RayMCTSQFSearch):
                mcts.root.simulator.simulator.reset_from(b, agents)
                mcts.root.simulator.turn = placement_simulator.turn
                mcts = RayMCTSQFSearch(placement_simulator, mcts.qf, n_threads=mcts.n_threads, use_graph=mcts.use_graph, lam=mcts.lam, workers=mcts.workers)
                mcts.search(max_time=180.0, verbose=False, c=2.5)
                act = mcts.get_optimal_path()[1].prev_act
                act = np.reshape(act, [54, 3])

            else:
                obs = collector.flatten_observation(placement_simulator.graph_observation)
                obs = {k:torch.tensor(v).float().unsqueeze(0) for k,v in obs.items()}
                act = placement_agents[placement_simulator.player_idx].action(obs, obs)['placement']

            placement_simulator.step({'placement':act}) 

        wins = placement_simulator.reward(n=10)
        check_wins = np.sum(wins[players==0])
        total_wins += check_wins
        placement_simulator.reset()
        b = placement_simulator.simulator.board
        agents = placement_simulator.simulator.players
        placement_agents = [GNNEpsilonGreedyPlacementAgent(b, qf, lambda e:0., pidx=0), GNNEpsilonGreedyPlacementAgent(b, qf, lambda e:0., pidx=1), GNNEpsilonGreedyPlacementAgent(b, qf, lambda e:0., pidx=2), GNNEpsilonGreedyPlacementAgent(b, qf, lambda e:0., pidx=3)]
        placement_agents = [HeuristicInitialPlacementAgent(b), HeuristicInitialPlacementAgent(b), HeuristicInitialPlacementAgent(b), HeuristicInitialPlacementAgent(b)]
        placement_agents = [InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b)]

        print('ITR =', i)
        print('WINS =', total_wins)

    total_wins /= itrs
    print('Win Rate = ', total_wins)

    placement_simulator.render()
