import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
import copy

from catanbot.rl.collectors.base import Collector
from catanbot.rl.collectors.initial_placement_collector import InitialPlacementCollector

from catanbot.core.independent_action_simulator import IndependentActionsCatanSimulator
from catanbot.core.board import Board
from catanbot.agents.independent_actions_agents import IndependentActionsHeuristicAgent, IndependentActionsAgent
from catanbot.board_placement.agents.base import InitialPlacementAgent
from catanbot.board_placement.agents.epsilon_greedy_agent import EpsilonGreedyPlacementAgent
from catanbot.core.independent_action_simulator import IndependentActionsCatanSimulator
from catanbot.board_placement.initial_placement_simulator import InitialPlacementSimulator
from catanbot.constants import DEV, SETTLEMENT, CITY, ROAD

class GraphInitialPlacementCollector(InitialPlacementCollector):
    """
    Rollout collector for initial placements. This one uses the graph observations from board.
    """
    def __init__(self, simulator, reward_scale = 1.0, reset_board=False):
        self.simulator = simulator
        self.reset_board = reset_board
        self.reward_scale = reward_scale

    def get_rollout(self, rollout_n = 10, i=0, verbose=False):
        self.simulator.reset(reset_board=self.reset_board)

        rollout = self.setup_rollout_dict()

        while not self.simulator.terminal:
            obs = self.simulator.graph_observation
            obs_flat = self.flatten_observation(obs)
            pidx = self.simulator.player_idx
            act = self.simulator.players[pidx].action(obs, obs_flat)
            self.simulator.step(act)
            nobs = self.simulator.graph_observation
            rew = self.simulator.reward(rollout_n) * self.reward_scale
            term = self.simulator.terminal

            act_flat = self.flatten_action(act)
            nobs_flat = self.flatten_observation(nobs)

            obs_vertices = obs_flat['vertices']
            obs_edges = obs_flat['edges']
            obs_player = obs_flat['player']

            nobs_vertices = nobs_flat['vertices']
            nobs_edges = nobs_flat['edges']
            nobs_player = nobs_flat['player']

            for k, v in zip(['observation_vertices', 'observation_edges', 'observation_player',  'action', 'reward', 'terminal', 'next_observation_vertices', 'next_observation_edges', 'next_observation_player', 'pidx'], [obs_vertices, obs_edges, obs_player, act_flat, rew, term, nobs_vertices, nobs_edges, nobs_player, pidx]):
                rollout[k] = np.append(rollout[k], [v], axis=0)

        for k in rollout.keys():
            rollout[k] = torch.tensor(rollout[k])
            if k != 'pidx':
                rollout[k] = rollout[k].float()

        if verbose:
            print('Finished rollout {}'.format(i), end='\r')
            sys.stdout.flush()

        return rollout

    def flatten_action(self, act):
        act_flat = np.concatenate([act[k].flatten() for k in ['placement']])
        return act_flat

    def flatten_observation(self, obs):
        player_obs_flat = np.concatenate([obs['player'][k].flatten() for k in ['pidx', 'resources', 'dev', 'trade_ratios']])
        player_obs_flat = np.concatenate([player_obs_flat, obs['vp'], obs['army'], obs['road']])

        return {
            'vertices':obs['board']['vertices'],
            'edges':obs['board']['edges'],
            'player':player_obs_flat
        }

    def setup_rollout_dict(self):
        obs_space = self.simulator.graph_observation_space
        act_space = self.simulator.action_space

        rollout = {
            'observation_vertices':np.zeros([0, *obs_space['board']['vertices']]),
            'observation_edges':np.zeros([0, *obs_space['board']['edges']]),
            'observation_player':np.zeros([0, obs_space['total']]),
            'action':np.zeros([0, act_space['total']]),
            'reward':np.zeros([0, 4]),
            'terminal':np.zeros(0).astype(bool),
            'next_observation_vertices':np.zeros([0, *obs_space['board']['vertices']]),
            'next_observation_edges':np.zeros([0, *obs_space['board']['edges']]),
            'next_observation_player':np.zeros([0, obs_space['total']]),
            'pidx':np.zeros([0]).astype(int)
        }

        return rollout

class GraphInitialPlacementComparisonCollector(GraphInitialPlacementCollector):
    """
    Special collector for Alphago-style self-play. We take in two Q-functions for eval and simulate games to get their relative strengths.
    """
    def get_rollouts(self, qf1, qf2, n):
        rollouts = [self.get_rollout(qf1, qf2) for _ in range(n)]
        out = self.setup_rollout_dict()

        if n == 0:
            return out

        wins = np.stack([r[1] for r in rollouts], axis=0).sum(axis=0)
        if wins.sum() > 0:
            wins /= wins.sum()
        for k in rollouts[0][0].keys():
            out[k] = torch.cat([r[0][k] for r in rollouts], dim=0)

        return out, wins 

    def get_rollout(self, qf1, qf2, rollout_n = 1):
        """
        Implement as follows:
            1. Save current player q functions.
            2. Replace 2 players with qf1 other two with qf2.
            3. Simulate n games and collect standard rollout info, but also collect win rates

        NOTE: we assume sparse reward structure given in the final rollout step.
        """
        x = np.random.permutation(np.arange(4))
        qf1_players = x[:2]
        qf2_players = x[2:]
        old_qfs = [p.network for p in self.simulator.players]

        for pidx in qf1_players:
            self.simulator.players[pidx].network = qf1
        for pidx in qf2_players:
            self.simulator.players[pidx].network = qf2

        rollout = super().get_rollout(rollout_n=rollout_n)
        scores = rollout['reward'].sum(axis=0)
        qf1_score = scores[qf1_players].sum()
        qf2_score = scores[qf2_players].sum()
        if scores.sum() > 0:
            qf1_score /= scores.sum()
            qf2_score /= scores.sum()
        wins = np.array([qf1_score, qf2_score])

        for pidx, qf in enumerate(old_qfs):
            self.simulator.players[pidx].network = qf

        return rollout, wins

if __name__ == '__main__':
    b = Board()
    b.reset()
    agents = [IndependentActionsAgent(b), IndependentActionsAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsAgent(b)]
#    agents = [IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b)]
#    agents = [HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b)]
    s = IndependentActionsCatanSimulator(board=b, players = agents, max_vp=10)
    placement_agents = [InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b)]
    placement_simulator = InitialPlacementSimulator(s, placement_agents)

    import pdb;pdb.set_trace()
    collector = GraphInitialPlacementCollector(placement_simulator)

    rollouts = collector.get_rollouts(10)
    print('done!') 
