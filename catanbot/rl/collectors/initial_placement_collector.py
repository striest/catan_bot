import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
import copy

from catanbot.rl.collectors.base import Collector
from catanbot.core.independent_action_simulator import IndependentActionsCatanSimulator
from catanbot.core.board import Board
from catanbot.agents.independent_actions_agents import IndependentActionsHeuristicAgent, IndependentActionsAgent
from catanbot.constants import DEV, SETTLEMENT, CITY, ROAD

class InitialPlacementCollector(Collector):
    """
    Rollout collector for initial placements. It's identical to base except for action space.
    """
    def __init__(self, simulator, reward_scale = 1.0, reset_board=False):
        self.simulator = simulator
        self.reset_board = reset_board
        self.reward_scale = reward_scale

    def get_rollout(self, rollout_n = 10, i=0, verbose=False):
        self.simulator.reset(reset_board=self.reset_board)

        rollout = self.setup_rollout_dict()

        while not self.simulator.terminal:
            obs = self.simulator.observation
            obs_flat = self.flatten_observation(obs)
            pidx = self.simulator.player_idx
            act = self.simulator.players[pidx].action(obs, obs_flat)
            self.simulator.step(act)
            nobs = self.simulator.observation
            rew = self.simulator.reward(rollout_n) * self.reward_scale
            term = self.simulator.terminal

            act_flat = self.flatten_action(act)
            nobs_flat = self.flatten_observation(nobs)

            for k, v in zip(['observation', 'action', 'reward', 'terminal', 'next_observation', 'pidx'], [obs_flat, act_flat, rew, term, nobs_flat, pidx]):
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
        #NOTE: Placeholder to just use board state
        board_obs_flat = np.concatenate([obs['board'][k].flatten() for k in ['tiles', 'roads', 'settlements']])
        player_obs_flat = np.concatenate([obs['player'][k].flatten() for k in ['pidx', 'resources', 'dev', 'trade_ratios']])
        obs_flat = np.concatenate([board_obs_flat, player_obs_flat, obs['vp'], obs['army'], obs['road']])
        return obs_flat

class DebugCollector(InitialPlacementCollector):
    """
    Collector for debugging. Cycles between a few random boards.
    """
    def __init__(self, simulator, reward_scale = 1.0, reset_board=False, nboards = 1):
        self.simulator = simulator
        self.reset_board = reset_board
        self.reward_scale = reward_scale
        self.stored_boards = []
        self.nboards = nboards
        for i in range(nboards):
            self.simulator.reset(reset_board=True)
            self.stored_boards.append(copy.deepcopy(self.simulator.simulator.board))
            self.simulator.render()
    
    def get_rollout(self, rollout_n = 10, i=0, verbose=False):
        if i % 100 == 0:
            print('Rollout {}'.format(i))

        idx = np.random.randint(0, self.nboards)
#        self.simulator.initial_board = np.random.choice(self.stored_boards)
        self.simulator.initial_board = self.stored_boards[idx]
        self.simulator.reset(reset_board=False)

        rollout = self.setup_rollout_dict()

        while not self.simulator.terminal:
            obs = self.simulator.observation
            obs_flat = self.flatten_observation(obs)

            """
            obs_flat = 1 + np.random.normal(scale=0.1, size=obs_flat.shape)
            nobs_flat = 1 + np.random.normal(scale=0.1, size=obs_flat.shape)

            if idx:
                obs_flat[:600] = 0
                nobs_flat[:600] = 0
            else:
                obs_flat[600:] = 0
                nobs_flat[600:] = 0
            """

            pidx = self.simulator.player_idx
            act = self.simulator.players[pidx].action(obs, obs_flat)
            self.simulator.step(act)
            nobs = self.simulator.observation
            rew = self.simulator.reward(rollout_n) * self.reward_scale
            term = self.simulator.terminal

            act_flat = self.flatten_action(act)
            nobs_flat = self.flatten_observation(nobs)

#            obs_flat[:act_flat.shape[0]] = act_flat
#            nobs_flat[:act_flat.shape[0]] = act_flat

            for k, v in zip(['observation', 'action', 'reward', 'terminal', 'next_observation', 'pidx'], [obs_flat, act_flat, rew, term, nobs_flat, pidx]):
                rollout[k] = np.append(rollout[k], [v], axis=0)

        for k in rollout.keys():
            rollout[k] = torch.tensor(rollout[k])
            if k != 'pidx':
                rollout[k] = rollout[k].float()

        if verbose:
            print('Finished rollout {}'.format(i), end='\r')
            sys.stdout.flush()

        return rollout

if __name__ == '__main__':
    b = Board()
    b.reset()
    agents = [IndependentActionsAgent(b), IndependentActionsAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsAgent(b)]
#    agents = [IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b)]
#    agents = [HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b)]
    s = IndependentActionsCatanSimulator(board=b, players = agents, max_vp=10)
    s.reset_from(b)

    collector = Collector(s)

    import pdb;pdb.set_trace()
    rollouts = collector.get_rollouts(10)
    print('done!') 
