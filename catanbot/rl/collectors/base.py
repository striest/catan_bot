import ray
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

from catanbot.core.independent_action_simulator import IndependentActionsCatanSimulator
from catanbot.core.board import Board
from catanbot.agents.independent_actions_agents import IndependentActionsHeuristicAgent, IndependentActionsAgent
from catanbot.constants import DEV, SETTLEMENT, CITY, ROAD

class ParallelizeCollector:
    """
    Wrap collectors such that they run in parallel.
    """
    def __init__(self, collector, nthreads):
        ray.init(ignore_reinit_error=True)
        self.workers = []
        self.nthreads = nthreads
        self.collector = collector
        for _ in range(self.nthreads):
            self.workers.append(CollectorWorker.remote(self.collector))

    def get_rollouts(self, n, rollout_kwargs={}):
        tasks = []
        for i in range(n):
            tasks.append(self.workers[i % self.nthreads].get_rollout.remote(collector=self.collector, i=i, rollout_kwargs=rollout_kwargs))

        out = self.collector.setup_rollout_dict()

        if n == 0:
            return out

        rollouts = ray.get(tasks)
        for k in rollouts[0].keys():
            temp = [r for r in rollouts]
            out[k] = torch.cat([r[k] for r in rollouts], dim=0)

        return out

@ray.remote
class CollectorWorker:
    def __init__(self, collector):
        self.collector = copy.deepcopy(collector)

    def get_rollout(self, collector, i, rollout_kwargs):
        collector_copy = copy.deepcopy(collector)
        return collector_copy.get_rollout(**rollout_kwargs)

class Collector:
    """
    Rollout collector for Catan games. While it's similar to regular RL, there are some bookkeeping things that are different.
    """
    def __init__(self, simulator, reward_scale, reset_board=False):
        self.simulator = simulator
        self.board_copy = copy.deepcopy(self.simulator.board) #Keep a copy of the board if we need to run a lot of rollouts from the same board.
        self.reset_board = reset_board
        self.reward_scale = reward_scale

    def get_rollouts(self, n):
        rollouts = [self.get_rollout() for _ in range(n)]
        out = self.setup_rollout_dict()

        if n == 0:
            return out

        for k in rollouts[0].keys():
            temp = [r for r in rollouts]
            out[k] = torch.cat([r[k] for r in rollouts], dim=0)

        return out

    def get_rollout(self):
        """
        A rollout should be a dictionary (with sub-dictionaries) of the following
        Observation:
            board
            player
            vp
            army
            road
        Action:
            settlements
            roads
            tiles
            dev
        Reward:
            A 4-tensor
        Terminal:
            Bool
        Next Observation:
            board
            player
            vp
            army
            road
        Ok, so the gross thing is that we have dicts of dicts of dicts and we need the tensors to be inside all of them. Flatten here.
        """
        if self.reset_board:
            self.simulator.reset_with_initial_placements()
        else:
            self.simulator.reset_from(self.board_copy)
            self.simulator.initial_placements()

        rollout = self.setup_rollout_dict()

        while not self.simulator.terminal:
            obs = self.simulator.observation
            pidx = self.simulator.turn
            obs_flat = self.flatten_observation(obs)
            act = self.simulator.players[self.simulator.turn].action()
            self.simulator.step(act)
            nobs = self.simulator.observation
            rew = self.simulator.reward * self.reward_scale
            term = self.simulator.terminal

            act_flat = self.flatten_action(act)
            nobs_flat = self.flatten_observation(nobs)

            for k, v in zip(['observation', 'action', 'reward', 'terminal', 'next_observation', 'pidx'], [obs_flat, act_flat, rew, term, nobs_flat, pidx]):
                rollout[k] = np.append(rollout[k], [v], axis=0)

        for k in rollout.keys():
            rollout[k] = torch.tensor(rollout[k]).float()

        return rollout

    def flatten_action(self, act):
        act_flat = np.concatenate([act[k].flatten() for k in ['settlements', 'roads', 'tiles', 'dev']])
        return act_flat

    def flatten_observation(self, obs):
        board_obs_flat = np.concatenate([obs['board'][k].flatten() for k in ['tiles', 'roads', 'settlements']])
        player_obs_flat = np.concatenate([obs['player'][k].flatten() for k in ['pidx', 'resources', 'dev', 'trade_ratios']])
        obs_flat = np.concatenate([board_obs_flat, player_obs_flat, obs['vp'], obs['army'], obs['road']])
        return obs_flat

    def setup_rollout_dict(self):
        obs_space = self.simulator.observation_space
        act_space = self.simulator.action_space

        rollout = {
            'observation':np.zeros([0, obs_space['total']]),
            'action':np.zeros([0, act_space['total']]),
            'reward':np.zeros([0, 4]),
            'terminal':np.zeros(0).astype(bool),
            'next_observation':np.zeros([0, obs_space['total']]),
            'pidx':np.zeros([0]).astype(int)
        }

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
