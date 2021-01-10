import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import random

from catanbot.board_placement.agents.base import InitialPlacementAgent
from catanbot.constants import DEV, SETTLEMENT, CITY, ROAD

class EpsilonGreedyPlacementAgent(InitialPlacementAgent):
    """
    An agent for intial placements
    """
    def __init__(self, board, network, epsilon_scheduler, pidx):
        """
        Unlike the regular agent, we need access to board state to mask invalid actions.
        """
        self.pidx = pidx
        self.board = board
        self.network = network
        self.epsilon_scheduler = epsilon_scheduler
        self.epoch = 0

    @property
    def action_space(self):
        return {
            'placement':np.array([54, 3]),
            'total':np.array(54*3)
        }

    def action_dist(self, obs, obs_flat):
        """
        Take the action that argmaxes Q with probability 1 unless epsilon is rolled.
        Also handle in batch.
        """
        mask = torch.tensor(self.action_mask()).flatten()
        inp = torch.tensor(obs_flat) if not isinstance(obs_flat, torch.Tensor) else obs_flat
        inp = inp.float()
        if len(inp.shape) == 1:
            inp = inp.unsqueeze(0)
        epsilon = self.epsilon_scheduler(self.epoch)
        rands = torch.rand(inp.shape[0])
        unif = (torch.ones(54*3) * mask)
        unif /= unif.sum()

        qs = self.network(inp)[:, self.pidx]
        qs[:, ~mask] = -1e6 #Don't pick invalid actions

        acts = torch.zeros(inp.shape[0], 54*3)
        acts[torch.arange(inp.shape[0]), torch.argmax(qs, dim=1)] = 1.
        acts[rands < epsilon] = unif

        dist = torch.distributions.OneHotCategorical(probs=acts)
        return dist

    def action(self, obs, obs_flat):
        """
        Take the action that argmaxes Q unless epsilon.
        """
        act_dists = self.action_dist(obs, obs_flat)
        act = act_dists.sample().view(54, 3)
        return {
            'placement':act.detach().numpy()
        }

class GNNEpsilonGreedyPlacementAgent(EpsilonGreedyPlacementAgent):
    """
    Different class bc graph input is a little different.
    """
    def action_dist(self, obs, obs_flat):
        mask = torch.tensor(self.action_mask()).flatten()
        inp = {k:v.float() if isinstance(v, torch.Tensor) else torch.tensor(v).float() for k, v in obs_flat.items()}

        #TODO: Fix to assert dim=3 instead of 2
        if len(inp['vertices'].shape) == 2:
            inp['vertices'] = inp['vertices'].unsqueeze(0)
        if len(inp['edges'].shape) == 2:
            inp['edges'] = inp['edges'].unsqueeze(0)
        if len(inp['player'].shape) == 1:
            inp['player'] = inp['player'].unsqueeze(0)

        epsilon = self.epsilon_scheduler(self.epoch)
        rands = torch.rand(inp['vertices'].shape[0])
        unif = (torch.ones(54*3) * mask)
        unif /= unif.sum()

        qs = self.network(inp)[:, self.pidx]
        qs[:, ~mask] = -1e6 #Don't pick invalid actions

        acts = torch.zeros(inp['vertices'].shape[0], 54*3)
        acts[torch.arange(inp['vertices'].shape[0]), torch.argmax(qs, dim=1)] = 1.
        acts[rands < epsilon] = unif

        dist = torch.distributions.OneHotCategorical(probs=acts)
        return dist 
        
class EpsilonGreedyPlacementAgentWithMask(EpsilonGreedyPlacementAgent):
    """
    Restrict available actions to be above a certain production threshold (prob 7+)
    """
    def __init__(self, board, network, epsilon_scheduler, pidx, prod_threshold=7):
        """
        Unlike the regular agent, we need access to board state to mask invalid actions.
        """
        self.pidx = pidx
        self.board = board
        self.network = network
        self.epsilon_scheduler = epsilon_scheduler
        self.epoch = 0
        self.production_threshold = prod_threshold

    def action_dist(self, obs, obs_flat):
        """
        Take the action that argmaxes Q with probability 1 unless epsilon is rolled.
        Also handle in batch.
        OK, so big thing is that the production-masked action only works if the agent's board is correctly updated.
        So only mask if not batch.
        """
        production_mask = self.board.compute_production()[:, 1] > self.production_threshold
        production_mask = torch.tensor(production_mask).unsqueeze(1).repeat(1, 3).flatten()
        mask = torch.tensor(self.action_mask()).flatten()
        inp = torch.tensor(obs_flat) if not isinstance(obs_flat, torch.Tensor) else obs_flat
        inp = inp.float()
        if len(inp.shape) == 1:
            inp = inp.unsqueeze(0)
            if torch.any(production_mask & mask):
                mask = mask & production_mask
        epsilon = self.epsilon_scheduler(self.epoch)
        rands = torch.rand(inp.shape[0])
        unif = (torch.ones(54*3) * mask)
        unif /= unif.sum()

        qs = self.network(inp)[:, self.pidx]
        qs[:, ~mask] = -1e6 #Don't pick invalid actions

        acts = torch.zeros(inp.shape[0], 54*3)
        acts[torch.arange(inp.shape[0]), torch.argmax(qs, dim=1)] = 1.
        acts[rands < epsilon] = unif

        dist = torch.distributions.OneHotCategorical(probs=acts)
        return dist
