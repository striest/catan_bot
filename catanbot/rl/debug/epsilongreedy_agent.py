import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import random

from catanbot.board_placement.agents.base import InitialPlacementAgent
from catanbot.constants import DEV, SETTLEMENT, CITY, ROAD

class EpsilonGreedyGridworldAgent:
    """
    An agent for gridworld.
    """
    def __init__(self, network, epsilon_scheduler, pidx):
        """
        Unlike the regular agent, we need access to board state to mask invalid actions.
        """
        self.pidx = pidx
        self.network = network
        self.epsilon_scheduler = epsilon_scheduler
        self.epoch = 0

    @property
    def action_space(self):
        return {
            'total':np.array(5)
        }

    def action_dist(self, obs, obs_flat):
        """
        Take the action that argmaxes Q with probability 1 unless epsilon is rolled.
        Also handle in batch.
        """
        inp = torch.tensor(obs_flat) if not isinstance(obs_flat, torch.Tensor) else obs_flat
        inp = inp.float()
        if len(inp.shape) == 1:
            inp = inp.unsqueeze(0)
        epsilon = self.epsilon_scheduler(self.epoch)
        rands = torch.rand(inp.shape[0])
        unif = torch.ones(5)
        unif /= unif.sum()

        qs = self.network(inp)[:, self.pidx]

        acts = torch.zeros(inp.shape[0], 5)
        acts[torch.arange(inp.shape[0]), torch.argmax(qs, dim=1)] = 1.
        acts[rands < epsilon] = unif

        dist = torch.distributions.OneHotCategorical(probs=acts)
        return dist

    def action(self, obs, obs_flat):
        """
        Take the action that argmaxes Q unless epsilon.
        """
        act_dists = self.action_dist(obs, obs_flat)
        act = act_dists.sample()
        return {
            'placement':act.detach().numpy()
        }
