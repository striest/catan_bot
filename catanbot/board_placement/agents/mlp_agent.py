import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import random

from torch.distributions import OneHotCategorical

from catanbot.rl.networks.mlp import MLP
from catanbot.board_placement.agents.base import InitialPlacementAgent
from catanbot.constants import DEV, SETTLEMENT, CITY, ROAD

class MLPPlacementAgent(InitialPlacementAgent):
    """
    An agent for intial placements
    """
    def __init__(self, board, network):
        """
        Unlike the regular agent, we need access to board state to mask invalid actions.
        """
        self.board = board
        self.network = network

    @property
    def action_space(self):
        return {
            'placement':np.array([54, 3]),
            'total':np.array(54*3)
        }


    def action_dist(self, obs, obs_flat):
        inp = torch.tensor(obs_flat).float()
        if len(inp.shape) == 1:
            inp = inp.unsqueeze(0)
        out = self.network(inp)
        out = torch.nn.functional.softmax(out, dim=1)
        out = out.view(-1, 54, 3)
        mask = torch.tensor(self.action_mask())
        out = (out * mask) + (1e-8 * mask) #Force valid actions to have a small positive probablity
        out = out.flatten(start_dim=1).squeeze()
        dist = OneHotCategorical(probs = out)

        return dist

    def action(self, obs, obs_flat):
        dist = self.action_dist(obs, obs_flat)
        act = dist.sample().view(54, 3)
        return {
            'placement':act.detach().numpy()
        }
        
    def action_deterministic(self, obs, obs_flat):
        dist = self.action_dist(obs, obs_flat)
        act = torch.zeros(dist.probs.shape)
        if len(obs_flat.shape) == 1:
            act[dist.probs.argmax()] = 1.
        else:
            act[torch.arange(len(dist.probs)), dist.probs.argmax(dim=1)] = 1.

        act = act.view(54, 3)
        return {
            'placement':act.detach().numpy()
        }
        
