import gym
import numpy as np
import torch
from torch import nn

from catanbot.rl.networks.mlp import MLP

class GridworldQMLP(MLP):
    """
    Basically a copy of the initial placements QMLP
    """
    def __init__(self, env, hiddens = [32, 32], hidden_activation = nn.Tanh, logscale=False, scale = 1.0, bias=None, gpu=False):
        self.obs_dim = env.observation_space['total']
        self.logscale = logscale
        self.scale = scale
        self.action_mask = np.zeros(5).astype(bool)

        super(GridworldQMLP, self).__init__(self.obs_dim, 4*5, hiddens, hidden_activation, bias=bias, gpu=gpu)

    def forward(self, obs):
        val = super().forward(obs).squeeze()
        val = val.view(-1, 4, 5)
        if self.logscale:
            return torch.exp(val) * self.scale
        else:
            return val * self.scale
