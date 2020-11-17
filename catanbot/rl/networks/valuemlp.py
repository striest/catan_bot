import gym
import torch
from torch import nn

from catanbot.rl.networks.mlp import MLP

class InitialPlacementsQMLP(MLP):
    """
    For initial placements only. Predict the Q value of each action on the board.
    output[i, j] is the Q value of building on the i-th settlement spot in direction j.
    Note: I'm not reshaping to 54 x 3 here.
    """
    def __init__(self, env, hiddens = [32, 32], hidden_activation = nn.Tanh, logscale=False, scale = 1.0, bias=None, gpu=False):
        self.obs_dim = env.observation_space['total']
        self.logscale = logscale
        self.scale = scale
        self.action_mask = env.players[0].action_mask().flatten()

        super(InitialPlacementsQMLP, self).__init__(self.obs_dim, 4*54*3, hiddens, hidden_activation, bias=bias, gpu=gpu)

    def forward(self, obs):
        val = super().forward(obs).squeeze()
        val = val.view(-1, 4, 54*3)
        val[:, :, ~self.action_mask] = -1e6
        if self.logscale:
            return torch.exp(val) * self.scale
        else:
            return val * self.scale

class InitialPlacementsDoubleQMLP(nn.Module):
    """
    Use the TD3 min of 2 Q functions trick to reduce overestimation.
    """
    def __init__(self, qf1, qf2):
        super(InitialPlacementsDoubleQMLP, self).__init__()

        self.qf1 = qf1
        self.qf2 = qf2

        self.logscale = self.qf1.logscale
        self.scale = self.qf1.scale
        self.gpu = self.qf1.gpu
        self.action_mask = self.qf1.action_mask

    def forward(self, x):
        y1 = self.qf1(x)
        y2 = self.qf2(x)

        y = torch.minimum(y1, y2)
        return y

class VMLP(MLP):
    """
    Unlike traditional V network, now we estimate V for wach player.
    """
    def __init__(self, env, hiddens = [32, 32], hidden_activation = nn.Tanh, logscale=False, scale = 1.0, bias=None, gpu=False):
        self.obs_dim = env.observation_space['total']
        self.logscale = logscale
        self.scale = scale

        super(VMLP, self).__init__(self.obs_dim, 4, hiddens, hidden_activation, bias=bias, gpu=gpu)

    def forward(self, obs):
        val = super().forward(obs).squeeze()
        if self.logscale:
            return torch.exp(val) * self.scale
        else:
            return val * self.scale

