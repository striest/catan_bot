import time

import matplotlib.pyplot as plt
import numpy as np
import random

from catanbot.constants import DEV, SETTLEMENT, CITY, ROAD

class InitialPlacementAgent:
    """
    An agent for intial placements
    """
    def __init__(self, board):
        self.board = board

    def action_mask(self):
        """
        Use board to get actions that are actually available.
        """
        act_mask = np.zeros([54, 3]).astype(bool)

        settlement_locs = np.zeros(54).astype(bool)
        settlement_locs[self.board.compute_settlement_spots()] = 1
        settlement_avail = np.expand_dims(settlement_locs, axis=1)

        road_choices = self.board.settlements[:, 5:8]
        road_avail = (road_choices != -1)
        road_open = self.board.roads[road_choices, 0] == 0

        act_mask = (road_avail & road_open & settlement_avail)

        return act_mask

    @property
    def action_space(self):
        return {
            'placement':np.array([54, 3]),
            'total':np.array(54*3)
        }

    def action(self, obs, flat_obs):
        return {
            'placement':np.random.rand(54, 3) * self.action_mask()
        }

    def action_determinsitic(self, obs, flat_obs):
        """
        NOTE: THIS ISNT ACTUALLY DETERMINISTIC
        """
        return {
            'placement':np.random.rand(54, 3) * self.action_mask()
        }

class HeuristicInitialPlacementAgent(InitialPlacementAgent):
    def action(self, obs, flat_obs):
        production = self.board.compute_production()[:, 1]
        prod = np.stack([production * 3], axis=1)
        return {
            'placement':prod * self.action_mask()
        }
        
class MakeDeterministic(InitialPlacementAgent):
    """
    Wraps a policy to take deterministic actions (expects determininsm to be implemented in the subclass)
    """
    def __init__(self, agent):
        super().__init__(agent.board)
        self.agent = agent
        self.agent.board = self.board

    def action(self, obs, flat_obs):
        self.agent.board = self.board #TODO: Need to find where the actual bug is. (agent board gets reset somewhere)
        return self.agent.action_deterministic(obs, flat_obs)
    
