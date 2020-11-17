import matplotlib.pyplot as plt
import numpy as np
import random

from catanbot.agents.base import Agent

class IndependentActionsAgent(Agent):
    """
    Agent for hte new action space that takes random actions.
    """
    def action(self):
        return {
            'settlements':np.random.rand(54),
            'roads':np.random.rand(72),
            'dev':np.random.rand(1),
            'tiles':np.random.rand(19)
        }

    @property
    def action_space(self):
        return {
            'settlements':np.array(54),
            'roads':np.array(72),
            'tiles':np.array(19),
            'dev':np.array(1),
            'total':np.array(54+72+19+1)
        }

class IndependentActionsHeuristicAgent(Agent):
    """
    The same heuristic agent but for the new action space.
    Behavior is as follows:
        1. Build settlements/cities whenever possible.
        2. Only build roads toward free space
        3. Buy dev cards more when you have more cards.
        4. Place the robber on the highest-producing tile you're not on.
    """
    def action(self):
#        import pdb;pdb.set_trace()
        city_act = np.zeros(54)

        prods = self.board.compute_production()[:, 1]
        prod_scores = (prods ** 2) / (15. ** 2)

        port_scores = self.board.settlements[:, 2] > 0
        settlement_scores = prod_scores + 0.5 * port_scores + 0.5 #Add 0.5 because you always want to build more.

        #TODO: Come up with road scores
        #Build towards the highest-value open settlement spot
        #Should also weigh number of roads
        open_spots = self.board.compute_settlement_spots()
        scores = np.zeros(54)
        scores[open_spots] = settlement_scores[open_spots]

        #We can get the approx value of a spot using an iterative diffusion-type thing
        #Every settlement gives a discounted score to its neighbor
        neighbors = np.copy(self.board.settlement_neighbors)

        discount = 0.8
        n_itrs = 4
        for _ in range(n_itrs):
            neighbor_scores = scores[neighbors]
            neighbor_scores[neighbors == -1] = 0.
            scores = np.mean(neighbor_scores, axis=1)
            scores[open_spots] = settlement_scores[open_spots]

        road_scores = np.mean(scores[self.board.roads[:, 1:]], axis=1) + 0.4

        #TODO: This heuristic should weight the number of players on a given resource.
        o = self.board.settlements[self.board.settlements[:, 0] == 1+self.pidx][:, -3:].flatten()
        o = np.unique(o[o > -1])
        occupied_tiles = np.zeros(19)
        occupied_tiles[o] = 1.

        tile_scores = np.zeros(19)
        tile_prods = np.clip(6 - np.abs(7 - self.board.tiles[:, 1]), 0, 5)
        for tile_idx in self.board.settlements[:, 8:].T:
            mask = (tile_idx != -1)
            settlement_val = np.copy(self.board.settlements[:, 1])
            settlement_val[self.board.settlements[:, 0] == 1+self.pidx] = 0.
            settlement_prods = tile_prods[tile_idx] * settlement_val * mask
            for t, p in zip(tile_idx, settlement_prods):
                tile_scores[t] += p

        tile_scores /= (0.1 + np.sum(tile_scores))
        tile_scores *= (1 - occupied_tiles)

        if np.argmax(self.board.tiles[:, 4]) in occupied_tiles:
            tile_scores += 0.5

        dev_score = np.array([((self.resources.sum() - 2)/5).clip(0, 1)])

        return {
            'settlements':settlement_scores,
            'roads':road_scores,
            'tiles':tile_scores,
            'dev':dev_score
        }
