import matplotlib.pyplot as plt
import numpy as np
import random

from catanbot.agents.base import Agent

class HeuristicAgent(Agent):
	"""
	A Catan-playing agent that acts according to a rough heuristic as follows.
	
	1. Build settlements/cities whenever possible
	2. Only build roads towards free space
	3. Buy dev cards whenever possible 

	"""

	def action(self):
		acts = self.compute_actions()

		#City the highest producing spot
		if (acts[:, 0] == 2).any():
			cacts = acts[acts[:, 0] == 2]
			clocs = cacts[:, 1]
			tiles = self.board.settlements[clocs, -3:]
			cvals = np.zeros(clocs.shape)
			for i in range(len(clocs)):
				t = tiles[i]
				t = t[t != -1]
				cvals[i] = (6-abs(7 - self.board.tiles[t, 1])).clip(0).sum()

			return cacts[np.argmax(cvals)]

		#Settlement on highest procducing spot
		if (acts[:, 0] == 1).any():
			cacts = acts[acts[:, 0] == 1]
			clocs = cacts[:, 1]
			tiles = self.board.settlements[clocs, -3:]
			cvals = np.zeros(clocs.shape)
			for i in range(len(clocs)):
				t = tiles[i]
				t = t[t != -1]
				cvals[i] = (6-abs(7 - self.board.tiles[t, 1])).clip(0).sum()

			return cacts[np.argmax(cvals)]

		#build roads towards available settlement spots, but only if you arent working toward a settlement
		if (acts[:, 0] == 3).any():
			racts = acts[acts[:, 0] == 3]
			rlocs = racts[:, 1]
			avail_spots = self.board.compute_settlement_spots()
			roads = self.board.roads[rlocs]
			r_vals = np.zeros(len(roads))
			for i in range(len(roads)):
				road = roads[i]
				avail = np.intersect1d(road[1:], avail_spots)
				val = 0
				for loc in avail:
					tloc = self.board.settlements[loc, -3:]
					tloc = tloc[tloc != -1]
					val += (6-abs(7 - self.board.tiles[tloc, 1])).clip(0).sum()
				r_vals[i] = val
			return racts[np.argmax(r_vals)]


		return random.choice(acts)
