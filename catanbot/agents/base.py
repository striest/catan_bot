import matplotlib.pyplot as plt
import numpy as np
import random

from catanbot.constants import DEV, SETTLEMENT, CITY, ROAD

class Agent:
	"""
	A Catan-playing agent
	State:
		Resource cards (1 = ore, 2 = wheat, 3 = sheep, 4 = wood, 5 = brick)
		Dev cards (1 = knight, 2 = VP, 3 = year of plenty, 4 = road building, 5 = monopoly)
	Action space:
		build settlement/city on a settlement spot
		build road on a free road spot
		buy a dev card
		play a dev card
		I think it's safe to implement actions as play dev and then purchase.
	"""

	def __init__(self, board):
		self.board = board
		self.resources = np.zeros(6).astype(int)
		self.dev = np.zeros(6).astype(int)
		self.trade_ratios = np.ones(6).astype(int) * 4
		self.trade_ratios[0] = 100000
		self.pidx = -1

	def reset(self):
		self.resources *= 0
		self.dev *= 0
		self.trade_ratios = np.ones(6) * 4
		self.trade_ratios[0] = 1000000

	def action(self):
		acts = self.compute_actions()
#		print('Actions =\n{}'.format(acts))
		return random.choice(acts)

	def __repr__(self):
		r = {a:b for a, b in zip(['Ore', 'Wheat', 'Sheep', 'Wood', 'Brick'], self.resources[1:])}
		d = {a:b for a, b in zip(['Knight', 'VP', 'YOP', 'RB', 'M'], self.dev[1:])}
		tr = {a:b for a, b in zip(['Ore', 'Wheat', 'Sheep', 'Wood', 'Brick'], self.trade_ratios[1:])}
		return 'RESOURCES:\n{}\nDEV:\n{}\nRATIOS:\n{}\n'.format(r, d, tr)
	
	def compute_actions(self):
		"""
		Using a func to compute valid moves for a given self.
		Moves are:
		0. Buy a dev card
		1. Place a settlement
		2. Place a city
		3. Place a road
		4. Pass
		Obviously, this is all dependent on resources, available spots, etc.
		Also, I'm consodering playing dev cards as a different phase.
		Return actions as a matrix where row are [move type, loc, cost]
		"""
		moves = [np.array([4, 0, 0, 0, 0, 0, 0, 0])]

		costs = np.stack([self.trade_for(r) for r in (DEV, SETTLEMENT, CITY, ROAD)], axis=0)

		#Compute dev card
		if (self.resources - costs[0] >= 0).all() and self.board.has_dev_cards():
			moves.append(np.concatenate([np.array([0, 0]), costs[0]], axis=0))

		#Compute settlements
		if (self.resources - costs[1] >= 0).all():
			avail_spots = self.board.compute_settlement_spots()
			road_spots = self.board.roads[self.board.roads[:, 0] == 1 + self.pidx][:, 1:].flatten()
			valid_spots = np.intersect1d(avail_spots, road_spots)
			for spot in valid_spots:
				moves.append(np.concatenate([np.array([1, spot]), costs[1]], axis=0))

		#Compute cities
		if (self.resources - costs[2] >= 0).all():
			spots = np.argwhere((self.board.settlements[:, 0] == 1+self.pidx) & (self.board.settlements[:, 1] == 1)).flatten()
			for spot in spots:
				moves.append(np.concatenate([np.array([2, spot]), costs[2]], axis=0))

		#Compute roads
		if (self.resources - costs[3] >= 0).all():
			road_spots = np.argwhere(self.board.roads[:, 0] == 1 + self.pidx).flatten()
			settlement_spots = np.concatenate([np.argwhere((self.board.settlements[:, 5:8] == r).any(axis=1)) for r in road_spots]).flatten()
			one_hop_road_spots = np.concatenate([np.argwhere((self.board.roads[:, 1:3] == s).any(axis=1)) for s in settlement_spots]).flatten()
			avail_road_spots = np.argwhere(self.board.roads[:, 0] == 0).flatten()
			valid_road_spots = np.intersect1d(avail_road_spots, one_hop_road_spots)

			for spot in valid_road_spots:
				moves.append(np.concatenate([np.array([3, spot]), costs[3]], axis=0))

		return np.stack(moves, axis=0)
	
	def trade_for(self, target):
		"""
		Computes the most resource-efficient way for self at pidx to get the resources in target. Returns a big val if impossible.
		"""
		resource_diffs = self.resources - target
		cost = np.zeros(6).astype(int)
		if (resource_diffs >= 0).all():
			cost = target.copy()
		else:
			cost = target.copy()
			ncnt = -resource_diffs[resource_diffs < 0].sum()
			surpluses = (resource_diffs//self.trade_ratios).clip(0)
			pcnt = surpluses.sum()
			t_order = np.argsort(self.trade_ratios)
			cost[resource_diffs < 0] = 0
			if pcnt < ncnt:
				cost += 100000
			else:
				ntrades = 0
				tidx = 0
				while ntrades < ncnt:
					curr_r = t_order[tidx]
					if surpluses[curr_r] <= 0:
						tidx += 1
						continue
					cost[curr_r] += self.trade_ratios[curr_r]
					surpluses[curr_r] -= 1
					ntrades += 1
		return cost
