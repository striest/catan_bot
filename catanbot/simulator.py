import matplotlib.pyplot as plt
import numpy as np
import random
import time

from catanbot.board import Board
from catanbot.agents.base import Agent
from catanbot.agents.heuristic_agent import HeuristicAgent
from catanbot.constants import DEV, SETTLEMENT, CITY, ROAD

class CatanSimulator:
	"""
	Actually simulates the Catan Game
	"""
	def __init__(self, board, players = None, max_vp = 10, max_steps=500):
		self.board = board
		if players is None:
			self.players = [Agent(self.board) for _ in range(4)]
		else:
			self.players = players
		for i in range(4):
			self.players[i].pidx = i
		self.turn = 0
		self.nsteps = 0
		self.max_steps = max_steps
		self.roll = -1
		self.vp = np.zeros(4)
		self.max_vp = max_vp

	def simulate(self, verbose=False):
		"""
		simulates a full game and returns the winner
		"""
		while not self.terminal:
			act = self.players[self.turn].action()
			if verbose:
				print('Act = {}'.format(act))
			self.step(act)
		return (self.vp == self.max_vp).astype(int)
	

	@property
	def terminal(self):
		return self.vp.max() >= self.max_vp or self.nsteps > self.max_steps

	def base_reset(self):
		self.board.reset()
		for p in self.players:
			p.reset()
		self.turn = 0
		self.nsteps = 0
		self.vp *= 0

	def reset_from(self, board, players=None, nsteps=0, resources=None):
		"""
		Start game from seed board (with initial placements)
		"""
		self.board = board
		if players is not None:
			self.players = players
		self.vp = self.vp * 0
		for i, p in enumerate(self.players):
			if players is None:	
				p.reset()
			self.vp[i] = board.settlements[board.settlements[:, 0] == i+1][:, 1].sum()
			p.board = board

		self.turn = 0
		self.nsteps = 0	
		self.assign_resources()

	def reset_with_initial_placements(self):
		self.base_reset()
		for i in range(8):
			s = np.random.choice(self.board.compute_settlement_spots())
			self.board.place_settlement(s, 1 + i%4, False)
			self.vp[i%4] += 1
			roads = self.board.settlements[s, 5:8]
			roads = roads[roads != -1]
			roads = roads[self.board.roads[roads][:, 0] == 0]
			self.board.place_road(np.random.choice(roads), 1 + i%4)
			if i > 3:
				tiles = self.board.settlements[s, 8:]
				tiles = tiles[tiles != -1]
				resources = self.board.tiles[tiles, 0]
				for r in resources:
					if r > 0:
						self.players[i%4].resources[r] += 1
		self.assign_resources()

	def step(self, action):
		pval = self.turn + 1
		avail_actions = self.compute_actions(self.turn)
		assert (action == avail_actions).all(axis=1).any(), 'invalid move'
		atype = action[0]
		aloc = action[1]
		acost = action[2:]

		#0=buy dev, 1=settlement, 2=city, 3=road, 4=pass
		if atype == 0:
			#print('bought a dev card (TODO: implement)')
			self.players[self.turn].resources -= acost
			card = self.board.get_dev_card()
			self.players[self.turn].dev[card] += 1
			if card == 2:#is VP
#				print('got VP')
				self.vp[self.turn] += 1
		elif atype == 1:
			self.board.place_settlement(aloc, pval, False)
			self.players[self.turn].resources -= acost
			self.vp[self.turn] += 1
		elif atype == 2:
			self.board.place_settlement(aloc, pval, True)
			self.players[self.turn].resources -= acost
			self.vp[self.turn] += 1
		elif atype == 3:
			self.board.place_road(aloc, pval)
			self.players[self.turn].resources -= acost

		self.turn = (self.turn + 1) % 4
		self.nsteps += 1
		self.assign_resources()
		
	def assign_resources(self):
		roll = np.random.randint(1, 7) + np.random.randint(1, 7)
		self.roll = roll
		resources = self.board.generate_resources(roll)
		for i, player in enumerate(self.players):
			player.resources += resources[i + 1]

	def compute_actions(self, pidx):
		"""
		Using a func to compute valid moves for a given player.
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
		player = self.players[pidx]

		costs = np.stack([self.trade_for(pidx, r) for r in (DEV, SETTLEMENT, CITY, ROAD)], axis=0)

		#Compute dev card
		if (player.resources - costs[0] >= 0).all() and self.board.has_dev_cards():
			moves.append(np.concatenate([np.array([0, 0]), costs[0]], axis=0))

		#Compute settlements
		if (player.resources - costs[1] >= 0).all():
			avail_spots = self.board.compute_settlement_spots()
			road_spots = self.board.roads[self.board.roads[:, 0] == 1 + pidx][:, 1:].flatten()
			valid_spots = np.intersect1d(avail_spots, road_spots)
			for spot in valid_spots:
				moves.append(np.concatenate([np.array([1, spot]), costs[1]], axis=0))

		#Compute cities
		if (player.resources - costs[2] >= 0).all():
			spots = np.argwhere((self.board.settlements[:, 0] == 1+pidx) & (self.board.settlements[:, 1] == 1)).flatten()
			for spot in spots:
				moves.append(np.concatenate([np.array([2, spot]), costs[2]], axis=0))

		#Compute roads
		if (player.resources - costs[3] >= 0).all():
			road_spots = np.argwhere(self.board.roads[:, 0] == 1 + pidx).flatten()
			settlement_spots = np.concatenate([np.argwhere((self.board.settlements[:, 5:8] == r).any(axis=1)) for r in road_spots]).flatten()
			one_hop_road_spots = np.concatenate([np.argwhere((self.board.roads[:, 1:3] == s).any(axis=1)) for s in settlement_spots]).flatten()
			avail_road_spots = np.argwhere(self.board.roads[:, 0] == 0).flatten()
			valid_road_spots = np.intersect1d(avail_road_spots, one_hop_road_spots)

			for spot in valid_road_spots:
				moves.append(np.concatenate([np.array([3, spot]), costs[3]], axis=0))

		return np.stack(moves, axis=0)
	
	def trade_for(self, pidx, target):
		"""
		Computes the most resource-efficient way for player at pidx to get the resources in target. Returns a big val if impossible.
		"""
		player = self.players[pidx]
		resource_diffs = player.resources - target
		cost = np.zeros(6).astype(int)
		if (resource_diffs >= 0).all():
			cost = target.copy()
		else:
			cost = target.copy()
			ncnt = -resource_diffs[resource_diffs < 0].sum()
			surpluses = (resource_diffs//player.trade_ratios).clip(0)
			pcnt = surpluses.sum()
			t_order = np.argsort(player.trade_ratios)
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
					cost[curr_r] += player.trade_ratios[curr_r]
					surpluses[curr_r] -= 1
					ntrades += 1
		return cost

	def render(self):
		fig, axs = plt.subplots(1, 3, figsize = (18, 5))
		plt.title('Turn = {} ({}), roll = {}'.format(self.nsteps, 'RGYB'[self.turn], self.roll))
		self.board.render_base(fig, axs[0], display_ids=True)
		self.board.render_pips(fig, axs[1])
		txt = ""
		for i in range(4):
			txt += 'Player {} ({}):\n'.format(i+1, 'RGYB'[i])
			txt += str(self.players[i])
			txt += '\n\n'

		txt = txt[:-1]
		txt += str(self.vp)
		txt += '\n'

		axs[2].text(0, 0, txt, fontsize=6)
		plt.show()

if __name__ == '__main__':
	import copy
	b = Board()
	agents = [HeuristicAgent(b), Agent(b), Agent(b), Agent(b)]
	agents = [HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b)]
	s = CatanSimulator(board=b, players = agents, max_vp=8)
	s.reset_with_initial_placements()
	s_c = copy.deepcopy(s)

	print('INIT')
	prod = s.board.compute_production()
	print(prod)
	print(prod[np.argsort(prod[:, 1])[-15:]])
	s.render()

	t = time.time()
	cnt = 0
	games = 50
	turns = 0
	wins = np.zeros(4)
	print(s.simulate())
	s.render()
	exit(0)
	while cnt < games:
		print('Game {}'.format(cnt + 1))
		print('Turn = {}, ({})'.format(s.nsteps+1, 'RGYB'[s.turn]))
		print('Resources = {}'.format(s.players[s.turn].resources))
		act = s.players[s.turn].action()
		print('Act = {}'.format(act))
		s.step(act)

		if s.terminal:
#			s.render()
			wins[np.argmax(s.vp)] += 1
			turns += s.nsteps
			#s.reset_with_initial_placements()
			s = copy.deepcopy(s_c)
			cnt += 1

#			s.render()
	
	print('Simulated {} games ({} turns) in {:.2f}s'.format(games, turns, time.time() - t))
	print('Win rates = {}'.format(wins))
	s.render()


