import matplotlib.pyplot as plt
import numpy as np
import random
import time

from board import Board
from agent import Agent
from constants import DEV, SETTLEMENT, CITY, ROAD

class CatanSimulator:
	"""
	Actually simulates the Catan Game
	"""
	def __init__(self):
		self.board = Board()
		self.players = [Agent(self.board) for _ in range(4)]
		self.turn = None
		self.nsteps = 0
		self.roll = -1
		self.vp = np.zeros(4)

	def base_reset(self):
		self.board.reset()
		for p in self.players:
			p.reset()
		self.turn = 0
		self.nsteps = 0
		self.vp *= 0

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

		#0=buy dev, 1=settlement, 2=city, 3=road, 4=pass
		if atype == 0:
			#print('bought a dev card (TODO: implement)')
			self.players[self.turn].resources -= DEV
		elif atype == 1:
			self.board.place_settlement(aloc, pval, False)
			self.players[self.turn].resources -= SETTLEMENT
			self.vp[self.turn] += 1
		elif atype == 2:
			self.board.place_settlement(aloc, pval, True)
			self.players[self.turn].resources -= CITY
			self.vp[self.turn] += 1
		elif atype == 3:
			self.board.place_road(aloc, pval)
			self.players[self.turn].resources -= ROAD

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
		Return actions as a list of two-tuples of move type, loc.
		"""
#		import pdb;pdb.set_trace()
		moves = [np.array([4, 0])]
		player = self.players[pidx]
		#Compute dev card
		if (player.resources - DEV >= 0).all():
			moves.append(np.array([0, 0]))

		#Compute settlements
		if (player.resources - SETTLEMENT >= 0).all():
			avail_spots = self.board.compute_settlement_spots()
			road_spots = self.board.roads[self.board.roads[:, 0] == 1 + pidx][:, 1:].flatten()
			valid_spots = np.intersect1d(avail_spots, road_spots)
			for spot in valid_spots:
				moves.append(np.array([1, spot]))

		#Compute cities
		if (player.resources - CITY >= 0).all():
			spots = np.argwhere((self.board.settlements[:, 0] == 1+pidx) & (self.board.settlements[:, 1] == 1)).flatten()
			for spot in spots:
				moves.append(np.array([2, spot]))

		#Compute roads
		if (player.resources - ROAD >= 0).all():
			road_spots = np.argwhere(self.board.roads[:, 0] == 1 + pidx).flatten()
			settlement_spots = np.concatenate([np.argwhere((self.board.settlements[:, 5:8] == r).any(axis=1)) for r in road_spots]).flatten()
			one_hop_road_spots = np.concatenate([np.argwhere((self.board.roads[:, 1:3] == s).any(axis=1)) for s in settlement_spots]).flatten()
			avail_road_spots = np.argwhere(self.board.roads[:, 0] == 0).flatten()
			valid_road_spots = np.intersect1d(avail_road_spots, one_hop_road_spots)

			for spot in valid_road_spots:
				moves.append(np.array([3, spot]))

		return np.stack(moves, axis=0)

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

		axs[2].text(0, 0, txt, fontsize=8)
		plt.show()

if __name__ == '__main__':
	s = CatanSimulator()
	s.reset_with_initial_placements()
	for i in range(4):
		s.players[i].resources += 1

	print('INIT')
	s.render()

	t = time.time()
	cnt = 0
	games = 1000
	while cnt < games:
		print('Game {}'.format(cnt + 1))
		print('Turn = {}, ({})'.format(s.nsteps+1, 'RGYB'[s.turn]))
		acts = s.compute_actions(s.turn)
		#print('Acts = {}'.format(acts))
		act = random.choice(acts)
		#print('Act = {}'.format(act))
		s.step(act)
		s.players[s.turn].resources += 1

		if s.vp.max() >= 10:
			s.reset_with_initial_placements()
			cnt += 1
	
	print('Simulated {} turns in {:.2f}s'.format(s.nsteps, time.time() - t))
	s.render()


