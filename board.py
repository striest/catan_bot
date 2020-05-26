import matplotlib.pyplot as plt
import numpy as np

class Board:
	"""
	Base class for catan board. Should store positions of roads, settlements, ports, and production tiles
	I think the best option is to put all the positions in numpy arrays and keep track of each entitiy's state

	For reference, I'll use the following for resources:
	0: Desert
	1: Ore
	2: Wheat
	3: Sheep
	4: Wood
	5: Brick

	For players:
	1: Red
	2: Bue
	3: Yellow
	4: Green

	For Settlements:
	1: Settlement
	2: City
	"""
	def __init__(self):
		self.tiles = np.zeros((19, 4)).astype(int) #resouece type, dice val, x, y
		self.roads = np.zeros((72, 3)).astype(int) #occupied_player, settlement 1, settlement 2
		self.settlements = np.zeros((54, 11)).astype(int) #player, settlement type, port, x_pos, y_pos, r1, r2, r3, t1, t2, t3

		self.tiles[:, 2] = np.array([6, 12, 18, 3, 9, 15, 21, 0, 6, 12, 18, 24, 3, 9, 15, 21, 6, 12, 18])
		self.tiles[:, 3] = np.array([24, 24, 24, 18, 18, 18, 18, 12, 12, 12, 12, 12, 6, 6, 6, 6, 0, 0, 0])

		self.settlements[:, 3] = np.array([6, 12, 18, 3, 9, 15, 21, 0, 6, 12, 18, 24, 3, 9, 15, 21, 27, 6, 12, 18, 24, 9, 15, 21, 9, 15, 21, 6, 12, 18, 24, 3, 9, 15, 21, 27, 0, 6, 12, 18, 24, 3, 9, 15, 21, 6, 12, 18, 3, 0, -3, -3, 0, 3])
		self.settlements[:, 4] = np.array([28, 28, 28, 22, 22, 22, 22, 16, 16, 16, 16, 16, 10, 10, 10, 10, 10, 4, 4, 4, 4, -2, -2, -2, 26, 26, 26, 20, 20, 20, 20, 14, 14, 14, 14, 14, 8, 8, 8, 8, 8, 2, 2, 2, 2, -4, -4, -4, 26, 20, 14, 10, 4, -2])
		self.settlements[:, 5] = np.array([-1, -1, -1, 48, 49, 50 ,51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 64, 65, 66, 67, 69, 70, 71, 1, 2, -1, 4, 5, 6, -1, 8, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 3, 7, 57, 63, 68])
		self.settlements[:, 6] = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, -1, 42, 43, 44, -1, 46, 47, -1, 49, 50, 51, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, -1, -1, -1, 48, 52, 57, 36, 41, 45])
		self.settlements[:, 7] = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, -1, -1, -1, -1, -1, -1, ])
		self.settlements[:, 8] = np.array([-1, -1, -1, 0, 1, 2, -1, 3, 4, 5, 6, -1, 8, 9, 10, 11, -1, 13, 14, 15, -1, 17, 18, -1, -1, -1, -1, 0, 1, 2, -1, 3, 4, 5, 6, -1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, -1, -1, -1, 7, 12, 16])
		self.settlements[:, 9] = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, 16, 17, 18, -1, -1, -1, -1, 1, 2, -1, 4, 5, 6, -1, 8, 9, 10, 11, -1, 12, 13, 14, 15, -1, 16, 17, 18, -1, -1, -1, -1, 0, 3, 7, -1, -1, -1])
		self.settlements[:, 10] = np.array([-1, -1, -1, -1, 0, 1, 2, -1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, 12, 13, 14, 15, -1, 16, 17, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1])

		self.roads[:, 1] = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 36, 12, 13, 14, 15, 41, 17, 18, 19, 45, 21, 22, 3, 4, 5, 6, 7, 8, 9, 10, 11, 50, 12, 13, 14, 15, 16, 36, 17, 18, 19, 20, 41, 21, 22, 23])
		self.roads[:, 2] = np.array([48, 24, 25, 49, 27, 28, 29, 50, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 51, 37, 38, 39, 40, 52, 42, 43, 44, 53, 46, 47, 48, 24, 25, 26, 49, 27, 28, 29, 30, 51, 31, 32, 33, 34, 35, 52, 37, 38, 39, 40, 53, 42, 43, 44])
		
		self.tile_dist = np.array([0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5])
		self.value_dist = np.array([2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12])

			
		
	def reset(self):
		tile_idxs = np.random.permutation(19)
		value_idxs = np.random.permutation(18)
		self.tiles[:, 0] = self.tile_dist[tile_idxs]
		self.tiles[:, 1][self.tiles[:, 0] != 0] = self.value_dist[value_idxs]
		self.roads[:, 0] *= 0
		self.settlements[:, [0, 1, 2]] *= 0

	def generate_resources(self, val):
		"""
		Computes resources for all players given a particular roll
		Returns as a 4x5 array where A[i, j] is the amount of resource j that player i gets
		"""
		out = np.zeros((5, 6)).astype(int)
		r_tiles = np.argwhere(self.tiles[:, 1] == val)
		for r_idx in r_tiles:
			s = self.settlements[(self.settlements[:, 8:] == r_idx).any(axis=1)]
			r = self.tiles[r_idx].flatten()
			out[s[:, 0], r[0]] += s[:, 1]
		return out

	def place_settlement(self, loc, player, city):
		"""
		Puts a settlement on the location specified
		Args:
		loc: the id to place the settlement on
		player: the player whose settlement to place
		city: True for city, False for settlement
		"""
		assert loc > -1 and loc < 54, 'settlement spots from 0-53'
		assert player > 0 and player < 5, 'player from 1-4'
		self.settlements[loc, 0] = player
		self.settlements[loc, 1] = 2 if city else 1

	def place_road(self, loc, player):
		assert loc > -1 and loc < 72, 'road spots from 0-72'
		assert player > 0 and player < 5, 'player from 1-4'
		self.roads[loc, 0] = player

	def compute_settlement_spots(self):
		"""
		Valid settlement spot are those that are not taken, and are two hops away from anyother 
		"""
		invalid_idxs = np.argwhere(self.settlements[:, 0] != 0)
		roads = self.settlements[invalid_idxs, 5:8].flatten()
		roads = roads[roads != -1]
		invalid_idxs = np.unique(self.roads[roads, 1:])
		mask = np.ones(self.settlements.shape[0], dtype=bool)
		mask[invalid_idxs] = False
		return np.arange(self.settlements.shape[0], dtype=int)[mask]

	def render_base(self, fig = None, ax = None, display_ids=False):
		if fig is None or ax is None:
			fig, ax = plt.subplots()
		
		#resource tiles
		for i in range(19):
			tile = self.tiles[i]
			ax.scatter(tile[2], tile[3], s=512, color = self.get_color(tile[0]))
			ax.text(tile[2], tile[3], '{}/{}'.format(tile[1], 6 - abs(7 - tile[1])), color='k', ha='center', va='center')

		#settlement spots
		for i, s in enumerate(self.settlements):
			ax.scatter(s[3], s[4], s = 80 if s[0] else 16, c=self.get_player_color(s[0]))
			if s[1] == 2:
				ax.scatter(s[3], s[4], s = 32, c='w')
			if display_ids:
				ax.text(s[3], s[4], i)

		#roads
		for idx, i in enumerate(self.roads):
			x = np.array([self.settlements[i[1]][3], self.settlements[i[2]][3]])
			y = np.array([self.settlements[i[1]][4], self.settlements[i[2]][4]])
			if i[0] != 0:
				ax.plot(x, y, c=self.get_player_color(i[0]))
			if display_ids:
				ax.text(x.mean(), y.mean(), idx, fontsize=6)

		ax.set_xlim(-6, 30)
		ax.set_ylim(-6, 30)
		return fig, ax

	def render_pips(self, fig, ax):
		"""
		Renders the pip values of each settlemetn square
		"""
		if fig is None or ax is None:
			fig, ax = plt.subplots()

		fig, ax = self.render_base(fig, ax)

		for s in self.settlements:
			p = 0
			for tidx in s[8:]:
				if tidx != -1:
					v = self.tiles[tidx][1]
					p += max(6 - abs(7 - v), 0)
			ax.text(s[3], s[4], p)

	def render(self):
		fig, axs = plt.subplots(1, 2, figsize = (14, 6))
		self.render_base(fig, axs[0])
		self.render_pips(fig, axs[1])

	def get_color(self, i):
		table = ['k', '0.75', '#fbff00', '#7da832', '#633906', '#ff8c00']
		return table[int(i)]

	def get_player_color(self, i):
		table = ['k', 'r', 'g', 'y', 'b']
		return table[i]

if __name__ == '__main__':
	b = Board()
	b.reset()

	b.place_settlement(3, 1, False)
	b.place_settlement(23, 2, False)
	b.place_settlement(18, 3, False)
	b.place_settlement(34, 4, False)
	b.place_settlement(13, 1, False)
	b.place_settlement(25, 2, True)
	b.place_settlement(46, 3, False)
	b.place_settlement(51, 4, True)
	b.place_road(3, 1)
	b.place_road(23, 2)
	b.place_road(18, 3)
	b.place_road(34, 4)

	for i in range(2, 13):
		print('roll = {}'.format(i))
		print(b.generate_resources(i))

	b.render()
	plt.show()

	spots = b.compute_settlement_spots()
	print(spots)
	fig, ax = b.render_base()
	ax.scatter(b.settlements[spots][:, 3], b.settlements[spots][:, 4], marker='x', c='r')
	plt.show()
	

	"""
	for s in b.settlements:
		print('s {} = {}'.format(cnt, s))
		fig, ax = b.render()
		r1idx = s[4]
		r2idx = s[5]
		r3idx = s[6]
		t1idx = s[7]
		t2idx = s[8]
		t3idx = s[9]

		ax.scatter(s[2], s[3])

		if t1idx != -1:
			ax.scatter(b.tiles[t1idx][2], b.tiles[t1idx][3], s=512, color='r')
		if t2idx != -1:
			ax.scatter(b.tiles[t2idx][2], b.tiles[t2idx][3], s=512, color='g')
		if t3idx != -1:
			ax.scatter(b.tiles[t3idx][2], b.tiles[t3idx][3], s=512, color='b')

		if r1idx != -1:
			ax.plot([b.settlements[b.roads[r1idx][1]][2], b.settlements[b.roads[r1idx][2]][2]], [b.settlements[b.roads[r1idx][1]][3], b.settlements[b.roads[r1idx][2]][3]], c='r')
		
		if r2idx != -1:
			ax.plot([b.settlements[b.roads[r2idx][1]][2], b.settlements[b.roads[r2idx][2]][2]], [b.settlements[b.roads[r2idx][1]][3], b.settlements[b.roads[r2idx][2]][3]], c='g')
		
		if r3idx != -1:
			ax.plot([b.settlements[b.roads[r3idx][1]][2], b.settlements[b.roads[r3idx][2]][2]], [b.settlements[b.roads[r3idx][1]][3], b.settlements[b.roads[r3idx][2]][3]], c='b')
		
		plt.show()
		cnt += 1
	"""
