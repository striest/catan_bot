import matplotlib.pyplot as plt
import numpy as np

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
		self.vp = 0

	def reset(self):
		self.resources *= 0
		self.dev *= 0
		self.vp = 0

	def __repr__(self):
		r = {a:b for a, b in zip(['Ore', 'Wheat', 'Sheep', 'Wood', 'Brick'], self.resources[1:])}
		d = {a:b for a, b in zip(['Knight', 'VP', 'YOP', 'RB', 'M'], self.dev[1:])}
		return 'RESOURCES:\n{}\nDEV:\n{}\n'.format(r, d)
