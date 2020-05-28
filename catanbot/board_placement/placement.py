import time
import copy
import matplotlib.pyplot as plt
import numpy as np;np.set_printoptions(precision=4, linewidth=1e6, suppress=True)

from catanbot.board import Board
from catanbot.agents.heuristic_agent import HeuristicAgent
from catanbot.simulator import CatanSimulator

class MCTSNode:
	def __init__(self, parent, parent_action, turn_number, state, is_road, simulator):
		self.parent = parent
		self.parent_action = parent_action
		self.board = state
		self.children = []
		self.stats = np.zeros(4)
		self.turn_number = turn_number
		self.is_road = is_road #Whether it's a settlement placement or road placement
		self.depth = 0 if self.parent is None else self.parent.depth + 1
		self.simulator = simulator

	@property
	def is_leaf(self):
		return not self.children

	@property
	def is_visited(self):
		return self.stats.sum() > 0

	def expand(self):
		"""
		Expand all valid moves 
		"""
		
		#If you're a road-placement node, children will be settlement-placement for next player.
		#Heuristically, only take the ones with value 8 and up
		ch = []
		if not self.is_road:
			avail = self.board.compute_settlement_spots()
			prod = self.board.compute_production()
			prod = prod[prod[:, 1] > 7]
			actions = np.intersect1d(avail, prod[:, 0])
			for a in actions:
				c_board = copy.deepcopy(self.board)
				pidx = self.player_from_turn(self.turn_number + 1)
				c_board.place_settlement(a, pidx, False)
				ch.append(MCTSNode(self, a, self.turn_number + 1, c_board, True, self.simulator))
		else:
			prev_loc = self.parent_action
			pidx = self.player_from_turn(self.turn_number)
			roads = self.board.settlements[prev_loc, 5:8]
			roads = roads[roads != -1]
			roads = roads[self.board.roads[roads][:, 0] == 0]
			for a in roads:
				c_board = copy.deepcopy(self.board)
				pidx = self.player_from_turn(self.turn_number)
				c_board.place_road(a, pidx)
				ch.append(MCTSNode(self, a, self.turn_number, c_board, False, self.simulator))
		return ch

	def children_ucb(self, c=1.0):
		"""
		Computes the UCB for all the children of this node.
		UCB = wc + c*sqrt(ln(sp)/sc)
		wc = child's win ratio, sp = #parent simulations, sc = #child simulations
		note that UCB can be inf (or like 10e10 for practical purposes)
		"""
		ucbs = np.zeros(len(self.children))
		for idx, ch in enumerate(self.children):
			if ch.stats.sum() == 0:
				ucbs[idx] = 1e10
			else:
				ch_wins = ch.stats[self.player_from_turn(ch.turn_number)-1]
				ch_total = ch.stats.sum()
				par_total = self.stats.sum()
				ucbs[idx] = (ch_wins/ch_total) + c*np.sqrt(np.log(par_total)/ch_total)
		return ucbs		
			
	def player_from_turn(self, turn):
		return int(4 - np.floor(abs(4.5 - turn)))

	def rollout(self, n_times=1):
		"""
		Complete initial placements from this node and evaluate.
		should probably do placements somewhat randomly.
		"""
		rollout_board = copy.deepcopy(self.board)
		if self.is_road:
#			print('place a road first')
			prev_loc = self.parent_action
			pidx = self.player_from_turn(self.turn_number)
			roads = rollout_board.settlements[prev_loc, 5:8]
			roads = roads[roads != -1]
			roads = roads[rollout_board.roads[roads][:, 0] == 0]
			rollout_board.place_road(np.random.choice(roads), pidx)	

		for turn in range(self.turn_number+1, 9):
#			print('Placing player {}'.format(self.player_from_turn(turn)))
			pidx = self.player_from_turn(turn)
			avail = rollout_board.compute_settlement_spots()
			prod = self.board.compute_production()
			prod = prod[prod[:, 1] > 7]
			good = np.intersect1d(avail, prod[:, 0])
			if good.size > 0:
				s = np.random.choice(good)
			else:
				s = np.random.choice(avail)
#			print(good, s)

			rollout_board.place_settlement(s, pidx, False)
			roads = rollout_board.settlements[s, 5:8]
			roads = roads[roads != -1]
			roads = roads[rollout_board.roads[roads][:, 0] == 0]
			rollout_board.place_road(np.random.choice(roads), pidx)
			#rollout_board.render();plt.show()
		results = np.zeros(4)
		for i in range(n_times):
			c_board = copy.deepcopy(rollout_board)
			self.simulator.reset_from(c_board)
			#self.simulator.render()
			results += self.simulator.simulate()
			#self.simulator.render()
		return results

	def __repr__(self, c=1.0):
		return 'state = {}, act = {}, stats = {}, turn = {}({}), ucbs = {}'.format(self.board, self.parent_action, self.stats, self.turn_number, 'RGYBX'[self.player_from_turn(self.turn_number-1)], self.children_ucb(c=c))

class MCTSSearch:
	"""
	Perform MCTS to get good placements for Catan
	"""
	def __init__(self, simulator, n_samples):
		self.simulator = simulator
		self.root = MCTSNode(None, None, 0, copy.deepcopy(simulator.board), False, simulator)
		self.n_samples = n_samples

	def search(self, n_rollouts = None, max_time = None, c=1.0):
		"""
		1. Search for a leaf node
		2. If the leaf hasn't been visited, collect a rollout and propagate up.
		3. If the leaf has been visited, expand it
		"""
		assert not (n_rollouts is None and max_time is None), 'Need to set one of rollouts or time'
		if n_rollouts is None:
			n_rollouts = float('inf')
		if max_time is None:
			max_time = float('inf')

		prev = time.time()
		r = 0
		t_running = 0
		while r < n_rollouts and t_running < max_time:
		#	print(self)
			t_itr = time.time()-prev
			t_remaining =  (t_running / (r+1)) * (n_rollouts - r) if max_time == float('inf') else max_time - t_running
			print('Rollout #{} (t={:.2f}s) time elapsed = {:.2f}s, time remaining = {:.2f}s'.format(r, t_itr, t_running, t_remaining))
			t_running += t_itr
			prev = time.time()
#			import pdb;pdb.set_trace()
			curr = self.find_leaf(c=c)
#			print(curr)
			maxdepth = (curr.turn_number == 8 and curr.is_road)
			if maxdepth:
				print('max depth')
			if curr.is_visited and not maxdepth:
				ch = curr.expand()
				curr.children = ch
			else:
				result = curr.rollout(n_times = self.n_samples)
				r += self.n_samples
				while curr:
					curr.stats += result
					curr = curr.parent

	def find_leaf(self, c=1.0):
		"""
		find leaf in MCTS tree: take action with highest UCB to get there
		"""	
		curr = self.root
		while not curr.is_leaf:
			ucbs = curr.children_ucb(c=c)
			idx = np.argmax(ucbs)
			curr = curr.children[idx]
		return curr

	def get_optimal_path(self):
		path = [self.root]
		curr = self.root
		while not curr.is_leaf:
			ucbs = curr.children_ucb(c=0)
			idx = np.argmax(ucbs)
			curr = curr.children[idx]
			path.append(curr)
		return path

	def __repr__(self):
		return self.dump(self.root, 0)

	def dump(self, node, depth, c=1.0):
		out = '\t' * depth
		out += node.__repr__(c)
		out += '\n'
		for ch in node.children:
			out += self.dump(ch, depth +1, c)
		return out

if __name__ == '__main__':
	b = Board()
	b.reset()
	agents = [HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b)]
	s = CatanSimulator(board=b, players = agents, max_vp=8)

	search = MCTSSearch(s, n_samples=10)	
	search.search(max_time=900, c=0.75)
	print(search.dump(search.root, 0, c=0))
	print('_-' * 100, end='\n\n')
	acts = search.get_optimal_path()
	for n in acts:
		print(n)
	b.render()
	plt.show()
