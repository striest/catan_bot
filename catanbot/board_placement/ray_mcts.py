import sys
import time
import copy
import ray
import matplotlib.pyplot as plt
import numpy as np;np.set_printoptions(precision=4, linewidth=1e6, suppress=True)

from catanbot.board import Board
from catanbot.agents.heuristic_agent import HeuristicAgent
from catanbot.simulator import CatanSimulator
from catanbot.board_placement.placement import MCTSNode

class RayMCTS:
	"""
	Perform parallel MCTS using Ray.
	"""	
	def __init__(self, simulator, n_samples, n_threads=1):
		ray.init(ignore_reinit_error=True)
		self.simulator = simulator
		self.original_board = simulator.board
		self.original_agents = simulator.players
		self.root = MCTSNode(None, None, simulator.turn, copy.deepcopy(simulator.board), copy.deepcopy(simulator.players), False, simulator)
		self.n_samples = n_samples
		self.n_threads = n_threads
		self.workers = []
		self.sigstop = False
		for _ in range(self.n_threads):
			self.workers.append(RayWorker.remote(self.simulator))

	def update_root(self, board):
		"""
		Updates the search tree to be rooted at board. Creates a new root if it doesn't exist in the tree
		"""
		search_results = self.root.search_for(board)

		if not search_results:
			#Make a new root
			turn_number = int(board.roads[:, 0].clip(0, 1).sum())
			self.simulator.reset_from(board, self.simulator.players)
			self.simulator.turn = turn_number
			is_road = board.settlements[:, 1].clip(0, 1).sum() > board.roads[:, 0].clip(0, 1).sum()
			if is_road:
				print('Settlement placed without corresponding road - stopping...')
				exit(1)

			node_out = MCTSNode(None, None, self.simulator.turn, copy.deepcopy(board), copy.deepcopy(self.simulator.players), False, self.simulator)
		else:
			node_out = search_results[0]
			print('Reused {} rollouts'.format(node_out.stats.sum()))

		self.root = node_out

	def search(self, n_rollouts = None, max_time = None, c=1.0, verbose=False):
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
		parallel_time = 0
		r_t_running = 0
		while r < n_rollouts and t_running < max_time and not self.sigstop:
		#	print(self)
			t_itr = time.time()-prev
			t_remaining =  (t_running / (r+1)) * (n_rollouts - r) if max_time == float('inf') else max_time - t_running
			t_running += t_itr
			prev = time.time()
			leaves, paths = self.find_leaves(c=c, verbose=verbose)
			tasks = []
			taskids = []

			#Generate expansions/rollouts. Note that we have to detach the leaf from the tree to keep Ray from copying the tree. Reattach after the rollouts are collected
			widx = 0
			for leaf, path in zip(leaves, paths):
				maxdepth = (leaf.turn_number == 8 and not leaf.is_road)
#				print(curr)
				leafcopy = copy.copy(leaf)
				leafcopy.parent = None
				if leaf.is_visited and not maxdepth:
					tasks.append(self.workers[widx].expand.remote(leafcopy))
					taskids.append('e')
				else:
					tasks.append(self.workers[widx].rollout.remote(leafcopy, n_times=self.n_samples))
					taskids.append('r')
				widx += 1

			p_time = time.time()
			tasks = ray.get(tasks)
			parallel_time += time.time() - p_time

			#Put expansions/rollouts into the tree
			for leaf, task, taskid in zip(leaves, tasks, taskids):
				if taskid == 'e':
					leaf.children = task
					#Idk why, but reset the parent for the children (the parent is probably copied when it moves to the Ray actor)
					for ch in leaf.children:
						ch.parent = leaf
				else:
					r += 1
					while leaf:
						leaf.stats = leaf.stats + task
						leaf = leaf.parent

			if verbose:
				print('Rollout #{} (t={:.2f}s) time elapsed = {:.2f}s, time remaining = {:.2f}s rollout/expand time = {:.2f}'.format(r, t_itr, t_running, t_remaining, parallel_time))
				best = self.get_optimal_path()
				best = [n.parent_action for n in best[1:]]
				print('best = {}'.format(best))
				placements = [path[:2] for path in paths]
				placements.sort(key=lambda x:x[0] * 100 + x[1] if len(x) >1 else 0)
				print('explored {}'.format(placements))
				print('avg depth = {}'.format(np.array([len(p) for p in paths]).mean()))

		#Restore the simulator if you need to call MCTS again
		self.simulator.reset_from(self.original_board, players = self.original_agents)

	def find_leaves(self, c=1.0, verbose=False):
		leaves = []
		paths = []
		path_unique = set()
		for i in range(self.n_threads):
			leaf, path = self.find_leaf(c=c, bias=i)
			leaves.append(leaf)
			paths.append(path)
			pathstr = ' '.join([str(v) for v in path])
			path_unique.add(pathstr)
		if verbose:
			print('{} unique/{} total'.format(len(path_unique), self.n_threads))
		return leaves, paths

	def find_leaf(self, c=1.0, bias=0.0):
		"""
		find leaf in MCTS tree: take action with highest UCB to get there
		"""	
		curr = self.root
		path = []
		while not curr.is_leaf:
			#Randomly sample argmaxes to get better coverage when using multiple trees
			ucbs = curr.children_ucb(c=c, bias=bias)
			_max = np.max(ucbs)
			maxidxs = np.argwhere(ucbs >= _max).flatten()
			idx = np.random.choice(maxidxs)
			curr = curr.children[idx]
			path.append(curr.parent_action)
		return curr, path

	def get_optimal_path(self):
		path = [self.root]
		curr = self.root
		while not curr.is_leaf:
			ucbs = curr.children_ucb(c=0, bias=0.0)
			idx = np.argmax(ucbs)
			curr = curr.children[idx]
			path.append(curr)
		return path

	def get_top_k(self, k=5):
		"""
		Get the top k placements (first settlement and road)
		"""
		curr = self.root
		options = []
		for settlement in curr.children:
			for idx, road in enumerate(settlement.children):
				options.append((settlement, road, settlement.children_ucb(c=0)[idx]))
		options.sort(key=lambda x:x[2], reverse=True)
		return options[:k]

	def __repr__(self):
		return self.dump(self.root, 0)

	def dump(self, node, depth, c=1.0):
		out = '\t' * depth
		out += node.__repr__(c)
		out += '\n'
		for i, ch in enumerate(node.children):
			out += str(depth+1) + ':' + str(i) + self.dump(ch, depth +1, c)
		return out

@ray.remote
class RayWorker:
	"""
	Worker class for doing rollouts/expansions. Necessary to make it a class because each needs a local simulator.
	"""
	def __init__(self, simulator):
		self.simulator = simulator

	def expand(self, node):
		oldsim = node.simulator
		node.simulator = self.simulator
		result = node.expand(threshold=6)
		node.simulator = oldsim
		return result

	def rollout(self, node, n_times):
		oldsim = node.simulator
		node.simulator = self.simulator
		result = node.rollout(n_times = n_times)
		node.simulator = oldsim
		return result

if __name__ == '__main__':
	b = Board()
	b.reset()
	agents = [HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b)]
	s = CatanSimulator(board=b, players = agents, max_vp=10)
	s.render()
	
	search = RayMCTS(s, n_samples=1, n_threads=16)	
	search.search(max_time=180, c=1.0, verbose=True)
	print(search.dump(search.root, 0, c=0))
	print('_-' * 100, end='\n\n')
	acts = search.get_optimal_path()
	for n in acts:
		print(n.__repr__(c=0))
		if n.turn_number > 0:
			if n.is_road:
				b.place_settlement(n.parent_action, n.player_from_turn(n.turn_number), False)
			else:
				b.place_road(n.parent_action, n.player_from_turn(n.turn_number))
	topk = search.get_top_k()	
	print('_'*20, 'TOP 5', '_'*20)
	for k in topk:
		print('Settlement = {}, Road = {}, Win Rate = {:.4f}, Stats = {}'.format(k[0].parent_action, k[1].parent_action, k[2], k[1].stats))
	b.render()
	plt.show()
