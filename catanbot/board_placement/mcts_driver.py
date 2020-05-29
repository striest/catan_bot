import sys
import time
import copy
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np;np.set_printoptions(precision=4, linewidth=1e6, suppress=True)

from catanbot.board import Board
from catanbot.agents.heuristic_agent import HeuristicAgent
from catanbot.simulator import CatanSimulator
from catanbot.board_placement.placement import MCTSSearch

class MPIMCTSDriver:
	"""
	Class for running MCTS on parallel threads
	"""
	def __init__(self, simulator, n_samples):
		self.comm = MPI.COMM_WORLD
		self.rank = self.comm.Get_rank()
		self.nprocs = self.comm.Get_size()

		print('Rank = {}, nprocs = {}'.format(self.rank, self.nprocs))

		if self.rank == 0:
			#Driver
			self.searcher = MCTSSearch(s, 1)
		else:
			self.searcher = None
	

		self.searcher = self.comm.bcast(self.searcher, root=0)

	def search(self, max_time = 60.0, c=1.0, gather_every = 10.0, verbose = True):
		if self.rank == 0:
			start = time.time()
			if verbose:
				print('starting...')
				sys.stdout.flush()
		nitrs = int(max_time/gather_every)
		for i in range(nitrs):
			c_start = time.time()
			if verbose:
				print('collecting...')
				sys.stdout.flush()
			self.collect(max_time = gather_every, c=c)
			c_time = time.time() - c_start
			g_start = time.time()
			if verbose:
				print('gathering...')
				sys.stdout.flush()
			self.gather_and_broadcast(verbose = verbose)
			g_time = time.time() - g_start
			if verbose and self.rank == 0:
				print('Itr {}/{}'.format(i, nitrs))
				print('Time elapsed = {:.2f}, collect_time = {:.2f}, gather_time = {:.2f}'.format(time.time() - start, c_time, g_time))
				best = self.searcher.get_optimal_path()
				print('Best path = {}'.format([n.parent_action for n in best[1:]]))
				sys.stdout.flush()

		if self.rank == 0:
			end = time.time() - start
			nrollouts = size(self.searcher.root)
			print(self.searcher.dump(self.searcher.root, 0, c=0))
			print('N_rollouts = {} in {:.2f}s ({:4f}s/rollout)'.format(nrollouts, end, end/nrollouts))
			best = self.searcher.get_optimal_path()
			print('Best path = {}'.format([n.parent_action for n in best[1:]]))
			topk = self.searcher.get_top_k()	
			print('_'*20, 'TOP 5', '_'*20)
			for k in topk:
				print('Settlement = {}, Road = {}, Win Rate = {:.4f}, Stats = {}'.format(k[0].parent_action, k[1].parent_action, k[2], k[1].stats))
			sys.stdout.flush()

	def collect(self, n_rollouts = None, max_time = None, c=1.0, verbose=False):
		self.searcher.search(n_rollouts, max_time, c, verbose=verbose)
		if verbose:
			sys.stdout.flush()

	def gather_and_broadcast(self, verbose=False):
		if self.rank == 0:
			trees = None
			trees = self.comm.gather(self.searcher.root, 0)
		else:
			trees = self.searcher.root
			self.comm.gather(self.searcher.root, 0)
			
		if self.rank == 0:
			out = trees[0]
			if verbose:
				diffs = 0
				s_old = size(out)
			for t_new in trees[1:]:
				out = merge_trees(out, t_new)
				if verbose:
					s_new = size(t_new)
					s_merge = size(out)
					new_cnt = 2*s_merge - (s_new + s_old)
					s_old = s_merge
					diffs += new_cnt
			if verbose:
				print('Got {} unique nodes on this gather ({} total)'.format(diffs, s_merge))
				sys.stdout.flush()

			self.searcher.root = out

		self.searcher.root = self.comm.bcast(self.searcher.root, root=0)	

	def dump(self, c=0.0):
		return 'Rank = {}, searcher = {}'.format(self.rank, self.searcher.dump(self.searcher.root, 0, c=c))
		

def merge_trees(t1, t2):
	"""
	Combines two search trees. Note that we can enforce an ordering on the children
	We can simplify this by enforcing that we are merging trees at the same depth

	If nodes same, merge statistics.
	For children, pair off and recurse.
	Guarantees correctness if root is same (it always will be for parallel MCTS, so make this a submethod of the driver class).

	Note that this will replace t1 with the merge of t1 and t2.

	"""
	assert t1.parent_action == t2.parent_action, 'You meesed up Sam (trying to merge node with different parent action)'
	t1.stats += t2.stats
	t1_dict = {n.parent_action: n for n in t1.children}	
	t1_acts = set(t1_dict.keys()) #i don't think python will con constant time lookups for dict keys
	for ch2 in t2.children:
		p_act = ch2.parent_action
		if p_act in t1_acts:
			t1_dict[p_act] = merge_trees(t1_dict[p_act], ch2)
		else:
			t1.children.append(ch2)
		
	return t1	

def size(t1):
	"""
	Computes the size of the tree
	"""
	if t1.is_leaf:
		return 1
	else:
		return 1 + sum([size(ch) for ch in t1.children])

if __name__ == '__main__':
	b = Board()
	b.reset()
	agents = [HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b)]
	s = CatanSimulator(board=b, players = agents, max_vp=8)

	driver = MPIMCTSDriver(s, 1)

	driver.search(max_time = 300.0, gather_every=30.0)

	if driver.rank == 0:
		s.render()
