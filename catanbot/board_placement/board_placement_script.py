import argparse
import numpy as np
import matplotlib.pyplot as plt

from catanbot.board import Board
from catanbot.agents.heuristic_agent import HeuristicAgent
from catanbot.simulator import CatanSimulator
from catanbot.board_placement.placement import MCTSSearch, MCTSNode

if __name__ == '__main__':
	plt.ion()
	board_str = input('Put in board string as a comma-separated list of RD, R=resource type(Ore=1, Wheat=2, Sheep=3, Wood=4, Brick=5, Desert=0), D=dice value (0 for desert):\n')
	b = Board()
	b.reset_from_string(board_str)
	b.render();plt.draw()

	turn_loc = int(input('Input turn order (1-4):\n'))
	print(turn_loc)

	#PLAYER INPUT 1

	for i in range(1, turn_loc):
		placement_str = input('Input placement for player {} (As <Settlement ID>, <Road ID>):\n'.format(i))
		tokens = placement_str.split(',')
		sloc = int(tokens[0])
		rloc = int(tokens[1])
		b.place_settlement(sloc, i, False)
		b.place_road(rloc, i)
		b.render();plt.draw()

	#MCTS 1	

	tmax = int(input('Input how long to run MCTS (in seconds):\n'))
	c = float(input('Input exploration factor for MCTS (default is 1):\n'))
	agents = [HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b)]
	s = CatanSimulator(board=b, players = agents, max_vp=8)
	s.reset_from(b, agents)
	s.render();plt.draw()
	mcts = MCTSSearch(s, n_samples=1)
	mcts.root.turn_number = turn_loc - 1
	mcts.search(max_time=tmax, c=c)
	print(mcts.dump(mcts.root, 0, c=0))
	print('_-' * 100, end='\n\n')
	acts = mcts.get_optimal_path()
	print('_'*20, 'OPTIMAL PATH', '_'*20)
	for n in acts:
		print(n.__repr__(c=0))

	topk = mcts.get_top_k()	
	print('_'*20, 'TOP 5', '_'*20)
	for k in topk:
		print('Settlement = {}, Road = {}, Win Rate = {:.4f}, Stats = {}'.format(k[0].parent_action, k[1].parent_action, k[2], k[1].stats))
	

	#PLAYER INPUT 2

	for i in range(turn_loc, 5):
		placement_str = input('Input placement for player {} (As <Settlement ID>, <Road ID>):\n'.format(i))
		tokens = placement_str.split(',')
		sloc = int(tokens[0])
		rloc = int(tokens[1])
		b.place_settlement(sloc, i, False)
		b.place_road(rloc, i)
		b.render();plt.draw()

	for i in range(4, turn_loc, -1):
		placement_str = input('Input placement for player {} (As <Settlement ID>, <Road ID>):\n'.format(i))
		tokens = placement_str.split(',')
		sloc = int(tokens[0])
		rloc = int(tokens[1])
		tiles = b.settlements[sloc, 8:]
		tiles = tiles[tiles != -1]
		resources = b.tiles[tiles, 0]
		for r in resources:
			if r > 0:
				agents[i-1].resources[r] += 1
		b.place_settlement(sloc, i, False)
		b.place_road(rloc, i)
		print(agents)
		b.render();plt.draw()

	#MCTS 2 (A smarter man would reuse the tree from part 1)
	tmax = int(input('Input how long to run MCTS (in seconds):\n'))
	c = float(input('Input exploration factor for MCTS (default is 1):\n'))
	s = CatanSimulator(board=b, players = agents, max_vp=8)
	s.reset_from(b, agents)
	s.render();plt.draw()
	mcts = MCTSSearch(s, n_samples=1)
	mcts.root.turn_number = 8 - turn_loc
	mcts.search(max_time=tmax, c=c)
	print(mcts.dump(mcts.root, 0, c=0))
	print('_-' * 100, end='\n\n')
	acts = mcts.get_optimal_path()
	for n in acts:
		print(n.__repr__(c=0))
	b.render();plt.draw()
	
	topk = mcts.get_top_k()	
	print('_'*20, 'TOP 5', '_'*20)
	for k in topk:
		print('Settlement = {}, Road = {}, Win Rate = {:.4f}, Stats = {}'.format(k[0].parent_action, k[1].parent_action, k[2], k[1].stats))

	plt.ioff()
	b.render();plt.show()
