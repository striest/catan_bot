import pygame
import argparse
import signal
import copy
from math import sin, cos, pi, floor
import matplotlib.pyplot as plt

from catanbot.board import Board
from catanbot.ui.renderer import BoardRenderer
from catanbot.ui.button import Button

from catanbot.agents.heuristic_agent import HeuristicAgent
from catanbot.simulator import CatanSimulator
from catanbot.board_placement.placement import MCTSSearch, MCTSNode
from catanbot.board_placement.ray_mcts import RayMCTS

class BoardPlacementUI:
	"""
	Stores an internal board state and lets you edit it through a GUI

	#Probably need some FSM to move through the clicks asynchronously.
	"""
	def __init__(self, screen, nthreads, exploration, maxtime):
		self.screen = screen
		self.board = Board()
		self.board.reset()
		self.renderer = BoardRenderer(screen)
		self.mcts_button = Button(screen, (52, 210, 235), 'Run MCTS', (100, 675), (100, 50))
		self.clear_mcts_button = Button(screen, (52, 210, 235), 'Clear MCTS Results', (225, 675), (175, 50))
		self.show_prod_button = Button(screen, (52, 210, 235), 'Toggle Production Values', (425, 675), (225, 50))
		self.place_button = Button(screen, (52, 210, 235), 'Place', (675, 675), (75, 50))

		self.nthreads = nthreads
		self.exploration = exploration
		self.max_time = maxtime
		self.turn_number = 0

		self.agents = [HeuristicAgent(self.board), HeuristicAgent(self.board), HeuristicAgent(self.board), HeuristicAgent(self.board)]
		s = CatanSimulator(board=self.board, players = self.agents, max_vp=10)
		s.reset_from(self.board, self.agents)

		mcts = RayMCTS(s, n_samples=1, n_threads=self.nthreads)
		mcts.root.turn_number = 0
		self.mcts = mcts
		self.running_mcts = False

		self.mcts_results = None

	def handle_click(self, event):
		print('_' * 50)
		print('Mouse coordinate = {}'.format(event.pos))
		print('Board coordinate = {}'.format(self.renderer.inv_scale_fn(event.pos)))
		o_code, o_idx = self.renderer.get_object(self.board, self.renderer.inv_scale_fn(event.pos))

		if o_code == 1:
			#Edit a hex
			self.board.tiles[o_idx, 0] = (self.board.tiles[o_idx, 0] + 1) % 6
		elif o_code == 2:
			#Edit a token
			self.board.tiles[o_idx, 1] = (self.board.tiles[o_idx, 1] + 1) % 13
			if self.board.tiles[o_idx, 1] == 7 or self.board.tiles[o_idx, 1] == 1:
				#There's no 1 or 7 token
				self.board.tiles[o_idx, 1] = self.board.tiles[o_idx, 1] + 1
		elif o_code == 3:
			#Edit a settlement
			if self.board.settlements[o_idx, 0] == 0:
				self.board.settlements[o_idx, 0] = self.get_player()
			else:
				#Don't remove other players' settlemetns
				if self.board.settlements[o_idx, 0] == self.get_player():
					self.board.settlements[o_idx, 0] = 0

			if self.board.settlements[o_idx, 0] != 0:
				self.board.settlements[o_idx, 1] = 1
			else:
				self.board.settlements[o_idx, 1] = 0
		elif o_code == 4:
			if self.board.roads[o_idx, 0] == 0:
				self.board.roads[o_idx, 0] = self.get_player()
			else:
				self.board.roads[o_idx, 0] = 0

		elif o_code == 5:
			port = self.board.port_locations[o_idx]
			s1 = port[0]
			s2 = port[1]
			r_new = (self.board.settlements[s1, 2] + 1) % 7
			if r_new == 0:
				r_new += 1
			self.board.settlements[s1, 2] = r_new
			self.board.settlements[s2, 2] = r_new

		if self.mcts_button.collidepoint(event.pos):
			topk = self.run_mcts()
			self.mcts_results = topk

		if self.clear_mcts_button.collidepoint(event.pos):
			self.mcts_results = None

		if self.show_prod_button.collidepoint(event.pos):
			self.renderer.show_prod = not self.renderer.show_prod

		if self.place_button.collidepoint(event.pos):
			if self.turn_number < 9:
				self.turn_number += 1

			#Update the mcts root, as the player has locked in the settlement
			self.mcts.simulator.reset_from(self.board, self.agents)
			self.mcts.update_root(self.board)

	def get_player(self):
		"""
		Get current player from turn number
		"""
		return int(4 - floor(abs(3.5 - self.turn_number)))

	def run_mcts(self):
		"""
		Sets up the backend to run MCTS. Also needs to ask the user for the time and c and threads to run mcts for and which player to run for, as we don't enforce valid board placements.
		Returns info for the top 5 results and updates the search tree
		"""
		self.mcts.sigstop = False
		self.running_mcts = True

		self.mcts.search(max_time=self.max_time, c=self.exploration, verbose=True)

		self.running_mcts = False

		print(self.mcts.dump(self.mcts.root, 0, c=0))
		return self.get_top5()

	def get_top5(self):
		print('_-' * 100, end='\n\n')
		acts = self.mcts.get_optimal_path()
		print('_'*20, 'OPTIMAL PATH', '_'*20)
		for n in acts:
			print(n.__repr__(c=0))

		topk = self.mcts.get_top_k()	
		print('_'*20, 'TOP 5', '_'*20)
		for k in topk:
			print('Settlement = {}, Road = {}, Win Rate = {:.4f}, Stats = {}'.format(k[0].parent_action, k[1].parent_action, k[2], k[1].stats))
		
		return topk

	def handle_stop(self, sig, frame):
		if self.running_mcts:
			self.mcts.sigstop = True
			self.running_mcts = False
			print('Stopping MCTS ...')
		else:
			print('Exiting program...')
			exit(0)
		
	def render(self):
		self.renderer.render(self.board)
		self.mcts_button.render()
		self.clear_mcts_button.render()
		self.show_prod_button.render()
		self.place_button.render()
		font = pygame.font.SysFont(None, 24)
		text = '{} to place'.format(['Nobody', 'Red', 'Green', 'Blue', 'Yellow'][self.get_player()])
		img = font.render(text, True, (0, 0, 0))
		self.screen.blit(img, (25, 25))
		if self.mcts_results:
			self.renderer.draw_mcts_results(self.board, self.mcts_results)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Args for MCTS')
	parser.add_argument('--nthreads', type=int, required=False, default=1, help='The number of processes to use for the tree search')
	parser.add_argument('--c', type=float, required = False, default=1.0, help='Exploration factor for MCTS (1.0 default, more=more exploration, less=more exploitation)')
	parser.add_argument('--max_time', type=float, required = False, default=300.0, help='Optional max time to run MCTS for')
	args = parser.parse_args()
	pygame.init()

	# Set up the drawing window
	screen = pygame.display.set_mode([900, 800])
	pygame.display.set_caption('Catan Placement Assist')

	ui = BoardPlacementUI(screen, nthreads = args.nthreads, exploration = args.c, maxtime = args.max_time)
	signal.signal(signal.SIGINT, ui.handle_stop)

	# Run until the user asks to quit
	running = True
	while running:
	# Did the user click the window close button?
		for event in pygame.event.get():
			if event.type == pygame.MOUSEBUTTONUP:
				ui.handle_click(event)
			if event.type == pygame.QUIT:
				running = False

		# Flip the display
		pygame.display.flip()
		ui.render()


	# Done! Time to quit.
	pygame.quit()
