import pygame
import time
import argparse
import signal
import copy
import torch
from math import sin, cos, pi, floor
import matplotlib.pyplot as plt

from catanbot.core.board import Board
from catanbot.ui.renderer import BoardRenderer
from catanbot.ui.button import Button

from catanbot.agents.independent_actions_agents import IndependentActionsHeuristicAgent, IndependentActionsAgent
from catanbot.core.independent_action_simulator import IndependentActionsCatanSimulator
from catanbot.board_placement.agents.base import InitialPlacementAgent, MakeDeterministic
from catanbot.core.independent_action_simulator import IndependentActionsCatanSimulator
from catanbot.board_placement.initial_placement_simulator import InitialPlacementSimulator, InitialPlacementSimulatorWithPenalty
from catanbot.rl.collectors.initial_placement_collector import InitialPlacementCollector
from catanbot.board_placement.initial_placement_mcts import RayMCTSQFSearch

class BoardPlacementUI:
    """
    Stores an internal board state and lets you edit it through a GUI

    #Probably need some FSM to move through the clicks asynchronously.
    """
    def __init__(self, screen, nthreads, nsamples, exploration, maxtime, network):
        self.screen = screen
        self.board = Board()
        self.board.reset()
        self.renderer = BoardRenderer(screen)
        self.mcts_button = Button(screen, (52, 210, 235), 'Run MCTS', (50, 675), (100, 50))
        self.clear_mcts_button = Button(screen, (52, 210, 235), 'Clear MCTS Results', (175, 675), (175, 50))
        self.show_prod_button = Button(screen, (52, 210, 235), 'Toggle Production Values', (375, 675), (225, 50))
        self.place_button = Button(screen, (52, 210, 235), 'Place', (625, 675), (75, 50))
        self.randomize_button = Button(screen, (52, 210, 235), 'Randomize', (725, 675), (125, 50))

        self.nthreads = nthreads
        self.nsamples = nsamples
        self.exploration = exploration
        self.max_time = maxtime
        self.turn_number = 0

        self.agents = [IndependentActionsHeuristicAgent(self.board), IndependentActionsHeuristicAgent(self.board), IndependentActionsHeuristicAgent(self.board), IndependentActionsHeuristicAgent(self.board)]
        s = IndependentActionsCatanSimulator(board=self.board, players = self.agents, max_vp=10)
        s.reset_from(self.board, self.agents)
        placement_agents = [InitialPlacementAgent(self.board), InitialPlacementAgent(self.board), InitialPlacementAgent(self.board), InitialPlacementAgent(self.board)]
        placement_simulator = InitialPlacementSimulator(s, placement_agents)
        collector = InitialPlacementCollector(placement_simulator, reset_board=True, reward_scale=1.)
        self.qf = network
        self.mcts = RayMCTSQFSearch(placement_simulator, self.qf, n_threads=self.nthreads)
        self.running_mcts = False
        self.mcts_results = None

        self.keystroke_buf = ''
        self.edit_hex_idx = -1
        self.rchr_to_rval = {
            'd': 0,
            'o': 1,
            'w': 2,
            's': 3,
            'l': 4, 
            'b': 5,
        }
        self.edit_port_idx = -1
        self.pchr_to_pval = {
            'o': 1,
            'w': 2,
            's': 3,
            'l': 4,
            'b': 5,
            'a': 6
        }

    def handle_keystroke(self, event):
        try:
            c = chr(event.key)
            if c.isalnum():
                self.keystroke_buf += c.lower()
        except:
            print('not ascii')

        if len(self.keystroke_buf) >= 2 and len(self.keystroke_buf) <= 3 and self.keystroke_buf[-1].isalpha() and self.keystroke_buf[:-1].isnumeric() and self.edit_hex_idx >= 0:
            rchr = self.keystroke_buf[-1]
            if rchr in self.rchr_to_rval.keys():
                rtype = self.rchr_to_rval[rchr]
                rval = int(self.keystroke_buf[:-1])
                if (rval == 0 and rtype == 0) or (rval >= 2 and rval <= 12 and rval != 7):
                    #At this point, we have a valid hex string
                    print(self.edit_hex_idx, rtype, rval)
                    self.board.tiles[self.edit_hex_idx, 0] = rtype
                    self.board.tiles[self.edit_hex_idx, 1] = rval
                    self.keystroke_buf = ''
                else:
                    print('resource number invalid')
            else:
                print('hex type invalid (d=desert, o=ore, w=wheat, s=sheep, l=wood(lumber), b=brick)')
        else:
            print('invalid hex string')

        if len(self.keystroke_buf) == 1 and self.edit_port_idx >= 0:
            pchr = self.keystroke_buf[0]
            if pchr in self.pchr_to_pval.keys():
                ptype = self.pchr_to_pval[pchr]
                port = self.board.port_locations[self.edit_port_idx]
                s1 = port[0]
                s2 = port[1]
                self.board.settlements[s1, 2] = ptype
                self.board.settlements[s2, 2] = ptype
                self.keystroke_buf = ''
            else:
                print('port type invalid. (a=any (3 for 1), o=ore, w=wheat, s=sheep, l=wood(lumber), b=brick)')
        else:
            print('invalid port char')

    def handle_click(self, event):
        print('_' * 50)
        print('Mouse coordinate = {}'.format(event.pos))
        print('Board coordinate = {}'.format(self.renderer.inv_scale_fn(event.pos)))
        o_code, o_idx = self.renderer.get_object(self.board, self.renderer.inv_scale_fn(event.pos))

        if o_code == 1 or o_code == 2:
            if self.edit_hex_idx == o_idx:
                self.edit_hex_idx = -1
            else:
                self.edit_hex_idx = o_idx
            self.keystroke_buf = ''
            self.edit_port_idx = -1

#        if o_code == 1:
#            #Edit a hex
#            self.board.tiles[o_idx, 0] = (self.board.tiles[o_idx, 0] + 1) % 6
#        elif o_code == 2:
#            #Edit a token
#            self.board.tiles[o_idx, 1] = (self.board.tiles[o_idx, 1] + 1) % 13
#            if self.board.tiles[o_idx, 1] == 7 or self.board.tiles[o_idx, 1] == 1:
#                #There's no 1 or 7 token
#                self.board.tiles[o_idx, 1] = self.board.tiles[o_idx, 1] + 1
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
            self.edit_hex_idx = -1
            self.edit_port_idx = -1

        elif o_code == 4:
            if self.board.roads[o_idx, 0] == 0:
                self.board.roads[o_idx, 0] = self.get_player()
            else:
                self.board.roads[o_idx, 0] = 0
            self.edit_hex_idx = -1
            self.edit_port_idx = -1

        elif o_code == 5:
            self.edit_hex_idx = -1
            self.keystroke_buf = ''
            if self.edit_port_idx == o_idx:
                self.edit_port_idx = -1
            else:
                self.edit_port_idx = o_idx

        if self.mcts_button.collidepoint(event.pos):
            self.mcts.root.simulator.simulator.reset_from(self.board, self.agents)
            self.mcts.root.simulator.turn = self.turn_number
            self.mcts = RayMCTSQFSearch(self.mcts.root.simulator, self.qf, n_threads=self.nthreads)
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
#            self.mcts.simulator.reset_from(self.board, self.agents)
#            self.mcts.update_root(self.board)
            self.mcts.root.simulator.simulator.reset_from(self.board, self.agents)
            self.mcts.root.simulator.turn = self.turn_number
            self.mcts = RayMCTSQFSearch(self.mcts.root.simulator, self.qf, n_threads=self.nthreads)

        if self.randomize_button.collidepoint(event.pos):
            self.turn_number = 0
            self.board.reset()
            self.mcts.root.simulator.simulator.reset_from(self.board, self.agents)
            self.mcts.root.simulator.turn = self.turn_number
            self.mcts = RayMCTSQFSearch(self.mcts.root.simulator, self.qf, n_threads=self.nthreads)
#            self.mcts.simulator.reset_from(self.board, self.agents)
#            self.mcts.update_root(self.board)
#            self.mcts.root.turn_number = 0
            self.running_mcts = False

            self.mcts_results = None

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
        print(self.mcts.root.simulator.player_idx)
        self.mcts.root.simulator.render()
        self.mcts.sigstop = False
        self.running_mcts = True

        self.mcts.search(max_time=self.max_time, c=self.exploration, verbose=True)

        self.running_mcts = False

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
            print('Settlement = {}, Road = {}, Win Rate = {:.4f}, Stats = {}'.format(k[1][0], k[1][1], k[2], k[0]))
        
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
        if self.edit_hex_idx >= 0:
            self.renderer.highlight_tile(self.board.tiles[self.edit_hex_idx])
        if self.edit_port_idx >= 0:
            self.renderer.highlight_port(self.board, self.board.port_locations[self.edit_port_idx])
        self.mcts_button.render()
        self.clear_mcts_button.render()
        self.show_prod_button.render()
        self.place_button.render()
        self.randomize_button.render()
        font = pygame.font.SysFont(None, 24)
        text = '{} to place'.format(['Nobody', 'Red', 'Green', 'Blue', 'Yellow'][self.get_player()])
        img = font.render(text, True, (0, 0, 0))
        self.screen.blit(img, (25, 25))
        if self.mcts_results:
            self.renderer.draw_mcts_results(self.board, self.mcts_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Args for MCTS')
    parser.add_argument('--nthreads', type=int, required=False, default=1, help='The number of processes to use for the tree search')
    parser.add_argument('--nsamples', type=int, required=False, default=1, help='The number of games to play at each rollout')
    parser.add_argument('--c', type=float, required = False, default=1.0, help='Exploration factor for MCTS (1.0 default, more=more exploration, less=more exploitation)')
    parser.add_argument('--max_time', type=float, required = False, default=300.0, help='Optional max time to run MCTS for')
    parser.add_argument('--net_path', type=str, required=True, help='Path to the neural network')
    args = parser.parse_args()
    pygame.init()

    net = torch.load(args.net_path)

    # Set up the drawing window
    screen = pygame.display.set_mode([900, 800])
    pygame.display.set_caption('Catan Placement Assist')

    ui = BoardPlacementUI(screen, nthreads = args.nthreads, nsamples = args.nsamples, exploration = args.c, maxtime = args.max_time, network=net)
    signal.signal(signal.SIGINT, ui.handle_stop)

    # Run until the user asks to quit
    running = True
    while running:
    # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                ui.handle_click(event)
            if event.type == pygame.KEYDOWN:
                ui.handle_keystroke(event)
                print(ui.keystroke_buf)
            if event.type == pygame.QUIT:
                running = False

        # Flip the display
        pygame.display.flip()
        ui.render()
        time.sleep(0.1)


    # Done! Time to quit.
    pygame.quit()
