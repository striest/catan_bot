import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

from catanbot.core.independent_action_simulator import IndependentActionsCatanSimulator
from catanbot.core.board import Board
from catanbot.agents.independent_actions_agents import IndependentActionsHeuristicAgent, IndependentActionsAgent
from catanbot.board_placement.agents.base import InitialPlacementAgent

from catanbot.rl.replaybuffers.simple_replay_buffer import SimpleReplayBuffer
from catanbot.rl.collectors.initial_placement_collector import InitialPlacementCollector
from catanbot.constants import DEV, SETTLEMENT, CITY, ROAD
from catanbot.util import argsort2d

class InitialPlacementSimulator:
    """
    For Catan, initial placements are essentially their own MDP. As such, we make a simulator specifically for it.
    MDP def'n:
        States: same as simulator state
        Actions: two tensors, a 54-tensor for placement position and a 3-tensor for road position.
        Reward: 0 if not finished placing. 1 for simulated winner if playing. (should also probably penalize for invalid actions)
        Terminal: After 8 actions
        Transition: Same as board transitions.
    """
    def __init__(self, simulator, placement_agents):
        self.simulator = simulator
        self.initial_board = copy.deepcopy(self.simulator.board)
        self.players = placement_agents #Different agents than the ones in simulator.
        self.turn = 0

    @property
    def player_idx(self):
        """
        Gets idx of current player to place ([0,3])
        """
        return 3 - np.floor(np.abs(3.5 - self.turn)).astype(int)

    @property
    def observation_space(self):
        return self.simulator.observation_space

    @property
    def action_space(self):
        return self.players[0].action_space

    @property
    def observation(self):
        """
        We have to hack the turn order to get the correct player observation
        """
        temp = self.simulator.turn
        self.simulator.turn = self.player_idx
        obs = self.simulator.observation
        self.simulator.turn = temp
        return obs

    @property
    def terminal(self):
        return self.turn >= 8

    def reset(self, reset_board=False):
        if reset_board:
            self.simulator.base_reset()
        else:
            self.simulator.reset_from(self.initial_board)

        for player in self.players:
            player.board = self.simulator.board

        self.turn = 0

    def reward(self, n=1):
        """
        Can simulate multiple games from this point.
        """
        if not self.terminal:
            return np.zeros(4)
        else:
            scores = np.zeros(4)
            for _ in range(n):
                s_copy = copy.deepcopy(self.simulator)
                scores += s_copy.simulate()
            if np.sum(scores) > 0:
                scores /= np.sum(scores)
            return scores
                
    def step(self, action):
        """
        Action is a settlement placement and road placement.
        If you pick something invalid, give a random loc
        """
        act_mask = np.zeros([54, 3]).astype(bool)

        settlement_locs = np.zeros(54).astype(bool)
        settlement_locs[self.simulator.board.compute_settlement_spots()] = 1
        settlement_avail = np.expand_dims(settlement_locs, axis=1)

        road_choices = self.simulator.board.settlements[:, 5:8]
        road_avail = (road_choices != -1)
        road_open = self.simulator.board.roads[road_choices, 0] == 0

        act_mask = (road_avail & road_open & settlement_avail)

        assert np.sum(act_mask * action['placement']) > 0, 'Policy chose invalid action.'

        act = np.unravel_index(np.argmax(action['placement']), action['placement'].shape)
        settlement_act = act[0]
        road_act = self.simulator.board.settlements[settlement_act, 5 + act[1]]

        self.simulator.board.place_settlement(settlement_act, self.player_idx+1, False)
        self.simulator.board.place_road(road_act, self.player_idx+1)
        self.simulator.vp[self.player_idx] += 1

        if self.turn > 3:
            tiles = self.simulator.board.settlements[settlement_act, 8:]
            tiles = tiles[tiles != -1]
            resources = self.simulator.board.tiles[tiles, 0]
            for r in resources:
                if r > 0:
                    self.simulator.players[self.player_idx].resources[r] += 1
        self.simulator.update_trade_ratios() 

        self.turn += 1 

    def random_act(self):
        return {
            'settlement':np.random.rand(54),
            'road':np.random.rand(3)
        }

    def render(self):
        self.simulator.render()

class InitialPlacementSimulatorWithPenalty(InitialPlacementSimulator):
        
    def reward(self, n=1):
        """
        Can simulate multiple games from this point.
        Penalize agents for taking an action that doesn't net 7+ production.
        """
        if not self.terminal:
            return np.zeros(4)
        else:
            scores = np.zeros(4)
            for _ in range(n):
                s_copy = copy.deepcopy(self.simulator)
                scores += s_copy.simulate()
            if np.sum(scores) > 0:
                scores /= np.sum(scores)

            prods = self.simulator.board.compute_production()[:, 1]
            settlement_spots = self.simulator.board.settlements[:, 0]

            scores = np.zeros(4)
            for i in range(4):
                prod_i = prods[settlement_spots == i+1]
                if np.any(prod_i < 7):
                    scores[i] = -1.

            return scores
        
if __name__ == '__main__':
    b = Board()
    b.reset()
#    agents = [IndependentActionsAgent(b), IndependentActionsAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsAgent(b)]
    agents = [IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b)]
#    agents = [HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b)]
    s = IndependentActionsCatanSimulator(board=b, players = agents, max_vp=10)
    placement_agents = [InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b)]
    placement_simulator = InitialPlacementSimulator(s, placement_agents)

    import pdb;pdb.set_trace()
    collector = InitialPlacementCollector(placement_simulator)
    rollout = collector.get_rollouts(10)
    buf = SimpleReplayBuffer(placement_simulator, capacity = int(1e6))

    buf.insert(rollout)
    print(buf.sample(10))



