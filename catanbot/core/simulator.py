import matplotlib.pyplot as plt
import numpy as np
import copy
import random
import time

from catanbot.core.board import Board
from catanbot.agents.base import Agent
from catanbot.agents.heuristic_agent import HeuristicAgent
from catanbot.constants import DEV, SETTLEMENT, CITY, ROAD
from catanbot.util import to_one_hot

class CatanSimulator:
    """
    Actually simulates the Catan Game
    """
    def __init__(self, board, players = None, max_vp = 10, max_steps=500):
        self.board = board
        if players is None:
            self.players = [Agent(self.board) for _ in range(4)]
        else:
            self.players = players
        for i in range(4):
            self.players[i].pidx = i
        self.turn = 0
        self.nsteps = 0
        self.max_steps = max_steps
        self.roll = -1
        self.vp = np.zeros(4)
        self.max_vp = max_vp

        self.road_len = np.zeros(4, dtype=int)
        self.longest_road_pidx = -1
        self.knights_played = np.zeros(4, dtype=int)
        self.largest_army_pidx = -1

    def simulate(self, verbose=False):
        """
        simulates a full game and returns the winner
        """
        while not self.terminal:
            ts = time.time()
            act = self.players[self.turn].action()
            ta = time.time()
            if verbose:
                print('Act = {}'.format(act))
            self.step(act)
            te = time.time()

        return (self.vp == self.max_vp).astype(int)

    @property
    def action_space(self):
        return self.players[0].action_space

    @property
    def observation_space(self):
        return {
            'board':self.board.observation_space,
            'player':self.players[self.turn].observation_space,
            'vp':np.array(4),
            'army':np.array(4),
            'road':np.array(4),
            'total':np.array(self.board.observation_space['total'] + self.players[self.turn].observation_space['total'] + 12)
        }

    @property
    def observation(self):
        """
        For neural networks, combine the observations from current player and board
        Add current VP, largest army and longest road.
        """
        board_obs = self.board.observation
        player_obs = self.players[self.turn].observation
        vp = self.vp
        army = np.zeros(4)
        road = np.zeros(4)
        if self.largest_army_pidx != -1:
            army[self.largest_army_pidx] = 1.
        if self.longest_road_pidx != -1:
            road[self.longest_road_pidx] = 1.

        return {
            'board':board_obs,
            'player':player_obs,
            'vp':vp,
            'army':army,
            'road':road
        }

    @property
    def graph_observation_space(self):
        return {
            'board':self.board.graph_observation_space,
            'player':self.players[self.turn].observation_space,
            'vp':np.array(4),
            'army':np.array(4),
            'road':np.array(4),
            'total':np.array(self.players[self.turn].observation_space['total'] + 12)
        } 

    @property
    def graph_observation(self):
        board_obs = self.board.graph_observation()
        player_obs = self.players[self.turn].observation
        vp = self.vp
        army = np.zeros(4)
        road = np.zeros(4)
        if self.largest_army_pidx != -1:
            army[self.largest_army_pidx] = 1.
        if self.longest_road_pidx != -1:
            road[self.longest_road_pidx] = 1.

        return {
            'board':board_obs,
            'player':player_obs,
            'vp':vp,
            'army':army,
            'road':road
        }

    @property
    def observation_flat(self):
        obs = self.observation
        board_obs_flat = np.concatenate([obs['board'][k].flatten() for k in ['tiles', 'roads', 'settlements']])
        player_obs_flat = np.concatenate([obs['player'][k].flatten() for k in ['pidx', 'resources', 'dev', 'trade_ratios']])
        obs_flat = np.concatenate([board_obs_flat, player_obs_flat, obs['vp'], obs['army'], obs['road']])
        return obs_flat 

    @property
    def reward(self):
        """
        Unlike traditional RL, reward is a 4-tensor per player.
        simple reward though. Start w/ sparse and give 1 to winner.
        """
        rew = np.zeros(4)
        if self.terminal:
            rew[np.argmax(self.vp)] = 1.

        return rew
    
    @property
    def terminal(self):
        return self.vp.max() >= self.max_vp or self.nsteps > self.max_steps

    def base_reset(self):
        self.board.reset()
        for p in self.players:
            p.reset()
        self.turn = 0
        self.nsteps = 0
        self.vp *= 0
        self.update_trade_ratios()

        self.road_len = np.zeros(4, dtype=int)
        self.longest_road_pidx = -1
        self.knights_played = np.zeros(4, dtype=int)
        self.largest_army_pidx = -1

    def reset_from(self, board, players=None, nsteps=0, resources=None):
        """
        Start game from seed board (with initial placements)
        """
        self.board = copy.deepcopy(board)
        if players is not None:
            self.players = players
        self.vp = self.vp * 0
        for i, p in enumerate(self.players):
            if players is None:    
                p.reset()
            self.vp[i] = board.settlements[board.settlements[:, 0] == i+1][:, 1].sum()
            p.board = board

        self.turn = 0
        self.nsteps = 0    
        self.assign_resources()
        self.update_trade_ratios()

        self.road_len = np.zeros(4, dtype=int)
        self.longest_road_pidx = -1
        self.knights_played = np.zeros(4, dtype=int)
        self.largest_army_pidx = -1

    def initial_placements(self):
        """
        Place initial settlements on the board.
        """
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
        self.update_trade_ratios()

        self.road_len = np.zeros(4, dtype=int)
        self.longest_road_pidx = -1
        self.knights_played = np.zeros(4, dtype=int)
        self.largest_army_pidx = -1

    def reset_with_initial_placements(self):
        self.base_reset()
        self.initial_placements()

    def step(self, action, check = False):
        pval = self.turn + 1

        #Checking actions is expensive. Avoid for simulation if necessary
        if check:
            avail_actions = self.compute_actions(self.turn)
            assert (action == avail_actions).all(axis=1).any(), 'invalid move'
        atype = action[0]
        aloc = action[1]
        acost = action[2:]

        #0=buy dev, 1=settlement, 2=city, 3=road, 4=pass
        if atype == 0:
            #print('bought a dev card (TODO: implement)')
            self.players[self.turn].resources -= acost
            card = self.board.get_dev_card()
            self.players[self.turn].dev[card] += 1
            if card == 1:#Is knight
                #PLACEHOLDER: Assume knights are immediately played
                self.knights_played[self.turn] += 1
                self.check_largest_army()

            if card == 2:#is VP
#                print('got VP')
                self.vp[self.turn] += 1
        elif atype == 1:
            self.board.place_settlement(aloc, pval, False)
            self.players[self.turn].resources -= acost
            self.vp[self.turn] += 1
            self.update_trade_ratios()
        elif atype == 2:
            self.board.place_settlement(aloc, pval, True)
            self.players[self.turn].resources -= acost
            self.vp[self.turn] += 1
        elif atype == 3:
            self.board.place_road(aloc, pval)
            self.players[self.turn].resources -= acost
            self.check_longest_road()

        self.turn = (self.turn + 1) % 4
        self.nsteps += 1
        self.assign_resources()

    def check_largest_army(self):
        kmax = self.knights_played.max()
        kmaxidx = np.argmax(self.knights_played)
        if kmax >= 3:
            if self.largest_army_pidx == -1:
                self.largest_army_pidx = kmaxidx
                self.vp[self.largest_army_pidx] += 2
            elif self.knights_played[self.largest_army_pidx] < kmax:
                self.vp[self.largest_army_pidx] -= 2
                self.largest_army_pidx = kmaxidx
                self.vp[self.largest_army_pidx] += 2

    def check_longest_road(self):
        """
        Updates values for longest road. If longest road less than 5, nobody gets points.
        Incumbent player keeps longest road in case of ties.
        """
        self.road_len = self.board.compute_longest_road()
        rmax = self.road_len.max()
        rmaxidx = np.argmax(self.road_len)
        if rmax >= 5:
            if self.longest_road_pidx == -1:
                self.longest_road_pidx = rmaxidx
                self.vp[self.longest_road_pidx] += 2
            elif self.road_len[self.longest_road_pidx] < rmax:
                self.vp[self.longest_road_pidx] -= 2
                self.longest_road_pidx = rmaxidx
                self.vp[self.longest_road_pidx] += 2
        
    def assign_resources(self):
        roll = np.random.randint(1, 7) + np.random.randint(1, 7)
        self.roll = roll
        resources = self.board.generate_resources(roll)
        for i, player in enumerate(self.players):
            player.resources += resources[i + 1]
        return roll

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
        Return actions as a matrix where row are [move type, loc, cost]
        """
        moves = [np.array([4, 0, 0, 0, 0, 0, 0, 0])]
        player = self.players[pidx]

        costs = np.stack([self.trade_for(pidx, r) for r in (DEV, SETTLEMENT, CITY, ROAD)], axis=0)

        #Compute dev card
        if (player.resources - costs[0] >= 0).all() and self.board.has_dev_cards():
            moves.append(np.concatenate([np.array([0, 0]), costs[0]], axis=0))

        #Compute settlements
        if (player.resources - costs[1] >= 0).all() and self.board.count_settlements_for(pidx+1) < 5:
            avail_spots = self.board.compute_settlement_spots()
            road_spots = self.board.roads[self.board.roads[:, 0] == 1 + pidx][:, 1:].flatten()
            valid_spots = np.intersect1d(avail_spots, road_spots)
            for spot in valid_spots:
                moves.append(np.concatenate([np.array([1, spot]), costs[1]], axis=0))

        #Compute cities
        if (player.resources - costs[2] >= 0).all() and self.board.count_cities_for(pidx+1) < 4:
            spots = np.argwhere((self.board.settlements[:, 0] == 1+pidx) & (self.board.settlements[:, 1] == 1)).flatten()
            for spot in spots:
                moves.append(np.concatenate([np.array([2, spot]), costs[2]], axis=0))

        #Compute roads
        if (player.resources - costs[3] >= 0).all() and self.board.count_roads_for(pidx+1) < 15:
            road_spots = np.argwhere(self.board.roads[:, 0] == 1 + pidx).flatten()
            settlement_spots = np.concatenate([np.argwhere((self.board.settlements[:, 5:8] == r).any(axis=1)) for r in road_spots]).flatten()
            one_hop_road_spots = np.concatenate([np.argwhere((self.board.roads[:, 1:3] == s).any(axis=1)) for s in settlement_spots]).flatten()
            avail_road_spots = np.argwhere(self.board.roads[:, 0] == 0).flatten()
            valid_road_spots = np.intersect1d(avail_road_spots, one_hop_road_spots)

            for spot in valid_road_spots:
                moves.append(np.concatenate([np.array([3, spot]), costs[3]], axis=0))

        return np.stack(moves, axis=0)

    def update_trade_ratios(self):
        for i, player in enumerate(self.players):
            player.trade_ratios = self.compute_trade_ratios(i)

    def compute_trade_ratios(self, pidx):
        """
        Compute the trade ratios/ports for player pidx
        """
        ratios = 4 * np.ones(6).astype(int)
        ratios[0] = 100000
        settlements = self.board.settlements[self.board.settlements[:, 0] == 1+pidx]
        ports = settlements[settlements[:, 2] != 0][:, 2]
        if 6 in ports:
            ratios[1:] = 3
        ports = ports[ports < 6]
        ratios[ports] = 2
        return ratios
    
    def trade_for(self, pidx, target):
        """
        Computes the most resource-efficient way for player at pidx to get the resources in target. Returns a big val if impossible.
        """
        player = self.players[pidx]
        resource_diffs = player.resources - target
        cost = np.zeros(6).astype(int)
        if (resource_diffs >= 0).all():
            cost = target.copy()
        else:
            cost = target.copy()
            ncnt = -resource_diffs[resource_diffs < 0].sum()
            surpluses = (resource_diffs//player.trade_ratios).clip(0)
            pcnt = surpluses.sum()
            t_order = np.argsort(player.trade_ratios)
            cost[resource_diffs < 0] = player.resources[resource_diffs < 0]
            if pcnt < ncnt:
                cost += 100000
            else:
                ntrades = 0
                tidx = 0
                while ntrades < ncnt:
                    curr_r = t_order[tidx]
                    if surpluses[curr_r] <= 0:
                        tidx += 1
                        continue
                    cost[curr_r] += player.trade_ratios[curr_r]
                    surpluses[curr_r] -= 1
                    ntrades += 1
        return cost

    def render(self):
        print('rendering...')
        fig, axs = plt.subplots(1, 3, figsize = (18, 6))
        plt.title('Turn = {} ({}), roll = {}'.format(self.nsteps+1, 'RGBY'[self.turn], self.roll))
        self.board.render_base(fig, axs[0], display_ids=True)
        self.board.render_pips(fig, axs[1])
        txt = ""
        for i in range(4):
            txt += 'Player {} ({}):\n'.format(i+1, 'RGBY'[i])
            txt += str(self.players[i])
            txt += '\n\n'

        txt = txt[:-1]
        txt += 'Victory Points = {}\n'.format(self.vp)
        txt += '\n'
        txt += 'Longest Road: {} Current = {}\n'.format(self.road_len, 'RGBYN'[self.longest_road_pidx])
        txt += 'Largest Army: {} Current = {}\n'.format(self.knights_played, 'RGBYN'[self.largest_army_pidx])

        axs[2].text(0, 0, txt, fontsize=6)
        plt.show()

if __name__ == '__main__':
    import pdb;pdb.set_trace()
    import copy
    b = Board()
    agents = [HeuristicAgent(b), Agent(b), Agent(b), Agent(b)]
    agents = [HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b)]
    s = CatanSimulator(board=b, players = agents, max_vp=10)
    s.reset_with_initial_placements()
    s_c = copy.deepcopy(s)

    print('INIT')
    s.render()

    t = time.time()
    cnt = 0
    games = 50
    turns = 0
    wins = np.zeros(4)
#    print(s.simulate())
    
    #s.render()
#    exit(0)
    while cnt < games:
        print('Game {}'.format(cnt + 1))
        print('Turn = {}, ({})'.format(s.nsteps+1, 'RGBY'[s.turn]))
        print('Resources = {}'.format(s.players[s.turn].resources))
        act = s.players[s.turn].action()
        print('Act = {}'.format(act))
        s.step(act)

        if s.terminal:
            print('LONGEST ROAD:', s.board.compute_longest_road())
            s.render()
            wins[np.argmax(s.vp)] += 1
            turns += s.nsteps
            s.reset_with_initial_placements()
            #s = copy.deepcopy(s_c)
            cnt += 1

        s.render()
    
    print('Simulated {} games ({} turns) in {:.2f}s'.format(games, turns, time.time() - t))
    print('Win rates = {}'.format(wins))
    s.render()


