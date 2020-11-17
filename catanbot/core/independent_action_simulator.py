import matplotlib.pyplot as plt
import numpy as np
import random
import time

from catanbot.core.board import Board
from catanbot.agents.base import Agent
from catanbot.agents.independent_actions_agents import IndependentActionsHeuristicAgent, IndependentActionsAgent
from catanbot.constants import DEV, SETTLEMENT, CITY, ROAD
from catanbot.core.simulator import CatanSimulator

class IndependentActionsCatanSimulator(CatanSimulator):
    """
    This simulator improves on the old one in the following ways:
        1. Agents can move the robber around
        2. Action space is different. Now, agents provide four vectors of size 54, 72, 19, 1 (nsettlements, nroads, ntiles, dev).
           The simulator will check if the actions are valid and execute the ones in descending order of score.
    """
    def compute_settlements(self, pidx):
        """
        Returns all valid locations to player at pidx to build a settlement, and its cost.
        """
#        print('computing settlements...')
        player = self.players[pidx]
        cost = self.trade_for(pidx, SETTLEMENT)
        if (player.resources - cost >= 0).all() and self.board.count_settlements_for(pidx+1) < 5:
            avail_spots = self.board.compute_settlement_spots()
            road_spots = self.board.roads[self.board.roads[:, 0] == 1 + pidx][:, 1:].flatten()
            valid_spots = np.intersect1d(avail_spots, road_spots)
            out = np.zeros(54)
            out[valid_spots] = 1.
            return out, cost
        else:
            return np.zeros(54), cost

    def compute_cities(self, pidx):
        """
        Returns all the valid locations for player at pidx to build a city, and its cost.
        """
#        print('computing cities...')
        player = self.players[pidx]
        cost = self.trade_for(pidx, CITY)
        if (player.resources - cost >= 0).all() and self.board.count_cities_for(pidx+1) < 4:
            spots = np.argwhere((self.board.settlements[:, 0] == 1+pidx) & (self.board.settlements[:, 1] == 1)).flatten()
            out = np.zeros(54)
            out[spots] = 1.
            return out, cost
        else:
            return np.zeros(54), cost

    def compute_roads(self, pidx):
        """
        Returns all valid locations for player pidx to build a road, and the associated resource cost.
        """
#        print('computing roads...')
        player = self.players[pidx]
        cost = self.trade_for(pidx, ROAD)
        if (player.resources - cost >= 0).all() and self.board.count_roads_for(pidx+1) < 15:
            road_spots = np.argwhere(self.board.roads[:, 0] == 1 + pidx).flatten()
            settlement_spots = np.concatenate([np.argwhere((self.board.settlements[:, 5:8] == r).any(axis=1)) for r in road_spots]).flatten()
            one_hop_road_spots = np.concatenate([np.argwhere((self.board.roads[:, 1:3] == s).any(axis=1)) for s in settlement_spots]).flatten()
            avail_road_spots = np.argwhere(self.board.roads[:, 0] == 0).flatten()
            valid_road_spots = np.intersect1d(avail_road_spots, one_hop_road_spots)
            
            out = np.zeros(72)
            out[valid_road_spots] = 1.
            return out, cost

        else:
            return np.zeros(72), cost

    def compute_devs(self, pidx):
        """
        Returns whether it's valid for player pidx to buy a dev card and its associated cost.
        """
#        print('computing devs...')
        player = self.players[pidx]
        cost = self.trade_for(pidx, DEV)
        if (player.resources - cost >= 0).all() and self.board.has_dev_cards():
            return np.ones(1), cost
        else:
            return np.zeros(1), cost

    def assign_resources(self):
        roll = np.random.randint(1, 7) + np.random.randint(1, 7)
        self.roll = roll
        resources = self.board.generate_resources(roll)
        for i, player in enumerate(self.players):
            player.resources += resources[i + 1]
        return roll

    def robber_act(self, robber_act, thr=0.5, force=False):
        """
        Given the action vector, place the robber on the argmax square that's over thr and isn't currently blocked.
        returns whether the robber was moved.

        Args:
            robber_act: a 19-elem array scoring the value of robbing that tile
            thr: Don't act if highest val below this threshold
            force: Force an option (i.e. for 7 roll)
        """
        curr_spot = np.argmax(self.board.tiles[:, 4])
        options = np.argsort(robber_act)
        options = options[options != curr_spot]

        if not force:
            options = options[robber_act[options] > thr]

        if options.size == 0:
            return False
        else:
#            print('moved robber')
            self.board.place_robber(options[-1])
            return True

    def city_act(self, pidx, city_act, build_locs, cost, city, thr=0.5):
        """
        Build a city on this tile if over a certain threshold. Take the highest score. (also assumes validity)
        Returns
            Whether the city was built

        args:
            city_act: A 54-elem tensor scoring each potential city location
            build_locs: A 54-elem tensor saying whether it's possible to buils a city there
            cost: the resource cost for building a city
            city: Whether to place city or settlement
            thr: Don't build unless above the threshold.
        """
        assert np.sum(build_locs) > 0, 'tried to build a city when no valid locations exist.'
        valid_idxs = np.argwhere(build_locs).flatten()
        scores = city_act[valid_idxs]
        prefs = np.argsort(-scores) #Sort descending
        if scores[prefs[0]] > thr:
#            print('bought {}'.format('city' if city else 'settlement'))
            self.board.place_settlement(valid_idxs[prefs[0]], pidx+1, city)
            self.players[pidx].resources -= cost
            self.vp[pidx] += 1
            return True
        else:
            return False
        
    def road_act(self, pidx, road_act, build_locs, cost, thr=0.5):
        """
        Build a city on this tile if over a certain threshold. Take the highest score. (also assumes validity)
        Returns
            Whether the city was built

        args:
            city_act: A 54-elem tensor scoring each potential city location
            build_locs: A 54-elem tensor saying whether it's possible to buils a city there
            cost: the resource cost for building a city
            city: Whether to place city or settlement
            thr: Don't build unless above the threshold.
        """
        assert np.sum(build_locs) > 0, 'tried to build a road when no valid locations exist.'
        valid_idxs = np.argwhere(build_locs).flatten()
        scores = road_act[valid_idxs]
        prefs = np.argsort(-scores) #Sort descending
        if scores[prefs[0]] > thr:
#            print('bought road')
            self.board.place_road(valid_idxs[prefs[0]], pidx+1)
            self.players[pidx].resources -= cost
            return True
        else:
            return False

    def dev_act(self, pidx, dev_act, cost, thr=0.5):
        """
        Buy dev if weight above threshold
        """
        if dev_act < thr:
            return False
        else:
#            print('bought dev card')
            self.players[pidx].resources -= cost

            card = self.board.get_dev_card()
            self.players[pidx].dev[card] += 1

            if card == 2:#is VP
#                print('got VP')
                self.vp[pidx] += 1
            
    def discard_cards(self):
        """
        Remove cards from players with more than 7 resources.
        I think it's a small enough loop that vectorization is unnecessary
        TODO: Vecotrize it anyway
        """
        for player in self.players:
            r_count = np.sum(player.resources)
            if r_count > 7:
                d = r_count//2
                for _ in range(d):
                    player.resources[np.random.choice(np.argwhere(player.resources > 0).flatten())] -= 1

    def step(self, action, check = False):
        """
        Unlike the old board, we'll step using the following scheme.
        
        1. Check for knights and move robber to the argmax tile if above a threshold
        2. Roll the die and assign resources.
        3. If roll was 7, do robber placement again.
        4. look at the city/settlement action. Build cities/settlements for elems over a threshold in descending order.
        5. look at roads. Similarly, build until all vals below certain value/outof resources
        6. build settlements (do this after roads to allow for road+settlement in a single turn).
        7. look to buy dev cards.

        action given as a dict of {'settlements':[54], 'roads':[72], 'dev':[1], 'tiles':[19]}
        """
        pval = self.turn + 1

        #PHASE 1: Robber via knight
        if self.players[self.turn].dev[1] > 0: #If player has knight, they can move it.
            played = self.robber_act(action['tiles'], force=False)
            if played:
                self.players[self.turn].dev[1] -=1
                self.knights_played[self.turn] += 1
                self.check_largest_army()

        #PHASE 2: Assign resources
        roll = self.assign_resources()
#        print('Roll = {}'.format(roll))

        #PHASE 3: Move robber if 7.
        if roll == 7:
            self.robber_act(action['tiles'], force=True)
            self.discard_cards()

        #PHASE 4: Build cities
        build_locs, cost = self.compute_cities(self.turn)
        while np.sum(build_locs) > 0:
            played = self.city_act(self.turn, action['settlements'], build_locs, cost, city=True)
            build_locs, cost = self.compute_cities(self.turn)
            if not played:
                break
       
        #PHASE 4b: Build settlements
        build_locs, cost = self.compute_settlements(self.turn)
        while np.sum(build_locs) > 0:
            played = self.city_act(self.turn, action['settlements'], build_locs, cost, city=False)
            build_locs, cost = self.compute_settlements(self.turn)
            if played:
                self.update_trade_ratios()
            else:
                break

        #PHASE 5: Build roads
        build_locs, cost = self.compute_roads(self.turn)
        while np.sum(build_locs) > 0:
            played = self.road_act(self.turn, action['roads'], build_locs, cost)
            build_locs, cost = self.compute_roads(self.turn)
            if played:
                self.check_longest_road()
            else:
                break

        #PHASE 6: Buy dev card 
        can_buy, cost = self.compute_devs(self.turn)
        while can_buy:
            played = self.dev_act(self.turn, action['dev'], cost)
            can_buy, cost = self.compute_devs(self.turn)
            if not played:
                break

        self.turn = (self.turn + 1) % 4
        self.nsteps += 1

if __name__ == '__main__':
    import copy
    b = Board()
    agents = [IndependentActionsAgent(b), IndependentActionsAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsAgent(b)]
#    agents = [IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b)]
#    agents = [HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b)]
    s = IndependentActionsCatanSimulator(board=b, players = agents, max_vp=10)
    s.reset_with_initial_placements()
    s_c = copy.deepcopy(s)

    print('INIT')
    s.render()

    t = time.time()
    cnt = 0
    games = 500
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
#            print(s.observation)
#            s.render()
            wins[np.argmax(s.vp)] += 1
            turns += s.nsteps
            s.reset_with_initial_placements()
            #s = copy.deepcopy(s_c)
            cnt += 1


#        s.render()
    
    print('Simulated {} games ({} turns) in {:.2f}s'.format(games, turns, time.time() - t))
    print('Win rates = {}'.format(wins))
    s.render()


