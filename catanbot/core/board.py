import matplotlib.pyplot as plt
import numpy as np
import copy

from catanbot.util import to_one_hot

class Board:
    """
    Base class for catan board. Should store positions of roads, settlements, ports, and production tiles
    I think the best option is to put all the positions in numpy arrays and keep track of each entitiy's state

    For reference, I'll use the following for resources:
    0: Desert
    1: Ore
    2: Wheat
    3: Sheep
    4: Wood
    5: Brick

    For players:
    1: Red
    2: Bue
    3: Yellow
    4: Green

    For Settlements:
    1: Settlement
    2: City

    For Dev Cards:
    1: Knight
    2: VP
    3: YOP
    4: RB
    5: M
    """
    def __init__(self):
        self.tiles = np.zeros((19, 5)).astype(int) #resouece type, dice val, x, y, is_blocked
        self.roads = np.zeros((72, 3)).astype(int) #occupied_player, settlement 1, settlement 2
        self.settlements = np.zeros((54, 11)).astype(int) #player, settlement type, port, x_pos, y_pos, r1, r2, r3, t1, t2, t3
        self.port_locations = np.zeros((9, 4)).astype(int) #The two settlement spots that access this port, then the coordinates for rendering: s1, s2, x, y
        self.dev_cards = np.zeros(25)
        self.dev_idx = 0

        self.tiles[:, 2] = np.array([6, 12, 18, 3, 9, 15, 21, 0, 6, 12, 18, 24, 3, 9, 15, 21, 6, 12, 18])
        self.tiles[:, 3] = np.array([24, 24, 24, 18, 18, 18, 18, 12, 12, 12, 12, 12, 6, 6, 6, 6, 0, 0, 0])

        self.settlements[:, 3] = np.array([6, 12, 18, 3, 9, 15, 21, 0, 6, 12, 18, 24, 3, 9, 15, 21, 27, 6, 12, 18, 24, 9, 15, 21, 9, 15, 21, 6, 12, 18, 24, 3, 9, 15, 21, 27, 0, 6, 12, 18, 24, 3, 9, 15, 21, 6, 12, 18, 3, 0, -3, -3, 0, 3])
        self.settlements[:, 4] = np.array([28, 28, 28, 22, 22, 22, 22, 16, 16, 16, 16, 16, 10, 10, 10, 10, 10, 4, 4, 4, 4, -2, -2, -2, 26, 26, 26, 20, 20, 20, 20, 14, 14, 14, 14, 14, 8, 8, 8, 8, 8, 2, 2, 2, 2, -4, -4, -4, 26, 20, 14, 10, 4, -2])
        self.settlements[:, 5] = np.array([-1, -1, -1, 48, 49, 50 ,51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 64, 65, 66, 67, 69, 70, 71, 1, 2, -1, 4, 5, 6, -1, 8, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 3, 7, 57, 63, 68])
        self.settlements[:, 6] = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, -1, 42, 43, 44, -1, 46, 47, -1, 49, 50, 51, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, -1, -1, -1, 48, 52, 57, 36, 41, 45])
        self.settlements[:, 7] = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, -1, -1, -1, -1, -1, -1, ])
        self.settlements[:, 8] = np.array([-1, -1, -1, 0, 1, 2, -1, 3, 4, 5, 6, -1, 8, 9, 10, 11, -1, 13, 14, 15, -1, 17, 18, -1, -1, -1, -1, 0, 1, 2, -1, 3, 4, 5, 6, -1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, -1, -1, -1, 7, 12, 16])
        self.settlements[:, 9] = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, 16, 17, 18, -1, -1, -1, -1, 1, 2, -1, 4, 5, 6, -1, 8, 9, 10, 11, -1, 12, 13, 14, 15, -1, 16, 17, 18, -1, -1, -1, -1, 0, 3, 7, -1, -1, -1])
        self.settlements[:, 10] = np.array([-1, -1, -1, -1, 0, 1, 2, -1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, 12, 13, 14, 15, -1, 16, 17, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1])

        self.roads[:, 1] = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 36, 12, 13, 14, 15, 41, 17, 18, 19, 45, 21, 22, 3, 4, 5, 6, 7, 8, 9, 10, 11, 50, 12, 13, 14, 15, 16, 36, 17, 18, 19, 20, 41, 21, 22, 23])
        self.roads[:, 2] = np.array([48, 24, 25, 49, 27, 28, 29, 50, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 51, 37, 38, 39, 40, 52, 42, 43, 44, 53, 46, 47, 48, 24, 25, 26, 49, 27, 28, 29, 30, 51, 31, 32, 33, 34, 35, 52, 37, 38, 39, 40, 53, 42, 43, 44])

        self.port_locations[:, 0] = np.array([48, 1, 6, 35, 20, 22, 45, 52, 7])
        self.port_locations[:, 1] = np.array([0, 25, 30, 16, 44, 46, 53, 36, 49])
        self.port_locations[:, 2] = np.array([4, 14, 23, 28, 23, 14, 4, -2, -2])
        self.port_locations[:, 3] = np.array([29, 29, 23, 12, 2, -5, -5, 6, 18])
        
        self.tile_dist = np.array([0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5])
        self.value_dist = np.array([2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12])
        self.dev_dist = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5])
        self.port_dist = np.array([1, 2, 3, 4, 5, 6, 6, 6, 6]) #6 for a 3:1 port    

        neighbor_roads = self.settlements[:, 5:8]
        neighbors = np.ones([self.settlements.shape[0], 3]).astype(int) * -1

        self.production_info = self.compute_production()
        self.structure_tensor = self.get_structure_tensor()

        for i, road_idx_vec in enumerate(neighbor_roads.T):
            roads = self.roads[road_idx_vec]

            #Find the road endpoint that isn't you
            neighbor_settlement_idx = roads[np.arange(54), 1+np.argmax(roads[:, 1:] != np.tile(np.arange(54), [2, 1]).T, axis=1)]
            neighbor_settlement_idx[road_idx_vec == -1] = -1
            neighbors[:, i] = neighbor_settlement_idx

        self.settlement_neighbors = neighbors

    @property
    def observation(self):
        """
        For input to neural networks.
        An agent should have access to:
            1. Tile info (resources, roll#, robber status)
            2. Road info (occupation)
            3. Settlement info (occupation, occupation type, port)
        return as a dict of numpy arrs with these headings.
        """
        tile_info = np.copy(self.tiles[:, [1, 4]])
        tile_production = 6 - np.abs(7 - self.tiles[:, 1])
        tile_production[self.tiles[:, 0] == 0] = 0.
        tile_production_info = np.expand_dims(tile_production, axis=1)
        tile_resource_info = to_one_hot(self.tiles[:, 0], 6)
#        tile_resource_info = np.copy(self.tiles[:, [0]])
        tile_info = np.concatenate([tile_resource_info, tile_info, tile_production_info], axis=1)

        road_occ_info = np.copy(self.roads[:, [0]])
        road_info = to_one_hot(road_occ_info, 5)
#        road_info = road_occ_info

        settlement_occ_info = to_one_hot(self.settlements[:, 0], 5)
        settlement_port_info = to_one_hot(self.settlements[:, 2], 7)

        settlement_production_info = self.compute_production()[:, [1]]

#        settlement_occ_info = np.copy(self.settlements[:, [0]])
#        settlement_port_info = np.copy(self.settlements[:, [2]])
        settlement_info = np.copy(self.settlements[:, [1]])
        settlement_info = np.concatenate([settlement_occ_info, settlement_info, settlement_port_info, settlement_production_info], axis=1)
#        settlement_info = np.concatenate([settlement_occ_info, settlement_production_info], axis=1)

        return {
            'tiles':tile_info,
            'roads':road_info,
            'settlements':settlement_info,
        }

    @property
    def observation_space(self):
        return {
            'tiles':np.array(19*9),
            'roads':np.array(72*5),
            'settlements':np.array(54*14),
            'total':np.array(19*9+54*14+72*5)
        }

        """
        return {
            'tiles':np.array(1),
            'roads':np.array(11),
            'settlements':np.array(54*2),
            'total':np.array(1+54*2+1)
        }
        """

    def equals(self, b2):
        """
        Two boards are equal if the tiles, settlements, ports, roads and dev cards are the same. (I don't want to override the built-in equals just in case).
        """
        return np.array_equal(self.tiles, b2.tiles) and np.array_equal(self.roads, b2.roads) and np.array_equal(self.settlements, b2.settlements) and np.array_equal(self.dev_cards, b2.dev_cards)
        
    def reset(self, fair=True):
        tile_idxs = np.random.permutation(19)
        value_idxs = np.random.permutation(18)
        dev_idxs = np.random.permutation(25)
        port_idxs = np.random.permutation(9)
        self.tiles[:, 0] = self.tile_dist[tile_idxs]
        self.tiles[:, 1][self.tiles[:, 0] != 0] = self.value_dist[value_idxs]
        self.tiles[:, 1][self.tiles[:, 0] == 0] = 0
        if fair:
            self.tiles[:, 1] = self.randomize_fair(tile_idxs)

        self.tiles[:, 4] = 0
        self.tiles[np.argmin(self.tiles[:, 0]), 4] = 1 #Robber starts on desert
        self.roads[:, 0] *= 0
        self.settlements[:, [0, 1, 2]] *= 0
        self.dev_cards = self.dev_dist[dev_idxs]
        self.dev_idx = 0
        #idk why I can't just broadcast it
        for i, locs in enumerate(self.port_locations):
            self.settlements[locs[:2], 2] = self.port_dist[port_idxs[i]]

        self.production_info = self.compute_production()
        self.structure_tensor = self.get_structure_tensor()

    def randomize_fair(self, tile_idxs):
        """
        Randomize production values such that no settlement spot has more than one 6 or 8 (collectively)
        """
        adjacencies = np.ones([19, 6]).astype(int) * -1
        adjacencies[:, 0] = np.array([1, 0, 1, 0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14])
        adjacencies[:, 1] = np.array([3, 2, 5, 4, 1, 2, 5, 8, 4, 5, 6, 10, 8, 9, 10, 11, 13, 14, 15])
        adjacencies[:, 2] = np.array([4, 4, 6, 7, 3, 4, 10, 12, 7, 8, 9, 15, 13, 12, 13, 14, 17, 16, 17])
        adjacencies[:, 3] = np.array([-1, 5, -1, 8, 5, 6, 11, -1, 9, 10, 11, -1, 16, 14, 15, 18, -1, 18, -1])
        adjacencies[:, 4] = np.array([-1, -1, -1, -1, 8, 9, -1, -1, 12, 13, 14, -1, -1, 16, 17, -1, -1, -1, -1])
        adjacencies[:, 5] = np.array([-1, -1, -1, -1, 9, 10, -1, -1, 13, 14, 15, -1, -1, 17, 18, -1, -1, -1, -1])

        value_idxs = np.random.permutation(18)
        value_ordering = self.value_dist[value_idxs]
        desert_idx = np.argmax(self.tile_dist[tile_idxs] == 0)
        value_ordering = np.insert(value_ordering, desert_idx, 0)
        high_prod_idxs = np.argwhere((value_ordering == 6) | (value_ordering == 8)).flatten()
        np.random.shuffle(high_prod_idxs)
        adj = adjacencies[high_prod_idxs] 
        for i, idx in enumerate(high_prod_idxs):
            if idx in adj:
                #6/8 borders another. Look for available swap spot
                adj_flat = np.unique(adj.flatten())
                adj_flat = adj_flat[(adj_flat >= 0)]
                invalid_idxs = np.concatenate([adj_flat, high_prod_idxs, np.array([desert_idx])])
                valid_idxs = np.setdiff1d(np.arange(19), invalid_idxs)
                if valid_idxs.size > 0:
                    valid_idx = np.random.choice(valid_idxs)
                else:
                    valid_idx = idx
                value_ordering[[idx, valid_idx]] = value_ordering[[valid_idx, idx]]
                high_prod_idxs[i] = valid_idx
                adj = adjacencies[high_prod_idxs] 

        return value_ordering

    def reset_from_string(self, tile_string, port_string):
        """
        Resets the Catan board according to a comma-separated string of the form:
        <R1><V1>, <R2><V2>, ..., where R is one of {0-5} for resources, and V is one of {0-12} (0 for the desert)
        Ports are a comma-separated string going clockwise from the top left.
        """
        self.reset()
        tokens = tile_string.split(',')
        for i, token in enumerate(tokens):
            t = token.strip()
            r = int(t[0])
            d = int(t[1:])
            self.tiles[i, 0] = r
            self.tiles[i, 1] = d

        tokens = port_string.split(',')
        for i, token in enumerate(tokens):
            t = token.strip()
            r = int(t[0])
            self.settlements[self.port_locations[i, 0], 2] = r
            self.settlements[self.port_locations[i, 1], 2] = r
        self.production_info = self.compute_production()
        self.structure_tensor = self.get_structure_tensor()

    def has_dev_cards(self):
        return self.dev_idx < 25

    def get_dev_card(self):
        out = self.dev_cards[self.dev_idx]
        self.dev_idx += 1
        return out

    def generate_resources(self, val):
        """
        Computes resources for all players given a particular roll
        Returns as a 4x5 array where A[i, j] is the amount of resource j that player i gets
        """
        out = np.zeros((5, 6)).astype(int)
        r_tiles = np.argwhere((self.tiles[:, 1] == val) & ~self.tiles[:, 4])
        for r_idx in r_tiles:
            s = self.settlements[(self.settlements[:, 8:] == r_idx).any(axis=1)]
            r = self.tiles[r_idx].flatten()
            out[s[:, 0], r[0]] += s[:, 1]
        return out

    def place_robber(self, tile):
        """
        Puts the robber on the specified tile. Unblocks all other tiles.
        """
        assert tile >= 0 and tile < 19, 'Not a valid tile to block'
        self.tiles[:, 4] = 0
        self.tiles[tile, 4] = 1

    def place_settlement(self, loc, player, city):
        """
        Puts a settlement on the location specified
        Args:
        loc: the id to place the settlement on
        player: the player whose settlement to place
        city: True for city, False for settlement
        """
        assert loc > -1 and loc < 54, 'settlement spots from 0-53'
        assert player > 0 and player < 5, 'player from 1-4'
        self.settlements[loc, 0] = player
        self.settlements[loc, 1] = 2 if city else 1

    def place_road(self, loc, player):
        assert loc > -1 and loc < 72, 'road spots from 0-72'
        assert player > 0 and player < 5, 'player from 1-4'
        self.roads[loc, 0] = player

    def compute_settlement_spots(self):
        """
        Valid settlement spot are those that are not taken, and are two hops away from anyother 
        """
        invalid_idxs = np.argwhere(self.settlements[:, 0] != 0)
        roads = self.settlements[invalid_idxs, 5:8].flatten()
        roads = roads[roads != -1]
        invalid_idxs = np.unique(self.roads[roads, 1:])
        mask = np.ones(self.settlements.shape[0], dtype=bool)
        mask[invalid_idxs] = False
        return np.arange(self.settlements.shape[0], dtype=int)[mask]

    def compute_production(self):
        """
        Computes the pips of production at each settlement spot (returns as a [spot x [idx, production]] matrix)
        """
        out = np.zeros(54)
        for i, s in enumerate(self.settlements):
            p = 0
            for tidx in s[8:]:
                if tidx != -1:
                    v = self.tiles[tidx][1]
                    p += max(6 - abs(7 - v), 0)
            out[i] = p
        return np.stack([np.arange(54), out], axis=1).astype(int)

    def count_settlements_for(self, player):
        player_spots = self.settlements[self.settlements[:, 0] == player]
        s = player_spots[player_spots[:, 1] == 1]
        return s.shape[0]

    def count_cities_for(self, player):
        player_spots = self.settlements[self.settlements[:, 0] == player]
        s = player_spots[player_spots[:, 1] == 2]
        return s.shape[0]
    
    def count_roads_for(self, player):
        player_spots = self.roads[self.roads[:, 0] == player]
        return player_spots.shape[0]

    def compute_longest_road(self):
        """
        Compute each player's longest road.
        """
        maxlen = np.array([self.compute_longest_road_for(i+1) for i in range(4)])
        return maxlen

    def compute_longest_road_for(self, player):
        """
        Given the sparsity of the Catan graph, it should be more efficient to use BFS from each vertex. (O(V*E), E ~ V, O(V^2))
        NOTE: Other player settlements can split longest road
        """
        edges = self.roads[self.roads[:, 0] == player]
        #Building the adjacency list makes my life a lot easier
        alist = {}
        for e in edges:
            if not e[1] in alist.keys():
                alist[e[1]] = set()
            if not e[2] in alist.keys():
                alist[e[2]] = set()

            alist[e[1]].add(e[2])
            alist[e[2]].add(e[1])
        
        maxlen = self.bfs_longest_path(alist)

        if maxlen == -1:
            maxlen = self.exhaustive_longest_path(alist)
            
        return maxlen

    def exhaustive_longest_path(self, alist):
        """
        Brute-forces all paths from vertices of degree 1 to find longest path when there are cycles.
        """
        paths = []
        start_points = []
        maxlen = 0
        maxpath = None
        for source in alist:
            if len(alist[source]) == 1:
                start_points.append(source)

        for source in start_points:
            paths = self.get_paths(source, alist, set())
            for path in paths:
                if len(path) - 1 > maxlen:
                    maxpath = path
                maxlen = max(maxlen, len(path) - 1)

        return maxlen

    def get_paths(self, curr, alist, visited):
        """
        Expand a path by creating a new path for each unvisited neighbor
        """
        out = []
        neighbors = alist[curr]
        for v in neighbors:
            if v not in visited:
                visited_new = copy.deepcopy(visited)
                visited_new.add(curr)
                paths = self.get_paths(v, alist, visited_new)
                for path in paths:
                    out.append([curr] + path)
        if not out:
            return [[curr]]
        else:
            return out
            
                                    
    def bfs_longest_path(self, alist):
        """
        Given an adjacency list, find the longest path. Since this doesn't work if the graph has a cycle, return -1 if a cycle is detected.
        """
        maxlen = 0
        for source in alist.keys():
            visited = {source}
            prevs = {source:-1}
            frontier = [source]
            is_acyclic = True
            curr = 0
            #BFS
            while frontier:
                curr += 1
                f_new = []
                for u in frontier:
                    neighbors = alist[u]
                    for v in neighbors:
                        if not v in visited:
                            visited.add(v)
                            prevs[v] = u
                            f_new.append(v)
                        #Check for edge-backedge cycle. Ignore if it's that.
                        elif v != prevs[u]:
                            return -1
                frontier = f_new    
            maxlen = max(maxlen, curr - 1)
        return maxlen


    def render_base(self, fig = None, ax = None, display_ids=False):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        
        #resource tiles
        for i in range(19):
            tile = self.tiles[i]
            ax.scatter(tile[2], tile[3], s=512, color = self.get_color(tile[0]))
            ax.text(tile[2], tile[3], '{}/{}'.format(tile[1], 6 - abs(7 - tile[1])), color='r' if tile[4] else 'k', ha='center', va='center')

        #settlement spots
        for i, s in enumerate(self.settlements):
            buf = ''
            ax.scatter(s[3], s[4], s = 256 if s[0] else 64, c=self.get_player_color(s[0]), marker = '.' if s[1] != 2 else '*')
            if s[2] != 0:
                ax.scatter(s[3], s[4], s = 24, c='k')
                ax.scatter(s[3], s[4], s = 4, c='w')
            if display_ids:
                buf += str(i)
            ax.text(s[3], s[4], buf, ha = 'center' if len(buf) > 1 else 'right')

        for i, p in enumerate(self.port_locations):
            loc = self.port_locations[i, [2, 3]]
            resource = self.settlements[self.port_locations[i, 0], 2]
            rlabel = ['None', 'Ore', 'Wheat', 'Sheep', 'Wood', 'Brick', 'Any'][resource]

            ax.text(loc[0], loc[1], rlabel, fontsize=8)
        

        #roads
        for idx, i in enumerate(self.roads):
            x = np.array([self.settlements[i[1]][3], self.settlements[i[2]][3]])
            y = np.array([self.settlements[i[1]][4], self.settlements[i[2]][4]])
            if i[0] != 0:
                ax.plot(x, y, c=self.get_player_color(i[0]))
            if display_ids:
                ax.text(x.mean(), y.mean(), idx, fontsize=6, ha='center')

        ax.set_xlim(-6, 30)
        ax.set_ylim(-6, 30)
        return fig, ax

    def render_pips(self, fig, ax):
        """
        Renders the pip values of each settlemetn square
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        fig, ax = self.render_base(fig, ax)

        for s in self.settlements:
            p = 0
            for tidx in s[8:]:
                if tidx != -1:
                    v = self.tiles[tidx][1]
                    p += max(6 - abs(7 - v), 0)
            ax.text(s[3], s[4], p)

    def render(self):
        fig, axs = plt.subplots(1, 2, figsize = (14, 6))
        self.render_base(fig, axs[0], display_ids=True)
        self.render_pips(fig, axs[1])

    def get_color(self, i):
        table = ['k', '0.75', '#fbff00', '#7da832', '#633906', '#ff8c00']
        return table[int(i)]

    def get_player_color(self, i):
        table = ['k', 'r', 'g', 'b', 'y']
        return table[i]

    @property
    def graph_observation_space(self):
        obs = self.graph_observation()
        return {
            'vertices':np.array(obs['vertices'].shape),
            'edges':np.array(obs['edges'].shape),
        }

    #GNN Nonsense
    def graph_observation(self):
        """
        For input to graph neural networks (hopefully this can help with generalization)
        For this GNN problem, we can assume constant graph structure. As such, use a different func to get incidence/adjecency matrices.
        V: A 54 x n tensor containing:
            1. Occupancy info
            2. Settlement type info
            3. Production info (resource types, production values, roll numbers)
            4. Port info

        E: A 72 x n tensor containing:
            1. Occupancy info (thats it)
            
        """
        vertex_occupancy = self.settlements[:, 0]
        vertex_type = self.settlements[:, 1]
        vertex_port = self.settlements[:, 2]
        vertex_tiles = self.settlements[:, -3:]

        vertex_occupancy = to_one_hot(vertex_occupancy, 5)
        vertex_port = to_one_hot(vertex_port, 7)
        vertex_production = self.production_info[:, [1]]

        vertex_tile_info = self.tiles[vertex_tiles]
        vertex_tile_type = vertex_tile_info[:, :, 0]
        vertex_tile_dice_val = vertex_tile_info[:, :, [1]]
        vertex_tile_production = np.clip(6-np.abs(7 - vertex_tile_dice_val), 0, 5)
        vertex_tile_blocked = vertex_tile_info[:, :, [-1]]

        vertex_tile_type = np.stack([to_one_hot(x, 6) for x in vertex_tile_type.T], axis=1)

        vertex_tile_feats = np.concatenate([vertex_tile_type, vertex_tile_dice_val, vertex_tile_production, vertex_tile_blocked], axis=2)
        vertex_tile_feats[vertex_tiles == -1] = -1 #Mask placeholders (use -1, not 0)

        vertex_tile_feats = np.reshape(vertex_tile_feats, [54, -1])

        vertex_feats = np.concatenate([vertex_occupancy, vertex_production, vertex_port, vertex_tile_feats], axis=1)

        edge_occupancy = self.roads[:, 0]

        edge_occupancy_one_hot = to_one_hot(edge_occupancy, 5)

        edge_feats = edge_occupancy_one_hot

        return {
            'vertices': vertex_feats,
            'edges':edge_feats
        }

    def get_structure_tensor(self):
        """
        A 54 x 3 x 2 tensor that encodes the structure of the Catan board. Essentially, it's an adjacency list indexed as:
        [vertex dim x edge dim x {vertexid/edgeid}]
        """
        road_neighbors = self.settlements[:, 5:8]
        settlement_neighbors = self.roads[road_neighbors][:, :, 1:]
        mask = ~(settlement_neighbors == np.expand_dims(np.expand_dims(np.arange(54), axis=1), axis=2))
        settlement_neighbors = np.sum(mask * settlement_neighbors, axis=2).astype(int)
        settlement_neighbors[road_neighbors == -1] = -1

        return np.stack([settlement_neighbors, road_neighbors], axis=2)

if __name__ == '__main__':
    board = Board()
    board.reset()
    obs = board.graph_observation()
    import pdb;pdb.set_trace()

    for i in range(10):
        board.reset()
        board.render();plt.show()
   

    board.place_settlement(3, 1, False)
    board.place_settlement(23, 2, False)
    board.place_settlement(18, 3, False)
    board.place_settlement(34, 4, False)
    board.place_settlement(13, 1, False)
    board.place_settlement(25, 2, True)
    board.place_settlement(46, 3, False)
    board.place_settlement(51, 4, True)
    board.place_road(3, 1)
    board.place_road(23, 2)
    board.place_road(18, 3)
    board.place_road(34, 4)

    print(board.graph_observation())
    import pdb;pdb.set_trace()
    print(board.observation)
    print(board.observation_space)

    for i in range(2, 13):
        print('roll = {}'.format(i))
        print(b.generate_resources(i))

    board.render()
    plt.show()

    spots = board.compute_settlement_spots()
    print(spots)
    fig, ax = board.render_base()
    ax.scatter(board.settlements[spots][:, 3], board.settlements[spots][:, 4], marker='x', c='r')
    plt.show()
    

    """
    for s in b.settlements:
        print('s {} = {}'.format(cnt, s))
        fig, ax = b.render()
        r1idx = s[4]
        r2idx = s[5]
        r3idx = s[6]
        t1idx = s[7]
        t2idx = s[8]
        t3idx = s[9]

        ax.scatter(s[2], s[3])

        if t1idx != -1:
            ax.scatter(b.tiles[t1idx][2], b.tiles[t1idx][3], s=512, color='r')
        if t2idx != -1:
            ax.scatter(b.tiles[t2idx][2], b.tiles[t2idx][3], s=512, color='g')
        if t3idx != -1:
            ax.scatter(b.tiles[t3idx][2], b.tiles[t3idx][3], s=512, color='b')

        if r1idx != -1:
            ax.plot([b.settlements[b.roads[r1idx][1]][2], b.settlements[b.roads[r1idx][2]][2]], [b.settlements[b.roads[r1idx][1]][3], b.settlements[b.roads[r1idx][2]][3]], c='r')
        
        if r2idx != -1:
            ax.plot([b.settlements[b.roads[r2idx][1]][2], b.settlements[b.roads[r2idx][2]][2]], [b.settlements[b.roads[r2idx][1]][3], b.settlements[b.roads[r2idx][2]][3]], c='g')
        
        if r3idx != -1:
            ax.plot([b.settlements[b.roads[r3idx][1]][2], b.settlements[b.roads[r3idx][2]][2]], [b.settlements[b.roads[r3idx][1]][3], b.settlements[b.roads[r3idx][2]][3]], c='b')
        
        plt.show()
        cnt += 1
    """
