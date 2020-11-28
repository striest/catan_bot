import pygame
import numpy as np
from math import sin, cos, pi, atan2

from catanbot.core.board import Board

class BoardRenderer:
    """
    Takes in a board state and outputs the image to display
    """

    HEX_RADIUS = 2.8
    TOKEN_RADIUS = 1.2
    SETTLEMENT_RADIUS = 0.5
    ROAD_RADIUS = 0.3
    SUGGESTION_RADIUS = 0.2
    PORT_WIDTH = 2
    PORT_HEIGHT = 1.25 * PORT_WIDTH

    def __init__(self, screen, margin = 0.1):
        self.screen = screen
        self.margin = margin
        self.width, self.height = pygame.display.get_surface().get_size()
        self.font = pygame.font.SysFont(None, self.scale_fn(1.5))
        self.small_font = pygame.font.SysFont(None, self.scale_fn(1.0))
        self.show_prod = False


        self.tile_imgs = {
            0:'assets/images/desert.png',
            1:'assets/images/ore.png',
            2:'assets/images/wheat.png',
            3:'assets/images/sheep.png',
            4:'assets/images/lumber.png',
            5:'assets/images/brick.png'
        }

        self.port_imgs = {
            1:'assets/images/ships/ore.png',
            2:'assets/images/ships/wheat.png',
            3:'assets/images/ships/sheep.png',
            4:'assets/images/ships/lumber.png',
            5:'assets/images/ships/brick.png',
            6:'assets/images/ships/31.png',
        }

        self.port_text = {
            1: 'Ore 2:1',
            2: 'Wheat 2:1',
            3: 'Sheep 2:1',
            4: 'Wood 2:1',
            5: 'Brick 2:1',
            6: 'Any 3:1',
        }

        self.player_colors = {
            0:(0, 0, 0),
            1:(255, 0, 0),
            2:(0, 255, 0),
            3:(0, 0, 255),
            4:(255, 255, 0)
        }

    def render(self, board):
        """
        Note that the positions are hard-coded in the board state. We can just rescale them.
        """    
        self.screen.fill((255, 255, 255))
        self.draw_tiles(board)
        self.draw_roads(board)
        self.draw_settlements(board)
        self.draw_ports(board)

    def highlight_tile(self, tile):
        """
        Highlight a tile
        """
        resource = tile[0]
        die = tile[1]
        x_pos, y_pos = self.scale_fn((tile[2], tile[3]))
            
        hex_img = self.draw_hex(self.screen, (255, 255, 0), (x_pos, y_pos), self.scale_fn(self.HEX_RADIUS + 0.6), rot = pi/6)
        hex_img = self.draw_hex(self.screen, (0, 0, 0), (x_pos, y_pos), self.scale_fn(self.HEX_RADIUS + 0.2), rot = pi/6)
        hex_img = self.draw_hex(self.screen, (255, 255, 255), (x_pos, y_pos), self.scale_fn(self.HEX_RADIUS), rot = pi/6)
        img = pygame.image.load(self.tile_imgs[resource])
        img = pygame.transform.scale(img, hex_img.size)
        self.screen.blit(img, (x_pos - hex_img.width/2, y_pos - hex_img.height/2))
        self.draw_token(self.screen, (x_pos, y_pos), self.scale_fn(self.TOKEN_RADIUS), die)

    def highlight_port(self, board, port):
        s1 = port[0]
        s2 = port[1]
        port_type = board.settlements[s1, 2]
        assert port_type == board.settlements[s2, 2], 'Sam messed up the eventing for port cycling'
        x_pos, y_pos = self.scale_fn((port[2], port[3]))

        port = pygame.draw.rect(self.screen, (255, 255, 0), pygame.Rect(x_pos - self.scale_fn(self.PORT_WIDTH/2 + 0.1), y_pos - self.scale_fn(self.PORT_HEIGHT/2 + 0.1), self.scale_fn(self.PORT_WIDTH+0.2), self.scale_fn(self.PORT_HEIGHT+0.2)))
        port = pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(x_pos - self.scale_fn(self.PORT_WIDTH/2), y_pos - self.scale_fn(self.PORT_HEIGHT/2), self.scale_fn(self.PORT_WIDTH), self.scale_fn(self.PORT_HEIGHT)))
        img = pygame.image.load(self.port_imgs[port_type])
        img = pygame.transform.scale(img, port.size)
        self.screen.blit(img, (x_pos - port.width/2, y_pos - port.height/2))

    def draw_mcts_results(self, board, results):
        for rank, result in enumerate(results):
            if isinstance(result[1], list):
                node, acts, win_rate = result
                s_idx = acts[0]
                r_idx = board.settlements[s_idx, 5+acts[1]]
                player = node.simulator.player_idx
            else:
                s_node, r_node, win_rate = result
                s_idx = s_node.parent_action
                r_idx = r_node.parent_action
                player = s_node.player_from_turn(s_node.turn_number)

            self.draw_mcts_suggestion(board, s_idx, r_idx, win_rate, player, rank + 1)

            text = 'Option {}: Win rate = {:.1f}%'.format(rank + 1, 100*win_rate)

            img = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(img, (self.scale_fn(31), self.scale_fn(0.5 + 1.5*rank)))

    def draw_mcts_suggestion(self, board, s_idx, r_idx, win_rate, player, rank):
        settlement = board.settlements[s_idx]
        road = board.roads[r_idx]
        s_place = settlement
        s2 = road[2] if road[1] == s_idx else road[1]
        s_to = board.settlements[s2]
        x_pos1, y_pos1 = self.scale_fn((board.settlements[s_idx, 3], board.settlements[s_idx, 4]))
        x_pos2, y_pos2 = self.scale_fn((board.settlements[s2, 3], board.settlements[s2, 4]))

#        pygame.draw.line(self.screen, (0, 0, 0), (x_pos1, y_pos1), (x_pos2, y_pos2), self.scale_fn(self.SUGGESTION_RADIUS + 0.2))
#        pygame.draw.line(self.screen, (128, 128, 128), (x_pos1, y_pos1), (x_pos2, y_pos2), self.scale_fn(self.SUGGESTION_RADIUS))
#        self.draw_village(self.screen, (0, 0, 0), (x_pos1, y_pos1), self.scale_fn(self.SETTLEMENT_RADIUS + 0.2))
#        self.draw_village(self.screen, (128, 128, 128), (x_pos1, y_pos1), self.scale_fn(self.SETTLEMENT_RADIUS))

        ang = atan2(y_pos2 - y_pos1, x_pos2 - x_pos1)

        self.draw_thin_triangle(self.screen, (0, 0, 0), ((2*x_pos1 + x_pos2)/3, (2*y_pos1 + y_pos2)/3), self.scale_fn(1.2), rot = ang)
        self.draw_thin_triangle(self.screen, (52, 210, 235), ((2*x_pos1 + x_pos2)/3, (2*y_pos1 + y_pos2)/3), self.scale_fn(1), rot = ang)

        img = self.font.render(str(rank), True, (0, 0, 0))
        iw, ih = img.get_rect().size
        self.screen.blit(img, ((3*x_pos1 + x_pos2)/4 - iw/2, (3*y_pos1 + y_pos2)/4 - ih/2))


    def draw_ports(self, board):
        for port in board.port_locations:
            s1 = port[0]
            s2 = port[1]
            port_type = board.settlements[s1, 2]
            assert port_type == board.settlements[s2, 2], 'Sam messed up the eventing for port cycling'
            x_pos, y_pos = self.scale_fn((port[2], port[3]))

            port = pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(x_pos - self.scale_fn(self.PORT_WIDTH/2), y_pos - self.scale_fn(self.PORT_HEIGHT/2), self.scale_fn(self.PORT_WIDTH), self.scale_fn(self.PORT_HEIGHT)))
            img = pygame.image.load(self.port_imgs[port_type])
            img = pygame.transform.scale(img, port.size)
            self.screen.blit(img, (x_pos - port.width/2, y_pos - port.height/2))

    def draw_settlements(self, board):
        """
        Draws settlements (uses the locs for the matplotlib render)
        """
        for settlement in board.settlements:
            player = settlement[0]
            s_type = settlement[1]
            x_pos, y_pos = self.scale_fn((settlement[3], settlement[4]))

            if player == 0:
                img = pygame.draw.circle(self.screen, self.player_colors[player], (x_pos, y_pos), self.scale_fn(0.5), 2)
            else:
                if s_type == 1:
                    self.draw_village(self.screen, (0, 0, 0), (x_pos, y_pos), self.scale_fn(self.SETTLEMENT_RADIUS + 0.2))
                    self.draw_village(self.screen, (self.player_colors[player]), (x_pos, y_pos), self.scale_fn(self.SETTLEMENT_RADIUS))
                elif s_type == 2:
                    self.draw_city(self.screen, (0, 0, 0), (x_pos, y_pos), self.scale_fn(self.SETTLEMENT_RADIUS + 0.2))
                    self.draw_city(self.screen, (self.player_colors[player]), (x_pos, y_pos), self.scale_fn(self.SETTLEMENT_RADIUS))

            if self.show_prod:
                tiles = settlement[8:]
                tiles = tiles[tiles > -1]
                tiles = board.tiles[tiles]
                vals = tiles[:, 1]
                pips = (6 - abs(7 - vals)).clip(0)
                prod = pips.sum()
                img = self.small_font.render(str(prod), True, (0, 0, 0))
                iw, ih = img.get_rect().size
                self.screen.blit(img, (x_pos - iw/2, y_pos - ih/2))
            else:
                if settlement[0] != 0:
                    img = self.small_font.render(str(settlement[0]), True, (0, 0, 0))
                    iw, ih = img.get_rect().size
                    self.screen.blit(img, (x_pos - iw/2, y_pos - ih/2))

    def draw_roads(self, board):
        """
        draws roads
        """
        for road in board.roads:
            player = road[0]
            s1 = road[1]
            s2 = road[2]
            x_pos1, y_pos1 = self.scale_fn((board.settlements[s1, 3], board.settlements[s1, 4]))
            x_pos2, y_pos2 = self.scale_fn((board.settlements[s2, 3], board.settlements[s2, 4]))

            if player != 0:
                pygame.draw.line(self.screen, (0, 0, 0), (x_pos1, y_pos1), (x_pos2, y_pos2), self.scale_fn(self.ROAD_RADIUS + 0.2))
                pygame.draw.line(self.screen, self.player_colors[player], (x_pos1, y_pos1), (x_pos2, y_pos2), self.scale_fn(self.ROAD_RADIUS))

    def draw_tiles(self, board):
        """
        Utilize the numpy representation of the board state to draw as image
        """
        for tile in board.tiles:
            resource = tile[0]
            die = tile[1]
            x_pos, y_pos = self.scale_fn((tile[2], tile[3]))

            hex_img = self.draw_hex(self.screen, (0, 0, 0), (x_pos, y_pos), self.scale_fn(self.HEX_RADIUS + 0.2), rot = pi/6)
            hex_img = self.draw_hex(self.screen, (255, 255, 255), (x_pos, y_pos), self.scale_fn(self.HEX_RADIUS), rot = pi/6)
            img = pygame.image.load(self.tile_imgs[resource])
            img = pygame.transform.scale(img, hex_img.size)
#            img = pygame.mask.from_surface(img).to_surface()
            self.screen.blit(img, (x_pos - hex_img.width/2, y_pos - hex_img.height/2))
            self.draw_token(self.screen, (x_pos, y_pos), self.scale_fn(self.TOKEN_RADIUS), die)
        

    def draw_token(self, surface, center, radius, value):
        """
        Draw the pip thing for Catan (i.e. the die roll and the pips below it)
        """
        color = (255, 0, 0) if value in [6, 8] else (0, 0, 0)
        pygame.draw.circle(surface, (0, 0, 0), center, radius + self.scale_fn(0.1))
        pygame.draw.circle(surface, (255, 255, 255), center, radius)
        img = self.font.render(str(value), True, color)
        iw, ih = img.get_rect().size
        self.screen.blit(img, (center[0] - iw/2, center[1] - self.scale_fn(1)))

        #Draw 1-5 pips
        n_pips = 6 - abs(7 - value)
        self.draw_pips(surface, center, color, n_pips)

    def draw_pips(self, surface, center, color, num):
        if num < 1:
             return
        else:
            yloc = center[1] + self.scale_fn(0.5)
            xlocs = [center[0] - self.scale_fn(0.4) * num//2 + self.scale_fn(0.4) * n + self.scale_fn(0.2) for n in range(num)]
            for xloc in xlocs:
                pygame.draw.circle(surface, color, (xloc, yloc), self.scale_fn(0.2))

    def draw_thin_triangle(self, surface, color, center, radius, rot = 0, width = 0):    
        points = []
        for ang, d in zip([0, 2*pi/3, 4*pi/3], [1.0, 0.4, 0.4]):
            pt = pygame.math.Vector2(center[0] + d * radius * cos(ang + rot), center[1] + d * radius * sin(ang + rot))
            points.append(pt)
        return pygame.draw.polygon(surface, color, points)

    def draw_hex(self, surface, color, center, radius, rot = 0, width=0):
        points = []
        for i in range(6):
            pt = pygame.math.Vector2(center[0] + radius * cos(rot + i*(pi/3)), center[1] + radius * sin(rot + i*(pi/3)))
            points.append(pt)
        return pygame.draw.polygon(surface, color, points, width)

    def draw_village(self, surface, color, center, radius):
        points = []
        for ang in [-pi/2, 7*pi/6, 5*pi/6, pi/6, 11*pi/6]:
            pt = pygame.math.Vector2(center[0] + radius * cos(ang), center[1] + radius * sin(ang))
            points.append(pt)
        return pygame.draw.polygon(surface, color, points)

    def draw_city(self, surface, color, center, radius):
        points = []
        for ang, d in zip([5*pi/3, 11*pi/6, pi/6, 5*pi/6, 13*pi/12, 3*pi/2], [1.2, 1.2, 1.2, 1.2, 1.2, 0.5]):
            pt = pygame.math.Vector2(center[0] + d * radius * cos(ang), center[1] + d * radius * sin(ang))
            points.append(pt)
        return pygame.draw.polygon(surface, color, points)

    def scale_fn(self, coord):
        """
        Since I have a renderer in matplotlib, this function takes matplotlib values and scales/translates them to the screen.
        Assumed input is an x, y tuple
        """

        if isinstance(coord, tuple):

            #Original plot is 36x36. leave margins
            x_scale = self.width/36 * (1 - 2*self.margin)
            y_scale = self.height/36 * (1 - 2*self.margin)
            #Translate then scale, so that we can always translate 6
            x_trans = 6 * (1 + self.margin)
            y_trans = 6 * (1 + self.margin)

            return ( int(x_scale*(x_trans+coord[0])), int(y_scale*(y_trans+coord[1])) )

        else:
            return int((coord * self.width/36 * (1 - 2*self.margin)))

    def inv_scale_fn(self, coord):
        """
        Invert the scale function to go from pixels to board units.
        """
        x_scale = self.width/36 * (1 - 2*self.margin)
        y_scale = self.height/36 * (1 - 2*self.margin)
        #Translate then scale, so that we can always translate 6
        x_trans = 6 * (1 + self.margin)
        y_trans = 6 * (1 + self.margin)
        return ((coord[0]/x_scale) - x_trans, (coord[1]/y_scale) - y_trans)

    def get_object(self, board, coord):
        """
        From board coordinates, get the correct object. lol this is going to be coded VERY poorly.
        Items to return:
            1. Tile (change the resource type in edit mode)
            2. Token (change the die valuein edit mode)
            3. Settlement (place a settlement)
            4. Road (place a road)
        return a (int, int) tuple to indicate the change type and the index to change
        """
        p = np.array(list(coord))
        p = np.expand_dims(p, axis = 0)

        tile_coords = board.tiles[:, [2, 3]]
        settlement_coords = board.settlements[:, [3, 4]]
        road_s_endpts = board.roads[:, [1, 2]]
        road_endpt1 = board.settlements[road_s_endpts[:, 0]][:, [3, 4]]
        road_endpt2 = board.settlements[road_s_endpts[:, 1]][:, [3, 4]]

        #First check if the item is a tile/token
        tile_dists = ((tile_coords - p) ** 2).sum(axis=1) ** 0.5
        closest_tile = np.argmin(tile_dists)
        closest_dist = np.min(tile_dists)
        if closest_dist < self.TOKEN_RADIUS:
            out_code = 2
            out_obj = board.tiles[closest_tile]
            print('Selected Token {} r = {}'.format(out_obj[1], out_obj[0]))
            return out_code, closest_tile
        elif closest_dist < (self.HEX_RADIUS + 0.2) * cos(pi/6): #reduce the circle radius to be circumscribed by the hexagon (but let the border be included in the radius)
            out_code = 1
            out_obj = board.tiles[closest_tile]
            print('Selected Hex {} r = {}'.format(out_obj[1], out_obj[0]))
            return out_code, closest_tile

        #next check if the item is a settlement spot
        s_dists = ((settlement_coords - p) ** 2).sum(axis = 1) ** 0.5
        closest_s = np.argmin(s_dists)
        closest_s_dist = np.min(s_dists)

        if closest_s_dist < self.SETTLEMENT_RADIUS + 0.2:
            out_code = 3
            out_obj = board.settlements[closest_s]
            print('Selected settlement spot {}'.format(closest_s))
            print(out_obj)
            return out_code, closest_s

        #Last, check if we're selecting a road spot
        #Line segment check: project point onto line. If on line, take length of normal. Else take min dist to either endpoint.
        l1 = road_endpt2 - road_endpt1
        l1_len = np.hypot(l1[:, 0], l1[:, 1])
        proj_line = p - road_endpt1
        proj = (proj_line * l1).sum(axis = 1)
        dot = np.where(l1_len > 0, proj/(l1_len**2), 0)
        closest_pt = road_endpt1 + np.expand_dims(dot, axis=1) * l1
        r_dists = ((closest_pt - p) ** 2).sum(axis = 1) ** 0.5
        r_dists[(dot > 1)|(dot < 0)] = 1e10
        closest_r = np.argmin(r_dists)
        closest_r_dist = np.min(r_dists)

        if closest_r_dist < self.ROAD_RADIUS:
            out_code = 4
            out_obj = board.roads[closest_r]
            print('Chose road {}'.format(out_obj))
            return out_code, closest_r

        for idx, port in enumerate(board.port_locations):
            s1 = port[0]
            s2 = port[1]
            port_type = board.settlements[s1, 2]
            x_pos, y_pos = (port[2], port[3])

            if p[0, 0] > x_pos - self.PORT_WIDTH/2 and p[0, 0] < x_pos + self.PORT_WIDTH/2 and p[0, 1] > y_pos - self.PORT_HEIGHT/2 and p[0, 1] < y_pos + self.PORT_HEIGHT/2:
                print('Selected port {}'.format(port))
                return 5, idx

        #No interaction with board
        return 0, -1

        


if __name__ == '__main__':
    print('implement this later')
