import numpy as np
import matplotlib.pyplot as plt

from catanbot.util import to_one_hot

from catanbot.rl.debug.gridworld_qmlp import GridworldQMLP
from catanbot.rl.debug.epsilongreedy_agent import EpsilonGreedyGridworldAgent

from catanbot.rl.collectors.initial_placement_collector import InitialPlacementCollector, DebugCollector

class Gridworld:
    """
    Basic class to test my multiagent DQN
    The task is to take four agents that move in alternating steps to trade places one clockwise.
    i.e. A   B     to     B    D

         C   D            A    C
    actions are to move in each cardinal direction, or stay in place.
    reward is manhattan distance to your goal.
    terminates if agents collide or go out of bounds.
    """
    def __init__(self, players, max_steps=24):
        self.players = players
        self.n = 5 #The board width.
        self.agent_goals = np.array([
            [self.n-1, self.n-1],
            [0, self.n-1],
            [0, 0],
            [self.n-1, 0],
        ])

        self.agent_positions = np.ones([4, 2]) * -1

        self.act_lookup = np.array([
            [0, 1],
            [0, -1],
            [-1, 0],
            [1, 0],
            [0, 0]
        ])
        
        self.is_collision = True
        self.oob = np.ones(4).astype(bool)
        self.max_steps = max_steps
        self.player_idx = -1
        self.t = -1

    @property
    def observation_space(self):
        return {
            'total':np.array(20)
        }

    @property
    def action_space(self):
        return {
            'total':np.array(5)
        }

    def reset(self, reset_board='useless arg to match Catan simulator format'):
        self.agent_positions = np.array([
            [0, 0],
            [0, self.n-1],
            [self.n-1, self.n-1],
            [self.n-1, 0]
        ])
        self.t = 0
        self.player_idx = 0
        self.is_collision = False
        self.oob = np.zeros(4).astype(bool)

    def step(self, act):
        """
        0 = up, 1 = down, 2 = left, 3 = right, 4 = stay
        """
        dx = self.act_lookup[np.argmax(act['placement'])]
        self.agent_positions[self.player_idx] += dx

        u = np.unique(self.agent_positions, axis=0)
        self.is_collision = len(u) < 4
        self.oob = np.any((self.agent_positions < 0) | (self.agent_positions >= self.n), axis=1)
        self.t += 1
        self.player_idx = (self.player_idx + 1) % 4

    def reward(self, rollout_n='placeholder to use the same collector API as the Catan simulator.'):
        dist_rew = (2*self.n - np.sum(np.abs(self.agent_goals - self.agent_positions), axis=1)) / (2*self.n) #max dist - Manhattan dist to goal, normalized to [0, 1]
        rew = dist_rew
        rew[self.oob] = -1
        if self.is_collision:
            rew[(self.player_idx-1)%4] = -1

        sparse_rew = np.all(self.agent_goals == self.agent_positions, axis=1).astype(float)
        sparse_rew[self.oob] = -1
        if self.is_collision:
            sparse_rew[(self.player_idx-1)%4] = -1 #Penalize previous player for collision, not current.

        return rew

    @property
    def observation(self):
        pidx_oh = to_one_hot(self.player_idx, max_val=4)
        obs = np.concatenate([self.agent_positions.flatten(), self.agent_goals.flatten(), pidx_oh])
        return {
            'board':{
                'tiles':self.agent_positions.flatten(),
                'roads':self.agent_goals.flatten(),
                 'settlements':np.array([])
            },
            'player':{
                'pidx':pidx_oh,
                'resources':np.array([]),
                'dev':np.array([]),
                'trade_ratios':np.array([]),
            },
            'vp':np.array([]),
            'army':np.array([]),
            'road':np.array([]),
        }

    @property
    def terminal(self):
#        return self.t >= self.max_steps
        return self.is_collision or np.any(self.oob) or self.t >= self.max_steps

    def render(self):
        c = 'rgby'
        for i, p in enumerate(self.agent_positions):
            plt.scatter(p[0], p[1], c=c[i], marker='x', s=100)

        for i, p in enumerate(self.agent_goals):
            plt.scatter(p[0], p[1], c=c[i], marker='.', s=50)

        plt.grid()
        plt.xlim(-1, 5)
        plt.ylim(-1, 5)
        plt.show()

if __name__ == '__main__':
    env = Gridworld('player_placeholder')
    env.reset()
    env.render()

    qf = GridworldQMLP(env)
    players = [EpsilonGreedyGridworldAgent(qf, lambda x:0.1, 0), EpsilonGreedyGridworldAgent(qf, lambda x:0.1, 1), EpsilonGreedyGridworldAgent(qf, lambda x:0.1, 2), EpsilonGreedyGridworldAgent(qf, lambda x:0.1, 3)]
    env.players = players

    collector = InitialPlacementCollector(env)

    rollout = collector.get_rollout()

    print(rollout)

    """
    for i in range(20):
        act = to_one_hot(np.random.randint(5), max_val=5)
        print('obs = \n{}'.format(env.observation))
        print('act = \n{}'.format(act))
        env.step(act)
        print('rew = \n{}'.format(env.reward()))
        print('terminal = \n{}'.format(env.terminal))
        env.render()

        if env.terminal:
            print('rendering')
            env.reset()
            env.render()
    """
