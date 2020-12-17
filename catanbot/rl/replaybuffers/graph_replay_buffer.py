import torch

from catanbot.rl.replaybuffers.base import ReplayBuffer


from catanbot.core.independent_action_simulator import IndependentActionsCatanSimulator
from catanbot.core.board import Board
from catanbot.agents.independent_actions_agents import IndependentActionsHeuristicAgent, IndependentActionsAgent
from catanbot.board_placement.agents.base import InitialPlacementAgent
from catanbot.board_placement.agents.epsilon_greedy_agent import EpsilonGreedyPlacementAgent
from catanbot.core.independent_action_simulator import IndependentActionsCatanSimulator
from catanbot.board_placement.initial_placement_simulator import InitialPlacementSimulator
from catanbot.rl.collectors.graph_initial_placement_collector import GraphInitialPlacementCollector

class GraphReplayBuffer:
    """
    Same as the normal replay buffer, but now the observation is an edge tensor and a vertex tensor (and both are 2D)
    """

    def __init__(self, env, capacity = int(1e7), gpu=False):
        self.capacity = int(capacity)
        self.vertex_obs_dim = env.graph_observation_space['board']['vertices'] #expect 54 x k
        self.edge_obs_dim = env.graph_observation_space['board']['edges'] #Expect 72 x k
        self.player_obs_dim = env.graph_observation_space['total']
        self.n = 0 #the index to start insering into
        self.gpu = gpu

        self.act_dim = env.action_space['total']
        self.buffer = {
            'observation_vertices': torch.tensor([float('inf')], device=self.device).repeat(self.capacity, *self.vertex_obs_dim),
            'observation_edges': torch.tensor([float('inf')], device=self.device).repeat(self.capacity, *self.edge_obs_dim),
            'observation_player': torch.tensor([float('inf')], device=self.device).repeat(self.capacity, self.player_obs_dim),
            'action': torch.tensor([float('inf')], device=self.device).repeat(self.capacity, self.act_dim),
            'reward':torch.tensor([float('inf')], device=self.device).repeat(self.capacity, 4),
            'next_observation_vertices': torch.tensor([float('inf')], device=self.device).repeat(self.capacity, *self.vertex_obs_dim),
            'next_observation_edges': torch.tensor([float('inf')], device=self.device).repeat(self.capacity, *self.edge_obs_dim),
            'next_observation_player': torch.tensor([float('inf')], device=self.device).repeat(self.capacity, self.player_obs_dim),
            'terminal': torch.tensor([True], device=self.device).repeat(self.capacity, 1),
            'pidx': torch.tensor([-1], device=self.device).repeat(self.capacity, 1)
        }    

    def insert(self, samples):
        """
        Assuming samples are being passed in as a dict of tensors.
        """
        assert len(samples['observation_vertices']) == len(samples['observation_edges']) == len(samples['observation_player']) == len(samples['action']) == len(samples['reward']) == len(samples['next_observation_vertices']) == len(samples['next_observation_edges']) == len(samples['next_observation_player']) == len(samples['terminal']) == len(samples['pidx']), \
        "expected all elements of samples to have same length, got: {} (\'returns\' should be a different length though)".format([(k, len(samples[k])) for k in samples.keys()])

        nsamples = len(samples['observation_vertices'])

        for k in self.buffer.keys():
            for i in range(nsamples):
                self.buffer[k][(self.n + i) % self.capacity] = samples[k][i]

        self.n += nsamples

    def nelements(self):
        return min(self.n, self.capacity)

    def sample(self, nsamples):
        """
        Get a batch of samples from the replay buffer.
        """

        #Don't want to sample placeholders, so min n and capacity.
        idxs = torch.LongTensor(nsamples).random_(0, min(self.n, self.capacity)) 

        out = {k:self.buffer[k][idxs] for k in self.buffer.keys()}
        
        if self.gpu:
            out = {k:out[k].cuda() for k in out.keys()}

        #Re-aggregate the observations
        out['observation'] = {
            'vertices':out['observation_vertices'],
            'edges':out['observation_edges'],
            'player':out['observation_player'],
        }
        out['next_observation'] = {
            'vertices':out['next_observation_vertices'],
            'edges':out['next_observation_edges'],
            'player':out['next_observation_player'],
        }

        return out

    @property
    def device(self):
        return 'cuda' if self.gpu else 'cpu'

    def __repr__(self):
        return "buffer = {} \nn = {}".format(self.buffer, self.n)

if __name__ == '__main__':
    b = Board()
    b.reset()
    agents = [IndependentActionsAgent(b), IndependentActionsAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsAgent(b)]
#    agents = [IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b), IndependentActionsHeuristicAgent(b)]
#    agents = [HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b), HeuristicAgent(b)]
    s = IndependentActionsCatanSimulator(board=b, players = agents, max_vp=10)
    placement_agents = [InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b), InitialPlacementAgent(b)]
    placement_simulator = InitialPlacementSimulator(s, placement_agents)

    collector = GraphInitialPlacementCollector(placement_simulator)

    rollouts = collector.get_rollouts(10)
    print('done!') 

    buf = GraphReplayBuffer(placement_simulator, capacity = 100)
    buf.insert(rollouts)
    print(buf)
    print(buf.sample(3))
