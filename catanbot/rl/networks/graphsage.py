import numpy as np
import torch
import matplotlib.pyplot as plt

from torch import nn, optim, distributions

from catanbot.core.board import Board

from catanbot.agents.independent_actions_agents import IndependentActionsHeuristicAgent, IndependentActionsAgent
from catanbot.board_placement.agents.base import InitialPlacementAgent, MakeDeterministic
from catanbot.board_placement.agents.epsilon_greedy_agent import GNNEpsilonGreedyPlacementAgent
from catanbot.core.independent_action_simulator import IndependentActionsCatanSimulator
from catanbot.board_placement.initial_placement_simulator import InitialPlacementSimulator
from catanbot.rl.collectors.graph_initial_placement_collector import GraphInitialPlacementCollector

class GraphSAGENet(nn.Module):
    """
    An implementation of GRaphSAGE (Hamilton et al. 2017) for Catan boards.
    We're going to make a few assumptions, since the graph structure of the Catan board is the same.

    Also, I'm going to use the pooling aggregator from the paper (instead of mean, LSTM)
    """
    def __init__(self, vertex_feat_size, edge_feat_size, out_size, structure, hidden_sizes = [32, 32, 32, 32], embedding_sizes=[32, 32, 32, 32, 32], activation = nn.Tanh):
        """
        We assume vertices passed in consistent order for this network.
        Args:
            vertex_feat_size: The number of features for each vertex
            edge_feat_size: The number of features for each edge
            out_size: The number of features for each vertex at the end
            structure: An n x k x 2 tensor that encodes the structure of the graph. First index is vertex number, second is edge number, last is vertex idx/edge idx
            hidden_sizes: The number of features for each vertex at each layer of the GNN
            embedding_sizes: The number of embedding features for the aggregator to use at each layer. This list should be one longer than hiddens
            acitvation: Which activation to use for the MLPs
        """
        super(GraphSAGENet, self).__init__()

        self.vertex_feat_size = vertex_feat_size
        self.edge_feat_size = edge_feat_size
        self.out_size = out_size
        self.structure = structure

        self.activation = activation()

        self.embedding_in_dims = [self.vertex_feat_size + self.edge_feat_size] + [e + self.edge_feat_size for e in  embedding_sizes[:-1]]
        self.embedding_out_dims = embedding_sizes
        self.hidden_in_dims = [h + e for h, e in zip([self.vertex_feat_size] + hidden_sizes, self.embedding_out_dims)]
        self.hidden_out_dims = hidden_sizes + [self.out_size]

        assert len(self.embedding_in_dims) == len(self.hidden_in_dims), 'Got a mismatch in hidden dims and embedding dims (is embedding_sizes 1 more than hidden_sizes?)'

        self.embedding_weights = nn.ModuleList()
        self.hidden_weights = nn.ModuleList()

        for i in range(len(self.embedding_in_dims)):
            self.embedding_weights.append(nn.Linear(self.embedding_in_dims[i], self.embedding_out_dims[i]))
            self.hidden_weights.append(nn.Linear(self.hidden_in_dims[i], self.hidden_out_dims[i])) #Don't forget that we concat the neighbor embedding to last hidden

    def forward(self, x):
        """
        Does the forward pass. Assumes x is an nxk matrix containing the initial features for all n vetices in adjacencies (fine for this problem instance).
        TODO: Dont overwrite the feats - append to list instead?
        """
        vertex_feats = x['vertices']
        edge_feats = x['edges']

        #Use the structure matrix to build all the embeddings you need
        for i, (e_weights, h_weights) in enumerate(zip(self.embedding_weights, self.hidden_weights)):
            neighbor_feats_vertex = vertex_feats[:, self.structure[:, :, 0]]
            neighbor_feats_edge = edge_feats[:, self.structure[:, :, 1]]
            neighbor_feats = torch.cat([neighbor_feats_vertex, neighbor_feats_edge], dim=3)
            neighbor_embeddings = self.activation(e_weights.forward(neighbor_feats))
#            neighbor_embeddings[self.structure[:, :, 0] == -1] = -1e6 #Should ignore placeholders in the max op. DON'T SWITCH TO MEAN
            neighbor_embedding = neighbor_embeddings.max(dim=2)[0]

            vertex_in = torch.cat([vertex_feats, neighbor_embedding], dim=2)
            vertex_feats = h_weights.forward(vertex_in)

            if i < len(self.embedding_weights)-1:
                vertex_feats = self.activation(vertex_feats)
                vertex_feats = vertex_feats / vertex_feats.norm(dim=2, keepdim=True) #normalize all layers but last.

        return vertex_feats

class GraphSAGEQMLP(nn.Module):
    """
    A Q function compatible with the standard QMLP interface.
    Since we aren't guaranteed to have a big enough receptive field from the GNN, pass everything though a linear layer at the end to get final.    
    """
    def __init__(self, env, gnn, logscale=False, scale=1.0, gpu=False):
        super(GraphSAGEQMLP, self).__init__()

        self.logscale = logscale
        self.scale = scale
        self.action_mask = env.players[0].action_mask().flatten()
        self.extra_feats = env.graph_observation_space['total']

        self.gnn = gnn
        self.activation = nn.Tanh()
        self.final_linear = nn.Linear(self.gnn.structure.shape[0] * self.gnn.out_size + self.extra_feats, 4 * 54 * 3)

        self.gpu = gpu

    def forward(self, obs):
        graph_embeddings = self.activation(self.gnn.forward(obs))
        graph_embeddings_flat = graph_embeddings.flatten(start_dim=1)
        extra_feats = obs['player']
        mlp_in = torch.cat([graph_embeddings_flat, extra_feats], dim=1)
        val = self.final_linear.forward(mlp_in)
        val = val.view(-1, 4, 54*3)
        val[:, :, ~self.action_mask] = -1e6
        if self.logscale:
            return torch.exp(val) * self.scale
        else:
            return val * self.scale

if __name__ == '__main__':
    board = Board()
    board.reset()

    agents = [IndependentActionsHeuristicAgent(board), IndependentActionsHeuristicAgent(board), IndependentActionsHeuristicAgent(board), IndependentActionsHeuristicAgent(board)]
    s = IndependentActionsCatanSimulator(board=board, players = agents, max_vp=10)
    placement_agents = [InitialPlacementAgent(board), InitialPlacementAgent(board), InitialPlacementAgent(board), InitialPlacementAgent(board)]
    placement_simulator = InitialPlacementSimulator(s, placement_agents)
    hidden_sizes = [64] * 4
    embedding_sizes = [64] * 5

    net = GraphSAGENet(40, 5, 16, board.structure_tensor, embedding_sizes=embedding_sizes, hidden_sizes = hidden_sizes)
    qf = GraphSAGEQMLP(placement_simulator, net)

    eps=0.
    placement_simulator.players = [GNNEpsilonGreedyPlacementAgent(board, qf, lambda e:eps, pidx=0), GNNEpsilonGreedyPlacementAgent(board, qf, lambda e:eps, pidx=1), GNNEpsilonGreedyPlacementAgent(board, qf, lambda e:eps, pidx=2), GNNEpsilonGreedyPlacementAgent(board, qf, lambda e:eps, pidx=3)]

    collector = GraphInitialPlacementCollector(placement_simulator)

    import pdb;pdb.set_trace()
    rollouts = collector.get_rollouts(10)

    exit(0)
    out = qf(obs)
    print(out, out.shape)

    opt = optim.Adam(net.parameters())

    for i in range(2000):
        print('EPOCH {}'.format(i+1))
        board.reset()
        obs = board.graph_observation()
        obs['vertices'] = torch.tensor(obs['vertices']).float()
        obs['edges'] = torch.tensor(obs['edges']).float()
        prod = torch.tensor(board.production_info[:, 1]).float()
        
        out = net(obs).flatten()
        err = prod - out
        loss = err.pow(2).mean()

        opt.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        opt.step()

        print('____LOSS____')
        print(loss)

    for i in range(3):
        board.reset()
        obs = board.graph_observation()
        obs['vertices'] = torch.tensor(obs['vertices']).float()
        obs['edges'] = torch.tensor(obs['edges']).float()
        prod = torch.tensor(board.production_info[:, 1]).float()
        
        out = net(obs).flatten()
        print(out)
        board.render();plt.show()
