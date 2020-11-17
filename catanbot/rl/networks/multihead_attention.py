import torch
from torch import nn

from catanbot.rl.networks.mlp import MLP

class MultiheadAttention(nn.Module):
    """
    Implement multihead attention as the following (on an m x 1 vector).
    1. Append (out_dim - 1) sine waves of frequency 2^k pi to the input (m x d matrix)
    2. Apply an NN to each column of the input (m x n) matrix
    3. Softmax across the rows of the matrix to get attention scores (m x n)
    4. Matrix-multiply input by attention scores ([n x m] * [m x 1] = [n x 1])
    """
    def __init__(self, embeddingsize, outsize, hidden_activation=nn.Tanh, gpu=False):
        super(MultiheadAttention, self).__init__()

        self.embedding_size = embeddingsize
        self.nn = MLP(embeddingsize + 1, outsize, hiddens=[32, ], bias=[False, True])
        self.activation = hidden_activation()

        self.gpu = gpu
        if self.gpu:
            self.cuda()

    def position_embedding(self, n):
        """
        An n x embeddingsize matrix. The position i, j = sin(pi * (2^i) * j)
        """
        x = torch.stack([(2**i) * torch.arange(n) for i in range(self.embedding_size)], dim=1).float()
        out = torch.sin(x)
        return out

    def forward(self, inp):
        position_embedding = self.position_embedding(inp.shape[1])
        position_embedding = position_embedding.repeat(inp.shape[0], 1, 1)
        nn_inp = torch.cat([inp.unsqueeze(2), position_embedding], dim=2)
        nn_out = self.nn(nn_inp)
        attention_weights = nn.functional.softmax(nn_out, dim=2)
        out = torch.bmm(inp.unsqueeze(1), attention_weights).squeeze(dim=1)
        return out

    def cuda(self):
        self.gpu = True
        self.nn.cuda()

if __name__ == '__main__':
    """
    Dataset: sin(x0) if exists x over 5 else cos(x0)
    """
    x = torch.normal(torch.zeros(128, 10), torch.ones(128, 10)) * 3
    mask = (x.max(dim=1)[0] > 5)
    y = mask.float() * torch.sin(x[:, 0]) + (~mask).float() * torch.cos(x[:, 0])
    y = y.unsqueeze(1)

    attn = MultiheadAttention(3, 5)
    mlp = MLP(5, 1, hiddens=[5, ])

    mlp2 = MLP(10, 1, hiddens=[32, 5, ])

    net = nn.Sequential(attn, mlp)
    net = nn.Sequential(mlp2)

    opt = torch.optim.Adam(net.parameters())

    epochs = 10000
    
    for i in range(epochs):
        print('___________EPOCH {}_________'.format(i+1))
        preds = net(x)
        err = y - preds
        loss = err.pow(2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        print('LOSS = {}'.format(loss))

    import pdb;pdb.set_trace()
    print(net)
