import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, insize, outsize, hiddens, hidden_activation=nn.Tanh, dropout=0.0, bias = None, gpu=False):
        """
        Note: no output activation included. Leave that for the individual policies.
        """
        super(MLP, self).__init__()

        layer_sizes = [insize] + list(hiddens) + [outsize]

        self.bias = bias
        if bias is None:
            self.bias = [True] * (len(layer_sizes) - 1)

        self.layers = nn.ModuleList()
        self.activation = hidden_activation()
        self.dropout = nn.Dropout(p=dropout)
        self.gpu = gpu


        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=self.bias[i]))

        if gpu:
            self.cuda()

    def forward(self, inp):
        out = self.layers[0](inp)
        for layer in self.layers[1:]:
            out = self.activation(out)
            out = self.dropout(out)
            out = layer.forward(out)

        return out

    def cuda(self):
        self.gpu = True
        self.activation.cuda()
        self.dropout.cuda()
        self.layers.cuda()
