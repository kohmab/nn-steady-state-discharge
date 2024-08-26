import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, nin=2, nout=2, nhid=16, nlay=4, activator=torch.nn.Tanh):
        super().__init__()

        self.nlayers = nlay

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(nin, nhid))
        self.layers.append(activator())
        for i in range(nlay - 2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.layers.append(activator())
        self.layers.append(nn.Linear(nhid, nout))

        for i, lay in enumerate(self.layers):
            if i % 2 == 0:
                nn.init.xavier_uniform_(lay.weight.data)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x
