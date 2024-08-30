import torch
from torch import nn

from parametersholder import ParametersHolder
from helpers import beam_field


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


class FeatTransform(nn.Module):
    def __init__(self, lb, ub, nin=2):
        if not (nin == len(lb) and nin == len(lb)):
            raise RuntimeError("wrong dimentions")
        super().__init__()
        self.B = nn.Linear(nin, nin, bias=False)
        self.nin = nin

        scales = torch.empty(nin)
        for i in range(nin):
            scales[i] = 2 * torch.pi / (ub[i] - lb[i])  # TODO vectorize? but ub lb can be not tensors
        self.scales = scales.float().to(device="cuda")

        shifts = torch.tensor(lb)
        self.shifts = shifts.float().to(device="cuda")  # TODO better way to move to gpu

        nn.init.normal_(self.B.weight.data)

    def forward(self, inp):
        inp = self.scales * (inp - self.shifts)

        inp = self.B(inp)

        out = torch.empty(inp.shape[0], 2 * self.nin, device="cuda").float()

        for i in range(self.nin):
            out[:, i] = torch.cos(inp[:, i])
            out[:, i + self.nin] = torch.sin(inp[:, i])

        return out


class AnalyticNet(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, z, r, params):
        inp = torch.hstack((z, r))
        max_field = params[:, [0]]
        return beam_field(inp, max_field)


class Mixer(nn.Module):
    def __init__(self, bs_and_ts: int, out_feat):
        super().__init__()
        self.fc = nn.Linear(bs_and_ts, out_feat)

    def forward(self, branch, trunk):
        return self.fc(branch * trunk)


class Pidonet(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = MLP(2,
                          ParametersHolder().branch_and_trunk_out_features,
                          ParametersHolder().branch_hidden_features,
                          ParametersHolder().branch_hidden_layers)

        self.trunk = MLP(4,
                         ParametersHolder().branch_and_trunk_out_features,
                         ParametersHolder().trunk_hidden_features,
                         ParametersHolder().trunk_hidden_layers,
                         activator=torch.nn.ReLU)

        self.mixer = Mixer(ParametersHolder().branch_and_trunk_out_features, 2)

    def forward(self, z, r, params):
        branch_in = torch.hstack((z, r))
        branch_out = self.branch(branch_in)

        trunk_out = self.trunk(params)

        out = self.mixer(branch_out, trunk_out)

        return out
