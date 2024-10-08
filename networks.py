import torch
from torch import nn

from helpers import beam_field, BesselJ0
from parametersholder import ParametersHolder


class AnalyticNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, r, params):
        inp = torch.hstack((z, r))
        max_field = params[:, [0]]
        return beam_field(inp, max_field)


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


class BesselFeatEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.Nbess = ParametersHolder().bessel_features
        self.wavenumbers = nn.Linear(1, self.Nbess, bias=False)

        self.wavenumbers.weight.data = torch.randn(self.Nbess, dtype=ParametersHolder().dtype)[:, None]
        # nn.init.normal_(self.amplitudes.weight.data)

    def forward(self, r):
        kr = self.wavenumbers(r)
        bessels = BesselJ0.apply(kr)
        return bessels


class BessFeatPIDeepONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bess_feat = BesselFeatEmb()
        self.branch = MLP(ParametersHolder().bessel_features + 1,
                          ParametersHolder().branch_and_trunk_out_features,
                          ParametersHolder().branch_hidden_features,
                          ParametersHolder().branch_hidden_layers)

        self.trunk = MLP(4,
                         ParametersHolder().branch_and_trunk_out_features,
                         ParametersHolder().trunk_hidden_features,
                         ParametersHolder().trunk_hidden_layers,
                         activator=torch.nn.SiLU)

        self.mixer = Mixer(ParametersHolder().branch_and_trunk_out_features, 2)

    def forward(self, z, r, params):
        bessel_feat = self.bess_feat(r)
        branch_in = torch.hstack((z, bessel_feat))
        branch_out = self.branch(branch_in)

        trunk_out = self.trunk(params)

        out = self.mixer(branch_out, trunk_out)

        return out
