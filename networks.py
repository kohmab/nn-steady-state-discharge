import torch
from torch import nn

from helpers import beam_field, BesselJ0
from configuration import PARAMETERS


class AnalyticNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, r, params):
        inp = torch.hstack((z, r))
        max_field = params[:, [0]]
        return beam_field(inp, max_field)


class MLP(nn.Module):
    def __init__(self, nin=2, nout=2, nhid=16, nlay=4, activator=torch.nn.Tanh, with_final_activator=False):
        super().__init__()

        self.nlayers = nlay

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(nin, nhid))
        self.layers.append(activator())
        for i in range(nlay - 2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.layers.append(activator())
        self.layers.append(nn.Linear(nhid, nout))
        if with_final_activator:
            self.layers.append(activator())

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
        self.trunk = MLP(2,
                         PARAMETERS.model.mixer.in_features,
                         PARAMETERS.model.trunk.hidden_features,
                         PARAMETERS.model.trunk.hidden_layers,
                         with_final_activator=True)

        self.branch = MLP(4,
                          PARAMETERS.model.mixer.in_features,
                          PARAMETERS.model.branch.hidden_features,
                          PARAMETERS.model.branch.hidden_layers,
                          activator=torch.nn.ReLU,
                          with_final_activator=True)

        self.mixer = Mixer(PARAMETERS.model.mixer.in_features, 2)

    def forward(self, z, r, params):
        trunk_in = torch.hstack((z, r))
        trunk_out = self.trunk(trunk_in)

        branch_out = self.branch(params)

        out = self.mixer(branch_out, trunk_out)

        return out


class BesselFeatEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.Nbess = PARAMETERS.model.bessel_features
        self.wavenumbers = nn.Linear(1, self.Nbess, bias=False)

        self.wavenumbers.weight.data = torch.arange(self.Nbess, dtype=PARAMETERS.torch.dtype)[:, None] / 10
        # nn.init.normal_(self.amplitudes.weight.data)

    def forward(self, r):
        kr = self.wavenumbers(r)
        bessels = BesselJ0.apply(kr)
        return bessels


# TODO FINISH
class HarmonmicsMixer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, vec1, vec2):
        return torch.flatten(vec1 @ vec2.T)


class FourierFeatEmb(nn.Module):

    def __init__(self):
        super().__init__()
        self.Nfeat = PARAMETERS.model.fourier_features
        self.wavenumbers = nn.Linear(1, self.Nfeat, bias=False)
        self.phases = nn.Parameter(torch.randn(self.Nfeat), requires_grad=True)

        self.wavenumbers.weight.data = torch.arange(self.Nfeat, dtype=PARAMETERS.torch.dtype)[:, None] / 10

    def forward(self, z):
        phase = self.wavenumbers(z) * self.phases
        return torch.cos(phase)


class BessFeatPIDeepONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bess_feat = BesselFeatEmb()
        self.fourier_feat = FourierFeatEmb()
        self.trunk = MLP(PARAMETERS.model.bessel_features + PARAMETERS.model.fourier_features,
                         PARAMETERS.model.mixer.in_features,
                         PARAMETERS.model.trunk.hidden_features,
                         PARAMETERS.model.trunk.hidden_layers)

        self.branch = MLP(4,
                          PARAMETERS.model.mixer.in_features,
                          PARAMETERS.model.branch.hidden_features,
                          PARAMETERS.model.branch.hidden_layers,
                          activator=torch.nn.SiLU)

        self.mixer = Mixer(PARAMETERS.model.mixer.in_features, 2)

    def forward(self, z, r, params):
        bessel_feat = self.bess_feat(r)
        fourier_feat = self.fourier_feat(z)
        trunk_in = torch.hstack((fourier_feat, bessel_feat))
        trunk_out = self.branch(trunk_in)

        branch_out = self.trunk(params)

        out = self.mixer(branch_out, trunk_out)

        return out
