import numpy as np
import torch
from pyDOE import lhs

from configuration import Configuration


def prepare_tensor(array, requires_grad=True):
    return (torch.from_numpy(array).
            to(device=Configuration().device, dtype=Configuration().dtype)
            .requires_grad_(requires_grad))


def beam_field(points, max_field=1.0):
    if torch.is_tensor(points):
        module = torch
    else:
        module = np

    z = points[:, [0]]
    r = points[:, [1]]

    ksi = 1. + 1.j * z

    field = max_field * module.exp(- module.square(r) / 2. / ksi) / ksi
    return module.hstack((field.real, field.imag))


def plasma_density(field_components, p, K0):
    if not torch.is_tensor(field_components):
        raise RuntimeError()

    abs_E_sqred = field_components[:, [0]].pow(2) + field_components[:, [1]].pow(2)

    density = torch.where(abs_E_sqred > 1., (abs_E_sqred.pow(p) - 1) * K0, 0)

    return density


def uniform(N, l, r, dim=1):
    return l + (r - l) * lhs(dim, N)


def grad(y, x):
    ones = torch.ones_like(y, device=Configuration().device, dtype=Configuration().dtype, requires_grad=False)
    return torch.autograd.grad(y, x, ones, create_graph=True)[0]
