import numpy as np
import torch
from pyDOE import lhs
from torch.autograd import Function

from parametersholder import ParametersHolder


def prepare_tensor(array, requires_grad=True):
    device = ParametersHolder().device
    dtype = ParametersHolder().dtype

    if not torch.is_tensor(array):
        tensor = torch.from_numpy(array)
    else:
        tensor = array

    return tensor.to(device=device, dtype=dtype).requires_grad_(requires_grad)


def beam_field(coordinates, max_field=1.0):
    if torch.is_tensor(coordinates):
        module = torch
    else:
        module = np

    z = coordinates[:, [0]]
    r = coordinates[:, [1]]

    ksi = 1. + 1.j * z

    field = max_field * module.exp(- module.square(r) / 2. / ksi) / ksi
    return module.hstack((field.real, field.imag))


def plasma_density(field_components, p, K0):
    if torch.is_tensor(field_components):
        module = torch
    else:
        module = np

    abs_E_sqred = field_components[:, [0]] ** 2 + field_components[:, [1]] ** 2

    density = module.where(abs_E_sqred > 1., (abs_E_sqred ** p - 1) * K0, 0.)

    return density


def uniform(N, l, r, dim=1):
    return l + (r - l) * lhs(dim, N)


def grad(y, x):
    ones = torch.ones_like(y, device=ParametersHolder().device, dtype=ParametersHolder().dtype, requires_grad=False)
    return torch.autograd.grad(y, x, ones, create_graph=True)[0]


class BesselJ0(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        result = torch.special.bessel_j0(input)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return -grad_output * torch.special.bessel_j1(input)
