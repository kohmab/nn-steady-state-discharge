import numpy as np
import torch
from torch.autograd import Function

from configuration import PARAMETERS

_DEVICE = PARAMETERS.torch.device
_DTYPE = PARAMETERS.torch.dtype

def prepare_tensor(array, requires_grad=True):
    if not torch.is_tensor(array):
        tensor = torch.from_numpy(array)
    else:
        tensor = array

    return tensor.to(device=_DEVICE, dtype=_DTYPE).requires_grad_(requires_grad)


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


def latin_hypercube_sampling(n_dimensions, n_samples):
    perm = torch.stack([torch.randperm(n_samples, device=_DEVICE) for _ in range(n_dimensions)], dim=1)

    random_points = torch.rand(n_samples, n_dimensions, device=_DEVICE, dtype=_DTYPE)

    samples = (perm.float() + random_points) / n_samples

    return samples


def uniform(num_points, lower_limits, upper_limits, dim=1):
    if not torch.is_tensor(lower_limits):
        lower_limits = torch.from_numpy(lower_limits).to(device=_DEVICE, dtype=_DTYPE)
    if not torch.is_tensor(upper_limits):
        upper_limits = torch.from_numpy(upper_limits).to(device=_DEVICE, dtype=_DTYPE)

    return lower_limits + (upper_limits - lower_limits) * latin_hypercube_sampling(dim, num_points)


def grad(y, x):
    ones = torch.ones_like(y, device=_DEVICE, dtype=_DTYPE, requires_grad=False)
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
