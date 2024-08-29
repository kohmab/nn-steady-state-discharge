import numpy as np
import torch

def to_torch(*args):
    result = []
    for arg in args:
        if not torch.is_tensor(arg):
            result.append(torch.from_numpy(arg).float())
        else:
            result.append(arg.float())
    return tuple(result)

def beam_field(points, max_field=1.):
    z = points[:, [0]]
    r = points[:, [1]]

    ksi = 1. + 1.j * z

    module = None
    if torch.is_tensor(r) ^ torch.is_tensor(z):
        raise RuntimeError("Incompatible argument types")
    if torch.is_tensor(r):
        module = torch
    else:
        module = np

    field = max_field * module.exp(- module.square(r)/ 2. / ksi) / ksi
    return module.hstack((field.real, field.imag))

