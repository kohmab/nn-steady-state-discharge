import torch
from torch.nn.functional import mse_loss

from configuration import Configuration
from helpers import grad


def eq_losses(model, z, r, params):
    out = model(z, r, params)

    Er, Ei = out[:, [0]], out[:, [1]]

    dEr_dz, dEi_dz = grad(Er, z), grad(Ei, z)

    rdEr_dr, rdEi_dr = r * grad(Er, r), r * grad(Ei, r)

    drdEr_dr, drdEi_dr = grad(rdEr_dr, r), grad(rdEi_dr, r)

    res_r = 2 * r * dEr_dz + drdEi_dr  # TODO add nonlinear Part!!!
    res_i = -2 * r * dEi_dz + drdEr_dr
    zero = torch.zeros_like(res_r, device=Configuration().device, dtype=Configuration().dtype)

    loss_eq = mse_loss(res_r, zero) + mse_loss(res_i, zero)

    return loss_eq


def upper_r_bc_losses(model, z, r, params, target):
    out = model(z, r, params)
    return mse_loss(out, target)


def lower_r_bc_losses(model, z, r, params):
    out = model(z, r, params)
    Er, Ei = out[:, [0]], out[:, [1]]

    dEr_dr, dEi_dr = grad(Er, r), grad(Ei, r)

    zero = torch.zeros_like(Er, device=Configuration().device, dtype=Configuration().dtype)

    return mse_loss(dEr_dr, zero) + mse_loss(dEi_dr, zero)


def ic_losses(model, z, r, params, target):
    out = model(z, r, params)
    return mse_loss(out, target)
