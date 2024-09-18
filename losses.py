import torch
from torch.nn.functional import mse_loss

from configuration import PARAMETERS
from helpers import grad, plasma_density
from hyperdomain import HYPERDOMAIN

_ZMIN = HYPERDOMAIN.zmin
_ZMAX = HYPERDOMAIN.zmax


def add_z_weights(z, *residual_tensors):
    idx = torch.argsort(z, dim=0, descending=True)

    res = torch.hstack(residual_tensors).abs().sum(dim=1)
    penalty = torch.clone(res).detach()
    for i in idx[1:]:
        penalty[i] += penalty[i - 1]
    res += penalty

    zero = torch.zeros_like(res, device=PARAMETERS.torch.device, dtype=PARAMETERS.torch.dtype)
    return mse_loss(res, zero)


def eq_residual(model, z, r, params):
    out = model(z, r, params)

    Er, Ei = out[:, [0]], out[:, [1]]

    dEr_dz, dEi_dz = grad(Er, z), grad(Ei, z)

    rdEr_dr, rdEi_dr = r * grad(Er, r), r * grad(Ei, r)

    drdEr_dr, drdEi_dr = grad(rdEr_dr, r), grad(rdEi_dr, r)

    p = params[:, [1]]
    K0 = params[:, [2]]
    nu = params[:, [3]]

    n = plasma_density(out, p, K0)
    nl_part_r = -n * (Ei - nu * Er)
    nl_part_i = -n * (Er + nu * Er)
    res_r = 2 * r * dEr_dz + drdEi_dr + nl_part_r
    res_i = -2 * r * dEi_dz + drdEr_dr + nl_part_i

    return res_r, res_i


def eq_losses(model, z, r, params):
    res_r, res_i = eq_residual(model, z, r, params)

    zero = torch.zeros_like(res_r, device=PARAMETERS.torch.device, dtype=PARAMETERS.torch.dtype)
    loss_eq = mse_loss(res_r, zero) + mse_loss(res_i, zero)

    # loss_eq = add_z_weights(z, res_r, res_i)

    return loss_eq


def upper_r_bc_losses(model, z, r, params, target):
    out = model(z, r, params)
    return mse_loss(out, target)


def axis_bc_losses(model, z, r, params):
    out = model(z, r, params)
    Er, Ei = out[:, [0]], out[:, [1]]

    dEr_dr, dEi_dr = grad(Er, r), grad(Ei, r)

    zero = torch.zeros_like(Er, device=PARAMETERS.torch.device, dtype=PARAMETERS.torch.dtype)

    return mse_loss(dEr_dr, zero) + mse_loss(dEi_dr, zero)


def ic_losses(model, z, r, params, target):
    out = model(z, r, params)
    return mse_loss(out, target)
