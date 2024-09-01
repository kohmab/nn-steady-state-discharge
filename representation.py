from typing import Tuple

import numpy as np
import torch
from torch import nn

from helpers import prepare_tensor, plasma_density


class GridModelExecutor:
    _model: nn.Module

    def __init__(self, model,
                 max_field: np.number = 1.,
                 p: np.number = 1.,
                 K0: np.number = 1.,
                 nu: np.number = 0.
                 ):
        self._model = model
        self._max_field = max_field
        self._p = p
        self._K0 = K0
        self._nu = nu
        self._params = prepare_tensor(np.asarray([max_field, p, K0, nu])).unsqueeze(0)
        self._out = None
        self._field = None
        self._plasma_density = None

    @staticmethod
    def _prepare_points(grid_z: np.ndarray, grid_r: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        z = prepare_tensor(grid_z.flatten()[:, None])
        r = prepare_tensor(grid_r.flatten()[:, None])
        return z, r

    def __call__(self, grid_z, grid_r):
        shape = grid_z.shape
        z, r = GridModelExecutor._prepare_points(grid_z, grid_r)
        params = self._params.repeat(z.numel(), 1)

        out = self._model(z, r, params).cpu().detach().numpy()

        self._field = (out[:, [0]] + 1j * out[:, [1]]).reshape(shape)
        self._plasma_density = plasma_density(out, self._p, self._K0).reshape(shape)

    @property
    def field(self):
        return self._field

    @property
    def plasma_density(self):
        return self._plasma_density
