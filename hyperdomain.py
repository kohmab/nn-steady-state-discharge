from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from configuration import PARAMETERS


class _BaseIdxContainer(ABC):
    def get_idx_for(self, name: str) -> int:
        return self.__dict__[name]


@dataclass(frozen=True)
class _AxisIdx(_BaseIdxContainer):
    z: int
    r: int
    max_field: int
    power_p: int
    coefficient_K0: int
    collision_frequency: int

    def __init__(self):
        object.__setattr__(self, 'z', 0)
        object.__setattr__(self, 'r', 1)
        object.__setattr__(self, 'max_field', 2)
        object.__setattr__(self, 'power_p', 3)
        object.__setattr__(self, 'coefficient_K0', 4)
        object.__setattr__(self, 'collision_frequency', 5)


@dataclass(frozen=True)
class _ParametersIdx(_BaseIdxContainer):
    max_field: int
    power_p: int
    coefficient_K0: int
    collision_frequency: int

    def __init__(self):
        object.__setattr__(self, 'max_field', 0)
        object.__setattr__(self, 'power_p', 1)
        object.__setattr__(self, 'coefficient_K0', 2)
        object.__setattr__(self, 'collision_frequency', 3)


@dataclass(frozen=True)
class _Hyperdomain:
    zmin: float
    zmax: float
    rmin: float
    rmax: float
    lb: np.ndarray
    ub: np.ndarray

    def __init__(self):
        object.__setattr__(self, 'zmin', PARAMETERS.limits.z[0])
        object.__setattr__(self, 'zmax', PARAMETERS.limits.z[1])
        object.__setattr__(self, 'rmin', PARAMETERS.limits.r[0])
        object.__setattr__(self, 'rmax', PARAMETERS.limits.r[1])
        limits = PARAMETERS.limits
        lb = torch.zeros(len(limits), device=PARAMETERS.torch.device, dtype=PARAMETERS.torch.dtype)
        ub = torch.zeros(len(limits), device=PARAMETERS.torch.device, dtype=PARAMETERS.torch.dtype)
        for name, val in limits.items():
            lb[_AxisIdx().get_idx_for(name)] = val[0]
            ub[_AxisIdx().get_idx_for(name)] = val[1]
        object.__setattr__(self, 'lb', lb)
        object.__setattr__(self, 'ub', ub)
        object.__setattr__(self, 'dim', 6)


HYPERDOMAIN = _Hyperdomain()
AXIDX = _AxisIdx()
PARIDX = _ParametersIdx()
