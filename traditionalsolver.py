# %%
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pyhank import HankelTransform
from tqdm import tqdm

# %%
# PARAMS = #TODO
NU = -0
K0 = 4
P = 5
EMAX = 2.

NR = 4096 * 2
RMAX = 12.
dZ = 1.e-3

ZEND = 3.


# %%
def gen_file_name():
    def get_var_name(var):
        for name, value in locals().items():
            if value is var:
                return name

    vars = (NU, K0, P, EMAX, NR, RMAX, dZ, ZEND)
    name_items = [f"{get_var_name(var)}={var}" for var in vars]

    return "_".join(name_items) + ".npy"


file_name = gen_file_name()


# %%
def beam_field_np(z, r):
    ksi = 1. + 1.j * z
    return EMAX * np.exp(- r ** 2 / 2. / ksi) / ksi


def plasma_density_np(E):
    absEsq = np.real(E * E.conjugate())
    density = np.where(absEsq > 1., K0 * (absEsq ** P - 1.), 0.0)
    return density


# %%
def find_start_z(tol=1e-8, start_z=-10 * EMAX, back_step=0.1):
    zl = start_z
    zr = 0.

    while zr - zl > tol:
        z = (zl + zr) / 2
        if np.abs(beam_field_np(z, 0)) > 1.:
            zr = z
        else:
            zl = z
    return z - back_step


z0 = find_start_z()

print(f"Start Z point is {z0}")

# %%
TRANSFORM = HankelTransform(order=0, max_radius=RMAX, n_points=NR)


# %%
class LaplaceOperator:
    _transformer: HankelTransform

    def __init__(self, transformer):
        self._transformer = transformer

    def __call__(self, fr: np.array) -> np.array:
        fr_hat = self._transformer.qdht(fr)
        lapl_hat = - self._transformer.kr ** 2 * fr_hat
        lapl = self._transformer.iqdht(lapl_hat)
        return lapl


# %%
class LinearOperator:
    _dz: float
    _transform: HankelTransform
    _operator_exponent: np.array

    def __init__(self, dz, transform):
        self._transform = transform
        self._dz = dz

        self._init_exponent()

    def _init_exponent(self):
        k = self._transform.kr
        self._operator_exponent = np.exp(k ** 2 / 2.j * self._dz)

    def __call__(self, fr: np.array) -> np.array:
        fr_hat = self._transform.qdht(fr)
        result_hat = fr_hat * self._operator_exponent
        result = self._transform.iqdht(result_hat)
        return result

    @property
    def dz(self):
        return self._dz

    @dz.setter
    def dz(self, dz):
        self._dz = dz
        self._init_exponent()


# %%
class NonlinearOperator:
    _dz: float

    _nonlinearity: Callable[[np.ndarray], np.ndarray]

    def __init__(self, dz, nonlinear_function):
        self._dz = dz
        self._nonlinearity = nonlinear_function

    def __call__(self, fr: np.array) -> np.array:
        nl = self._nonlinearity(fr)
        exp = np.exp(nl / 2.j * self._dz)
        return exp * fr


# %%
class SecondOrderSolver:
    _dz: float
    _linear_op: LinearOperator
    _nonlinear_op: NonlinearOperator

    def __init__(self, dz, transformer, nonlinear_function):
        self._dz = dz
        self._linear_op = LinearOperator(dz, transformer)
        self._nonlinear_op = NonlinearOperator(dz / 2., nonlinear_function)

    def __call__(self, fr: np.array) -> np.array:
        step1 = self._nonlinear_op(fr)
        step2 = self._linear_op(step1)
        step3 = self._nonlinear_op(step2)
        return step3


# %%
class FourthOrderSolver:
    _dz: float
    # see https://doi.org/10.1016/j.matcom.2004.08.002
    _magic_constant: float = (2. + 2 ** (1. / 3) +
                              1. / 2. ** (1. / 3)) / 3.
    _solver1: SecondOrderSolver
    _solver2: SecondOrderSolver

    def __init__(self, dz, transformer, nonlinear_function):
        self._dz = dz
        dz1 = dz * self._magic_constant
        dz2 = dz * (1 - 2. * self._magic_constant)
        self._solver1 = SecondOrderSolver(dz1,
                                          transformer,
                                          nonlinear_function)
        self._solver2 = SecondOrderSolver(dz2,
                                          transformer,
                                          nonlinear_function)

    def __call__(self, fr: np.array) -> np.array:
        step1 = self._solver1(fr)
        step2 = self._solver2(step1)
        step3 = self._solver1(step2)
        return step3


# %%
def nl(field: np.array) -> np.array:
    return (1. + 1.j * NU) * plasma_density_np(field)


# %%
solver = FourthOrderSolver(dZ, TRANSFORM, nl)
r = TRANSFORM.r
z = np.arange(z0, ZEND, dZ)
Z, R = np.meshgrid(z, r, indexing='ij')

result = np.zeros_like(R, dtype=np.complex64)
result[0, :] = beam_field_np(z0, r)

# %%
for i, zi in tqdm(enumerate(z[1:]), total=len(z) - 1):
    result[i + 1, :] = solver(result[i, :])

# %%
density = plasma_density_np(result)

# %%
mpl.use('webagg')
plt.close('all')
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)

contours = np.empty((2, 2), dtype=object)

contours[0, 0] = ax[0, 0].contourf(Z, R, result.real)
contours[0, 1] = ax[0, 1].contourf(Z, R, result.imag)
contours[1, 0] = ax[1, 0].contourf(Z, R, density, 256)
contours[1, 1] = ax[1, 1].contourf(Z, R, np.abs(result), 256)

ax[0, 0].set_title('Re numeric')
ax[0, 1].set_title('Im numeric')
ax[1, 0].set_title('Plasma density')
ax[1, 1].set_title('Abs numeric')

fig.colorbar(contours[0, 0], ax=ax[0, 0])
fig.colorbar(contours[0, 1], ax=ax[0, 1])
fig.colorbar(contours[1, 0], ax=ax[1, 0])
fig.colorbar(contours[1, 1], ax=ax[1, 1])
# %%

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
ax[0].plot(z, np.abs(result[:, 0]), 'r')
ax[1].plot(z, density[:, 0], 'k')
plt.grid()
plt.show()
