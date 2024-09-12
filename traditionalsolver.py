# %%
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pyhank import HankelTransform
from tqdm import tqdm

# %%
NU = 0.3
K0 = 4
P = 1
EMAX = 2.

NR = 4096
RMAX = 12.
dZ = 1.e-2

ZEND = 3.


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
solver = FourthOrderSolver(dZ, TRANSFORM, lambda x: np.zeros_like(x))
r = TRANSFORM.r
z = np.arange(z0, ZEND, dZ)
Z, R = np.meshgrid(z, r, indexing='ij')

result = np.zeros_like(R, dtype=np.complex64)
result[0, :] = beam_field_np(z0, r)
# %%
for i, zi in tqdm(enumerate(z[1:]), total=len(z) - 1):
    result[i + 1, :] = solver(result[i, :])

# %%
analytic = beam_field_np(Z, R)
# %%
# Create a figure and a 2x2 grid of subplots with shared x and y axes

mpl.use('webagg')
plt.close('all')
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)

# Initialize an array to store the contour plots
contours = np.empty((2, 2), dtype=object)

# Plot the real and imaginary parts of the numeric and analytic results
contours[0, 0] = ax[0, 0].contourf(Z, R, result.real)
contours[0, 1] = ax[0, 1].contourf(Z, R, result.imag)
contours[1, 0] = ax[1, 0].contourf(Z, R, analytic.real)
contours[1, 1] = ax[1, 1].contourf(Z, R, analytic.imag)

# Set titles for each subplot
ax[0, 0].set_title('Re numeric')
ax[0, 1].set_title('Im numeric')
ax[1, 0].set_title('Re analytic')
ax[1, 1].set_title('Im analytic')

fig.colorbar(contours[0, 0], ax=ax[0, 0])
fig.colorbar(contours[0, 1], ax=ax[0, 1])
fig.colorbar(contours[1, 0], ax=ax[1, 0])
fig.colorbar(contours[1, 1], ax=ax[1, 1])

fig, ax = plt.subplots()
residual = np.log10(np.abs(result/analytic - 1.))
cr = ax.contourf(Z, R, residual)
ax.set_title('Residual')
fig.colorbar(cr, ax=ax)

# Display the figure
plt.show()
