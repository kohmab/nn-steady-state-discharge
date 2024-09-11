# %%
import numpy as np
from pyhank import HankelTransform
import scipy.special
import matplotlib.pyplot as plt
import matplotlib as mpl

NU = 0.3
K0 = 4


def grid_plasma_density(E, p, k0, shape):
    tmp = E.reshape(-1, 1)
    components = np.zeros


# %%
transformer = HankelTransform(order=0, max_radius=18, n_points=1024)
r = transformer.r
k = transformer.kr

z = 4
ksi = 1 - 1j * z
f = np.exp(-r ** 2 / 2 / ksi) / ksi
analyt = 1 / ksi * (r ** 2 / ksi - 2) * f

ht = transformer.qdht(f)
hlapl = -ht * k ** 2
lapl = transformer.iqdht(hlapl)
plt.figure()
plt.plot(r, lapl.real, 'r')
plt.plot(r, lapl.imag, 'b')
plt.plot(r, analyt.real, 'k:')
plt.plot(r, analyt.imag, 'k:')
plt.plot(r, np.abs(f), 'g')
plt.ylim((-10, 10))

plt.figure()
plt.plot(r, np.abs(lapl / analyt), 'g')
plt.ylim((-10, 10))
print(np.mean(lapl / analyt))
plt.show()
