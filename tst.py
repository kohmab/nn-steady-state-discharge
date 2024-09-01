import torch

torch.set_printoptions(precision=2, sci_mode=False)
bs = 10
bes_s = 4
x = torch.randn(bs, bes_s, requires_grad=True)
a = torch.ones(1, bes_s)
from helpers import BesselJ0

# print(x)
# print(a)
# print(a*x)
x = torch.linspace(0, 10, 10, requires_grad=True)
f = BesselJ0.apply(x**2)*x+x
f.sum().backward()
print(x.grad)
print(-2*torch.special.bessel_j1(x**2)*x**2 + torch.special.bessel_j0(x**2) + 1 )
