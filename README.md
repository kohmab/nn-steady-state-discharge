This repository reflects experiments on training a neural network of type Pi-DeepONet for calculating a stationary discharge and an axially symmetric Gaussian beam.

An equation \[[1](https://doi.org/10.1016/0167-2789(95)00138-T)\] of the type of the nonlinear Schrödinger equation is solved for the field

$$\Delta_{\perp} E + 2i\frac{\partial E}{\partial z} = (1 - i\nu)nE,$$

where $\nu$ is the effective collision frequency. The plasma density  $n$ is determined by the expression.

$$ n = \begin{cases}      
      K_0(|E|^{2p} - 1) & \text{at~} |E| > 1,\\
      0 & \text{otherwise}.
    \end{cases}$$

Initial condition is

$$E(r,z_{min}) = E_{max} \frac{\exp(- \frac{r^2}{2\xi})}{\xi},~~ \xi = 1 + i  z_{min}.$$
