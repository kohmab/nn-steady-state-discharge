import torch


class EqLossesWithWeights:
    _eps: float

    def __init__(self, eps: float = 1.0):
        self._eps = eps

    def __call__(self, losses: torch.Tensor) -> torch.Tensor:
        sum = losses.clone().detach()
        for i in range(len(sum) - 1):
            sum[i + 1] += sum[i]
        weights = torch.exp(-self._eps * sum)
        return torch.mean(weights * losses)


if __name__ == "__main__":
    tlww = EqLossesWithWeights(eps=1.0)
    x = torch.ones(10).requires_grad_()
    l = x**2
    loss= tlww(l)
    loss.backward()
    print(tlww(l))
    print(x.grad)
