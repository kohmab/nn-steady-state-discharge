import torch


class EqLossesWithWeights:
    _eps: float

    def __init__(self, eps: float = 1.0):
        self._eps = eps

    def __call__(self, losses: torch.Tensor) -> torch.Tensor:
        sum = losses.clone().detach()
        for i in range(len(sum) - 1):
            sum[i+1] += sum[i]
        weights = torch.exp(-self._eps * sum)
        print(weights)
        return torch.mean(weights*losses)


if __name__ == "__main__":
    tlww = EqLossesWithWeights()
    l = torch.ones(9)*1e-8
    print(tlww(l))
