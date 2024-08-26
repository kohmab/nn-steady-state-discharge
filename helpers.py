import numpy as np
import torch

def to_torch(*args):
    result = []
    for arg in args:
        if not torch.is_tensor(arg):
            result.append(torch.from_numpy(arg).float())
        else:
            result.append(arg.float())
    return tuple(result)


a = np.zeros(3)
b = torch.zeros(1)
c = np.eye(1)
d = torch.eye(2)

print(to_torch(a, b, c, d))
