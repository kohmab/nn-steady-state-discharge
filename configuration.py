import numpy as np
import torch


class Config:
    lb = np.asarray([-4., 0.])  # z,r
    ub = np.asarray([8., 4.])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    Npoints = 1024 * 2
    Nbcpoints = 1024 * 2
    Nicpoints = 1024 * 2
