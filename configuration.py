import numpy as np
import torch


class Configuration:
    _instance = None
    _is_initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        torch.manual_seed(1234)

        self.z_limits = np.asarray([-4., 8.])
        self.r_limits = np.asarray([0., 4.])

        self.max_field_limits = np.asarray([1., 2.])

        self.power_limits = np.asarray([1., 1.])
        self.coefficient_K0_limits = np.asarray([0.1, 10.])
        self.collision_frequency_limits = np.asarray([0.0, 0.0])

        self.device = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32

        self.Npoints = 1024
        self.Npoints_for_boundary_conditions = 1024
        self.Npoints_for_initial_condition = 1024

        self.train_batch_size = 64
        self.test_batch_size = 64

        self.trunk_hidden_layers = 3
        self.branch_hidden_layers = 3
        self.trunk_hidden_features = 96
        self.branch_hidden_features = 96
        self.branch_and_trunk_out_features = 96

        Configuration._is_initialized = True

    def get_lb(self):
        return self._get_boundary(0)

    def get_ub(self):
        return self._get_boundary(1)

    def _get_boundary(self, index):
        result = []
        for k, v in vars(self).items():
            if isinstance(v, np.ndarray):
                result.append(v[index])
        return np.hstack(tuple(result))

    def __setattr__(self, __name, __value):
        if Configuration._is_initialized:
            return
        object.__setattr__(self, __name, __value)

    def __repr__(self):
        result = []
        for k, v in vars(self).items():
            result.append(f'{k}: {v}')
        return "parameters: " + ', '.join(result)


if __name__ == '__main__':
    c = Configuration()
    print(c)
    print(c.get_lb())
    print(c.__dict__)
