from abc import abstractmethod, ABC

import numpy as np
from torch.utils.data import Dataset, DataLoader
from configuration import Configuration
from helpers import uniform, beam_field, prepare_tensor


class AbstractDataset(Dataset, ABC):
    _points: np.ndarray
    _target: np.ndarray

    _N: int

    def __init__(self, Npoints):
        self._N = Npoints
        self._points = None
        self._target = None
        self._generate_data()

    @abstractmethod
    def _generate_data(self):
        pass

    @property
    def points(self):
        return self._points

    @property
    def target(self):
        return self._target

    @property
    def N(self):
        return self._N

    def __len__(self):
        return self._N

    def __getitem__(self, idx):
        indices = {"z": slice(0, 1), "r": slice(1, 2), "params": slice(2, None)}  # TODO move from here

        result = [prepare_tensor(self.points[idx, indices["z"]]),
                  prepare_tensor(self.points[idx, indices["r"]]),
                  prepare_tensor(self.points[idx, indices["params"]])]
        if self.target is not None:
            result.append(prepare_tensor(self.target[idx, :]))
        return tuple(result)


class DomainData(AbstractDataset):
    def __init__(self):
        super().__init__(Configuration().Npoints)

    def _generate_data(self):
        lb = Configuration().get_lb()
        ub = Configuration().get_ub()

        self._points = uniform(self.N, lb, ub, dim=len(lb))


class BcAtAxisData(AbstractDataset):
    def __init__(self):
        super().__init__(Configuration().Npoints_for_boundary_conditions)

    def _generate_data(self):
        lb = Configuration().get_lb()
        ub = Configuration().get_ub()
        ub[1] = lb[1]
        self._points = uniform(self.N, lb, ub, dim=len(lb))


class BcAtUpperRLimitData(AbstractDataset):
    def __init__(self):
        super().__init__(Configuration().Npoints_for_boundary_conditions)

    def _generate_data(self):
        lb = Configuration().get_lb()
        ub = Configuration().get_ub()
        lb[1] = ub[1]
        self._points = uniform(self.N, lb, ub, dim=len(lb))
        self._target = beam_field(self._points[:, 0:2], max_field=self._points[:, [2]])  # TODO use idx


class IcData(AbstractDataset):

    def __init__(self):
        super().__init__(Configuration().Npoints_for_initial_condition)

    def _generate_data(self):
        lb = Configuration().get_lb()
        ub = Configuration().get_ub()
        ub[0] = lb[0]
        self._points = uniform(self.N, lb, ub, dim=len(lb))
        self._target = beam_field(self._points[:, 0:2], max_field=self._points[:, [2]])  # TODO use idx


class Dataloaders:  # TODO think better
    def __init__(self, batch_size):
        self.domain = DataLoader(DomainData(), batch_size=batch_size)
        self.axis = DataLoader(BcAtAxisData(), batch_size=batch_size)
        self.upper = DataLoader(BcAtUpperRLimitData(), batch_size=batch_size)
        self.ic = DataLoader(IcData(), batch_size=batch_size)


if __name__ == "__main__":
    d = BcAtAxisData()
    dl = Dataloaders(Configuration().train_batch_size)
    print(next(iter(dl.ic)))
