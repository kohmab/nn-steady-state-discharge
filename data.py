from abc import abstractmethod, ABC
from typing import Dict

import numpy as np
from torch.utils.data import Dataset, DataLoader
from parametersholder import ParametersHolder
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

        result = [self.points[idx, indices["z"]],
                  self.points[idx, indices["r"]],
                  self.points[idx, indices["params"]]]
        if self.target is not None:
            result.append(self.target[idx, :])
        return tuple(result)


class DomainData(AbstractDataset):
    def __init__(self):
        super().__init__(ParametersHolder().Npoints)

    def _generate_data(self):
        lb = ParametersHolder().get_lb()
        ub = ParametersHolder().get_ub()

        self._points = uniform(self.N, lb, ub, dim=len(lb))


class BcAtAxisData(AbstractDataset):
    def __init__(self):
        super().__init__(ParametersHolder().Npoints_for_boundary_conditions)

    def _generate_data(self):
        lb = ParametersHolder().get_lb()
        ub = ParametersHolder().get_ub()
        ub[1] = lb[1]
        self._points = uniform(self.N, lb, ub, dim=len(lb))


class BcAtUpperRLimitData(AbstractDataset):
    def __init__(self):
        super().__init__(ParametersHolder().Npoints_for_boundary_conditions)

    def _generate_data(self):
        lb = ParametersHolder().get_lb()
        ub = ParametersHolder().get_ub()
        lb[1] = ub[1]
        self._points = uniform(self.N, lb, ub, dim=len(lb))
        self._target = beam_field(self._points[:, 0:2], max_field=self._points[:, [2]])  # TODO use idx


class IcData(AbstractDataset):

    def __init__(self):
        super().__init__(ParametersHolder().Npoints_for_initial_condition)

    def _generate_data(self):
        lb = ParametersHolder().get_lb()
        ub = ParametersHolder().get_ub()
        ub[0] = lb[0]
        self._points = uniform(self.N, lb, ub, dim=len(lb))
        self._target = beam_field(self._points[:, 0:2], max_field=self._points[:, [2]])  # TODO use idx


class JointDataLoaders:
    def __init__(self, batch_size, main_dataset, additional_datasets: Dict[str, Dataset]):
        self._dataloaders = {"main": DataLoader(main_dataset, batch_size=batch_size)}
        self._data = {"main": None}
        self._iterators = {"main": iter(self._dataloaders["main"])}
        self._len = len(self._dataloaders["main"])

        for name, dataset in additional_datasets.items():
            self._dataloaders[name] = DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=False)
            self._data[name] = None

        self._reset_all()

    def _reset_all(self):
        for name in self._dataloaders:
            self._reset_one(name)
        self.count = -1

    def _reset_one(self, name):
        self._iterators[name] = iter(self._dataloaders[name])

    def _prepare_batch(self, name):
        arrays_tuple = next(self._iterators[name])

        tensors = [prepare_tensor(array) for array in arrays_tuple]

        return tuple(tensors)

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        for name in self._iterators:
            try:
                self._data[name] = self._prepare_batch(name)
            except StopIteration:
                if name == "main":
                    self._reset_all()
                    raise StopIteration
                else:
                    self._reset_one(name)
                    self._data[name] = self._prepare_batch(name)

        self.count += 1

        return self.count

    @property
    def data(self):
        return self._data


# TODO remove garbage
if __name__ == "__main__":
    class TestDataset(Dataset):
        def __init__(self, N):
            self._N = N
            self._data = np.arange(N)

        def __len__(self):
            return self._N

        def __getitem__(self, idx):
            return self._data[idx]


    main = TestDataset(100)
    first = TestDataset(33)
    second = TestDataset(25)

    dl = JointDataLoaders(11, main, {"1": first, "2": second})

    for i in dl:
        print(i)
        print(dl.data)
