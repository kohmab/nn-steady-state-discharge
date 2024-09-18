from typing import Dict, Optional

from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from helpers import uniform, beam_field, prepare_tensor
from hyperdomain import *

_DEVICE = PARAMETERS.torch.device
_DTYPE = PARAMETERS.torch.dtype
_LOWER_BOUNDARIES = HYPERDOMAIN.lb
_UPPER_BOUNDARIES = HYPERDOMAIN.ub
_DIM = HYPERDOMAIN.dim


class AbstractDataset(Dataset, ABC):
    _points: Tensor
    _target: Optional[Tensor]

    _N: int

    def __init__(self, Npoints):
        self._N = Npoints
        self._points = None
        self._target = None
        self.refresh()

    def refresh(self):
        self._generate_points()
        self._generate_data()

    def _generate_points(self):
        self._points = uniform(self.N, _LOWER_BOUNDARIES, _UPPER_BOUNDARIES, dim=_DIM)

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
        result = [self.points[idx, [AXIDX.z]],
                  self.points[idx, [AXIDX.r]],
                  self.points[idx, AXIDX.max_field:]]
        if self.target is not None:
            result.append(self.target[idx, :])
        if idx == self._N - 1:
            self.refresh()
        return tuple(result)


class DomainData(AbstractDataset):

    def __init__(self, Npoints):
        super().__init__(Npoints)

    def _generate_data(self):
        return


class BcAtAxisData(AbstractDataset):

    def __init__(self, Npoints):
        super().__init__(Npoints)

    def _generate_data(self):
        self._points[1, :] = _LOWER_BOUNDARIES[1]


class BcAtUpperRLimitData(AbstractDataset):

    def __init__(self, Npoints):
        super().__init__(Npoints)

    def _generate_data(self):
        self._points[1, :] = _UPPER_BOUNDARIES[1]
        self._target = beam_field(self._points[:, AXIDX.z:AXIDX.max_field],
                                  max_field=self._points[:, [AXIDX.max_field]])


class IcData(AbstractDataset):

    def __init__(self, Npoints):
        super().__init__(Npoints)

    def _generate_data(self):
        self._points[0, :] = _LOWER_BOUNDARIES[0]
        self._target = beam_field(self._points[:, AXIDX.z:AXIDX.max_field],
                                  max_field=self._points[:, [AXIDX.max_field]])


class SegmentData:
    def __init__(self, Npoints):
        self._N = Npoints

    def __call__(self, lb, ub):
        points = uniform(self._N, lb, ub, _DIM).requires_grad_(True)
        return points[:, [AXIDX.z]], points[:, [AXIDX.r]], points[:, AXIDX.max_field:]


class JointDataLoaders:
    _main_key: str

    def __init__(self, batch_size, main_key, datasets: Dict[str, Dataset]):
        assert main_key in datasets

        self._main_key = main_key

        self._dataloaders = {}
        self._data = {}
        self._iterators = {}

        for name, dataset in datasets.items():
            self._dataloaders[name] = DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=False)
            self._data[name] = None

        self._len = len(self._dataloaders[self._main_key])

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
                if name == self._main_key:
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
