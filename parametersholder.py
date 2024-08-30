import numpy as np


class ParametersHolder:
    _instance = None
    _is_initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, configuration_dict=None):
        if ParametersHolder._is_initialized:
            return

        if configuration_dict is None:
            raise ValueError("You must provide a configuration dictionary. Parameters are not defined yet")

        for key, value in configuration_dict.items():
            setattr(self, key, value)

        ParametersHolder._is_initialized = True

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

    def __setattr__(self, __name, __value): # TODO remove
        if ParametersHolder._is_initialized:
            return
        object.__setattr__(self, __name, __value)

    def __repr__(self):
        result = []
        for k, v in vars(self).items():
            result.append(f'{k}: {v}')
        return "parameters: " + ', '.join(result)


if __name__ == '__main__':
    c = ParametersHolder()
    print(c)
    print(c.get_lb())
    print(c.__dict__)
