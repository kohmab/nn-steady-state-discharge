import numpy as np
import torch
import yaml
from box import Box

_CONFIG_FILE = 'configuration.yml'

with open(_CONFIG_FILE, 'r') as file:
    _CONFIG = yaml.safe_load(file)

for key, value in _CONFIG['limits'].items():
    _CONFIG[key] = np.asarray(value)

_CONFIG['torch']['device'] = torch.device(_CONFIG['torch']['device'] if torch.cuda.is_available() else 'cpu')
_CONFIG['torch']['dtype'] = eval(_CONFIG['torch']['dtype'])

PARAMETERS = Box(_CONFIG, frozen_box=True)
