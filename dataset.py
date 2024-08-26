import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs


d = lhs(2,10)
print(d)
