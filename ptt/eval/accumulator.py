# ------------------------------------------------------------------------------
# Accumulates results from a minibatch.
# ------------------------------------------------------------------------------

import numpy as np
import torch

class Accumulator:
    def __init__(self, keys=['loss']):
        self.keys = keys
        self.values = dict()
        self.init()

    def add(self, key, value, count=1):
        if isinstance(value, torch.Tensor):
            np_value = float(value.detach().cpu().numpy())
        else:
            np_value = value
        for _ in range(count):
            self.values[key].append(np_value)

    def mean(self, key):
        return np.mean(self.values[key])

    def std(self, key):
        return np.std(self.values[key])

    def sum(self, key):
        return sum(self.values[key])

    def init(self):
        for key in self.keys:
            self.values[key] = []
