################################################################################
# Accumulates results in a minibatch
################################################################################
import numpy as np

class Accumulator():
    def __init__(self, currents=['epoch']):
        self.dict = dict()

    def accumulate(self, key, value):
        if key in self.dict:
            self.dict[key].append(value)
        else:
            self.dict[key] = [value]

    def get_mean(self, key):
        return np.mean([x for x in self.dict[key]])

    def get_std(self, key):
        return np.std([x for x in self.dict[key]])

'''
acc = Accumulator()
for i in range(5):
    acc.accumulate('A', float(i))
assert acc.get_mean('A') == 2.0
assert 1.41 < acc.get_std('A') < 1.415
'''


#%%