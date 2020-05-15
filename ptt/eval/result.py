# ------------------------------------------------------------------------------
# A class which accumulates results for easy visualization.
# 'Result' stores the per-epoch results for a run, e.g. for a fold.
# 'ExperimentResult' calculates the average over all runs at the end.
# ------------------------------------------------------------------------------

import pandas as pd

class ExperimentResults():
    """Per-epoch results for all repetitions."""
    def __init__(self, global_result_lst, epoch_result_lst):
        pass
        # TODO

class Result():
    """Per-epoch results for 1 repetition."""
    def __init__(self, name='Results'):
        self.name = name
        self.results = dict()

    def add(self, epoch, metric, value, data='train'):
        assert isinstance(epoch, int)
        assert isinstance(metric, str)
        assert isinstance(data, str)
        assert isinstance(value, float) or isinstance(value, int)
        if metric not in self.results:
            self.results[metric] = dict()
        if epoch not in self.results[metric]:
            self.results[metric][epoch] = dict()
        self.results[metric][epoch][data] = value

    def get_epoch_metric(self, epoch, metric, data='train'):
        try:
            value = self.results[metric][epoch][data]
            return value
        except Exception:
            return None

    def to_pandas(self):
        data = [[metric, epoch, data, 
            self.results[metric][epoch][data]] 
            for metric in self.results.keys()
            for epoch in self.results[metric].keys()
            for data in self.results[metric][epoch].keys()]
        df = pd.DataFrame(data, columns = ['Metric', 'Epoch', 'Data', 'Value'])
        return df

    def get_min_epoch(self, metric, data='val'):
        return min(self.results[metric].keys(), key=lambda e: self.results[metric][e][data])

    def get_max_epoch(self, metric, data='val'):
        return max(self.results[metric].keys(), key=lambda e: self.results[metric][e][data])