# ------------------------------------------------------------------------------
# A class which accumulates results for easy plotting.
# 'EpochResult' stores the per-epoch results for a run, e.g. for a fold.
# 'GlobalResult' stores the final results for a run, e.g. the last model score.
# 'ExperimentResult' calculates the average over all runs at the end.
# ------------------------------------------------------------------------------

import pandas as pd

class ExperimentResults():
    def __init__(self, global_result_lst, epoch_result_lst):
        # TODO: perform averaging

class GlobalResults():
    """Summary results for one run."""
    def __init__(self, name, metrics, splits=['train', 'val', 'test']):
        self.name = name
        self.splits = splits
        self.results = {metric: {split: None for split in 
            self.splits} for metric in metrics}
    
    def add(self, value, metric, split='train'):
        self.results[metric][split] = value

    def get(self, metric, split='train'):
        return self.results[metric][split]

class EpochResult():
    """Per-epoch results."""
    def __init__(self, name, metrics, splits=['train', 'val', 'test']):
        self.name = name
        self.metrics = metrics
        self.results = dict()
        self.splits = splits

    def add(self, epoch, metric, value, split='train'):
        assert metric in self.metrics
        assert isinstance(epoch, int)
        assert isinstance(epoch, int)
        assert isinstance(value, float) or isinstance(value, int)
        if epoch not in self.results:
            self.results[epoch] = dict()
        if metric not in self.results[epoch]:
            self.results[epoch][metric] = dict()
        self.results[epoch][metric][split] = value

    def get_epoch_metric(self, epoch, metric, split='train'):
        try:
            value = self.results[epoch][metric][split]
            return value
        except:
            return None

    def to_pandas(self, metrics=None):
        if not metrics:
            metrics = self.metrics # Use all
        data = [[epoch, split, metric, 
            self.get_epoch_metric(epoch, metric, split=split)] 
            for epoch in self.results 
            for split in self.splits
            for metric in self.metrics if metric in metrics]
        data = [x for x in data if x[3]]
        df = pd.DataFrame(data, columns = ['Epoch', 'Split', 'Metric', 'Value'])
        return df

    def get_best_epoch(self, metric='dice', split='val'):
        return min(self.results.keys(), key=lambda e: self.results[e][metric][split])

