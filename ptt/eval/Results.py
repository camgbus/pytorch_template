################################################################################
# Stores results in a Pandas df
################################################################################
import pandas as pd

class Results():
    def __init__(self):
        self.df = pd.DataFrame(columns=['epoch', 
            'measure_name', 
            'measure_value'])

    def extend_measure(self, epoch, measure, value):
        assert isinstance(epoch, int)
        assert isinstance(measure, str)
        assert isinstance(value, float)
        self.df = self.df.append({'epoch': epoch, 
            'measure_name': measure, 
            'measure_value': value}, 
            ignore_index=True)
