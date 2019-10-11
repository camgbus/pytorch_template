################################################################################
# Stores results in a Pandas DataFrame
################################################################################

import pandas as pd

from ptt.utils.load_restore import save_df

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
    
    def save(self, name, path):
        save_df(self.df, name, path)