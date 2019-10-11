################################################################################
# Tracks experiments with different configurations
################################################################################

import os
import time

from ptt.utils.helper_functions import get_time_string
from ptt.utils.load_restore import pkl_dump, save_json, save_model_state, save_df
from ptt.visualization.plot_results import plot_results

class Experiment:

    def __init__(self, config, id=None, notes=''):
        self.time_str = get_time_string()
        if not id:
            self.id = self.time_str
        else:
            self.id = id
        self.config = config
        # Create directories and assign to field
        self.paths = self._build_paths()
        # Save config
        pkl_dump(self.config, path=self.paths['root'], name='config')
        # Set initial time
        self.time_start = time.time()
        self.review = {'notes': notes}

    def _build_paths(self):
        paths = dict()
        paths['root'] = os.path.join('obj', self.id)
        # Create root path
        os.makedirs(paths['root']) 
        # Creates subdirectories
        for subpath in ['results', 'model_states']:
            paths[subpath] = os.path.join(paths['root'], subpath)
            os.mkdir(paths[subpath])
        return paths

    def save_model_state(self, model, name):
        save_model_state(model, name, self.paths['model_states'])

    def finish(self, results = None, exception = None):
        elapsed_time = time.time() - self.time_start
        self.review['passed_time'] = '{0:.2f}'.format(elapsed_time/60)
        if results:
            self.review['state'] = 'SUCCESS'
            save_df(results.df, path=self.paths['results'], name='results')
        else:
            self.review['state'] = 'FAILED: ' + str(exception)
        save_json(self.review, 'review', self.paths['root'])
        if self.review['state'] == 'SUCCESS':
            plot_results(df=results.df, 
                save_path=os.path.join(self.paths['results'], 'plot'))
