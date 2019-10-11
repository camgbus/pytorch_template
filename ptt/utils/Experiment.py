import os
import time

from utils.helper_functions import get_time_string
from utils.load_restore import pkl_dump, save_json, save_model_state
from ptt.visualization.plot_results import plot_results

class Experiment:

    def __init__(self, config):
        self.time_str = get_time_string()
        self.config = config
        # Create directories and assign to field
        self.paths = self.build_paths()
        # Save config
        pkl_dump(self.config, path=self.paths['root'], name='config')
        # Set initial time
        self.time_start = time.time()
        self.review = dict()

    def build_paths(self):
        paths = dict()
        paths['root'] = os.path.join('obj', self.time_str)
        # Create root path
        os.makedirs(paths['root']) 
        # Creates subdirectories
        for subpath in ['outputs', 'results', 'model_states']:
            paths[subpath] = os.path.join(paths['root'], subpath)
            os.mkdir(paths[subpath])
        return paths

    def save_model_state(self, model, name):
        save_model_state(model, name, self.paths['model_states'])

    def finish(self, results = None, accumulator = None, exception = None):
        elapsed_time = time.time() - self.time_start
        self.review['passed_time'] = '{0:.2f}'.format(elapsed_time/60)
        if results:
            self.review['state'] = 'SUCCESS'
            pkl_dump(results, path=self.paths['results'], name='results')
        else:
            self.review['state'] = 'FAILED: ' + str(exception)
        save_json(self.review, self.paths['root'], 'review')
        if self.review['state'] == 'SUCCESS':
            plot_results(self.time_str)
