################################################################################
# Experiment class that tracks experiments with different configurations.
# The idea is that if multiple experiments are performed, all intermediate 
# stored files and model states are within a directory for that experiment. In 
# addition, the experiment directory contains the config.json file with the 
# original configuration, as well as the splitting of the dataset for each fold. 
# When multiple repetitions, for instance within cross-validation, are 
# performed, all files are within the experiment directory.
################################################################################

import os
import time

from ptt.utils.helper_functions import get_time_string
from ptt.utils.load_restore import pkl_dump, save_json, save_model_state, save_df
from ptt.visualization.plot_results import plot_results

class Experiment:
    """A bundle of experiments runs with the same configuration. 
    :param config: A dictionary contains a.o. the following keys:
    - cross_validation: are the repetitions cross-validation folds?
    - nr_runs: number of repetitions/cross-validation folds
    """
    def __init__(self, config=None, name=None, notes='', reload=False):
        self.time_str = get_time_string()
        if not name:
            self.name = self.time_str
        else:
            self.name = name
        # Define subdirectories and restore\save config
        self.paths = self._set_paths()
        if reload:
            self.config = load_json(path=self.paths['root'], name='config')
        else:
            self.config = config
            pkl_dump(self.config, path=self.paths['root'], name='config')
            self._build_paths()
        # Set initial time
        self.time_start = time.time()
        self.review = {'notes': notes}

    def _set_paths(self):
        paths = dict()
        paths['root'] = os.path.join('obj', self.name)
        for subpath in ['results', 'model_states', 'obj']:
            paths[subpath] = os.path.join(paths['root'], subpath)
        return paths
        
    def _build_paths(self):
        os.makedirs(paths['root'])
        for subpath in ['results', 'model_states', 'obj']:
            os.mkdir(paths[subpath])

    def save_state(self, state_dict):
        save_model_state(model, name, self.paths['model_states'])

    def restore_state(self, )

    # Override
    def plot_results(result, save_path):
        plot_results(df=result.df, save_path=save_path, 'plot'))

    def finish(self, results=None, exception=None):
        elapsed_time = time.time() - self.time_start
        self.review['elapsed_time'] = '{0:.2f}'.format(elapsed_time/60)
        if results:
            self.review['state'] = 'SUCCESS'
            pkl_dump(results, path=self.paths['results'], name='results')
            self.write_summary_measures(results)
            if isinstance(results, list):
                for result in results:
                    plot_results(result=result, save_path=os.path.join(self.paths['results']))
            else:
                 plot_results(result=results, save_path=os.path.join(self.paths['results']))
        else:
            self.review['state'] = 'FAILED: ' + str(exception)
            # TODO: store exception with better format, or whole error path
        save_json(self.review, self.paths['root'], 'review')
        try:
            shutil.rmtree(self.paths['tmp'])
        except:
            pass

    ############################################################################
    # Functions to override for a specific kind of experiment
    ############################################################################