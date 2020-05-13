# ------------------------------------------------------------------------------
# Experiment class that tracks experiments with different configurations.
# The idea is that if multiple experiments are performed, all intermediate 
# stored files and model states are within a directory for that experiment. In 
# addition, the experiment directory contains the config.json file with the 
# original configuration, as well as the splitting of the dataset for each fold. 
# When multiple repetitions, for instance within cross-validation, are 
# performed, all files are within the experiment directory.
# ------------------------------------------------------------------------------

import os
import time
import shutil

from ptt.utils.helper_functions import get_time_string
import ptt.utils.load_restore as lr
import ptt.utils.pytorch.pytorch_load_restore as ptlr

class Experiment:
    """A bundle of experiments runs with the same configuration. """
    def __init__(self, config=None, name=None, notes='', reload_exp=False):
        """
        :param config: A dictionary contains a.o. the following keys:
        - cross_validation: are the repetitions cross-validation folds?
        - nr_runs: number of repetitions/cross-validation folds
        """
        self.time_str = get_time_string()
        if not name:
            self.name = self.time_str
        else:
            self.name = name
        # Define subdirectories and restore\save config
        self.paths = self._set_paths()
        if reload_exp:
            self.config = lr.load_json(path=self.paths['root'], name='config')
        else:
            self.config = config
            lr.pkl_dump(self.config, path=self.paths['root'], name='config')
            self._build_paths()
        # Set initial time
        self.time_start = time.time()
        self.review = {'notes': notes}

    def _set_paths(self):
        paths = dict()
        paths['root'] = os.path.join('obj', self.name)
        for subpath in ['results', 'states', 'obj', 'tmp']:
            paths[subpath] = os.path.join(paths['root'], subpath)
        return paths
        
    def _build_paths(self):
        os.makedirs(self.paths['root'])
        for subpath in ['results', 'states', 'obj', 'tmp']:
            os.mkdir(self.paths[subpath])

    def finish(self, results=None, exception=None):
        elapsed_time = time.time() - self.time_start
        self.review['elapsed_time'] = '{0:.2f}'.format(elapsed_time/60)
        if results:
            self.review['state'] = 'SUCCESS'
            lr.pkl_dump(results, path=self.paths['results'], name='results')
            self._write_summary_measures(results)
            if isinstance(results, list):
                for result in results:
                    self._plot_results(result=result, save_path=os.path.join(self.paths['results']))
            else:
                 self._plot_results(result=results, save_path=os.path.join(self.paths['results']))
        else:
            self.review['state'] = 'FAILURE: ' + str(exception)
            # TODO: store exception with better format, or whole error path
        lr.save_json(self.review, self.paths['root'], 'review')
        try:
            shutil.rmtree(self.paths['tmp'])
        except:
            pass

    def save_state(self, state_name, pytorch_dict={}, np_dict={}, pkl_dict={}):
        if 'model' in pytorch_dict:
            ptlr.save_model_state(pytorch_dict['model'], name=state_name, path=self.paths['states'])
        if 'optimizer' in pytorch_dict:
            ptlr.save_optimizer_state(pytorch_dict['optimizer'], name=state_name, path=self.paths['states'])
        if 'scheduler' in pytorch_dict:
            ptlr.save_scheduler_state(pytorch_dict['scheduler'], name=state_name, path=self.paths['states'])
        for key, value in np_dict.items():
            lr.np_dump(value, key, path=self.paths['states'])
        for key, value in pkl_dict.items():
            lr.pkl_dump(value, key, path=self.paths['states'])

    def restore_state(self, state_name, pytorch_dict, np_dict, pkl_dict):
        success = True
        if 'model' in pytorch_dict:
            success = success and ptlr.load_model_state(pytorch_dict['model'], name=state_name, path=self.paths['states'])
        if 'optimizer' in pytorch_dict:
            success = success and ptlr.load_optimizer_state(pytorch_dict['optimizer'], name=state_name, path=self.paths['states'])
        if 'scheduler' in pytorch_dict:
            success = success and ptlr.load_scheduler_state(pytorch_dict['scheduler'], name=state_name, path=self.paths['states'])
        for key in np_dict.keys():
            np_dict[key] = lr.np_load(key, path=self.paths['states'])
        for key in pkl_dict.keys():
            pkl_dict[key] = lr.pkl_load(key, path=self.paths['states'])

    # Functions to override for a specific kind of experiment
    def _plot_results(self, result, save_path):
        pass

    def _write_summary_measures(self, results):
        pass
