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
from ptt.visualization.plot_results import plot_results
from ptt.experiment.data_splitting import split_dataset
from ptt.paths import storage_path
from ptt.data.data import Data

class Experiment:
    """A bundle of experiment runs with the same configuration. """
    def __init__(self, config=None, name='', notes='', reload_exp=True):
        """
        :param config: A dictionary contains a.o. the following keys:
        - cross_validation: are the repetitions cross-validation folds?
        - nr_runs: number of repetitions/cross-validation folds
        - test_ratio: Ratio of test data
        - val_ratio: Ratio of validation data from non-test data
        :param name: experiment name. If empty, a datestring is set as name.
        :param notes: optional notes about the experiment
        :param reload_exp: Reload or throw error when expriment name exists?
            Meant for performing runs separatedly.
        """
        self.time_str = get_time_string()
        self.review = {'time_str': self.time_str, 'notes': notes}
        self.splits = None
        if not name:
            self.name = self.time_str
        else:
            self.name = name
        # Set path in defined storage directory
        self.path = os.path.join(os.path.join(storage_path, 'exp'), self.name)
        # Restore files
        if reload_exp and os.path.exists(self.path):
            assert name is not None
            self.config = lr.load_json(path=self.path, name='config')
            self.review = lr.load_json(path=self.path, name='review')
        else:
            os.makedirs(self.path)
            self.config = config
            lr.save_json(self.config, path=self.path, name='config')
            lr.save_json(self.review, path=self.path, name='review')

    def set_data_splits(self, data):
        try:
            self.splits = lr.load_json(path=self.path, name='splits')
        except FileNotFoundError:
            print('Dividing dataset')
            # If the data consists of several datasets, then the splits are a
            # dictionary with one more label, that of the dataset name.
            if isinstance(data, Data):
                self.splits = dict()
                for ds_name, ds in data.datasets.items():
                    self.splits[ds_name] = split_dataset(ds, test_ratio=self.config.get('test_ratio', 0.0), 
                    val_ratio=self.config['val_ratio'], nr_repetitions=self.config['nr_runs'], 
                    cross_validation=self.config['cross_validation'])
            else:
                self.splits = split_dataset(data, test_ratio=self.config.get('test_ratio', 0.0), 
                    val_ratio=self.config['val_ratio'], nr_repetitions=self.config['nr_runs'], 
                    cross_validation=self.config['cross_validation'])
            lr.save_json(self.splits, path=self.path, name='splits')
            print('\n')

    def get_run(self, run_ix):
        return ExperimentRun(run_ix, self.path)

    def finish(self, results=None):
        """After running all runs, finish expeirment by recording average values"""
        # TODO first do eval.result.ExperimentResults
        pass

class ExperimentRun:
    """Experiment runs with different indexes for train, val, test. """
    def __init__(self, run_ix, exp_path):
        self.run_ix = run_ix
        self.paths = self._set_paths(exp_path)
        self.time_start = time.time()
        self.review = {'time_str': get_time_string()}

    def _set_paths(self, exp_path):
        paths = dict()
        paths['root'] = os.path.join(exp_path, str(self.run_ix))
        if os.path.exists(paths['root']):
            shutil.rmtree(paths['root'])
        os.mkdir(paths['root'])
        for subpath in ['results', 'states', 'obj', 'tmp']:
            paths[subpath] = os.path.join(paths['root'], subpath)
            os.mkdir(paths[subpath])
        return paths

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

    # Functions to override for a specific kind of experiment

    def save_state(self, state_name, pytorch_dict={}, np_dict={}, pkl_dict={}):
        if 'model' in pytorch_dict:
            ptlr.save_model_state(pytorch_dict['model'], name=state_name+'_model', path=self.paths['states'])
        if 'optimizer' in pytorch_dict:
            ptlr.save_optimizer_state(pytorch_dict['optimizer'], name=state_name+'_optimizer', path=self.paths['states'])
        if 'scheduler' in pytorch_dict:
            ptlr.save_scheduler_state(pytorch_dict['scheduler'], name=state_name+'_scheduler', path=self.paths['states'])
        for key, value in np_dict.items():
            lr.np_dump(value, key, path=self.paths['states'])
        for key, value in pkl_dict.items():
            lr.pkl_dump(value, key, path=self.paths['states'])

    def restore_state(self, state_name, pytorch_dict, np_dict, pkl_dict):
        success = True
        if 'model' in pytorch_dict:
            success = success and ptlr.load_model_state(pytorch_dict['model'], name=state_name+'_model', path=self.paths['states'])
        if 'optimizer' in pytorch_dict:
            success = success and ptlr.load_optimizer_state(pytorch_dict['optimizer'], name=state_name+'_optimizer', path=self.paths['states'])
        if 'scheduler' in pytorch_dict:
            success = success and ptlr.load_scheduler_state(pytorch_dict['scheduler'], name=state_name+'_scheduler', path=self.paths['states'])
        for key in np_dict.keys():
            np_dict[key] = lr.np_load(key, path=self.paths['states'])
        for key in pkl_dict.keys():
            pkl_dict[key] = lr.pkl_load(key, path=self.paths['states'])

    def _plot_results(self, result, save_path):
        plot_results(result, save_path=self.paths['results'])

    def _write_summary_measures(self, results):
        pass
