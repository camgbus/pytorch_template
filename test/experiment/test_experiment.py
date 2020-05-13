
import os
import shutil

from ptt.eval.Experiment import Experiment
from ptt.eval.Results import Results
from ptt.utils.load_restore import load_json, pkl_load, load_df

def test_success():
    notes='A test experiment which is successful'
    exp = Experiment({'test_param': 2}, id='TEST_SUCCESS', notes=notes)
    res = Results()
    res.extend_measure(1, 'A', 2.0)
    res.extend_measure(3, 'A', 10.0)
    exp.finish(results=res)
    path = os.path.join('obj', 'TEST_SUCCESS')
    review = load_json('review', path)
    assert review['notes'] == notes
    config = pkl_load('config', path)
    assert config['test_param'] == 2
    res_path = os.path.join(path, 'results')
    assert os.path.isfile(os.path.join(res_path, 'plot.png'))
    results = load_df('results', res_path)
    assert len(results) == 2
    shutil.rmtree(path)

def test_failure():
    notes='A test experiment which fails'
    exp = Experiment({'test_param': 2}, id='TEST_FAILURE', notes=notes)
    exp.finish(exception=Exception)
    path = os.path.join('obj', 'TEST_FAILURE')
    review = load_json('review', path)
    assert review['notes'] == notes
    assert 'FAILED' in review['state']
    shutil.rmtree(path)