import os
import shutil

from ptt.experiment.experiment import Experiment
from ptt.eval.result import Result
from ptt.utils.load_restore import load_json
from ptt.paths import storage_path

def test_success():
    notes='A test experiment which is successful'
    exp = Experiment({'test_param': 2}, name='TEST_SUCCESS', notes=notes)
    res = Result(name='some_result')
    res.add(1, 'A', 2.0)
    res.add(3, 'A', 10.0)
    exp.finish(results=res)
    path = os.path.join(storage_path, 'TEST_SUCCESS')
    review = load_json(path, 'review')
    assert review['notes'] == notes
    config = load_json(path, 'config')
    assert config['test_param'] == 2
    res_path = os.path.join(path, 'results')
    assert os.path.isfile(os.path.join(res_path, 'some_result.png'))
    shutil.rmtree(path)

def test_failure():
    notes='A test experiment which fails'
    exp = Experiment({'test_param': 2}, name='TEST_FAILURE', notes=notes)
    exp.finish(exception=Exception)
    path = os.path.join(storage_path, 'TEST_FAILURE')
    review = load_json(path, 'review')
    assert review['notes'] == notes
    assert 'FAILURE' in review['state']
    shutil.rmtree(path)

def test_reload():
    notes='A test experiment which is reloaded'
    # First experiment creation
    exp = Experiment({'test_param': 2}, name='TEST_RELOAD', notes=notes)
    res = Result(name='some_result')
    res.add(1, 'A', 2.0)
    exp.finish(results=res)
    # Experiment reload
    exp = Experiment(name='TEST_RELOAD', reload_exp=True)
    assert exp.review['notes'] == notes
    assert exp.config['test_param'] == 2
    exp.finish(results=res)
    path = os.path.join(storage_path, 'TEST_RELOAD')
    shutil.rmtree(path)