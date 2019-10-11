import shutil
import os

from ptt.eval.Results import Results
from ptt.visualization.plot_results import plot_results

def test_plotting():
    res = Results()
    res.extend_measure(1, 'A', 2.0)
    res.extend_measure(2, 'A', 3.0)
    res.extend_measure(3, 'A', 4.0)
    res.extend_measure(0, 'B', 5.0)
    res.extend_measure(3, 'B', 10.0)
    save_path = os.path.join('test_fgrs', 'test')
    plot_results(res.df, measures = ['A', 'B'], save_path=save_path, title='Test figure', ending='.png')
    assert os.path.isfile(save_path+'.png')
    shutil.rmtree('test_fgrs')