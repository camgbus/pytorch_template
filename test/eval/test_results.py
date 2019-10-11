import os
import shutil

from ptt.eval.Results import Results
from ptt.utils.load_restore import load_df

def test_results():
    res = Results()
    res.extend_measure(1, 'A', 2.0)
    res.extend_measure(2, 'A', 3.0)
    res.extend_measure(3, 'A', 4.0)
    res.extend_measure(0, 'B', 5.0)
    res.extend_measure(3, 'B', 10.0)
    assert len(res.df) == 5

def test_save_results():
    res = Results()
    res.extend_measure(1, 'A', 2.0)
    try:
        os.mkdir('test_res')
    except:
        pass
    res.save('test', 'test_res')
    df = load_df('test', 'test_res')
    assert len(df) == 1
    shutil.rmtree('test_res')

    
