from ptt.eval.Results import Results

def test_results():
    res = Results()
    res.extend_measure(1, 'A', 2.0)
    res.extend_measure(2, 'A', 3.0)
    res.extend_measure(3, 'A', 4.0)
    res.extend_measure(0, 'B', 5.0)
    res.extend_measure(3, 'B', 10.0)
    assert len(res.df) == 5