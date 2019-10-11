from ptt.eval.Accumulator import Accumulator

def test_acc():
    acc = Accumulator()
    for i in range(5):
        acc.accumulate('A', float(i))
    assert acc.get_mean('A') == 2.0
    assert 1.41 < acc.get_std('A') < 1.415