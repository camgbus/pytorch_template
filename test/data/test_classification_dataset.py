from ptt.data.dataset_classification import CIFAR10

def test_classification_dataset():
    ds = CIFAR10()
    assert ds.get_class_dist() == {'truck': 5000, 'automobile': 5000, 'ship': 5000, 'horse': 5000, 'bird': 5000, 'dog': 5000, 'cat': 5000, 'frog': 5000, 'deer': 5000, 'airplane': 5000}
    assert ds.get_class_dist(ixs=ds.hold_out_ixs) == {'truck': 1000, 'automobile': 1000, 'ship': 1000, 'horse': 1000, 'bird': 1000, 'dog': 1000, 'cat': 1000, 'frog': 1000, 'deer': 1000, 'airplane': 1000}
