import numpy as np
from ptt.data.dataset_classification import CIFAR10
from ptt.data.pytorch_dataset import ImgClassificationDataset
from ptt.utils.pytorch.compute_normalization_values import normalization_values

def test_normalization_values():
    dataset = CIFAR10()
    pt_dataset = ImgClassificationDataset(dataset)
    norm_values = normalization_values(pt_dataset)
    assert np.allclose(np.array(norm_values[0]), np.array([0.4913998, 0.4821584, 0.44653133]))
    assert np.allclose(np.array(norm_values[1]), np.array([0.2470308, 0.24348563, 0.2615871]))
    normed_dataset = ImgClassificationDataset(dataset, norm=norm_values)
    norm_values = normalization_values(normed_dataset)
    assert np.allclose(np.array(norm_values['mean']), np.array([0., 0., 0.]), atol=1e-05)
    assert np.allclose(np.array(norm_values['std']), np.array([1, 1, 1]))
