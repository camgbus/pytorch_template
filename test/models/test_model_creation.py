import os
import shutil
import numpy as np
from ptt.models.small_cnn import SmallCNN

def test_model_creation():
    # Test storing and restoring new weights
    weights_init_path = os.path.join(os.path.join('test', 'tmp'), 'test_model_state')
    model = SmallCNN(input_shape=(3, 32, 32), output_shape=2)
    model.initialize(weights_init_path)
    curr_param_dict = dict()
    for name, param in model.named_parameters():
        curr_param_dict[name] = param.detach().cpu().numpy()
    model = SmallCNN(input_shape=(3, 32, 32), output_shape=2)
    model.initialize(weights_init_path)
    for name, param in model.named_parameters():
        assert np.allclose(curr_param_dict[name], param.detach().cpu().numpy())
    shutil.rmtree(os.path.join('test', 'tmp'))

def test_model_summary():
    model = SmallCNN(input_shape=(3, 32, 32), output_shape=10)
    summary = model.model_summary()
    assert 'Total params: 62,006' in summary
    assert 'Trainable params: 62,006' in summary
    assert 'Non-trainable params: 0' in summary