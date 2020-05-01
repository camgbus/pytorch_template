from ptt.models.get_model import get_model
import pytest

#
#pytest.mark.skip(reason="no way of currently testing this")
#def test_dynamic_inst():
#    class_ref = get_model('InceptionV3')
#    config = {'pretrained': True, 'freeze_params': True, 'nr_outputs': 5}
#    model = class_ref(config)
#    assert 'torchvision.models.inception.Inception3' in str(type(model))
