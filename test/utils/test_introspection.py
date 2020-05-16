from ptt.utils.introspection import introspect
from ptt.utils.load_restore import join_path

def test_introspection():
    class_path = 'ptt.models.small_cnn.SmallCNN'
    exp = introspect(class_path)()
    assert exp.__class__.__name__ == 'SmallCNN'