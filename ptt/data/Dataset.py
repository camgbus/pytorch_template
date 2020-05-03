class Dataset:
    """A dataset stores instances."""
    def __init__(self, name, instances):
        self.name = name
        self.instances = instances

class Instance:
    """To define a dataset for a specific problem, inherit 
    from this Instance class.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

class ImageGeneratorInstance(Instance):
    """ x are parameteres, and y is a path to an image """
    def __init__(self, x, y):
        assert isinstance(x, tuple) or isinstance(x, list)
        assert all(isinstance(x_i, float) for x_i in x)
        assert isinstance(y, str)
        super().__init__(x, y)