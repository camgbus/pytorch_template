class Dataset:
    """A dataset stores instances."""
    def __init__(self, name, classes, instances, hold_out_ixs=[]):
        self.name = name
        self.classes = classes
        self.instances = instances
        self.hold_out_ixs = hold_out_ixs

class Instance:
    """To define a dataset for a specific problem, inherit 
    from this Instance class.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

class PathInstance(Instance):
    def __init__(self, x_path, y):
        super().__init__(x=x_path, y=y)