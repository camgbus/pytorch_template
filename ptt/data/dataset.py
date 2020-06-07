# ------------------------------------------------------------------------------
# Costum datasets are descendants of the Dataset class. Dataset objects are used
# to create index lists of subsets such as cross-validation folds. 
# ------------------------------------------------------------------------------

class Dataset:
    """A dataset stores instances."""
    def __init__(self, name, instances, classes=('0'), hold_out_ixs=[], 
        input_shape=(1, 32, 32), output_shape=(1, 32, 32), x_norm=None):
        """
        :param name: name of a dataset
        :param instances: list of Instance objects
        :param hold_out_ixs: instances which will not be evaluated until the end 
        :param input_shape: mean input shape
        :param output_shape: same as input_shape for segmentation or autoencoding, 
            numpber of classes for classification
        :param x_norm: normalization values for the input
        """
        self.name = name
        self.instances = instances
        self.size = len(instances)
        self.classes = classes
        self.hold_out_ixs = hold_out_ixs
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.x_norm = x_norm

    def get_class_dist(self, ixs=None):
        """Get class (category) distribution
        :param ixs: if not None, distribution for only these indexes. Otherwise
        distribution for all indexes not part of the hold-out dataset.
        """
        if ixs is None:
            ixs = [ix for ix in range(self.size) if ix not in self.hold_out_ixs]
        class_dist = {class_ix: 0 for class_ix in self.classes}
        for ex_ix, ex in enumerate(self.instances):
            if ex_ix in ixs:
                class_dist[self.classes[ex.class_ix]] += 1
        return class_dist

    def get_class_instance_ixs(self, class_name, exclude_ixs):
        return [ix for ix, ex in enumerate(self.instances) if 
            ex.class_ix==self.classes.index(class_name) and ix not in exclude_ixs]


class Instance:
    def __init__(self, x, y, name=None, class_ix=0, group_id=None):
        """
        :param x: NN input
        :param y: NN output
        :param name: instance name (e.g. file name) for case-wise evaluation
        :param class_ix: during splitting of the dataset, the resulting subsets are 
        stratesfied according to this value (i.e. there are about as many examples 
        of each class on each fold). For classification, class_ix==y.
        :param group_id: instances with same 'group id' should always 
        remain on the same dataset split. A group id could be, for instance, a 
        patient identifier.
        """
        self.x = x
        self.y = y
        self.name = name
        self.class_ix = class_ix
        self.group_id = group_id