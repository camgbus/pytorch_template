# ------------------------------------------------------------------------------
# Classes for creating new classification datasets.
# ------------------------------------------------------------------------------

import os

class ClassificationDataset:
    """A dataset stores instances."""
    def __init__(self, name, classes, instances, hold_out_ixs=[]):
        self.name = name
        self.classes = classes
        self.instances = instances
        self.hold_out_ixs = hold_out_ixs

    def get_class_dist(self):
        class_dist_primary = {class_name: 0 for class_name in self.classes}
        class_dist_holdout = {class_name: 0 for class_name in self.classes}
        for ex_ix, ex in enumerate(self.instances):
            if ex_ix in self.hold_out_ixs:
                class_dist_holdout[self.classes[ex.y]] += 1
            else:
                class_dist_primary[self.classes[ex.y]] += 1
        return class_dist_primary, class_dist_holdout

class ClassificationInstance:
    """To define a dataset for a specific problem, inherit 
    from this Instance class.
    """
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

class ClassificationPathInstance(ClassificationInstance):
    """Instance class where x is a path and y is an integer label corr. to
    an index of the dataset 'classes' field.
    """
    def __init__(self, name, x_path, y):
        assert isinstance(y, int)
        super().__init__(name=name, x=x_path, y=y)

class SplitClassImageDataset(ClassificationDataset):
    """Dataset with the structure root/split/class/filename,
    where 'split' is test for the hold-out test dataset and train for the rest.
    The instances are of the type 'PathInstance'.
    """
    def __init__(self, name, root_path):
        classes = []
        instances = []
        hold_out_start = None
        for split in ['train', 'test']:
            if split == 'test':
                hold_out_start = len(instances)
            split_path = os.path.join(root_path, split)
            for class_name in os.listdir(split_path):
                if class_name not in classes:
                    classes.append(class_name)
                class_path = os.path.join(split_path, class_name)
                for img_name in os.listdir(class_path):
                    instance = ClassificationPathInstance(name=img_name, x_path=os.path.join(class_path, img_name), y=classes.index(class_name))
                    instances.append(instance)
        super().__init__(name=name, classes=tuple(classes), instances=instances, hold_out_ixs=list(range(hold_out_start, len(instances))))

class CIFAR10(SplitClassImageDataset):
    def __init__(self, root_path):
        super().__init__(name='cifar10', root_path=root_path)
