import os
from ptt.data.Dataset import Dataset, PathInstance

def SplitClassImage(name, dataset_path):
    """Dataset with the structure root/split/class/filename,
    where split is test for the hold-out test dataset and train for the rest.
    """




def CIFAR10(dataset_path):

    os.path.join(dataset_path, train)




    classes
    instances
    hold_out_ixs



    return Dataset(name='CIFAR10', 
        classes=classes, 
        instances=instances, 
        hold_out_ixs=hold_out_ixs)



