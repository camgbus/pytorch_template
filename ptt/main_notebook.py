ON_JUPYTER = False
try:
    from IPython import get_ipython
    # Autoreload imported modules for Jupyter
    get_ipython().magic('load_ext autoreload') 
    get_ipython().magic('autoreload 2')
    ON_JUPYTER = True
except AttributeError:
    pass

from ptt.data.dataset_classification import CIFAR10
from ptt.models.small_cnn import SmallCNN
from ptt.data.pytorch_dataset import ImgClassificationDataset


# An example where the training takes place with all training data, and the
# model is tested on the hold-out data set.
batch_size=128


# Fetch data
ds = CIFAR10()
pytorch_data = {'train': ImgClassificationDataset(ds, ix_lst=None, resize=None, norm=ds.x_norm),
    'test': ImgClassificationDataset(ds, ix_lst=ds.hold_out_ixs, resize=None, norm=ds.x_norm)}
dataloaders = dict()
for split, pytorch_ds in pytorch_data.items():
    shuffle = not (split == 'test')
    print(shuffle)
    dataloaders[split] = torch.utils.data.DataLoader(pytorch_data[split], batch_size=batch_size, shuffle=shuffle)
    print(len(dataloaders[split]))



model = SmallCNN(input_shape=ds.input_shape, output_shape=ds.output_shape)