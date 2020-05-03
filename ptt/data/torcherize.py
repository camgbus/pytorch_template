import os
import numpy as np
from PIL import Image
import csv
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ClassificationPathsDS(Dataset):
    """Classification dataset that loads images from file paths.
    """
    def __init__(self, dataset, ix_lst, transform=None):
        self.instances = [ex for ix, ex in enumerate(dataset.instances) if ix in ix_lst]
        self.x_paths = [ex.x for ex in self.instances]
        self.y = [ex.y for ex in self.instances]
        self.labels = torch.LongTensor([float(targets[ix][0]) for ix in index_list])    
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        image = Image.open(self.x_paths[idx])
        image = self.transform(image)
        return image, self.labels[idx]







