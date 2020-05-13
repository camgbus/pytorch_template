from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PytorchDataset(Dataset):
    def __init__(self, dataset, ix_lst=None, transform=None, norm=None):
        if ix_lst is None:
            ix_lst = [ix for ix in range(len(dataset.instances)) if ix not in dataset.hold_out_ixs]
        self.instances = [ex for ix, ex in enumerate(dataset.instances) if ix in ix_lst]
        self.transform = transform
        if transform is None:
            self.transform = transforms.ToTensor()
        if norm:
            self.transform = transforms.Compose([self.transform, 
                transforms.Normalize(mean=norm['mean'], std=norm['std'])])

    def __len__(self):
        return len(self.instances)

class ImgClassificationDataset(PytorchDataset):
    """Classification dataset that loads images from file paths.
    """
    def __init__(self, dataset, ix_lst=None, transform=None, norm=None):
        super().__init__(dataset=dataset, ix_lst=ix_lst, transform=transform, norm=norm)
        self.x_paths = [ex.x for ex in self.instances]
        self.y = torch.LongTensor([float(ex.y) for ex in self.instances])

    def __getitem__(self, idx):
        image = Image.open(self.x_paths[idx])
        image = self.transform(image)
        return image, self.y[idx]







