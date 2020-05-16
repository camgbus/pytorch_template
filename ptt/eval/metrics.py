# ------------------------------------------------------------------------------
# Collection of metrics.
# ------------------------------------------------------------------------------

import torch
import numpy as np

def accuracy(outputs, targets):
    _, pred = torch.max(outputs.data, 1)
    total = outputs.size(0)
    correct = (pred == targets).sum().item()
    return correct/total

def dice(outputs, target, smooth = 1.):
    outputs = outputs.contiguous()
    target = target.contiguous()    
    intersection = (outputs * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (outputs.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def np_dice(mask_a, mask_b, smooth = 1.):
    assert mask_a.shape == mask_b.shape
    while len(mask_a.shape) < 4:
        mask_a = np.expand_dims(mask_a, axis=0)
        mask_b = np.expand_dims(mask_b, axis=0)
    a = torch.from_numpy(mask_a).float()
    b = torch.from_numpy(mask_b).float()
    return dice(a, b, smooth=smooth)
