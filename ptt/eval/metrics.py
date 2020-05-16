# ------------------------------------------------------------------------------
# Collection of metrics.
# ------------------------------------------------------------------------------

import torch
import numpy as np

def dice(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
     
    return loss.mean()

def np_dice(mask_a, mask_b, smooth = 1.):
    assert mask_a.shape == mask_b.shape
    while len(mask_a.shape) < 4:
        mask_a = np.expand_dims(mask_a, axis=0)
        mask_b = np.expand_dims(mask_b, axis=0)
    a = torch.from_numpy(mask_a).float()
    b = torch.from_numpy(mask_b).float()
    return dice(a, b, smooth=smooth)