# ------------------------------------------------------------------------------
# Functions to store and restore PyTorch objects.
# ------------------------------------------------------------------------------

import torch
import os

def save_model_state(model, name, path):
    """Saves a pytorch model."""
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, name)
    torch.save(model.state_dict(), full_path)

def load_model_state(model, name, path):
    """Restores a pytorch model."""
    if os.path.exists(path):
        full_path = os.path.join(path, name)
        if os.path.isfile(full_path):
            model.load_state_dict(torch.load(full_path))
            return True
    return False

def save_optimizer_state(optimizer, name, path):
    """
    Saves a pytorch optimizer state.

    This makes sure that, for instance, if learning rate decay is used the same
    state is restored which was left of at this point in time.
    """
    full_path = os.path.join(path, name)
    torch.save(optimizer.state_dict(), full_path)

def load_optimizer_state(optimizer, name, path):
    """Restores a pytorch optimizer state."""
    if os.path.exists(path):
        full_path = os.path.join(path, name)
        if os.path.isfile(full_path):
            optimizer.load_state_dict(torch.load(full_path))
            return True
    return False

def save_scheduler_state(scheduler, name, path):
    full_path = os.path.join(path, name)
    torch.save(scheduler.state_dict(), full_path)

def load_scheduler_state(scheduler, name, path):
    if os.path.exists(path):
        full_path = os.path.join(path, name)
        if os.path.isfile(full_path):
            scheduler.load_state_dict(torch.load(full_path))
            return True
    return False