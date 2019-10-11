
#%%
ON_JUPYTER = False
try:
    from IPython import get_ipython
    # Autoreload imported modules for Jupyter
    get_ipython().magic('load_ext autoreload') 
    get_ipython().magic('autoreload 2')
    ON_JUPYTER = True
except AttributeError:
    pass

import numpy as np
import os
import sys
import torch
assert torch.cuda.is_available()

from ptt.utils.argument_parsing import parse_args
from ptt.eval.Experiment import Experiment
from ptt.models.get_model import get_model


def run(experiment, args):
    # Set torch device
    torch.cuda.set_device(args.device)
    # Fetch data

    
    # Build model
    class_ref = get_model(name)
    config = {'pretrained': args.pretrained, 
        'freeze_params': args.freeze_params, 
        'nr_outputs': 10}
    model = class_ref(config)









#%%


if __name__ == '__main__':
    # Set random seeds
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not ON_JUPYTER:        
        # Use console arguments
        args = parse_args(sys.argv[1:])
        experiment = Experiment(args)
        try:
            run(experiment, args)
        except:
            experiment.finish(exception=e)


#%%
#args = {}