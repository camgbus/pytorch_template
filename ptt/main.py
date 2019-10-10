
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
from ptt.utils.Experiment import Experiment

#%%
device = 1

from pt.other import deff

torch.cuda.set_device(device)
deff()


#%%
print(torch.cuda.device_count())
#%%
cuda0 = torch.cuda.set_device(5)
print(torch.cuda.current_device())  # output: 0


#%%
print(torch.cuda.device_count())

#%%
if __name__ == '__main__':
    # Set random seeds
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not ON_JUPYTER:
        # Use console arguments
        if len(sys.argv[1:]) > 0:
            args = parse_args(sys.argv[1:])
            try:
                run(Experiment(args))
            except:
                experiment.finish(exception=e)
