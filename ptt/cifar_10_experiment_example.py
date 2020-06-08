# ------------------------------------------------------------------------------
# An example equivalent to cifar_10_experiment_notebook bot using the
# experiment formatting and cross-validation, not using hold-out data.
# ------------------------------------------------------------------------------

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ptt.argument_parsing import parse_args_as_dict
from ptt.experiment.experiment import Experiment
from ptt.data.dataset_classification import CIFAR10
from ptt.models.small_cnn import SmallCNN
from ptt.data.pytorch_dataset import ImgClassificationDataset
from ptt.agents.classification_agent import ClassificationAgent
from ptt.eval.result import Result

def get_data(config):
    return CIFAR10()

def run(exp, exp_run, data):
    config = exp.config

    # Transform data to PyTorch format and build dataloaders
    datasets = dict()
    for data_name, data_ixs in exp.splits[exp_run.run_ix].items():
        if len(data_ixs) > 0:
            datasets[data_name] = ImgClassificationDataset(data, ix_lst=data_ixs, resize=None, norm=data.x_norm)
    dataloaders = dict()
    for split, ds in datasets.items():
        shuffle = not(split == 'test')
        dataloaders[split] = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=shuffle)

    # Get model
    model = SmallCNN(input_shape=data.input_shape, output_shape=data.output_shape)
    model.to(config['device'])

    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])

    # Train model
    results = Result(name='training_trajectory')
    agent = ClassificationAgent(config=config, base_criterion=criterion, verbose=True)
    agent.train(results, model, optimizer, trainloader=dataloaders['train'], dataloaders=dataloaders)

    return results


if __name__ == '__main__':
    # Set random seeds
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Use console arguments
    config = parse_args_as_dict(sys.argv[1:])
    exp = Experiment(config=config, name=config['experiment_name'], 
        notes='An example of training a network for CIFAR10 classification', reload_exp=True)
    # Get data
    data = get_data(config)
    # Divide indexes into splits/folds
    exp.set_data_splits(data)
    # Iterate over repetitions and run
    for ix in range(config['nr_runs']):
        print('Running repetition {} of {}'.format(ix+1, config['nr_runs']))
        exp_run = exp.get_run(run_ix=ix)
        try:
            results = run(exp=exp, exp_run=exp_run, data=data)
            exp_run.finish(results=results)
        except Exception as e: 
            exp_run.finish(exception=e)