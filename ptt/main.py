import torch
assert torch.cuda.is_available()
import torch.nn as nn
import torch.optim as optim

from ptt.data.dataset_classification import CIFAR10
from ptt.models.small_cnn import SmallCNN
from ptt.data.pytorch_dataset import ImgClassificationDataset
from ptt.agents.classification_agent import ClassificationAgent
from ptt.eval.result import Result
from ptt.visualization.plot_results import plot_results




if __name__ == '__main__':
    config = {'batch_size':128, 'lr':1e-3, 'momentum':0.9, 'device':'cuda:0', 'nr_epochs': 5, 'tracking_interval': 1}

    # Fetch data, transform to PyTorch format and build dataloaders
    data = CIFAR10()
    datasets = {'train': ImgClassificationDataset(data, ix_lst=None, resize=None, norm=data.x_norm),
        'test': ImgClassificationDataset(data, ix_lst=data.hold_out_ixs, resize=None, norm=data.x_norm)}
    dataloaders = dict()
    for split, ds in datasets.items():
        shuffle = not(split == 'test')
        dataloaders[split] = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=shuffle)
    print('Got dataset')

    # Get model
    model = SmallCNN(input_shape=data.input_shape, output_shape=data.output_shape)
    model.to(config['device'])

    # Devine criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])

    # Train model
    results = Result(name='training_trajectory')
    agent = ClassificationAgent(config=config, base_criterion=criterion, verbose=True)
    agent.train(results, model, optimizer, trainloader=dataloaders['train'], dataloaders=dataloaders)

    # Visualize results
    save_path = os.path.join('test', 'test_obj')
    plot_results(res, measures=['accuracy'], save_path=save_path, title='CIFAR10 example accuracy', ending='.png')
    plot_results(res, measures=['loss'], save_path=save_path, ylog=True, title='CIFAR10 example loss', ending='.png')