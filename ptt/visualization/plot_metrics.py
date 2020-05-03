import os
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
from src.utils.seaborn.legend_utils import format_legend

def plot_learning_trajectory(result, metrics=None, save_path=None, ylog=False, figsize=(10,7)):
    """
    :param result: an instance of src.eval.results.PartialResult
    """
    plt.figure()
    sns.set(rc={'figure.figsize':figsize})
    pd = result.to_pandas(metrics=metrics)
    ax = sns.lineplot(data=pd, x='Epoch', y='Value', style='Split', hue='Metric')
    if ylog:
        ax.set_yscale('log')
    format_legend(ax, ['Metric', 'Split'])
    if save_path:
        plt.savefig(os.path.join(save_path, result.name+'_'+metrics[0]+'.png'), 
            facecolor='w', bbox_inches="tight", dpi = 300)

def compare_experiment_last(result_dict, metric_x, metric_y, save_path=None, save_name=None, ylog=False, figsize=(10,7), splits=['train', 'val', 'test']):
    """
    :param result_dict: an experiment_name: result obj dictionary
    Compare the last values for several experiments
    """
    max_shared_epoch = min([max(result.results.keys()) for result in result_dict.values()])
    print('Max shared epoch: {}'.format(max_shared_epoch))

    # Build data frame
    data = []
    for split in splits:
        data += [
            [value.get_epoch_metric(epoch=max_shared_epoch, metric=metric_x, split=split),
            value.get_epoch_metric(epoch=max_shared_epoch, metric=metric_y, split=split), split, key] 
            for key, value in result_dict.items()]
    df = pd.DataFrame(data, columns = [metric_x, metric_y, 'Split', 'Experiment'])
    # Plot
    plt.figure()
    sns.set(rc={'figure.figsize':figsize})
    ax = sns.scatterplot(data=df, x=metric_x, y=metric_y, style='Split', hue='Experiment')
    if ylog:
        ax.set_yscale('log')
    format_legend(ax, ['Experiment', 'Split'])
    if save_path:
        plt.savefig(os.path.join(save_path, save_name+'.png'), 
            facecolor='w', bbox_inches="tight", dpi = 300)