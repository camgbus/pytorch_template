################################################################################
# Plots results
################################################################################

import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results(df, measures, save_path=None, title=None, ending='.png'):
    # Filter out measures that are not to be shown
    df = df.loc[df['measure_name'].isin(measures)]
    # Start a new figure so that different plots do not overlap
    plt.figure()
    sns.set(rc={'figure.figsize':(10,5)})
    # Plot
    ax = sns.lineplot(x='epoch', 
        y='measure_value', 
        hue='measure_name', 
        style='measure_name', 
        alpha=0.7, 
        data=df)
    # Legend to the side
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1, 1), loc=2)
    # Set title
    if title:
        ax.set_title(title)
    # Save image
    if save_path:
        path, file_name = os.path.split(save_path)
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = file_name.split('.')[0]+ending
        plt.savefig(os.path.join(path, file_name), facecolor='w', 
            bbox_inches="tight", dpi = 300)
