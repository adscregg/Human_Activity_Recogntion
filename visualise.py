import matplotlib.pyplot as plt
import json
import os
import numpy as np
import seaborn as sns


def visualise(data_dir, subjects = None, save_name = None):
    if subjects is None:
        subjects = ["P" + f"{a:03}" for a in range(1,41)]
    files = os.listdir(data_dir)
    files_to_vis = filter(lambda x: x[8:12] in subjects, files)
    dict_counter = dict()
    for file in files_to_vis:
        if int(file[17:20]) in dict_counter:
            dict_counter[int(file[17:20])] += 1
        else:
            dict_counter[int(file[17:20])] = 1

    plt.bar(list(dict_counter.keys()), dict_counter.values())
    # plt.xticks([])
    plt.show()

def plot_curves(summary_dir, files, key, title, xlabel, ylabel):
    """
    Plot curves associated with the training of models, can plot, the learning rate, loss and accuracy curves for any model

    Parameters
    ----------
    summary_dir: str
        File path to the directory where the model summary .json files are stored

    files: list or iterable
        Summary files that contain the values to be plotted

    key: str
        The metric that will be plotted on the graph

    title: str
        Title of the figure

    xlabel: str
        Label of the x-axis

    ylabel: str
        Label of the y-axis

    Returns
    -------
    fig, ax:
        matplotlib figure object and matplotlib Axis object
    """
    fig, ax = plt.subplots()
    for file in files:
        with open(summary_dir + file) as f:
            summary = json.load(f)
        data = summary[key]
        ax.plot(data, label = summary['name'])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig, ax


def plot_confusion_matrix(summary_dir, file, normalise = True):
    fig, ax = plt.subplots()
    with open(summary_dir + file) as f:
        summary = json.load(f)
    mat = np.array(summary['confusion matrix'])
    if normalise:
        mat /= mat.sum(axis = 1, keepdims=True)
    nb_classes = mat.shape[0]
    sns.heatmap(mat, cmap = 'Greys', vmax = 1, vmin = 0, ax = ax, cbar = False)

    ticks = np.arange(0.5, nb_classes, 2).tolist()
    tick_labels = ['A' + f"{int(np.ceil(i)):03}" for i in ticks]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels, rotation = 90, fontsize = 12)
    ax.set_yticklabels(tick_labels, rotation = 0, fontsize = 12)

    ax.set_xlabel('Predicted', fontsize = 15, labelpad = 10)
    ax.set_ylabel('True', fontsize = 15, labelpad = 10)

    return fig, ax



if __name__ == '__main__':

    data_dir = './data/NTU_RGB+D/transformed_images'
    summary_dir = './model_summaries/'

    save_fig_dir = './figures/'

    fig2, ax2 = plot_confusion_matrix(summary_dir, 'shallow_scattering_5_subs.json')
    plt.show()

    # fig.savefig(save_fig_dir + 'test1.pdf')
    # fig1.savefig(save_fig_dir + 'happy1.pdf')
