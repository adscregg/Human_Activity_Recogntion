import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import json
import os
import numpy as np
import seaborn as sns

def _choose_col(filename):
    if '12' in filename:
        return 'r'
    elif '5' in filename:
        return 'b'
    elif '2' in filename:
        return 'g'
    else:
        return 'k'

def _grid(ax, major_x, minor_x, major_y, minor_y, xlim, ylim):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Change major ticks to show every 20.
    ax.xaxis.set_major_locator(MultipleLocator(major_x))
    ax.yaxis.set_major_locator(MultipleLocator(major_y))

    # Change minor ticks to show every 5. (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(minor_x))
    ax.yaxis.set_minor_locator(AutoMinorLocator(minor_y))

    # Turn grid on for both major and minor ticks and style minor slightly
    # differently.
    ax.grid(which='major', color='#CCCCCC', linestyle='-')
    ax.grid(which='minor', color='#CCCCCC', linestyle='-', alpha = 0.3)

def change_legend(L):
    for i, label in enumerate(L.get_texts()):
        label.set_text(legend_names[i])

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

def plot_curves(summary_dir, files, keys, title, ylabel, xlabel = 'Epochs'):
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
        col = _choose_col(file)
        with open(summary_dir + file) as f:
            summary = json.load(f)
        for key in keys:
            data = summary[key]
            if 'train' in key:
                ax.plot(data, label = summary['name'] + ' train', linestyle = '-', c = col)
            elif 'test' in key:
                ax.plot(data, label = summary['name'] + ' test', linestyle = '--', c = col)


    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    L = ax.legend(prop={'size': 6})
    return fig, ax, L


def plot_confusion_matrix(summary_dir, file, title = None, normalise = True):
    fig, ax = plt.subplots(figsize = (20,20))
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
    ax.set_xticklabels(tick_labels, rotation = 90, fontsize = 25)
    ax.set_yticklabels(tick_labels, rotation = 0, fontsize = 25)

    ax.set_xlabel('Predicted', fontsize = 50, labelpad = 10)
    ax.set_ylabel('True', fontsize = 50, labelpad = 10)

    if title is not None:
        ax.set_title(title, fontsize = 70)

    return fig, ax



if __name__ == '__main__':

    data_dir = './data/NTU_RGB+D/transformed_images'

    summary_dir = './model_summaries/'

    res_dir = './model_summaries/ResNet/'
    shuffle_dir = './model_summaries/ShuffleNet/'
    scatnet_lin_dir = './model_summaries/J2_L8_44/ScatNet_Linear/'
    scatnet_shallow_dir = './model_summaries/J2_L8_44/ScatNet_Shallow/'
    scatnet_deep_dir = './model_summaries/J2_L8_44/ScatNet_Deep/'
    hybrid_dir = './model_summaries/Hybrid/'

    save_fig_dir = './figures/'

    scatterings = ['scattering_2_subs.json','scattering_5_subs.json','scattering_12_subs.json','scattering_all_subs.json']
    resnets = ['resnet_2_subs.json','resnet_5_subs.json','resnet_12_subs.json','resnet_all_subs.json']
    shufflenets = ['shufflenet_2_subs.json','shufflenet_5_subs.json','shufflenet_12_subs.json','shufflenet_all_subs.json']
    hybrids = ['hybrid_2_subs.json','hybrid_5_subs.json','hybrid_12_subs.json','hybrid_all_subs.json']

    resnets_scratch = ['resnet_2_subs_scratch.json','resnet_5_subs_scratch.json','resnet_12_subs_scratch.json','resnet_all_subs_scratch.json']
    shufflenets_scratch = ['shufflenet_2_subs_scratch.json','shufflenet_5_subs_scratch.json','shufflenet_12_subs_scratch.json','shufflenet_all_subs_scratch.json']

    legend_names = ['2 Subjects (train)', '2 Subjects (test)', '5 Subjects (train)', '5 Subjects (test)',
     '12 Subjects (train)', '12 Subjects (test)', '20 Subjects (train)', '20 Subjects (test)']

    train_test_loss_keys = ['train loss history', 'test loss history']
    train_test_acc_keys = ['train accuracy history', 'test accuracy history']





    fig_scat_lin_acc, ax_scat_lin_acc, L_scat_lin_acc = plot_curves(scatnet_lin_dir, scatterings, train_test_acc_keys, 'Linear ScatNets', 'Accuracy')
    change_legend(L_scat_lin_acc)
    _grid(ax_scat_lin_acc, 10, 5, 0.1, 5, (0,100), (0,1))

    fig_scat_lin_loss, ax_scat_lin_loss, L_scat_lin_loss = plot_curves(scatnet_lin_dir, scatterings, train_test_loss_keys, 'Linear ScatNets', 'Loss')
    change_legend(L_scat_lin_loss)
    _grid(ax_scat_lin_loss, 10, 5, 1, 5, (0,100), (0,6))

    fig_scat_shallow_acc, ax_scat_shallow_acc, L_scat_shallow_acc = plot_curves(scatnet_shallow_dir, scatterings, train_test_acc_keys, 'Shallow ScatNets', 'Accuracy')
    change_legend(L_scat_shallow_acc)
    _grid(ax_scat_shallow_acc, 10, 5, 0.1, 5, (0,100), (0,1))

    fig_scat_shallow_loss, ax_scat_shallow_loss, L_scat_shallow_loss = plot_curves(scatnet_shallow_dir, scatterings, train_test_loss_keys, 'Shallow ScatNets', 'Loss')
    change_legend(L_scat_shallow_loss)
    _grid(ax_scat_shallow_loss, 10, 5, 1, 5, (0,100), (0,6))

    fig_scat_deep_acc, ax_scat_deep_acc, L_scat_deep_acc = plot_curves(scatnet_deep_dir, scatterings, train_test_acc_keys, 'Deep ScatNets', 'Accuracy')
    change_legend(L_scat_deep_acc)
    _grid(ax_scat_deep_acc, 10, 5, 0.1, 5, (0,100), (0,1))

    fig_scat_deep_loss, ax_scat_deep_loss, L_scat_deep_loss = plot_curves(scatnet_deep_dir, scatterings, train_test_loss_keys, 'Deep ScatNets', 'Loss')
    change_legend(L_scat_deep_loss)
    _grid(ax_scat_deep_loss, 10, 5, 1, 5, (0,100), (0,6))

    fig_res_acc, ax_res_acc, L_res_acc = plot_curves(res_dir, resnets, train_test_acc_keys, 'ResNets', 'Accuracy')
    change_legend(L_res_acc)
    _grid(ax_res_acc, 10, 5, 0.1, 5, (0,100), (0,1))

    fig_res_loss, ax_res_loss, L_res_loss = plot_curves(res_dir, resnets, train_test_loss_keys, 'ResNets', 'Loss')
    change_legend(L_res_loss)
    _grid(ax_res_loss, 10, 5, 1, 5, (0,100), (0,6))

    fig_shuffle_acc, ax_shuffle_acc, L_shuffle_acc = plot_curves(shuffle_dir, shufflenets, train_test_acc_keys, 'ShuffleNets', 'Accuracy')
    change_legend(L_shuffle_acc)
    _grid(ax_shuffle_acc, 10, 5, 0.1, 5, (0,100), (0,1))

    fig_shuffle_loss, ax_shuffle_loss, L_shuffle_loss = plot_curves(shuffle_dir, shufflenets, train_test_loss_keys, 'ShuffleNets', 'Loss')
    change_legend(L_shuffle_loss)
    _grid(ax_shuffle_loss, 10, 5, 1, 5, (0,100), (0,6))

    fig_hybrid_acc, ax_hybrid_acc, L_hybrid_acc = plot_curves(hybrid_dir, hybrids, train_test_acc_keys, 'Hybrid ScatNet CNN', 'Accuracy')
    change_legend(L_hybrid_acc)
    _grid(ax_hybrid_acc, 10, 5, 0.1, 5, (0,100), (0,1))

    fig_hybrid_loss, ax_hybrid_loss, L_hybrid_loss = plot_curves(hybrid_dir, hybrids, train_test_loss_keys, 'Hybrid ScatNet CNN', 'Loss')
    change_legend(L_hybrid_loss)
    _grid(ax_hybrid_loss, 10, 5, 1, 5, (0,100), (0,6))

    plt.close('all')


    fig_conf_mat_scat2_lin, ax_conf_mat_scat2_lin = plot_confusion_matrix(scatnet_lin_dir, 'scattering_2_subs.json')
    fig_conf_mat_scat5_lin, ax_conf_mat_scat5_lin = plot_confusion_matrix(scatnet_lin_dir, 'scattering_5_subs.json')
    fig_conf_mat_scat12_lin, ax_conf_mat_scat12_lin = plot_confusion_matrix(scatnet_lin_dir, 'scattering_12_subs.json')
    fig_conf_mat_scat20_lin, ax_conf_mat_scat20_lin = plot_confusion_matrix(scatnet_lin_dir, 'scattering_all_subs.json')

    fig_conf_mat_scat2_shallow, ax_conf_mat_scat2_shallow = plot_confusion_matrix(scatnet_shallow_dir, 'scattering_2_subs.json')
    fig_conf_mat_scat5_shallow, ax_conf_mat_scat5_shallow = plot_confusion_matrix(scatnet_shallow_dir, 'scattering_5_subs.json')
    fig_conf_mat_scat12_shallow, ax_conf_mat_scat12_shallow = plot_confusion_matrix(scatnet_shallow_dir, 'scattering_12_subs.json')
    fig_conf_mat_scat20_shallow, ax_conf_mat_scat20_shallow = plot_confusion_matrix(scatnet_shallow_dir, 'scattering_all_subs.json')

    fig_conf_mat_scat2_deep, ax_conf_mat_scat2_deep = plot_confusion_matrix(scatnet_deep_dir, 'scattering_2_subs.json')
    fig_conf_mat_scat5_deep, ax_conf_mat_scat5_deep = plot_confusion_matrix(scatnet_deep_dir, 'scattering_5_subs.json')
    fig_conf_mat_scat12_deep, ax_conf_mat_scat12_deep = plot_confusion_matrix(scatnet_deep_dir, 'scattering_12_subs.json')
    fig_conf_mat_scat20_deep, ax_conf_mat_scat20_deep = plot_confusion_matrix(scatnet_deep_dir, 'scattering_all_subs.json')

    fig_conf_mat_res2, ax_conf_mat_res2 = plot_confusion_matrix(res_dir, 'resnet_2_subs.json')
    fig_conf_mat_res5, ax_conf_mat_res5 = plot_confusion_matrix(res_dir, 'resnet_5_subs.json')
    fig_conf_mat_res12, ax_conf_mat_res12 = plot_confusion_matrix(res_dir, 'resnet_12_subs.json')
    fig_conf_mat_res20, ax_conf_mat_res20 = plot_confusion_matrix(res_dir, 'resnet_all_subs.json')

    fig_conf_mat_shuffle2, ax_conf_mat_shuffle2 = plot_confusion_matrix(shuffle_dir, 'shufflenet_2_subs.json')
    fig_conf_mat_shuffle5, ax_conf_mat_shuffle5 = plot_confusion_matrix(shuffle_dir, 'shufflenet_5_subs.json')
    fig_conf_mat_shuffle12, ax_conf_mat_shuffle12 = plot_confusion_matrix(shuffle_dir, 'shufflenet_12_subs.json')
    fig_conf_mat_shuffle20, ax_conf_mat_shuffle20 = plot_confusion_matrix(shuffle_dir, 'shufflenet_all_subs.json')

    plt.close('all')

    fig_conf_mat_hybrid2, ax_conf_mat_hybrid2 = plot_confusion_matrix(hybrid_dir, 'hybrid_2_subs.json')
    fig_conf_mat_hybrid5, ax_conf_mat_hybrid5 = plot_confusion_matrix(hybrid_dir, 'hybrid_5_subs.json')
    fig_conf_mat_hybrid12, ax_conf_mat_hybrid12 = plot_confusion_matrix(hybrid_dir, 'hybrid_12_subs.json')
    fig_conf_mat_hybrid20, ax_conf_mat_hybrid20 = plot_confusion_matrix(hybrid_dir, 'hybrid_all_subs.json')




    fig_scat_lin_acc.savefig(save_fig_dir + 'curves/J2_L8_44/ScatNet_Linear/scat_acc_lin.pdf', bbox_inches = 'tight')
    fig_scat_lin_loss.savefig(save_fig_dir + 'curves/J2_L8_44/ScatNet_Linear/scat_loss_lin.pdf', bbox_inches = 'tight')

    fig_scat_shallow_acc.savefig(save_fig_dir + 'curves/J2_L8_44/ScatNet_Shallow/scat_acc_shallow.pdf', bbox_inches = 'tight')
    fig_scat_shallow_loss.savefig(save_fig_dir + 'curves/J2_L8_44/ScatNet_Shallow/scat_loss_shallow.pdf', bbox_inches = 'tight')

    fig_scat_deep_acc.savefig(save_fig_dir + 'curves/J2_L8_44/ScatNet_Deep/scat_acc_deep.pdf', bbox_inches = 'tight')
    fig_scat_deep_loss.savefig(save_fig_dir + 'curves/J2_L8_44/ScatNet_Deep/scat_loss_deep.pdf', bbox_inches = 'tight')

    # fig_res_acc.savefig(save_fig_dir + 'curves/ResNet/res_acc.pdf', bbox_inches = 'tight')
    # fig_res_loss.savefig(save_fig_dir + 'curves/ResNet/res_loss.pdf', bbox_inches = 'tight')
    #
    # fig_shuffle_acc.savefig(save_fig_dir + 'curves/ShuffleNet/shuffle_acc.pdf', bbox_inches = 'tight')
    # fig_shuffle_loss.savefig(save_fig_dir + 'curves/ShuffleNet/shuffle_loss.pdf', bbox_inches = 'tight')
    #
    # fig_hybrid_acc.savefig(save_fig_dir + 'curves/Hybrid/hybrid_acc.pdf', bbox_inches = 'tight')
    # fig_hybrid_loss.savefig(save_fig_dir + 'curves/Hybrid/hybrid_loss.pdf', bbox_inches = 'tight')



    fig_conf_mat_scat2_lin.savefig(save_fig_dir + 'confusion_mats/J2_L8_44/ScatNet_Linear/conf_mat_scat_2_lin.pdf', bbox_inches = 'tight')
    fig_conf_mat_scat5_lin.savefig(save_fig_dir + 'confusion_mats/J2_L8_44/ScatNet_Linear/conf_mat_scat_5_lin.pdf', bbox_inches = 'tight')
    fig_conf_mat_scat12_lin.savefig(save_fig_dir + 'confusion_mats/J2_L8_44/ScatNet_Linear/conf_mat_scat_12_lin.pdf', bbox_inches = 'tight')
    fig_conf_mat_scat20_lin.savefig(save_fig_dir + 'confusion_mats/J2_L8_44/ScatNet_Linear/conf_mat_scat_20_lin.pdf', bbox_inches = 'tight')

    fig_conf_mat_scat2_shallow.savefig(save_fig_dir + 'confusion_mats/J2_L8_44/ScatNet_Shallow/conf_mat_scat_2_shallow.pdf', bbox_inches = 'tight')
    fig_conf_mat_scat5_shallow.savefig(save_fig_dir + 'confusion_mats/J2_L8_44/ScatNet_Shallow/conf_mat_scat_5_shallow.pdf', bbox_inches = 'tight')
    fig_conf_mat_scat12_shallow.savefig(save_fig_dir + 'confusion_mats/J2_L8_44/ScatNet_Shallow/conf_mat_scat_12_shallow.pdf', bbox_inches = 'tight')
    fig_conf_mat_scat20_shallow.savefig(save_fig_dir + 'confusion_mats/J2_L8_44/ScatNet_Shallow/conf_mat_scat_20_shallow.pdf', bbox_inches = 'tight')

    fig_conf_mat_scat2_deep.savefig(save_fig_dir + 'confusion_mats/J2_L8_44/ScatNet_Deep/conf_mat_scat_2_deep.pdf', bbox_inches = 'tight')
    fig_conf_mat_scat5_deep.savefig(save_fig_dir + 'confusion_mats/J2_L8_44/ScatNet_Deep/conf_mat_scat_5_deep.pdf', bbox_inches = 'tight')
    fig_conf_mat_scat12_deep.savefig(save_fig_dir + 'confusion_mats/J2_L8_44/ScatNet_Deep/conf_mat_scat_12_deep.pdf', bbox_inches = 'tight')
    fig_conf_mat_scat20_deep.savefig(save_fig_dir + 'confusion_mats/J2_L8_44/ScatNet_Deep/conf_mat_scat_20_deep.pdf', bbox_inches = 'tight')

    # fig_conf_mat_res2.savefig(save_fig_dir + 'confusion_mats/ResNet/conf_mat_res_2.pdf', bbox_inches = 'tight')
    # fig_conf_mat_res5.savefig(save_fig_dir + 'confusion_mats/ResNet/conf_mat_res_5.pdf', bbox_inches = 'tight')
    # fig_conf_mat_res12.savefig(save_fig_dir + 'confusion_mats/ResNet/conf_mat_res_12.pdf', bbox_inches = 'tight')
    # fig_conf_mat_res20.savefig(save_fig_dir + 'confusion_mats/ResNet/conf_mat_res_20.pdf', bbox_inches = 'tight')
    #
    # fig_conf_mat_shuffle2.savefig(save_fig_dir + 'confusion_mats/ShuffleNet/conf_mat_shuffle_2.pdf', bbox_inches = 'tight')
    # fig_conf_mat_shuffle5.savefig(save_fig_dir + 'confusion_mats/ShuffleNet/conf_mat_shuffle_5.pdf', bbox_inches = 'tight')
    # fig_conf_mat_shuffle12.savefig(save_fig_dir + 'confusion_mats/ShuffleNet/conf_mat_shuffle_12.pdf', bbox_inches = 'tight')
    # fig_conf_mat_shuffle20.savefig(save_fig_dir + 'confusion_mats/ShuffleNet/conf_mat_shuffle_20.pdf', bbox_inches = 'tight')
    #
    # fig_conf_mat_hybrid2.savefig(save_fig_dir + 'confusion_mats/Hybrid/conf_mat_hybrid_2.pdf', bbox_inches = 'tight')
    # fig_conf_mat_shuffle5.savefig(save_fig_dir + 'confusion_mats/Hybrid/conf_mat_hybrid_5.pdf', bbox_inches = 'tight')
    # fig_conf_mat_shuffle12.savefig(save_fig_dir + 'confusion_mats/Hybrid/conf_mat_hybrid_12.pdf', bbox_inches = 'tight')
    # fig_conf_mat_shuffle20.savefig(save_fig_dir + 'confusion_mats/Hybrid/conf_mat_hybrid_20.pdf', bbox_inches = 'tight')
