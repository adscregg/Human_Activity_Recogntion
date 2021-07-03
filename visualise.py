import matplotlib.pyplot as plt
import json
import os


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

if __name__ == '__main__':

    data_dir = './data/NTU_RGB+D/transformed_images'
    summary_dir = './model_summaries/'

    fig, ax = plot_curves(summary_dir, ['test.json'], 'train accuracy history','TEST', 'EPOCHS', 'ACCURACY')
    fig1, ax1 = plot_curves(summary_dir, ['test.json', 'test1.json'], 'train accuracy history','HAPPY', 'EPOCHS', 'ACCURACY')
    plt.show()
