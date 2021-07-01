import matplotlib.pyplot as plt
import os


def visualise(data_dir, subjects = None, save_name = None):
    if subjects is None:
        subjects = ["P" + f"{a:03}" for a in range(1,6)]
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

if __name__ == '__main__':

    data_dir = './data/NTU_RGB+D/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons'

    visualise(data_dir)
