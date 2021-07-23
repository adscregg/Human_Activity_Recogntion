#!/usr/bin/env python
# coding=utf-8
# https://github.com/shahroudy/NTURGB-D/blob/master/Python/txt2npy.py
'''
transform the skeleton data in NTU RGB+D dataset into the numpy arrays for a more efficient data loading
'''

import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import sys

user_name = 'user'
save_npy_path = './data/NTU_RGB+D/transformed_images/'
load_txt_path = './data/NTU_RGB+D/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons/'
missing_file_path = './missing_skeletons.txt'
step_ranges = list(range(0,100)) # just parse range, for the purpose of paralle running.


toolbar_width = 50
def _print_toolbar(rate, annotation=''):
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def _end_toolbar():
    sys.stdout.write('\n')

def _load_missing_file(path):
    missing_files = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            if line not in missing_files:
                missing_files[line] = True
    return missing_files

def _read_skeleton(file_path, save_skelxyz=True, save_rgbxy=False, save_depthxy=False):
    f = open(file_path, 'r')
    datas = f.readlines()
    f.close()
    max_body = 4
    njoints = 25

    # specify the maximum number of the body shown in the sequence, according to the certain sequence, need to pune the
    # abundant bodys.
    # read all lines into the pool to speed up, less io operation.
    nframe = int(datas[0][:-1])
    bodymat = dict()
    bodymat['file_name'] = file_path[-29:-9]
    nbody = int(datas[1][:-1])
    bodymat['nbodys'] = []
    bodymat['njoints'] = njoints
    for body in range(max_body):
        if save_skelxyz:
            bodymat['skel_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 3))
        if save_rgbxy:
            bodymat['rgb_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
        if save_depthxy:
            bodymat['depth_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
    # above prepare the data holder
    cursor = 0
    for frame in range(nframe):
        cursor += 1
        bodycount = int(datas[cursor][:-1])
        if bodycount == 0:
            continue
        # skip the empty frame
        bodymat['nbodys'].append(bodycount)
        for body in range(bodycount):
            cursor += 1
            skel_body = 'skel_body{}'.format(body)
            rgb_body = 'rgb_body{}'.format(body)
            depth_body = 'depth_body{}'.format(body)

            bodyinfo = datas[cursor][:-1].split(' ')
            cursor += 1

            njoints = int(datas[cursor][:-1])
            for joint in range(njoints):
                cursor += 1
                jointinfo = datas[cursor][:-1].split(' ')
                jointinfo = np.array(list(map(float, jointinfo)))
                if save_skelxyz:
                    bodymat[skel_body][frame,joint] = jointinfo[:3]
                if save_depthxy:
                    bodymat[depth_body][frame,joint] = jointinfo[3:5]
                if save_rgbxy:
                    bodymat[rgb_body][frame,joint] = jointinfo[5:7]
    # prune the abundant bodys
    for each in range(max_body):
        if each >= max(bodymat['nbodys']):
            if save_skelxyz:
                del bodymat['skel_body{}'.format(each)]
            if save_rgbxy:
                del bodymat['rgb_body{}'.format(each)]
            if save_depthxy:
                del bodymat['depth_body{}'.format(each)]
    return bodymat

def _translation_scale_invariant(arr):
    """
    Parameters
    -----------
    arr: np.array, shape = (N, M, C), N = rows, M = columns, C = channels
        The array to be made scale and translation invariant, C is usually 3 for an RGB image or 1 for a black and white image

    Returns
    ---------
    np.array, shape = (N, M, C)
        array which is invariant to additive shift or multiplicative scaling, np.uint8 values between 0 and 255, valid images have 1 or 3 channels
    """
    channels = arr.shape[-1] # The number of channels in the array, mostly 3 (RGB) or 1 (binary)
    mins = np.empty((1,1,3)) # create a placeholder array for shifting the coordinate values
    diffs = list() # empty list which will hold the variation between max and min of each channel
    for c in range(channels):
        channel = arr[:,:,c] # the c th channel
        c_min = channel.min() # min value in c th channel
        c_max = channel.max() # max value in c th channel
        diffs.append(c_max - c_min)
        mins[:,:,c] = c_min # store the min value for the c th channel, to remove translation factor of the image
    max_diffs = np.max(diffs) # largest variation to remove the scale of the image
    transformed = np.floor(((arr - mins)/max_diffs) * 255).astype(np.uint8) # apply tranformation
    return transformed

def _create_action_array(d):
    """
    Parameters
    -----------
    d: dictionary, contains information about the skeleton sequence in the NTU RGB+D dataset

    Returns
    ---------
    np.array, shape = (N, M, C)
        array where the RGB channel represent the XYZ coordinates of the joints, each row corresponds to a single joint in all frames,
        each column is all joints in a single frame
    """

    order = [4,5,6,7,21,22,8,9,10,11,23,24,0,1,20,2,3,16,17,18,19,12,13,14,15] # the order of the joints in the image, reordered from default to cature local spatial characteristics

    action_array = None # default value for action image

    nbodys = 1 if int(d['file_name'][17:21]) < 50 else 2 # the number of bodys there should be in a sequence according to the action class
    actual_bodys = min(np.max(d['nbodys']), nbodys) # the number of bodys to select for creating an action array

    vars = list() # initialize empty list, will hold variance for each channel of potential bodys
    for i in range(np.max(d['nbodys'])): # loop over all bodys present in a given sequence
        vars.append(np.sum([np.var(d[f'skel_body{i}'][:,:,c]) for c in range(3)])) # sum the variances for all channels together, heuristic for how dynamic a body is during a sequence

    select = np.argsort(vars)[-actual_bodys:] # select the largest variances and use them for the ceation of action array




    for i in select: # loop over the most number of bodies in that are present in the video sequence
        temp = d[f'skel_body{i}'][:,order,:] # get the ith skeleton and reorder the joints in the image
        if action_array is None: # check if the default value has been overwritten
            action_array = temp
        else:
            action_array = np.concatenate((action_array, temp), axis = 1) # extend the image by adding extra colums representing multipl bodies

    action_array = action_array.transpose(1,0,2) # swap the rows and columns so the rows correspond to a single joint and the columns correspond to all joints

    return action_array


if __name__ == '__main__':
    missing_files = _load_missing_file(missing_file_path)
    datalist = os.listdir(load_txt_path)
    alread_exist = os.listdir(save_npy_path)
    alread_exist_dict = dict(zip(alread_exist, len(alread_exist) * [True]))


    for each in tqdm(datalist):
        # _print_toolbar(ind * 1.0 / len(datalist),
        #                '({:>5}/{:<5})'.format(
        #                    ind + 1, len(datalist)
        #                ))
        S = int(each[1:4])
        if S not in step_ranges:
            continue
        if each[:20]+'.jpg' in alread_exist_dict:
            # print('file already existed!')
            continue
        if each[:20] in missing_files:
            print(f'{each[:20]} file missing')
            continue
        loadname = load_txt_path+each
        mat = _read_skeleton(loadname)
        arr = np.array(mat).item(0)

        array = _create_action_array(arr)
        image_array = _translation_scale_invariant(array)
        image = Image.fromarray(image_array)


        save_path = save_npy_path+'{}.jpg'.format(each[:-9])
        image.save(save_path)
        # if ind == 5:
        #     break
        # raise ValueError()
    # _end_toolbar()
