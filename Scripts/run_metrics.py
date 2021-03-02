import numpy as np
from numpy.fft import fft2, fftshift
import pandas as pd
import imageio
import os
import sys
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns
import cv2
import csv
import math
import argparse
from functools import partial
from tensorflow.keras.applications.vgg16 import VGG16

from pspec import get_pspec, rgb_2_darter
from config import *
from visual_metrics import *

###############################################################
### File used to calculate metrics and save them in a csv file
###############################################################


def load_info_from_filepath(file_path):
    '''
    Given an absolute file path
    returns the informations about the file in the form of a dict
    '''
    long_head, filename = os.path.split(file_path)
    head, directory = os.path.split(long_head)
    image = imageio.imread(img_path)
    dict_info={COL_FILENAME:filename, COL_DIRECTORY:directory,
        COL_IMG_HEIGHT:image.shape[0], COL_IMG_WIDTH:image.shape[1]}

    #if the directory contains stylized fishes
    if [p for p in DIR_STYLIZED_FISHES if p == directory]:
        crossing, color_ctrl, _, tvloss, *_  = filename.split("_")
        middle, fish_n = crossing.rsplit("x", maxsplit=1)
        dict_info[COL_FISH_NUMBER] = fish_n[4:] #4 to remove FISH
        dict_info[COL_HABITAT] = middle
        dict_info[COL_COLOR_CONTROL] = color_ctrl[12:]
        dict_info[COL_TV_LOSS] = tvloss[6:]
        dict_info[COL_LAYERS] = directory
        dict_info[COL_TYPE] = "fish stylized"
    #if the directory contains original fishes
    elif [p for p in DIR_ORIGINAL_FISHES if p == directory]:
        fish_n, _, original_size, specie, *_, end = filename.split("_")
        dict_info[COL_SPECIES] = specie
        dict_info[COL_FISH_NUMBER] = end.split('.')[0][1:]
        dict_info[COL_FISH_SEX] = end[0]
        dict_info[COL_TYPE] = "crop fish"
    #if the directory contains original habitats
    elif [p for p in DIR_ORIGINAL_HABITATS if p == directory]:
        middle, *_ = filename.split('_')
        dict_info[COL_HABITAT] = middle
        dict_info[COL_TYPE] = "habitat"
    #if the folder is the samuel folder
    elif [p for p in DIR_SAMUEL if p == directory]:
        specie, *_, fish = filename.split("_")
        dict_info[COL_FISH_NUMBER] = fish.split('.')[0][1:]
        dict_info[COL_FISH_SEX] = fish.split('.')[0][0]
        dict_info[COL_SPECIES] = specie
    return dict_info


def work_metrics(row, preprocess, metrics):
    '''
    function called to perform the boring work of tidying the data to prepare its inclusion to the main dataframe
    call the metric given in parameter under the form (func, args, kwargs)
    where func is the function to call, args is a list of parameters, and kwargs is a dict of parameters
    '''
    img_path = row.name
    head, image_name = os.path.split(img_path)
    image_name, ext = os.path.splitext(image_name)
    image = imageio.imread(img_path)
    image = preprocess(image)
    for (func, args, kwargs) in metrics:
        #call the metric
        dict_data = func(image, *args, **kwargs)
        #if the metric return something to save, save it
        data = dict_data.pop(DATA, None)
        format = dict_data.pop(FORMAT, "")
        save_dir = dict_data.pop(SAVING_DIR, "DefaultDir")
        col_name_path = dict_data.pop(NAME_COL_PATH, "path")
        if data is not None:
            output_dir = os.path.join(head, save_dir)
            dest = os.path.join(output_dir, image_name)+format
            if not( os.path.exists(output_dir) and os.path.isdir(output_dir) ):
                os.mkdir(output_dir)
            if format == ".npy":
                np.save(dest, data)
            elif format in [".png", ".jpg", ".tif"]:
                imageio.imwrite(dest, data, format)
            elif format == ".npz":
                np.savez(dest, *data)
            else:
                raise ValueError("format {} is not of a supported format".format(format))
            print("DATA saved at {}".format(dest))
            row[col_name_path] = dest
        #add the info to the dataframe
        for key, val in dict_data.items():
            row[key] = val
    return row


def get_files(main_path, max_depth, types_allowed, ignored_folders, only_endnodes, visu=False):
    '''
    initiate a csv file with minimum values in it
    max_depth is the depth of the search: 0 is for a file, 1 is to search one directory, 2 is dir and subdirs...
    only_endnodes is whether or not to ignore files that are not located at the endnode of the directory tree
    ignored_folders is a list of folders to ignore
    '''
    assert max_depth>=0, "You should not give a depth lower than 0"
    data = pd.DataFrame()
    to_visit=[(main_path, 0)]
    while to_visit:
        path, depth = to_visit.pop()
        if os.path.isdir(path) and depth<max_depth and (not [i for i in ignored_folders if i in path]):
            to_visit += [(os.path.join(path, next), depth+1) for next in os.listdir(path)]
        elif os.path.isfile(path) and path.endswith(types_allowed):
            if depth==max_depth or not only_endnodes:
                if visu: print("ADDING {}".format(path))
                dict_info = load_info_from_filepath(path)
                data.loc[os.path.abspath(path), [*dict_info.keys()]] = [*dict_info.values()]
    return data


def main(input, preprocess, metrics, recursivity, types_allowed, only_endnodes, ignored_folders, verbosity=1):
    '''
    create a csv file or load an existing one
    update the values in this csv by running new metrics
    save and return the data
    '''
    if input.endswith(".csv"):  #load the csv file if there is one
        data = pd.read_csv(input, index_col=0)
        output = input
    else:   #else create it by parcouring the files
        data = get_files(input, recursivity, types_allowed, ignored_folders, only_endnodes, verbosity>=1)
        if os.path.isdir(input): output = os.path.join(input, CSV_NAME+str(recursivity)+".csv")
        else: output = os.path.splitext(input)[0] + CSV_NAME + str(recursivity) + ".csv"
        #for each function add informations to the data
    data = data.apply(work_metrics, 1, args=[preprocess, metrics])
    #save the data
    data.to_csv(output, sep=',', index=True)
    print("DONE: CSV SAVED AT {}".format(output))
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="the path of the main directory containing the subdirs or the path of the csv to use.")
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=1,
                    help="increase output verbosity, default is 1.")
                    ####   ARGUMENTS FOR FOURIER SLOPE   ###
    parser.add_argument("-s", "--window_size", default=512, type=int,
                    help="Size of the square window, default 512")
    parser.add_argument('--no-resize', dest='resize', action='store_false',
                    help="add this argument to prevent resize, default resize the image to fit the window crop.")
    parser.set_defaults(auto_window=False)
    parser.set_defaults(resize=True)
                    ####   ARGUMENTS FOR LBP   ###
    parser.add_argument("-r", "--radius", default=1, type=int,
                    help="radius of the circle. Default 1.")
    parser.add_argument("-p", "--points", default=8, type=int,
                    help="Number of points on the circle. Default 8.")
    args = parser.parse_args()

    FFT_RANGE=(10, 110) #110 pour des fenetres 200x200!!!
    GLCM_DISTANCES=[1]
    ANGLES=[0, 45, 90, 135]
    RESIZED_IMG=(1536,512)
    GABOR_FREQ=[0.2, 0.4, 0.8]

    vgg16_model = Deep_Features_Model( VGG16(weights='imagenet', include_top=False), (RESIZED_IMG))

    preprocess = partial(preprocess_img, resize=None, to_darter=False, to_gray=False, normalize=False, standardize=False)
    metrics=[
        # (get_Haralick_descriptors, [GLCM_DISTANCES, ANGLES], {"visu":args.verbosity>=2}),
        # # (get_GLCM, [GLCM_DISTANCES, ANGLES], {}),
        # (get_FFT_slope, [FFT_RANGE, args.resize ,args.window_size], {"verbose":args.verbosity}),
        # # (vgg16_model.get_deep_features, [args.verbosity>=1], {}),
        # (get_statistical_features, [], {"visu":args.verbosity>=1}),
        # (get_LBP, [args.points, args.radius, RESIZED_IMG], {"visu":args.verbosity>=2}),
        # (vgg16_model.get_layers_gini, [args.verbosity>=2], {}),
        # (get_gini, [args.verbosity>=2], {}),
        (get_gabor_filters, [ANGLES, GABOR_FREQ], {"visu":args.verbosity>=2})
        ]

    d = main(args.input, preprocess, metrics, 2, (".jpg",".png",".tif"), ignored_folders=[], only_endnodes=True, verbosity=args.verbosity)
    print(d)
