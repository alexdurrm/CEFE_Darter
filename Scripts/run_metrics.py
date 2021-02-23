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

from pspec import get_pspec, rgb_2_darter
from config import *
from visual_metrics import get_LBP, get_FFT_slope

###############################################################
### File used to calculate metrics and save them in a csv file
###############################################################


def do_LBP_metric(img_path, P, R, verbosity=1, resize=None):
    '''
    calls get_LBP and saves its result
    return a dict of informations about the LBP obtained
    '''
    image = imageio.imread(img_path)
    image = rgb_2_darter(image)
    image = image[:, :, 0] + image[:, :, 1]
    if resize:
        image = cv2.resize(image, dsize=resize, interpolation=cv2.INTER_CUBIC)
    LBP_image = get_LBP(image, P, R, verbosity>=2)

    head, image_name = os.path.split(img_path)
    image_name, ext = os.path.splitext(image_name)
    output_dir = os.path.join(head, "LocalBinaryPattern_P{}_R{}".format(P,R))

    if not( os.path.exists(output_dir) and os.path.isdir(output_dir) ):
        os.mkdir(output_dir)
    output_filepath = os.path.join(output_dir, image_name+"_LBP.npy")
    np.save(output_filepath, LBP_image)
    if verbosity>=1: print("LBP image saved at {}".format(output_filepath))
    return {COL_PATH_LBP: output_filepath, COL_RADIUS_LBP:R , COL_POINTS_LBP:P, COL_RESIZE_LBP:bool(resize)}


def calculate_slope(path, fft_range, sample_dim ,resize, verbose=1, return_dict=False):
    '''
    Calculate the fourier slope of a given image
    fft_range is the range of frequencies for which is calculated the slope
    sample_dim is the size of the squared window sample
    if resize is true the image is resized to fit the sample dim but conserve its ratio
    return a dict containing more informations about the fourier slope
    '''
    image = imageio.imread(path)
    # resize the image to just fit the window while keeping its ratio
    if resize:
        resize_ratio = sample_dim/np.min(image.shape[0:2])
        new_x, new_y = (int(round(resize_ratio*dim)) for dim in image.shape[0:2])
        if verbose>=1:
            print("Resizing image from {}x{} to {}x{}".format(
                image.shape[0], image.shape[1], new_x, new_y ))
        image = cv2.resize(image, dsize=(new_y, new_x),
            interpolation=cv2.INTER_CUBIC)  #cv2 (x,y) are numpy (y,x)
    return get_FFT_slope(image, fft_range, sample_dim, verbose)
    
    
############################################################


def load_info_from_filepath(file_path):
    '''
    Given an absolute file path
    returns the informations about the file in the form of a dict
    '''
    long_head, filename = os.path.split(file_path)
    head, directory = os.path.split(long_head)
    dict_info={COL_FILENAME:filename, COL_DIRECTORY:directory}

    #if the directory contains stylized fishes
    if [p for p in DIR_STYLIZED_FISHES if p in long_head]:
        crossing, color_ctrl, _, tvloss, _, layers  = filename.split("_")
        middle, fish_n = crossing.rsplit("x", maxsplit=1)
        dict_info[COL_FISH_NUMBER] = fish_n
        dict_info[COL_HABITAT] = middle
        dict_info[COL_COLOR_CONTROL] = color_ctrl
        dict_info[COL_TV_LOSS] = tvloss
        dict_info[COL_LAYERS] = layers
        dict_info[COL_TYPE] = "fish stylized"
    #if the directory contains original fishes
    elif [p for p in DIR_ORIGINAL_FISHES if p in long_head]:
        fish_n, _, original_size, specie, *_, end = filename.split("_")
        dict_info[COL_SPECIES] = specie
        dict_info[COL_FISH_NUMBER] = end.split('.')[0][1:]
        dict_info[COL_FISH_SEX] = end[0]
        dict_info[COL_TYPE] = "crop fish" 
    #if the directory contains original habitats
    elif [p for p in DIR_ORIGINAL_HABITATS if p in long_head]:
        middle, *_ = filename.split('_')
        dict_info[COL_HABITAT] = middle
        dict_info[COL_TYPE] = "habitat"   
    #if the folder is the samuel folder
    elif [p for p in DIR_SAMUEL if p in long_head]:
        specie, *_, fish = filename.split("_")
        dict_info[COL_FISH_NUMBER] = fish.split('.')[0][1:]
        dict_info[COL_FISH_SEX] = fish.split('.')[0][0]
        dict_info[COL_SPECIES] = specie


def work_metrics(row, *metrics):
    '''
    function called to perform the boring work of tidying the data to prepare its inclusion to the main dataframe
    call the metric given in parameter under the form (func, args, kwargs)
    where func is the function to call, args is a list of parameters, and kwargs is a dict of parameters
    '''
    img_path = row.name
    for (func, args, kwargs) in metrics:
        dict_data = func(img_path, *args, **kwargs)
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
                data.loc[path, [*dict_info.keys()]] = [*dict_info.values()]
    return data

    
def linear_metric_work(input, metrics, recursivity, types_allowed, only_endnodes, ignored_folders, verbosity=1):
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
    data = data.apply(work_metrics, 1, args=metrics)
    # data.join(df_info)  #merge the data on the index
        
    #save the data
    data.to_csv(output, sep=',', index=True)
    print("DONE: CSV SAVED AT {}".format(output))
    return data
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="the path of the main directory containing the subdirs or the path of the csv to use.")
                    ####   ARGUMENTS FOR FOURIER SLOPE   ###
    parser.add_argument("-s", "--window_size", default=512, type=int,
                    help="Size of the square window, default 512")
    parser.add_argument('--no-resize', dest='resize', action='store_false',
                    help="add this argument to prevent resize, default resize the image to fit the window crop.")
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=1,
                    help="increase output verbosity, default is 1.")
    parser.set_defaults(auto_window=False)
    parser.set_defaults(resize=True)
                    ####   ARGUMENTS FOR LBP   ###
    parser.add_argument("-r", "--radius", default=1, type=int,
                    help="radius of the circle. Default 1.")
    parser.add_argument("-p", "--points", default=8, type=int,
                    help="Number of points on the circle. Default 8.")
    args = parser.parse_args()
    
    FFT_RANGE=(10, 110) #110 pour des fenetres 200x200!!!

    metrics=[
        (calculate_slope, [FFT_RANGE, args.window_size ,args.resize, args.verbosity, True], {}),
        (do_LBP_metric, [ args.points, args.radius], {"verbosity":args.verbosity, "resize":(1536, 512)})
        ]
    # recursive_metrics_work(INPUT, metrics, recursivity=2, types_allowed=(".jpg",".png",".tif"), only_endnodes=True)
    d = linear_metric_work(args.input, metrics, 1, (".jpg",".png",".tif"), ignored_folders=[], only_endnodes=True, verbosity=args.verbosity) 
    print(d)