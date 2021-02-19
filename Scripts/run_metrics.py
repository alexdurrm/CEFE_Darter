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
from pspec import get_pspec
from visual_metrics import do_LBP_metric

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="the path of the main directory containing the subdirs.")

                    ####   ARGUMENTS FOR FOURIER SLOPE   ###
parser.add_argument("-s", "--window_size", default=200, type=int,
                    help="Size of the square window, default 200")
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
MAIN_DIR=os.path.abspath(args.input_dir)
VERBOSITY=args.verbosity
#FOURIER SLOPE
RESIZE=args.resize
SAMPLE_DIM=args.window_size
FFT_RANGE=(10, 110) #110 pour des fenetres 200x200!!!
#LBP
POINTS = args.points
RADIUS = args.radius


def load_info_from_filepath(file_path):
    '''
    Given an absolute file path
    returns the informations about the file in the form of a dict
    '''
    head, filename = os.path.split(file_path)
    head, directory = os.path.split(head)
    dict_info={"filename":filename, "folder":directory}
    if "styletransferimages" in head.lower():
        if 'HABITAT' in directory:
            dict_info["middle"] = filename.split("_")[0]
        elif 'layers' in directory.lower():
            dict_info["middle"], dict_info["fish_n"] = filename.split("_")[0].rsplit("x", maxsplit=1)
            dict_info["color_control"] = filename.split("_")[1][12:]
        elif "FISH" in directory:
            dict_info["sex"] = filename.split("_")[-1][0]
            dict_info["fish_n"] = filename.split("_")[0]
    elif "crops" in head.lower():
        dict_info["species"] = directory
        dict_info["sex"] = filename.split("_")[-1][0]
        dict_info["fish_n"] = filename.split("_")[-1].split(".")[0][1:]

    return dict_info


def calculate_slopes(folder, fft_range, sample_dim ,resize, verbose=1,
                        types_allowed={".tif",".jpg",".png"}):
    '''
    Calculate the slopes for each images in a folder,
    fft_range is the range of frequencies for which is calculated the slope
    sample_dim is the size of the squared window sample
    if resize is true the image is resized to fit the sample dim but conserve its ratio
    returns a dict of slope value per image
    '''
    dict_slopes={}
    for file in folder:
        if os.path.isfile(file) and path.endswith(types_allowed):
            dict_slopes[file] = calculate_slope(path, fft_range, sample_dim ,resize, verbose, return_dict=False)
    return dict_slopes


def calculate_slope(path, fft_range, sample_dim ,resize, verbose=1, return_dict=False):
    '''
    Calculate the fourier slope of a given image
    fft_range is the range of frequencies for which is calculated the slope
    sample_dim is the size of the squared window sample
    if resize is true the image is resized to fit the sample dim but conserve its ratio
    returns the slope value of the image or a dict containing more informations about the fourier slope
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
    else:
        assert image.shape[0]>=sample_dim<=image.shape[0], "The dimention\
             for the samples should be lower or equal to the images shape, \
             you can specify a resize true to resize the image to the sample\
             dimension"
    # Define a sliding square window to iterate on the image
    stride = int(sample_dim/2)
    slopes = []
    tot_samples = 0
    for start_x in range(0, image.shape[0]-sample_dim+1, stride):
        for start_y in range(0, image.shape[1]-sample_dim+1, stride):
            tot_samples+=1
            sample = image[start_x: start_x+sample_dim, start_y: start_y + sample_dim]
            slopes.append( get_pspec(sample, bin_range=fft_range, visu=verbose>=2) )
    if verbose>=1: print("mean slope on {} samples: {}".format(tot_samples,np.mean(slopes)))
    if return_dict:
        return {"F_slope":np.mean(slopes), "F_n_samples":tot_samples, "F_window_size":sample_dim}
    else:
        return np.mean(slopes)


def recursive_metrics_work(path, metrics, recursivity=1, root_save=True,
                            types_allowed=(".jpg", ".tif", ".png"), only_endnodes=True):
    '''
    will look recursively in folders for images to analyse,
    for each found image will perform a set of metrics
    recursivity is the depth of the search we want:
        1 is for every images in a folder deep, 0 is for an image
    when the end node is reached will perform the metrics and return a dict of values
    metrics should be a list of tuples containing 4 values (func, args, kwargs)
        the function to use to get a value ( must be func(image,...) )
        the list of parameters to use on the metric function
        the dict of named parameters to use on the metric function
    returns a dataframe containing informations about the images
    '''
    assert recursivity>=0, "You should not set a recursivity less than 0"
    data=pd.DataFrame()
    #if end node is reached, perform the metrics on image path
    if recursivity <= 0:
        print("reached endnode {}".format(path))
        if os.path.isfile(path) and path.endswith(types_allowed):
            for (func, args, kwargs) in metrics:
                #for each function add informations to the data
                dict_info = func(path, *args, **kwargs)
                data.loc[path, dict_info.keys()] = dict_info.values()

    #if not an end node, call itself and append the results
    else:
        if os.path.isdir(path):
            for file in os.listdir(path):
                print("NEXT: "+file)
                next_path = os.path.join(path, file)
                values = recursive_metrics_work(next_path, metrics,
                        recursivity-1, False, types_allowed, only_endnodes)
                data = data.append(values, ignore_index=False, verify_integrity=True)
        elif not only_endnodes:
                values = recursive_metrics_work(path, metrics,
                        0, False, types_allowed, only_endnodes)
                data = data.append(values, ignore_index=False, verify_integrity=True)
    #the root of the recursive tree returns or save the gathered data
    if root_save:
        data.to_csv('{}/metrics_rec{}.csv'.format(path, recursivity), sep=',', index=True)
        print("csv saved in {}/metrics_rec{}.csv".format(path, recursivity))
    return data


if __name__ == '__main__':
    metrics=[
        (load_info_from_filepath, [], {}),
        (calculate_slope, [FFT_RANGE, SAMPLE_DIM ,RESIZE, VERBOSITY, True], {}),
        (do_LBP_metric, [POINTS, RADIUS], {"verbosity":VERBOSITY, "resize":(3000,1000)})
        ]
    recursive_metrics_work(MAIN_DIR, metrics, recursivity=2, types_allowed=(".jpg",".png",".tif"), only_endnodes=True)
