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
import argparse
from pspec import get_pspec

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="the path of the main directory containing the subdirs.")
parser.add_argument("-s", "--window_size", default=200, type=int,
                    help="Size of the square window, default 200. Is ignored if [-auto_window] is used")
parser.add_argument('--no-resize', dest='resize', action='store_false',
                    help="add this argument to prevent resize, default is resize.")
parser.set_defaults(resize=True)
parser.add_argument('--auto-window', dest='auto_window', action='store_true',
                    help="add this argument to choose automatically the window size,\
                     default is set by [-window_size] at 200 .")
parser.set_defaults(auto_window=False)
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=1,
                    help="increase output verbosity, default is 1.")
args = parser.parse_args()

MAIN_DIR=os.path.abspath(args.input_dir)
RESIZE=args.resize
SAMPLE_DIM=args.window_size
AUTO_WINDOW=args.auto_window
fft_range=(10, 110) #110 pour des fenetres 200x200!!!

def get_files(path):
    folders = []
    index = {}
    os.chdir(path)

    for item in os.listdir():
        if os.path.isdir(item): folders.append(item)

    for folder in folders:
        os.chdir(path + '/' + folder)
        fish = []
        for item in os.listdir():
            if item.endswith('.tif') or item.endswith('.jpg') or item.endswith('.png'):
                fish.append(item)
        index[folder] = fish

    return folders, index

def update_sample_dim(folders, index):
    '''Determine the largest possible square sample dimension that can fit all the images'''
    dims = []
    for folder in folders:
        os.chdir(MAIN_DIR + '/' + folder)
        if index[folder]:
            for file in index[folder]:
                image = imageio.imread(file)[:, :, 1]
                dims.append(np.min(image.shape[0:2]))
    global SAMPLE_DIM
    SAMPLE_DIM = np.min(dims)

def calculate_slopes(folder):
    '''Calculate the slopes for each images in a folder'''
    dict_slopes={}
    for file in folder:
        image = imageio.imread(file)

        # resize the image to just fit the window while keeping its ratio
        if RESIZE:
            resize_ratio = SAMPLE_DIM/np.min(image.shape[0:2])
            new_x = int(resize_ratio*image.shape[0])
            new_y = int(resize_ratio*image.shape[1])
            if args.verbosity>=1:
                print("Resizing image from {}x{} to {}x{}".format(
                    image.shape[0], image.shape[1], new_x, new_y ))

            image = cv2.resize(image, dsize=(new_y, new_x),
                interpolation=cv2.INTER_CUBIC)  #cv2 (x,y) are numpy (y,x)

        #Define a sliding square window of size sample_dim and stride sample_dim/2
        stride = int(SAMPLE_DIM/2)
        samp_on_X = int((image.shape[0]-SAMPLE_DIM)/stride)+1
        samp_on_Y = int((image.shape[1]-SAMPLE_DIM)/stride)+1
        tot_samples = samp_on_X * samp_on_Y
        #iterate on the image
        slope = 0
        for n_x in range(samp_on_X):
            for n_y in range(samp_on_Y):
                #Sample image and convert to darter color model
                start_x = n_x*stride
                start_y = n_y*stride
                sample = image[start_x: start_x+SAMPLE_DIM, start_y: start_y + SAMPLE_DIM]
                #Calculate the slope of the power spectrum
                slope += get_pspec(sample, bin_range=fft_range)
        slope /= tot_samples
        print("slope: "+str(slope))
        dict_slopes[file] = slope
    return dict_slopes

def basic_work_slopes(path, output_file):
    '''
    Function used to calculate the fourier slopes of all subdirectories in a main directory
    then save it in a csv file 
    uses calculate_slopes to get the slopes but takes care of the file structure and csv structure
    '''
    folders, index = get_files(path)
    if AUTO_WINDOW:
        update_sample_dim(folders, index)
    print("Evaluating Fourier slopes on {}x{} windows\n{}".format(SAMPLE_DIM, SAMPLE_DIM, "-"*11))

    data = pd.DataFrame(columns=["filename", "folder", "slope", "window_size"])
    for folder in folders:
        os.chdir(path + '/' + folder)
        #Go through list of files and calculate power spectrum for each image
        if index[folder]:
            slopes = calculate_slopes(index[folder])
            for file, slope in slopes.items():
                #add to the DataFrame
                middle = np.nan
                color_control = np.nan
                fish_n = np.nan
                if 'HABITAT' in folder:
                    middle = file.split("_")[0]
                elif 'layers' in folder.lower():
                    middle, fish_n = file.split("_")[0].split("x")
                    color_control = file.split("_")[1][12:]
                elif "FISH" in folder:
                    fish_n = file.split("_")[0]
                data.loc[file] = [file, folder, slope, SAMPLE_DIM]
    data.to_csv(MAIN_DIR+ '/' + output_file +'.csv', sep=',', index=False)
    if args.verbosity>=1: print("Saved "+ MAIN_DIR+ '/' + output_file +'.csv')

def work_compare_transfer(path, output_file):
    '''
    Function used to calculate the fourier slope of all subdirectories in a main directory
    then save it in a csv file
    '''
    folders, index = get_files(path)
    if AUTO_WINDOW:
        update_sample_dim(folders, index)
    print("Evaluating Fourier slopes on {}x{} windows\n{}".format(SAMPLE_DIM, SAMPLE_DIM, "-"*11))

    data = pd.DataFrame(columns=["filename", "folder", "slope", "window_size", "middle", "fish_n", "color_control", "sex"])
    for folder in folders:
        os.chdir(path + '/' + folder)
        #Go through list of files and calculate power spectrum for each image
        if index[folder]:
            slopes = calculate_slopes(index[folder])
            for file, slope in slopes.items():
                #add to the DataFrame
                middle = np.nan
                color_control = np.nan
                fish_n = np.nan
                sex = np.nan
                if 'HABITAT' in folder:
                    middle = file.split("_")[0]
                elif 'layers' in folder.lower():
                    middle, fish_n = file.split("_")[0].split("x")
                    color_control = file.split("_")[1][12:]
                elif "FISH" in folder:
                    sex = file.split("_")[-1][0]
                    fish_n = file.split("_")[0]
                data.loc[file] = [file, folder, slope, SAMPLE_DIM, middle, fish_n, color_control, sex]
    data.to_csv(MAIN_DIR+ '/' + output_file +'.csv', sep=',', index=False)
    if args.verbosity>=1: print("Saved "+ MAIN_DIR+ '/' + output_file +'.csv')

if __name__ == '__main__':
    output_file="Fourier_slope_resize{}_window{}".format(RESIZE, SAMPLE_DIM)
    #work_slopes(MAIN_DIR, output_file)
    work_compare_transfer(MAIN_DIR, output_file)
