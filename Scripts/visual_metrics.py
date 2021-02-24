from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from scipy.stats import skew, kurtosis, entropy
import argparse
import imageio
import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import cv2
import tensorflow.keras as K
from tensorflow.keras.applications.vgg16 import VGG16

from pspec import rgb_2_darter, get_pspec
from config import *


def get_LBP(image, P, R, resize=None, visu=False):
    '''
    calculates the Local Binary Pattern of a given image
    P is the number of neighbors points to use
    R is the radius of the circle around the central pixel
    visu is to visualise the result
    return the path to the saved image
    '''
    assert image.ndim==3, "image should be 3 dimensions: H,W,C"
    image = rgb_2_darter(image)
    image = image[:, :, 0] + image[:, :, 1]
    if resize:
        image = cv2.resize(image, dsize=resize, interpolation=cv2.INTER_CUBIC)
        
    lbp_image = local_binary_pattern(image, P, R)
    if visu:
        fig, (ax0, ax1, ax2) = plt.subplots(figsize=(6, 12), nrows=3)
        ax0.imshow(image, cmap='gray')
        ax0.set_title("original image")

        bar = ax1.imshow(lbp_image, cmap='gray')
        fig.colorbar(bar, ax=ax1, orientation="vertical")
        ax1.set_title("LBP with params P={} and R={}".format(P, R))

        ax2.hist(lbp_image.flatten(), bins=P)
        ax2.set_title("lbp values histogram")
        plt.show()
    return {DATA: lbp_image.astype(np.uint8), FORMAT:".jpg", SAVING_DIR:"LBP_P{}_R{}".format(P,R), NAME_COL_PATH:COL_PATH_LBP}


def get_statistical_features(image, visu=False):
    '''
    get an image and return the statistical features like
    mean value, standard deviation, skewness, kurtosis, and entropy
    (calculated on flattened image)
    '''
    image = rgb_2_darter(image)
    image = image[:, :, 0] + image[:, :, 1]
    dict_vals={}
    dict_vals[COL_STAT_MEAN]=np.mean(image, axis=None)
    dict_vals[COL_STAT_STD]=np.std(image, axis=None)
    dict_vals[COL_STAT_SKEW]=skew(image, axis=None)
    dict_vals[COL_STAT_KURT]=kurtosis(image, axis=None)
    dict_vals[COL_STAT_ENTROPY]=entropy(image, axis=None)
    if visu: print("mean: {} /std: {} / skewness: {} / kurtosis: {}".format(*dict_vals.values()))
    return dict_vals


def get_deep_features(image, base_model, visu=False):
    '''
    get the feature space of an image propagated through a given model
    return a list of np array, each element of the list represent an output of a layer ,input layer is ignored
    '''
    image = (image - np.mean(image, axis=(0,1))) / np.std(image, axis=(0,1))
    input_tensor = K.Input(shape=image.shape)
    base_model.layers[0] = input_tensor
    deep_features = K.Model(inputs=base_model.input, outputs=[l.output for l in base_model.layers[1:]])
    # To see the models' architecture and layer names, run the following
    pred = deep_features.predict(image[np.newaxis, ...])
    if visu:
        deep_features.summary()
        for p in pred:
            print(type(p))
            print(p.shape)
    return {DATA:pred, FORMAT:".npz", SAVING_DIR:"DeepFeatures_"+base_model.name, 
            COL_MODEL_NAME:base_model.name, NAME_COL_PATH:COL_PATH_DEEP_FEATURES}

   
def get_FFT_slope(image, fft_range, resize, sample_dim, verbose=1):
    '''
    Calculate the fourier slope of a given image
    fft_range is the range of frequencies for which is calculated the slope
    sample_dim is the size of the squared window sample
    return a dict containing informations about the fourier slope
    '''
    if resize:
        resize_ratio = sample_dim/np.min(image.shape[0:2])
        new_x, new_y = (int(round(resize_ratio*dim)) for dim in image.shape[0:2])
        if verbose>=1:
            print("Resizing image from {}x{} to {}x{}".format(
                image.shape[0], image.shape[1], new_x, new_y ))
        image = cv2.resize(image, dsize=(new_y, new_x),
            interpolation=cv2.INTER_CUBIC)  #cv2 (x,y) are numpy (y,x)
    else: assert image.shape[0]>=sample_dim<=image.shape[1], "sample dim should be less or equal than image shapes"
    # Define a sliding square window to iterate on the image
    stride = int(sample_dim/2)
    slopes = []
    tot_samples = 0
    for start_x in range(0, image.shape[0]-sample_dim+1, stride):
        for start_y in range(0, image.shape[1]-sample_dim+1, stride):
            tot_samples+=1
            sample = image[start_x: start_x+sample_dim, start_y: start_y + sample_dim]
            slopes.append( get_pspec(sample, bin_range=fft_range, visu=verbose>=2) )
    if verbose>=1: print("mean slope on {} samples: {}".format(tot_samples, np.mean(slopes)))
    return {COL_F_SLOPE:np.mean(slopes), COL_F_N_SAMPLE:tot_samples, COL_F_WIN_SIZE:sample_dim, COL_FFT_RANGE:fft_range}

    
def get_GLCM(image, distances, angles):
    '''
    get an image and calculates its grey level co-occurence matrix
    calculate it along different angles and distances
    '''
    image = rgb_2_darter(image).astype(np.uint8)
    image = image[:, :, 0] + image[:, :, 1]
    
    glcm = greycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False)
    return {DATA:glcm, FORMAT:".npy", SAVING_DIR:"GLCM"}

    
def get_Haralick_descriptors(image, distances, angles, visu=False):
    '''
    get an image and calculates its grey level co-occurence matrix
    calculate it along different angles and distances
    returns a few characteristics about this GLCM
    '''
    image = rgb_2_darter(image).astype(np.uint8)
    image = image[:, :, 0] + image[:, :, 1]
    
    dict_vals={}
    glcm = greycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False)
    
    dict_vals[COL_GLCM_MEAN] = np.mean(glcm, axis=(0,1))
    dict_vals[COL_GLCM_VAR] = np.var(glcm, axis=(0,1))
    dict_vals[COL_GLCM_CORR] = greycoprops(glcm, 'correlation')
    
    dict_vals[COL_GLCM_CONTRAST] = greycoprops(glcm, 'contrast')
    dict_vals[COL_GLCM_DISSIMIL] = greycoprops(glcm, 'dissimilarity')
    dict_vals[COL_GLCM_HOMO] = greycoprops(glcm, 'homogeneity')
    
    dict_vals[COL_GLCM_ASM] = greycoprops(glcm, 'ASM')
    dict_vals[COL_GLCM_ENERGY] = greycoprops(glcm, 'energy')
    
    dict_vals[COL_GLCM_MAXP] = np.max(glcm, axis=(0,1))
    dict_vals[COL_GLCM_ENTROPY] = entropy(glcm, axis=(0,1))
    if visu: 
        print(dict_vals)
    return dict_vals
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path of the image file to open")
    args = parser.parse_args()
    image = imageio.imread(os.path.abspath(args.image))

    vgg_model = VGG16(weights='imagenet',
                              include_top=False)
    get_deep_features(image, vgg_model, True)   
    get_FFT_slope(image, (10,110), 200, 1)
    get_LBP(image, 8, 1, True)
    get_statistical_features(image, visu=True)
    get_GLCM(image, [1], [0, 45, 90, 135] )
    get_Haralick_descriptors(image, [1], [0, 45, 90, 135] , visu=True)