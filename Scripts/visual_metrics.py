from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.filters import gabor
import skimage
from scipy.stats import skew, kurtosis, entropy
import argparse
import imageio
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import cv2
import tensorflow.keras as K
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.decomposition import PCA
from PHOG.anna_phog import anna_phog

from pspec import rgb_2_darter, get_pspec
from config import *


def preprocess_img(image, resize=None, to_darter=False, to_gray=False, normalize=False, standardize=False):
    if resize:
        image = cv2.resize(image, dsize=resize, interpolation=cv2.INTER_CUBIC)
    if to_darter:
        image = rgb_2_darter(image)
        if to_gray:
            image = image[:, :, 0] + image[:, :, 1]
    elif to_gray:
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        image = 0.2989 * r + 0.5870 * g + 0.1140 * b
    if normalize:
        image = (image - np.mean(image, axis=(0,1))) / np.std(image, axis=(0,1))
    if standardize:
        image = (image - np.min(image)) / (np.max(image)-np.min(image))
    return image


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
    return {DATA: lbp_image, FORMAT:".npy", SAVING_DIR:"LBP_P{}_R{}".format(P,R), NAME_COL_PATH:COL_PATH_LBP,
            COL_RADIUS_LBP:R ,COL_POINTS_LBP:P }


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
    return {DATA:glcm, FORMAT:".npy", SAVING_DIR:"GLCM", COL_GLCM_DIST: distances, COL_GLCM_ANGLES: angles}


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

    dict_vals[COL_GLCM_DIST] = distances
    dict_vals[COL_GLCM_ANGLES] = angles

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
        plt.plot(dict_vals[COL_GLCM_MEAN])
        plt.show()
        plt.plot(dict_vals[COL_GLCM_VAR])
        plt.show()
        plt.plot(dict_vals[COL_GLCM_CORR])
        plt.show()
        plt.plot(dict_vals[COL_GLCM_CONTRAST])
        plt.show()
        plt.plot(dict_vals[COL_GLCM_DISSIMIL])
        plt.show()
        plt.plot(dict_vals[COL_GLCM_HOMO])
        plt.show()
        plt.plot(dict_vals[COL_GLCM_ASM])
        plt.show()
        plt.plot(dict_vals[COL_GLCM_ENERGY])
        plt.show()
        plt.plot(dict_vals[COL_GLCM_MAXP])
        plt.show()
        plt.plot(dict_vals[COL_GLCM_ENTROPY])
        plt.show()
    return dict_vals


def get_gini(array, visu=False):
    '''
    Calculate the Gini coefficient of a numpy array.
    Author: Olivia Guest (oliviaguest)
    Original publication of this code available at https://github.com/oliviaguest/gini/blob/master/gini.py
    '''
    # All values are treated equally, arrays must be 1d:
    assert array.ndim <=3, "Gini can be calculated on an array of max 3 Dims, given {}".format(array.ndim)
    if array.ndim==3:
        array = rgb_2_darter(array)
        array = array[:, :, 0] + array[:, :, 1]
    if array.ndim==2:
        array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    gini_val = ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))
    if visu: print("GINI: {}".format(gini_val))
    return {COL_GINI_VALUE: gini_val}


class Deep_Features_Model:
    def __init__(self, base_model, input_shape):
        self.base_model = base_model
        self.input_shape = input_shape
        input_tensor = K.Input(shape=self.input_shape)
        self.base_model.layers[0] = input_tensor
        self.deep_features = K.Model(inputs=self.base_model.input, outputs=[l.output for l in self.base_model.layers[1:]])

    def get_deep_features(self, image, visu=False):
        '''
        get the feature space of an image propagated through the deep feature model
        return a list of np array, each element of the list represent an output of a layer ,input layer is ignored
        '''
        #resize and normalize the image
        if image.shape != self.input_shape:
            image = cv2.resize(image, dsize=self.input_shape, interpolation=cv2.INTER_CUBIC)
        image = (image - np.mean(image, axis=(0,1))) / np.std(image, axis=(0,1))
        #make the prediction
        pred = self.deep_features.predict(image[np.newaxis, ...])
        if visu:
            self.deep_features.summary()
            for p in pred:
                print(type(p))
                print(p.shape)
        return {DATA:pred, FORMAT:".npz", SAVING_DIR:"DeepFeatures_"+self.base_model.name,
                COL_MODEL_NAME:self.base_model.name, NAME_COL_PATH:COL_PATH_DEEP_FEATURES}

    def get_layers_gini(self, image, visu=False):
        deep_features = self.get_deep_features(image, visu)[DATA]
        sparseness=[get_gini(df[0])[COL_GINI_VALUE] for df in deep_features]
        if visu: print(sparseness)
        return {COL_SPARSENESS_DF:sparseness, COL_MODEL_NAME:self.base_model.name}


def get_gabor_filters(image, angles, frequencies, visu=False):
    '''
    produces a set of gabor filters and
    angles is the angles of the gabor filters, given in degrees
    return a map of the mean activation of each gabor filter
    '''
    image = rgb_2_darter(image)
    image = image[:, :, 0] + image[:, :, 1]

    assert image.ndim==2, "Should be a 2D array"
    activation_map = np.empty(shape=[len(angles), len(frequencies)])
    rad_angles = np.radians(angles)
    for t, theta in enumerate(rad_angles):
        for f, freq in enumerate(frequencies):
            real, _ = gabor(image, freq, theta)
            if visu:
                plt.imshow(real, cmap="gray")
                plt.title("gabor theta:{}  frequency:{}".format(t, f))
                plt.colorbar()
                plt.show()
            activation_map[t, f] = np.mean(real)
    if visu:
        ax = sns.heatmap(activation_map, annot=True, center=1, xticklabels=frequencies, yticklabels=angles)
        plt.show()
    return {COL_GABOR_ANGLES:angles, COL_GABOR_FREQ:frequencies, COL_GABOR_VALUES:activation_map}


def get_color_ratio(image, visu=False):
    '''
    return the color ratio slope between the two color channel
    '''
    image = rgb_2_darter(image)
    size_sample = np.min([image.shape[0]*image.shape[1], 1000])
    selection = np.random.choice(np.arange(size_sample), size=size_sample, replace=False)
    X = image[..., 0].flatten()[selection]
    Y = image[..., 1].flatten()[selection]

    slope, b = np.polyfit(X, Y, 1)
    print(slope, b)
    if visu:
        x=np.arange(0, np.max(X), 16)
        y=slope*x+b
        plt.plot(x, y)
        plt.scatter(X, Y)
        plt.show()
    return {COL_COLOR_RATIO:slope}


def get_PHOG(image, orientations=8, level=0, visu=False):
    '''
    return the pyramidal histogram oriented graph
    '''
    roi = [0, image.shape[0], 0, image.shape[1]]
    phog = anna_phog(image, orientations, 360, level, roi)
    if visu:
        plt.bar(range(phog.shape[0]), phog)
        plt.show()
    return {COL_PHOG_LEVELS:level, COL_PHOG_BINS:orientations,
        DATA:phog, FORMAT:".npy", SAVING_DIR:"PHOG", NAME_COL_PATH:COL_PATH_PHOG}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path of the image file to open")
    args = parser.parse_args()
    image = imageio.imread(os.path.abspath(args.image))

    # vgg_model_df = Deep_Features_Model(VGG16(weights='imagenet', include_top=False), (1536,512))
    # vgg_model_df.get_layers_gini(image, (1536,512), True)
    # get_FFT_slope(image, (10,110), 200, 1)
    # get_LBP(image, 8, 1, (1536,512), True)
    # get_statistical_features(image, visu=True)
    # get_GLCM(image, [1], [0, 45, 90, 135] )
    # get_Haralick_descriptors(image, [1], [0, 45, 90, 135] , visu=True)
    # get_gini(image)
    # get_gabor_filters(image, [0,45,90,135], [0.2, 0.4, 0.8],visu=True)
    #get_color_ratio(image, True)
    print(get_PHOG(image, 40, 2))
