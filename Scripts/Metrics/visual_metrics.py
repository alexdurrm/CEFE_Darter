from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.filters import gabor
import skimage
import argparse
import imageio
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import cv2
from PHOG.anna_phog import anna_phog

from pspec import rgb_2_darter, get_pspec
from config import *



def get_LBP(image, P, R, visu=False):
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
