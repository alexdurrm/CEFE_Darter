from scipy.stats import skew, kurtosis, entropy
import numpy as np
import pandas as pd
import argparse
import os

from MotherMetric import MotherMetric
from Preprocess import *


###statistical values

COL_STAT_MEAN="mean_stat"
COL_STAT_STD="std_stat"
COL_STAT_SKEW="skewness_stat"
COL_STAT_KURT="kurtosis_stat"
COL_STAT_ENTROPY="entropy_stat"
class StatMetrics(MotherMetric):
    def __init__(self, preprocess=None):
        if not preprocess:
            preprocess = Preprocess(img_type=IMG.DARTER, img_channel=CHANNEL.GRAY)
        super().__init__(preprocess)
    
    def function(self, image):
        df = self.preprocess.get_params()
        metrics = get_statistical_features(image)
        df.loc[0, metrics.columns] = metrics.loc[0]
        return df
        
        
        
def get_statistical_features(image, visu=False):
    '''
    get an image and return the statistical features like
    mean value, standard deviation, skewness, kurtosis, and entropy
    (calculated on flattened image)
    '''
    assert image.ndim==2, "Image should be 2D only"
    
    vals=pd.DataFrame()
    vals.loc[0, COL_STAT_MEAN]=np.mean(image, axis=None)
    vals.loc[0, COL_STAT_STD]=np.std(image, axis=None)
    vals.loc[0, COL_STAT_SKEW]=skew(image, axis=None)
    vals.loc[0, COL_STAT_KURT]=kurtosis(image, axis=None)
    vals.loc[0, COL_STAT_ENTROPY]=entropy(image, axis=None)
    if visu: print(vals)
    return vals



###GINI

COL_GINI_VALUE="gini_coefficient"
class GiniMetrics(MotherMetric):
    def __init__(self, preprocess=None):
        if not preprocess:
            preprocess = Preprocess(img_type=IMG.DARTER, img_channel=CHANNEL.GRAY)
        super().__init__(preprocess)
    
    def function(self, image):
        gini_df = self.preprocess.get_params()
        gini_df.loc[0, COL_GINI_VALUE] = [get_gini(image)]
        return gini_df
        
def get_gini(array, visu=False):
    '''
    Calculate the Gini coefficient of a numpy array.
    Author: Olivia Guest (oliviaguest)
    Original publication of this code available at https://github.com/oliviaguest/gini/blob/master/gini.py
    '''
    # All values are treated equally, arrays must be 1d:
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
    gini_val = (np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))
    if visu: print("GINI: {}".format(gini_val))
    return gini_val
    
    
    
###COLOR RATIO

COL_COLOR_RATIO="color_ratio"        
class ColorRatioMetrics(MotherMetric):
    def __init__(self, preprocess=None):
        if not preprocess:
            preprocess = Preprocess(img_type=IMG.DARTER, img_channel=CHANNEL.ALL)
        super().__init__(preprocess)
    
    def function(self, image):
        df = self.preprocess.get_params()
        df.loc[0,COL_COLOR_RATIO]=[get_color_ratio(image)]
        return df

def get_color_ratio(image, visu=False):
    '''
    return the color ratio slope between the two color channel
    '''
    assert image.ndim==3, "Image should be 3D"
    assert image.shape[2]==2, "Image should have two channels, here image is shape{}".format(image.shape)
    
    size_sample = np.min([image.size, 1000])
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
    return slope






if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path of the image file to open")
    args = parser.parse_args()
    image_path = os.path.abspath(args.image)
    
    stat = StatMetrics()
    gini = GiniMetrics()
    ratio = ColorRatioMetrics()
    
    print(stat(image_path))
    print(gini(image_path))
    print(ratio(image_path))