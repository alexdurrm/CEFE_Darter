from scipy.stats import skew, kurtosis, entropy
import numpy as np
import pandas as pd
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt

from MotherMetric import MotherMetric
from Preprocess import *
from config import *


###statistical values

COL_STAT_MEAN="mean_stat"
COL_STAT_STD="std_stat"
COL_STAT_SKEW="skewness_stat"
COL_STAT_KURT="kurtosis_stat"
COL_STAT_ENTROPY="entropy_stat"
COL_GINI_VALUE="gini_coefficient"
class StatMetrics(MotherMetric):
    def function(self, image):
        df = self.preprocess.get_params()
        metrics = get_statistical_features(image)
        df.loc[0, metrics.columns] = metrics.loc[0]
        df.loc[0, COL_GINI_VALUE] = [get_gini(image)]
        return df
        
    def visualize(self):
        '''
        plot a visualization of the metric
        '''
        stats = [COL_STAT_MEAN, COL_STAT_STD, COL_STAT_SKEW, COL_STAT_KURT, COL_STAT_ENTROPY, COL_GINI_VALUE]
        data_image = pd.read_csv(os.path.join(DIR_RESULTS, CSV_IMAGE), index_col=0)
        merge_data = self.data.merge(data_image, on=COL_IMG_PATH)
        
        sns.set_palette(sns.color_palette(FLAT_UI))
        for stat in stats:
            ax = sns.catplot(x=COL_DIRECTORY, y=stat, data=merge_data)
            ax.set_ylabels(stat)
            ax.set_yticklabels(fontstyle='italic')
            plt.xticks(rotation=45)
            plt.title("statistical descriptors")
            plt.show()

            ax = sns.catplot(x=COL_DIRECTORY, y=stat, data=merge_data, hue=COL_HABITAT)
            ax.set_ylabels(stat)
            plt.xticks(rotation=45)
            plt.show()



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
CSV_COLOR_RATIO="color_ratio.csv"
COL_COLOR_RATIO="color_ratio"        
class ColorRatioMetrics(MotherMetric):    
    def function(self, image):
        df = self.preprocess.get_params()
        df.loc[0,COL_COLOR_RATIO]=[get_color_ratio(image)]
        return df
        
    def visualize(self):
        '''
        plot a visualization of the metric
        '''
        sns.set_palette(sns.color_palette(FLAT_UI))

        data_image = pd.read_csv(os.path.join(DIR_RESULTS, CSV_IMAGE), index_col=0)
        merge_data = self.data.merge(data_image, on=COL_IMG_PATH)
        
        ax = sns.catplot(x=COL_DIRECTORY, y=COL_COLOR_RATIO, data=merge_data)
        ax.set_ylabels(COL_COLOR_RATIO)
        ax.set_yticklabels(fontstyle='italic')
        plt.xticks(rotation=45)
        plt.title("color ratio")
        plt.show()

    

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
    path = os.path.abspath(args.image)
    
    # preprocess = Preprocess(img_type=IMG.DARTER, img_channel=CHANNEL.GRAY)
    # stat = StatMetrics(preprocess=preprocess, path="Results\\stats.csv")
    # stat.metric_from_df(path)
    # stat.save()
    # stat.load()
    # stat.visualize()
    
    preprocess_all = Preprocess(img_type=IMG.DARTER, img_channel=CHANNEL.ALL)
    # print(ratio(path))
    ratio = ColorRatioMetrics(preprocess_all, path=os.path.join(DIR_RESULTS, CSV_COLOR_RATIO))
    ratio.metric_from_df(path)
    ratio.save()
    ratio.visualize()
    