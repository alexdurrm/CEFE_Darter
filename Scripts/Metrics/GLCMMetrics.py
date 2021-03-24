from skimage.feature import greycomatrix, greycoprops
from scipy.stats import entropy
import pandas as pd
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt

from MotherMetric import MotherMetric
from PHOG.anna_phog import anna_phog
from Preprocess import *
from config import *


#GLCM
CSV_HARALICK="haralick.csv"

COL_GLCM_MEAN="GLCM_mean"
COL_GLCM_VAR="GLCM_variance"
COL_GLCM_CORR="GLCM_correlation"
COL_GLCM_CONTRAST="GLCM_contrast"
COL_GLCM_DISSIMIL="GLCM_dissimilarity"
COL_GLCM_HOMO="GLCM_homogeneity"
COL_GLCM_ASM="GLCM_ASM"
COL_GLCM_ENERGY="GLCM_energy"
COL_GLCM_MAXP="GLCM_max_proba"
COL_GLCM_ENTROPY="GLCM_entropy"

COL_GLCM_ANGLE="GLCM_angle"
COL_GLCM_DIST="GLCM_dist"

class HaralickMetrics(MotherMetric):
    def __init__(self, distances, angles, *args, **kwargs):
        self.distances = distances
        self.angles = angles
        super().__init__(*args, **kwargs)

    def function(self, image):
        image = image.astype(np.uint8)
        df = pd.DataFrame()
        params = self.preprocess.get_params()
        haralick = get_Haralick_descriptors(image, self.distances, self.angles)
        idx = 0
        for d, distance in enumerate(self.distances):
            for a, angle in enumerate(self.angles):
                df.loc[idx, params.columns] = params.loc[0]
                df.loc[idx, [COL_GLCM_ANGLE, COL_GLCM_DIST]] = [angle, distance]
                for k, v in haralick.items():
                    df.loc[idx, k] = v[d, a]
                idx+=1
        return df

    def visualize(self):
        data_image = pd.read_csv(os.path.join(DIR_RESULTS, CSV_IMAGE), index_col=0)
        merge_data = self.data.merge(data_image, on=COL_IMG_PATH)
        haralick_descriptors = [COL_GLCM_MEAN, COL_GLCM_VAR, COL_GLCM_CORR, COL_GLCM_CONTRAST,
        COL_GLCM_DISSIMIL, COL_GLCM_HOMO, COL_GLCM_ASM, COL_GLCM_ENERGY,
        COL_GLCM_MAXP, COL_GLCM_ENTROPY]
        sns.set_palette(sns.color_palette(FLAT_UI))

        for col_descriptor in haralick_descriptors:
            ax = sns.catplot(x=COL_DIRECTORY, y=col_descriptor, data=merge_data,
                            row=COL_GLCM_ANGLE,
                            col=COL_GLCM_DIST)

            ax.set_ylabels(col_descriptor)
            ax.set_yticklabels(fontstyle='italic')
            plt.xticks(rotation=45)
            plt.title(col_descriptor)
            plt.show()

def get_Haralick_descriptors(image, distances, angles, visu=False):
    '''
    get an image and calculates its grey level co-occurence matrix
    calculate it along different angles and distances
    returns a few characteristics about this GLCM
    '''
    assert image.ndim==2, "Image should be 2D"

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


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path of the image file to open")
    args = parser.parse_args()
    image_path = os.path.abspath(args.image)

    preprocess = Preprocess(img_type=IMG.DARTER, img_channel=CHANNEL.GRAY)
    metric = HaralickMetrics(distances=[2,4], angles=[0,45,90,135], preprocess=preprocess, path=os.path.join(DIR_RESULTS, CSV_HARALICK))
    # metric.metric_from_df(image_path)
    # metric.save()
    metric.load()
    metric.visualize()
