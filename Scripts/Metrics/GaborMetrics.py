import pandas as pd
import os
import argparse
from skimage.filters import gabor

from MotherMetric import MotherMetric
from Preprocess import *


#GABOR
COL_GABOR_ANGLES="gabor_angles"
COL_GABOR_FREQ="gabor_frequencies"
COL_GABOR_VALUES="gabor_values"

class GaborMetrics(MotherMetric):
    def __init__(self, angles, frequencies, *args, **kwargs):
        self.angles = angles
        self.frequencies = frequencies
        super().__init__(*args, **kwargs)    
    
    def function(self, image):
        params = self.preprocess.get_params()
        df = pd.DataFrame()
        
        activation_map = get_gabor_filters(image, self.angles, self.frequencies)
        for a, angle in enumerate(self.angles):
            for f, freq in enumerate(self.frequencies):
                df.loc[a*f+f, params.columns] = params.loc[0]
                df.loc[a*f+f, [COL_GABOR_ANGLES, COL_GABOR_FREQ, COL_GABOR_VALUES]] = [angle, freq, activation_map[a, f]]
        return df


def get_gabor_filters(image, angles, frequencies, visu=False):
    '''
    produces a set of gabor filters and
    angles is the angles of the gabor filters, given in degrees
    return a map of the mean activation of each gabor filter
    '''
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
    return activation_map

    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path of the image file to open")
    args = parser.parse_args()
    image_path = os.path.abspath(args.image)

    preprocess = Preprocess(img_type=IMG.DARTER, img_channel=CHANNEL.GRAY)
    metric = GaborMetrics(angles=[0,45,90,135], frequencies=[0.2, 0.4, 0.8], preprocess=preprocess)
    print(metric(image_path))
    
    