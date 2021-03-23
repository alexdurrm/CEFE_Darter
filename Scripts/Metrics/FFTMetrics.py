import cv2
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from MotherMetric import MotherMetric
from Preprocess import *
from FourierAnalysisMaster.pspec import get_pspec
from config import *


COL_F_WIN_SIZE="window_size_F"
COL_FFT_RANGE_MIN="freq_range_Min_F"
COL_FFT_RANGE_MAX="freq_range_Max_F"
class FFTMetrics(MotherMetric):
    def __init__(self, fft_range, sample_dim, *args, **kwargs):
        self.fft_range = fft_range
        self.sample_dim = sample_dim
        super().__init__(*args, **kwargs)

###FFT SLOPE
CSV_FFT_SLOPE="FFT_slopes.csv"
COL_F_SLOPE_SAMPLE="slope_sample_F"
COL_F_SAMPLE_IDX="sample_idx_F"
### FFT_SLOPES
class FFTSlopes(FFTMetrics):
    def function(self, image):
        df = pd.DataFrame()
        params = self.preprocess.get_params()
        slopes = get_FFT_slopes(image, self.fft_range, self.sample_dim)
        for idx, slope in enumerate(slopes):
            df.loc[idx, params.columns] = params.iloc[0]
            df.loc[idx, [COL_F_SAMPLE_IDX, COL_F_SLOPE_SAMPLE, COL_F_WIN_SIZE, COL_FFT_RANGE_MIN, COL_FFT_RANGE_MAX]] = [idx, slope, self.sample_dim, *self.fft_range]
        return df

#### MEAN_FFT_SLOPES
CSV_MEAN_FFT_SLOPE="mean_fft_slope.csv"
COL_F_MEAN_SLOPE = "mean_fourier_slope"
COL_F_N_SAMPLE = "samples_used_F"
class MeanFFTSlope(FFTMetrics):
    def function(self, image):
        df = self.preprocess.get_params()
        slopes = get_FFT_slopes(image, self.fft_range, self.sample_dim)
        df.loc[0, [COL_F_MEAN_SLOPE, COL_F_N_SAMPLE, COL_F_WIN_SIZE, COL_FFT_RANGE_MIN, COL_FFT_RANGE_MAX]] = [np.mean(slopes), len(slopes), self.sample_dim, *self.fft_range]
        return df

    def visualize(self):
        '''
        used to plot the violin graph the fourier slopes categorized per folder
        '''
        data_image = pd.read_csv(os.path.join(DIR_RESULTS, CSV_IMAGE), index_col=0)
        merge_data = self.data.merge(data_image, on=COL_IMG_PATH)
        sns.set_palette(sns.color_palette(FLAT_UI))
        ax = sns.catplot(x=COL_DIRECTORY, y=COL_F_MEAN_SLOPE, data=merge_data,
                        #col=COL_COLOR_CONTROL,
                        #hue=COL_FISH_SEX,
                        #split=True,
                        kind="violin")

        ax.set_ylabels('Slope of Fourier Power Spectrum ')
        ax.set_yticklabels(fontstyle='italic')
        plt.xticks(rotation=45)
        plt.title("mean Fourrier slopes per folder")
        plt.show()


def get_FFT_slopes(image, fft_range, sample_dim, verbose=1):
    '''
    Calculate the fourier slopes of a given image for a given sample dimension
    '''
    assert image.ndim==2, "Image should be 2D"
    assert image.shape[0]>=sample_dim<=image.shape[1], "sample dim should be less or equal than image shapes"

    # Define a sliding square window to iterate on the image
    stride = int(sample_dim/2)
    slopes = []
    for start_x in range(0, image.shape[0]-sample_dim+1, stride):
        for start_y in range(0, image.shape[1]-sample_dim+1, stride):
            sample = image[start_x: start_x+sample_dim, start_y: start_y + sample_dim]
            slopes.append( get_pspec(sample, bin_range=fft_range, color_model=False) )
    return slopes


### FFT
CSV_FFT_BINS = "fft_bins.csv"
COL_FREQ_F = "frequency_F"
COL_AMPL_F = "amplitude_F"
class FFT_bins(FFTMetrics):
    def function(self, image):
        assert image.ndim==2, "Image should be 2D"
        assert image.shape[0]>=self.sample_dim<=image.shape[1], "sample dim should be less or equal than image shapes"
        df = pd.DataFrame()
        params = self.preprocess.get_params()

        # Define a sliding square window to iterate on the image
        stride = int(self.sample_dim/2)
        slopes = []
        idx = 0
        for start_x in range(0, image.shape[0]-self.sample_dim+1, stride):
            for start_y in range(0, image.shape[1]-self.sample_dim+1, stride):
                sample = image[start_x: start_x+self.sample_dim, start_y: start_y + self.sample_dim]
                bins, ampl = get_pspec(sample, bin_range=self.fft_range, return_bins=True, color_model=False)
                for f, a in zip(bins, ampl):
                    df.loc[idx, params.columns] = params.iloc[0]
                    df.loc[idx, [COL_F_SAMPLE_IDX, COL_FREQ_F, COL_AMPL_F, COL_F_WIN_SIZE, COL_FFT_RANGE_MIN, COL_FFT_RANGE_MAX]] = [idx, f, a, self.sample_dim, *self.fft_range]
                    idx+=1
        return df

    def visualize(self):
        data_image = pd.read_csv(os.path.join(DIR_RESULTS, CSV_IMAGE), index_col=0)
        merge_data = self.data.merge(data_image, on=COL_IMG_PATH)
        sns.set_palette(sns.color_palette(FLAT_UI))
        g = sns.relplot(data=merge_data, x=COL_FREQ_F, y=COL_AMPL_F, hue=COL_DIRECTORY, kind="line")
        g.set(xscale="log")
        g.set(yscale="log")
        plt.show()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path of the image file to open")
    args = parser.parse_args()
    image_path = os.path.abspath(args.image)

    preprocess = Preprocess(img_type=IMG.DARTER, img_channel=CHANNEL.GRAY, resize=(512,1536))

    # mean_slope = MeanFFTSlope(fft_range=[10,80], sample_dim=120, preprocess=preprocess, path=os.path.join(DIR_RESULTS, CSV_MEAN_FFT_SLOPE))
    # mean_slope.metric_from_df(image_path)
    # mean_slope.save()
    # mean_slope.load()
    # mean_slope.visualize()

    fft_bins = FFT_bins(fft_range=[10,110], sample_dim=512, preprocess=preprocess, path=os.path.join(DIR_RESULTS, CSV_FFT_BINS))
    # fft_bins.metric_from_df(image_path)
    # fft_bins.save()
    fft_bins.load()
    fft_bins.visualize()
