import cv2
import pandas as pd
import os

from MotherMetric import MotherMetric
from Preprocess import *
from pspec import get_FFT_slopes, get_pspec


COL_F_WIN_SIZE="window_size_F"
COL_FFT_RANGE_MIN="freq_range_Min_F"
COL_FFT_RANGE_MAX="freq_range_Max_F"
class FFTMetrics(MotherMetric):
    def __init__(self, fft_range, sample_dim, standardize=True, preprocess=None):
        if not preprocess:
            preprocess = Preprocess(img_type=IMG.DARTER, img_channel=CHANNEL.GRAY)
        super().__init__(preprocess)
        self.fft_range = fft_range
        self.sample_dim = sample_dim

###FFT SLOPE
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
COL_F_MEAN_SLOPE = "mean_fourier_slope"
COL_F_N_SAMPLE = "samples_used_F"
class MeanFFTSlope(FFTMetrics):        
    def function(self, image):
        df = self.preprocess.get_params()
        slopes = get_FFT_slopes(image, self.fft_range, self.sample_dim)
        df.loc[0, [COL_F_MEAN_SLOPE, COL_F_N_SAMPLE, COL_F_WIN_SIZE, COL_FFT_RANGE_MIN, COL_FFT_RANGE_MAX]] = [np.mean(slopes), len(slopes), self.sample_dim, *self.fft_range]
        return df

### FFT
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
                for freq, ampl in zip(bins, ampl):
                    df.loc[idx, [COL_FREQ_F, COL_AMPL_F, COL_F_WIN_SIZE, COL_FFT_RANGE_MIN, COL_FFT_RANGE_MAX]] = [freq, ampl, self.sample_dim, *self.fft_range]
                    idx+=1
        return df


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path of the image file to open")
    args = parser.parse_args()
    image_path = os.path.abspath(args.image)
    
    fft_bins = FFT_bins(fft_range=[10,80], sample_dim=120)
    mean_slope = MeanFFTSlope(fft_range=[10,80], sample_dim=120)
    fft_slopes = FFTSlopes(fft_range=[10,80], sample_dim=120)
    
    print( fft_bins(image_path) )
    print( mean_slope(image_path) )
    print( fft_slopes(image_path) )