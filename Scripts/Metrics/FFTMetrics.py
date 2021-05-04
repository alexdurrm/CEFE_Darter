import cv2
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from MotherMetric import MotherMetric
from Preprocess import *
from FourierAnalysisMaster.pspec import get_pspec
from config import *

CSV_NAME=""
COL_F_WIN_SIZE="window_size_F"
COL_FFT_RANGE_MIN="freq_range_Min_F"
COL_FFT_RANGE_MAX="freq_range_Max_F"
class FFTMetrics(MotherMetric):
	def __init__(self, fft_range, sample_dim, *args, **kwargs):
		self.fft_range = fft_range
		self.sample_dim = sample_dim
		super().__init__(*args, **kwargs)
		self.data.index.name = CSV_NAME

###FFT SLOPE
CSV_FFT_SLOPE_NAME="slope fft"
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

	def visualize(self):
		'''
		used to plot the slopes of the images
		'''
		data_image = pd.read_csv(os.path.join(DIR_RESULTS, CSV_IMAGE), index_col=0)
		merge_data = self.data.merge(data_image, on=COL_IMG_PATH)
		sns.set_palette(sns.color_palette(FLAT_UI))
		ax = sns.relplot(x=COL_F_SAMPLE_IDX, y=COL_F_SLOPE_SAMPLE, data=merge_data,
						units=COL_IMG_PATH,
						# col= ,
						hue=COL_DIRECTORY,
						#split=True,
						kind="line")

		ax.set_ylabels('Slope of Fourier Power Spectrum ')
		ax.set_yticklabels(fontstyle='italic')
		plt.xticks(rotation=45)
		plt.title("mean Fourrier slopes per folder")
		plt.show()


#### MEAN_FFT_SLOPES
CSV_MEAN_FFT_SLOPE_NAME="mean fft slope"
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
CSV_FFT_BINS_NAME="bins fft"
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
	#DEFAULT PARAMETERS
	NORMALIZE=False
	STANDARDIZE=False
	INPUT_SHAPE=(512,1536)
	IMG_TYPE=IMG.DARTER
	IMG_CHANNEL=CHANNEL.GRAY
	SAMPLE_DIM=120
	FFT_RANGE=[10,110]
	VERBOSE=1

	#parsing parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("action", help="type of action needed", choices=["visu", "work"])
	parser.add_argument("metric", choices=["slope", "mean_slope", "fft"], help="type of metric: slope get many slopes coefficient per image, mean_slope mean those slope values into 1, fft does not calculate the slope coefficient and return the whole line")
	parser.add_argument("input_path", help="path of the image file to open")
	parser.add_argument("-o", "--output_dir", default=DIR_RESULTS, help="directory where to put the csv output, default: {}".format(DIR_RESULTS))
	parser.add_argument("-x", "--resize_X", default=INPUT_SHAPE[0], type=int, help="shape to resize image x, default: {}".format(INPUT_SHAPE[0]))
	parser.add_argument("-y", "--resize_Y", default=INPUT_SHAPE[1], type=int, help="shape to resize image y, default: {}".format(INPUT_SHAPE[1]))
	parser.add_argument("-d", "--sample_dim", default=SAMPLE_DIM, type=int, help="size of the squared sample cropped from the image to calculate the fft, default: {}".format(SAMPLE_DIM))
	parser.add_argument("-m", "--fft_range_min", default=FFT_RANGE[0], type=int, help="minimum freq used for fft, default: {}".format(FFT_RANGE[0]))
	parser.add_argument("-M", "--fft_range_max", default=FFT_RANGE[1], type=int, help="maximum freq used for fft, default: {}".format(FFT_RANGE[1]))
	parser.add_argument("-n", "--normalize", default=NORMALIZE, type=lambda x: bool(eval(x)), help="if image should be normalized, default: {}".format(NORMALIZE))
	parser.add_argument("-s", "--standardize", default=STANDARDIZE, type=lambda x: bool(eval(x)), help="if image should be standardized, default: {}".format(STANDARDIZE))
	parser.add_argument("-v", "--verbose", default=VERBOSE, type=int, choices=[0,1,2], help="set the level of visualization, default: {}".format(VERBOSE))
	parser.add_argument("-t", "--type_img", default=IMG_TYPE.name, type=lambda x: IMG[x], choices=list(IMG), help="the type of image needed, default: {}".format(IMG_TYPE))
	parser.add_argument("-c", "--channel_img", default=IMG_CHANNEL.name, type=lambda x: CHANNEL[x], choices=list(CHANNEL), help="The channel used for the image, default: {}".format(IMG_CHANNEL))

	args = parser.parse_args()
	input_shape = (args.resize_X, args.resize_Y)
	fft_range = [args.fft_range_min, args.fft_range_max]
	path = os.path.abspath(args.input_path)

	#preparing the metric
	preprocess = Preprocess(resizeX=input_shape[0], resizeY=input_shape[1], normalize=args.normalize, standardize=args.standardize,
		img_type=args.type_img, img_channel=args.channel_img)

	if args.metric == "slope":
		CSV_NAME = CSV_FFT_SLOPE_NAME
		file_path = os.path.join(args.output_dir, CSV_FFT_SLOPE)
		metric = FFTSlopes(fft_range=fft_range, sample_dim=args.sample_dim, preprocess=preprocess)
	elif args.metric=="mean_slope":
		CSV_NAME = CSV_MEAN_FFT_SLOPE_NAME
		file_path = os.path.join(args.output_dir, CSV_MEAN_FFT_SLOPE)
		metric = MeanFFTSlope(fft_range=fft_range, sample_dim=args.sample_dim, preprocess=preprocess)
	elif args.metric=="fft":
		CSV_NAME = CSV_FFT_BINS_NAME
		file_path = os.path.join(args.output_dir, CSV_FFT_BINS)
		metric = FFT_bins(fft_range=fft_range, sample_dim=args.sample_dim, preprocess=preprocess)

	if args.action == "visu":
		metric.load(file_path)
		metric.visualize()
	elif args.action=="work":
		metric.metric_from_csv(path)
		metric.save(file_path)
