from scipy.stats import kurtosis, entropy
import pandas as pd
import os
import numpy as np
import argparse

from Utils.Preprocess import *
from Utils.FileManagement import *
from Metrics.ImageMetrics import *
from Metrics.PHOG.anna_phog import anna_phog
from Metrics.Fourier import get_pspec, get_Fourier_slope
from Metrics.DeepNetworks import get_deep_features
from AutoEncoders.CommonAE import get_MSE

###############################################################################
#
#		Mother metric is the parent class for data processing of this poject
#
###############################################################################

class MotherMetric:
	'''
	Parent class for each metric, takes care of the data processing
	'''
	def __init__(self, preprocess=None, load_from=None):
		if preprocess:
			self.preprocess = preprocess
		else:
			self.preprocess = Preprocess()
		#if given a valid path load the data
		self.data = pd.DataFrame()
		if load_from:
			self.load(load_from)

	def __call__(self, path=None, *args, **kwargs):
		'''
		prepare the image and call the function on it, then store the result
		'''
		if not path:
			image = self.preprocess.get_image()
		else:
			image = self.preprocess(path)
		df = self.function(image, *args, **kwargs)
		self.data = self.data.append(df, ignore_index=True)
		return df

	def metric_from_path_list(self, path_list, *args, **kwargs):
		'''
		calculate the metrics for every path in the given list of images
		'''
		for path in path_list:
			_ = self.__call__(path, *args, **kwargs)

	def load(self, data_path):
		'''
		load a csv_file as object data
		'''
		try:
			self.data = pd.read_csv(data_path, index_col=0)
		except FileNotFoundError:
			print("nothing to load, continuing with current data")

	def save(self, output_path):
		'''
		save the data as csv, will erase previous
		'''
		#if the directory to store results do not exist create it
		directory = os.path.dirname(output_path)
		if not os.path.exists(directory):
			os.makedirs(directory)
		self.data.to_csv(output_path, index=True)
		print("saved at {}".format(output_path))

	def clear(self):
		'''
		clear the data
		'''
		del self.data
		self.data = pd.DataFrame()

	def function(self):
		raise NotImplementedError

###############################################################################
#
#								AUTOENCODER
#
###############################################################################

# COL_MODEL_NAME_AE="autoencoder_name"
#
# COL_ITERATION_AE="prediction_interation"
#
# COL_GINI_AE="gini_pxl_space"
# COL_KURTO_AE="kurtois_pxl_space"
# COL_ENTRO_AE="entropy_pxl_space"
#
# COL_GINI_LATENT_AE="gini_latent_space"
# COL_KURTO_LATENT_AE="kurtois_latent_space"
# COL_ENTRO_LATENT_AE="entropy_latent_space"
# COL_MEAN_ACTIVATION_AE="mean_activation_latent_space"
#
# COL_LATENT_DIST_AE="latent_distance_to_start"
#
# COL_MSE_AE="MSE_pxl_comparerd_to_start"
# COL_SSIM_AE="SSIM_pxl_compared_to_start"
#
# class AutoencoderMetrics(MotherMetric):
# 	def __init__(self, model_path, *args, **kwargs):
# 		import tensorflow.keras as K
# 		self.model = K.models.load_model(model_path, compile=False)
# 		model.compile("Adam", "mse")
# 		super().__init__(*args, **kwargs)
#
# 	def function(self, image, verbose):
# 		df = pd.DataFrame()
# 		params = self.preprocess.get_params()
#
# 		i=0
# 		divergence_generator = autoencoder_generate_retro_prediction(self.model, image, 5)
# 		start_latent, start_pxl = next(divergence_generator)
# 		axis_latent = tuple(i for i in range(0, start_latent.ndim))
# 		for latent, prediction in divergence_generator:
# 			df.loc[i, params.columns] = params.iloc[0]
# 			df.loc[i, [COL_MODEL_NAME_AE, COL_ITERATION_AE]] = [self.model.name, i]
# 			df.loc[i, [COL_GINI_AE, COL_KURTO_AE, COL_ENTRO_AE]] = [get_gini(prediction), kurtosis(prediction, axis=None), entropy(prediction, axis=None)]
# 			df.loc[i, [COL_GINI_LATENT_AE, COL_KURTO_LATENT_AE, COL_ENTRO_LATENT_AE, COL_MEAN_ACTIVATION_AE]] = [get_gini(prediction), kurtosis(prediction, axis=None), entropy(prediction, axis=None), np.mean(prev_latent)]
# 			df.loc[i, [COL_MSE_AE, COL_SSIM_AE]] = [get_MSE(start_pxl, prediction), get_SSIM(start_pxl, prediction)]
# 			df.loc[i, [COL_LATENT_DIST_AE]] = [np.mean(np.square(start_latent - latent), axis=(axis_latent))]
# 			i+=1
# 		return df

###############################################################################
#
#								NEW AUTOENCODER
#
###############################################################################

COL_MODEL_NAME_AE="autoencoder_name"
COL_MSE_AE="mse_prediction"
COL_SSIM_AE="ssim_prediction"

COL_GINI_ACTIVATION_AE="gini_activation_layer"
COL_KURTO_ACTIVATION_AE="kurtois_activation_layer"
COL_ENTRO_ACTIVATION_AE="entropy_activation_layer"
COL_MEAN_ACTIVATION_AE="mean_activation_layer"
COL_L0_ACTIVATION_AE="L0_activation_layer"

class AutoencoderMetrics(MotherMetric):
	def __init__(self, model_path, *args, **kwargs):
		import tensorflow.keras as K
		self.model = K.models.load_model(model_path, compile=False)
		self.model.compile("Adam", "mse")
		super().__init__(*args, **kwargs)

	def function(self, image):
		df = pd.DataFrame()
		params = self.preprocess.get_params()
		df.loc[0, params.columns] = params.iloc[0]

		test = image[np.newaxis, ...]
		prediction = self.model.predict(test)[0]
		df.loc[0, [COL_MODEL_NAME, COL_MSE_AE, COL_SSIM_AE]] = [self.model.name, get_MSE(image, prediction), get_SSIM(image, prediction)]
		deep_features = get_deep_features(self.model.encoder, test)

		for i, layer in enumerate(deep_features):
			df.loc[0, COL_GINI_ACTIVATION_AE+"_"+str(i)] = get_gini(layer)
			df.loc[0, COL_KURTO_ACTIVATION_AE+"_"+str(i)] = kurtosis(layer, axis=None)
			df.loc[0, COL_ENTRO_ACTIVATION_AE+"_"+str(i)] = entropy(layer, axis=None)
			df.loc[0, COL_MEAN_ACTIVATION_AE+"_"+str(i)] = np.mean(layer, axis=None)
			df.loc[0, COL_L0_ACTIVATION_AE+"_"+str(i)] = get_L0(layer)
		return df

###############################################################################
#
#								DEEP FEATURES
#
###############################################################################

COL_MODEL_NAME="model_name_deep_features"
COL_LAYER_DF="layer_deep_feature"
COL_SPARSENESS_DF="gini_deep_features"
COL_KURTOSIS_DF="deep_feature_kurtosis"
COL_ENTROPY_DF="deep_feature_entropy"
COL_MEAN_DF="deep_feature_mean"
COL_STD_DF="deep_feature_std"
COL_SKEW_DF="deep_feature_skewness"
class DeepFeatureMetrics(MotherMetric):
	"""
	DeepFeatureMetrics is a class used to calculate and store
	different metrics derived from the neurons activation of a specific neural net
	to a given image
	"""
	def __init__(self, base_model, input_shape, *args, **kwargs):
		import tensorflow.keras as K
		if base_model == "vgg16":
			from tensorflow.keras.applications.vgg16 import VGG16
			self.base_model = VGG16(weights='imagenet', include_top=False)
		elif base_model == "vgg19":
			from tensorflow.keras.applications.vgg19 import VGG19
			self.base_model = VGG19(weights='imagenet', include_top=False)
		else:
			raise ValueError("base_model should be vgg16 or vgg19")
		self.input_shape = input_shape
		input_tensor = K.Input(shape=self.input_shape)
		self.base_model.layers[0] = input_tensor
		self.deep_features = K.Model(inputs=self.base_model.input, outputs=[l.output for l in self.base_model.layers[1:]])
		super().__init__(*args, **kwargs)

	def function(self, image):
		deep_feat = get_deep_features(self.base_model, image[np.newaxis,...])
		params = self.preprocess.get_params()
		df = pd.DataFrame()
		for layer_idx, layerfeatures in enumerate(deep_feat):
			df.loc[layer_idx, params.columns] = params.iloc[0]
			df.loc[layer_idx, [COL_MODEL_NAME, COL_LAYER_DF]] = [self.base_model.name, layer_idx]
			df.loc[layer_idx, [COL_SPARSENESS_DF, COL_MEAN_DF, COL_STD_DF, COL_SKEW_DF, COL_KURTOSIS_DF, COL_ENTROPY_DF]] = [get_gini(layerfeatures), *get_statistical_features(layerfeatures)]
		return df

###############################################################################
#
#								FFT SLOPES
#
###############################################################################

COL_F_WIN_SIZE="window_size_F"
COL_FFT_RANGE_MIN="freq_range_Min_F"
COL_FFT_RANGE_MAX="freq_range_Max_F"
class FFTMetrics(MotherMetric):
	def __init__(self, fft_range, sample_dim, *args, **kwargs):
		self.fft_range = fft_range
		self.sample_dim = sample_dim
		super().__init__(*args, **kwargs)

def get_FFT_slopes_samples(image, fft_range, sample_dim, verbose=1):
	'''
	Calculate the fourier slopes of a sample window of an image
	'''
	assert image.ndim==3 and image.shape[-1]==1, "Image should be 3D with only one channel, here {}".format(image.shape)
	image = image[..., 0]
	assert image.shape[0]>=sample_dim<=image.shape[1], "sample dim should be less or equal than image shapes, here {}, {}".format(sample_dim, image.shape)

	# Define a sliding square window to iterate on the image
	stride = int(sample_dim/2)
	slopes = []
	for sample in fly_over_image(image, [sample_dim, sample_dim], [stride, stride], return_coord=False):
		slopes.append( get_Fourier_slope(sample, bin_range=fft_range, kaiser=True, n_bins=20) )
	return slopes

COL_F_SLOPE_SAMPLE="slope_sample_F"
COL_F_SAMPLE_IDX="sample_idx_F"
### FFT_SLOPES
class FFTSlopes(FFTMetrics):
	def function(self, image):
		df = pd.DataFrame()
		params = self.preprocess.get_params()
		slopes = get_FFT_slopes_samples(image, self.fft_range, self.sample_dim)
		for idx, slope in enumerate(slopes):
			df.loc[idx, params.columns] = params.iloc[0]
			df.loc[idx, [COL_F_SAMPLE_IDX, COL_F_SLOPE_SAMPLE, COL_F_WIN_SIZE, COL_FFT_RANGE_MIN, COL_FFT_RANGE_MAX]] = [idx, slope, self.sample_dim, *self.fft_range]
		return df


###############################################################################
#
#								MEAN FFT SLOPE
#
###############################################################################

COL_F_MEAN_SLOPE = "mean_fourier_slope"
COL_F_N_SAMPLE = "samples_used_F"
class MeanFFTSlope(FFTMetrics):
	def function(self, image):
		df = self.preprocess.get_params()
		slopes = get_FFT_slopes_samples(image, self.fft_range, self.sample_dim)
		df.loc[0, [COL_F_MEAN_SLOPE, COL_F_N_SAMPLE, COL_F_WIN_SIZE, COL_FFT_RANGE_MIN, COL_FFT_RANGE_MAX]] = [np.mean(slopes), len(slopes), self.sample_dim, *self.fft_range]
		return df

###############################################################################
#
#									FFT BINS
#
###############################################################################

COL_FREQ_F = "frequency_F"
COL_AMPL_F = "amplitude_F"
class FFT_bins(FFTMetrics):
	def function(self, image):
		assert image.ndim==3 and image.shape[-1]==1, "Image should be 3D with only one channel, here {}".format(image.shape)
		image = image[..., 0]
		assert image.shape[0]>=self.sample_dim<=image.shape[1], "sample dim should be less or equal than image shapes"
		df = pd.DataFrame()
		params = self.preprocess.get_params()

		# Define a sliding square window to iterate on the image
		stride = int(self.sample_dim/2)
		slopes = []
		idx, sample_idx = 0, 0
		for sample in fly_over_image(image, [self.sample_dim, self.sample_dim], [stride, stride], return_coord=False):
			bins, ampl = get_pspec(sample, bin_range=self.fft_range)
			for f, a in zip(bins, ampl):
				df.loc[idx, params.columns] = params.iloc[0]
				df.loc[idx, [COL_F_SAMPLE_IDX, COL_FREQ_F, COL_AMPL_F, COL_F_WIN_SIZE, COL_FFT_RANGE_MIN, COL_FFT_RANGE_MAX]] = [sample_idx, f, a, self.sample_dim, *self.fft_range]
				idx+=1
			sample_idx+=1
		return df

###############################################################################
#
#									GABOR
#
###############################################################################

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
		idx=0
		for a, angle in enumerate(self.angles):
			for f, freq in enumerate(self.frequencies):
				df.loc[idx, params.columns] = params.loc[0]
				df.loc[idx, [COL_GABOR_ANGLES, COL_GABOR_FREQ, COL_GABOR_VALUES]] = [angle, freq, activation_map[a, f]]
				idx+=1
		return df

###############################################################################
#
#									HARALICK
#
###############################################################################

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
		assert image.shape[-1]==1, "given image for haralick descriptors should have 1 channel"
		image = image.astype(np.uint8)
		df = pd.DataFrame()
		params = self.preprocess.get_params()
		mean, var, corr, contrast, dissimil, homo, asm, energy, maxi, entro = get_Haralick_descriptors(image[:,:,0], self.distances, self.angles)

		idx = 0
		for d, distance in enumerate(self.distances):
			for a, angle in enumerate(self.angles):
				df.loc[idx, params.columns] = params.loc[0]
				df.loc[idx, [COL_GLCM_ANGLE, COL_GLCM_DIST]] = [angle, distance]
				df.loc[idx, [COL_GLCM_MEAN, COL_GLCM_VAR, COL_GLCM_CORR]] = [mean, var, corr]
				df.loc[idx, [COL_GLCM_CONTRAST, COL_GLCM_DISSIMIL, COL_GLCM_HOMO]] = [contrast, dissimil, homo]
				df.loc[idx, [COL_GLCM_ASM, COL_GLCM_ENERGY, COL_GLCM_MAXP, COL_GLCM_ENTROPY]] = [asm, energy, maxi, entro]
				idx+=1
		return df

###############################################################################
#
#									LBP
#
###############################################################################

COL_POINTS_LBP="points_LBP"
COL_RADIUS_LBP="radius_LBP"
COL_BIN_LBP="bin_val_LBP"
COL_COUNT_LBP="count_LBP_value"
class LBPHistMetrics(MotherMetric):
	def __init__(self, points, radius, nbins, *args, **kwargs):
		assert len(points)==len(radius), "points and radius are used zipped, should be the same length"
		self.points = points
		self.radius = radius
		self.nbins = nbins
		super().__init__(*args, **kwargs)

	def function(self, image, visu=False):
		'''
		calculates the Local Binary Pattern of a given image
		P is the number of neighbors points to use
		R is the radius of the circle around the central pixel
		visu is to visualise the result
		'''
		assert image.ndim==3 and image.shape[-1]==1, "for lbp image should be one channel"
		df = pd.DataFrame()
		params = self.preprocess.get_params()
		idx=0
		for P, R in zip(self.points, self.radius):
			lbp_image = local_binary_pattern(image[:,:,0], P, R)
			vals, bins = np.histogram(lbp_image, bins=self.nbins)
			for vbin, val in zip(bins, vals):
				df.loc[idx, params.columns] = params.iloc[0]
				df.loc[idx, [COL_POINTS_LBP, COL_RADIUS_LBP, COL_BIN_LBP, COL_COUNT_LBP]] = [P, R, vbin, val]
				idx+=1
			if visu:
				fig, (ax0, ax1, ax2) = plt.subplots(figsize=(6, 12), nrows=3)
				ax0.imshow(image, cmap='gray')
				ax0.set_title("original image")

				bar = ax1.imshow(lbp_image, cmap='gray')
				fig.colorbar(bar, ax=ax1, orientation="vertical")
				ax1.set_title("LBP with params P={} and R={}".format(P, R))

				ax2.hist(lbp_image.flatten(), bins=self.bins)
				ax2.set_title("lbp values histogram")
				plt.show()

		return df

###############################################################################
#
#								BEST LBP
#
###############################################################################

COL_RANK_LBP="rank_lbp_value"
COL_VALUE_LBP="value_LBP"
class BestLBPMetrics(MotherMetric):
	def __init__(self, points, radius, n_best=20, *args, **kwargs):
		assert len(points)==len(radius), "points and radius are used zipped, should be the same length"
		self.points = points
		self.radius = radius
		self.n_best = n_best
		super().__init__(*args, **kwargs)

	def function(self, image, visu=False):
		'''
		calculates the Local Binary Pattern of a given image
		P is the number of neighbors points to use
		R is the radius of the circle around the central pixel
		visu is to visualise the result
		return the path to the saved image
		'''
		df = pd.DataFrame()
		params = self.preprocess.get_params()
		idx=0
		for P, R in zip(self.points, self.radius):
			best_lbp = get_most_common_lbp(image, P, R, self.n_best)
			for rank, lbp in enumerate(best_lbp):
				lbp_val, lbp_count = lbp
				df.loc[idx, params.columns] = params.iloc[0]
				df.loc[idx, [COL_POINTS_LBP, COL_RADIUS_LBP, COL_RANK_LBP, COL_VALUE_LBP, COL_COUNT_LBP]] = [P, R, rank, lbp_val, lbp_count]
				idx+=1
		return df


###############################################################################
#
#									PHOG
#
###############################################################################

COL_PHOG_LEVELS="phog_level"
COL_PHOG_ORIENTATIONS="phog_bins"
COL_PHOG_VALUE="phog_val"
COL_PHOG_BIN="phog_bin"
class PHOGMetrics(MotherMetric):
	def __init__(self, orientations=8, level=0, *args, **kwargs):
		self.orientations=orientations
		self.level=level
		super().__init__(*args, **kwargs)

	def function(self, image):
		df = pd.DataFrame()
		params = self.preprocess.get_params()

		roi = [0, image.shape[0], 0, image.shape[1]]
		phog = anna_phog(image.astype(np.uint8), self.orientations, 360, self.level, roi)

		for i, value in enumerate(phog):
			df.loc[i, params.columns] = params.loc[0]
			df.loc[i, [COL_PHOG_BIN, COL_PHOG_ORIENTATIONS, COL_PHOG_LEVELS, COL_PHOG_VALUE]] = [i, self.orientations, self.level, value]
		return df

###############################################################################
#
#							STATISTICAL MOMENTS
#
###############################################################################

COL_STAT_MEAN="mean_stat"
COL_STAT_STD="std_stat"
COL_STAT_SKEW="skewness_stat"
COL_STAT_KURT="kurtosis_stat"
COL_STAT_ENTROPY="entropy_stat"
COL_GINI_VALUE="gini_coefficient"
class StatMetrics(MotherMetric):
	'''
	Class used to calculate a list of statistical moments of an image
	'''
	def function(self, image):
		df = self.preprocess.get_params()
		metrics = get_statistical_features(image)
		df.loc[0, [COL_STAT_MEAN, COL_STAT_STD, COL_STAT_SKEW, COL_STAT_KURT, COL_STAT_ENTROPY]] = metrics
		df.loc[0, COL_GINI_VALUE] = [get_gini(image)]
		return df

###############################################################################
#
#								COLOR RATIO
#
###############################################################################

COL_COLOR_RATIO="color_ratio"
class ColorRatioMetrics(MotherMetric):
	'''
	Class used to calculate the ratio between 2 color channels of an image
	'''
	def function(self, image, visu=False):
		df = self.preprocess.get_params()
		df.loc[0,COL_COLOR_RATIO]=[get_color_ratio(image, visu)]
		return df


###############################################################################
#
#									MAIN
#
###############################################################################

def main(args):
	params_preprocess={
		"resize":args.resize,
		"normalize":args.normalize,
		"standardize":args.standardize,
		"img_type":args.type_img,
		"img_channel":args.channel_img
	}
	preprocess = Preprocess(**params_preprocess)
	load_from = args.output_path if not args.override else None
	if args.command == "color_ratio":
		metric = ColorRatioMetrics(preprocess, load_from=load_from)
	elif args.command == "stats":
		metric = StatMetrics(preprocess, load_from=load_from)
	elif args.command == "phog":
		metric = PHOGMetrics(orientations=args.angles, level=args.level, preprocess=preprocess, load_from=load_from)
	elif args.command == "best_lbp":
		metric = BestLBPMetrics(points=args.points, radius=args.radius, n_best=args.bins, preprocess=preprocess, load_from=load_from)
	elif args.command == "lbp":
		metric = LBPHistMetrics(points=args.points, radius=args.radius, nbins=args.bins, preprocess=preprocess, load_from=load_from)
	elif args.command == "haralick":
		metric = HaralickMetrics(distances=args.distances, angles=args.angles, preprocess=preprocess, load_from=load_from)
	elif args.command == "gabor":
		metric = GaborMetrics(angles=args.angles, frequencies=args.frequencies, preprocess=preprocess, load_from=load_from)
	elif args.command == "fft_bins":
		metric = FFT_bins(fft_range=args.fft_range, sample_dim=args.sample_dim, preprocess=preprocess, load_from=load_from)
	elif args.command == "fft_slopes":
		metric = FFTSlopes(fft_range=args.fft_range, sample_dim=args.sample_dim, preprocess=preprocess, load_from=load_from)
	elif args.command == "mean_fft_slopes":
		metric = MeanFFTSlope(fft_range=args.fft_range, sample_dim=args.sample_dim, preprocess=preprocess, load_from=load_from)
	elif args.command == "deep_features":
		metric = DeepFeatureMetrics(args.model, args.resize, preprocess=preprocess, load_from=load_from)
	elif args.command == "autoencoder":
		metric = AutoencoderMetrics(args.model_path, preprocess=preprocess, load_from=load_from)
	elif args.command == "list":
		metric = get_files(args.input_path, args.depth, tuple(args.formats), [], only_endnodes=args.endnodes, visu=args.verbose>=1)
		metric.to_csv(args.output_path, index=True)

	if args.command != "list":	#list is the only metric that do not take a list image as input
		data_image = pd.read_csv(args.input_path, index_col=0)
		metric.metric_from_path_list(data_image[COL_IMG_PATH])
		metric.save(args.output_path)


if __name__ == '__main__':

	#DEFAULT PARAMETERS
	RESIZE=(None, None)
	NORMALIZE=False
	STANDARDIZE=False
	IMG_TYPE=IMG.RGB
	IMG_CHANNEL=CHANNEL.ALL
	VERBOSE=1
	OVERRIDE=True

	#parsing parameters
	parser = argparse.ArgumentParser(description="Launch the specified metric calculations on the given csv and store the results in another csv")

	#common parameters
	parser.add_argument("input_path", help="path of the file to open")
	parser.add_argument("output_path", help="where to save the csv")
	parser.add_argument("-v", "--verbose", default=VERBOSE, type=int, choices=[0,1,2], help="set the level of visualization, default: {}".format(VERBOSE))
	parser.add_argument("-o", "--override", default=OVERRIDE, type=lambda x: bool(eval(x)), help="If the output file already exists, override it, else append the results. default: {}".format(OVERRIDE))

	#parameters for preprocess
	parser.add_argument("-r", "--resize", type=int, nargs=2, default=RESIZE, help="resize image to this value, default is {}".format(RESIZE))
	parser.add_argument("-n", "--normalize", default=NORMALIZE, type=lambda x: bool(eval(x)), help="if image should be normalized, default: {}".format(NORMALIZE))
	parser.add_argument("-s", "--standardize", default=STANDARDIZE, type=lambda x: bool(eval(x)), help="if image should be standardized, default: {}".format(STANDARDIZE))
	parser.add_argument("-t", "--type_img", default=IMG_TYPE.name, type=lambda x: IMG[x], choices=list(IMG), help="the type of image needed, default: {}".format(IMG_TYPE))
	parser.add_argument("-c", "--channel_img", default=IMG_CHANNEL.name, type=lambda x: CHANNEL[x], choices=list(CHANNEL), help="The channel used for the image, default: {}".format(IMG_CHANNEL))
	subparsers = parser.add_subparsers(title="command", dest="command", help='action to perform')

	#commands with no other parameters
	cr_parser = subparsers.add_parser("color_ratio", description="take 2channel images and calculate the slope of the ratio C1/C2")
	stats_parser = subparsers.add_parser("stats", description="calculate statistical moments on the given images")

	#parameters for listing
	DEPTH = 2
	FORMATS = (".jpg",".png",".tif",".tiff")
	ONLY_ENDNODES = False
	list_parser = subparsers.add_parser("list", description="will list the images and retrives informations about them, then store into a csv")
	list_parser.add_argument("-d", "--depth", type=int, choices=[0, 1, 2, 3], default=DEPTH, help="Depth of the path searched, 0 is image, 1 is folder, 2 is subfolders ,etc. default: {}".format(DEPTH))
	list_parser.add_argument("-f", "--formats", default=FORMATS, nargs="+", type=str, help="Image extensions accepted, default: {}".format(FORMATS))
	list_parser.add_argument("-e", "--endnodes", default=ONLY_ENDNODES, type=lambda x: bool(eval(x)), help="If True accepts only images at endnode directories, else accept all encountered images, default: {}".format(ONLY_ENDNODES))

	#parameters for PHOG
	ORIENTATIONS=8
	LEVEL=2
	PHOG_parser = subparsers.add_parser("phog")
	PHOG_parser.add_argument("-a", "--angles", default=ORIENTATIONS, type=int, help="Number of angles to use, default: {}".format(ORIENTATIONS))
	PHOG_parser.add_argument("-l", "--level", default=LEVEL, type=int, help="Number of levels for the pyramid of histogram oriend gradients, default: {}".format(LEVEL))


	POINTS=[8,16]
	RADIUS=[2,4]
	#parameters for lbp
	N_BINS=256
	LBP_parser = subparsers.add_parser("lbp")
	LBP_parser.add_argument("-p", "--points", nargs="+", type=int, default=POINTS, help="Number of points used in pair with radius, default: {}".format(POINTS))
	LBP_parser.add_argument("-r", "--radius", nargs="+", type=int, default=RADIUS, help="Radius used in pair with number of points, default: {}".format(RADIUS))
	LBP_parser.add_argument("-b", "--bins",  type=int, default=N_BINS, help="Number of bins, default: {}".format(N_BINS))
	#parameters for best_lbp
	N_BEST_BINS=100
	best_LBP_parser = subparsers.add_parser("best_lbp")
	best_LBP_parser.add_argument("-p", "--points", nargs="+", type=int, default=POINTS, help="Number of points used in pair with radius, default: {}".format(POINTS))
	best_LBP_parser.add_argument("-r", "--radius", nargs="+", type=int, default=RADIUS, help="Radius used in pair with number of points, default: {}".format(RADIUS))
	best_LBP_parser.add_argument("-b", "--bins",  type=int, default=N_BEST_BINS, help="Number of best bins, default: {}".format(N_BEST_BINS))

	#parameters for haralick
	DISTANCES=[2,4]
	ANGLES=[0,45,90,135]
	haralick_parser = subparsers.add_parser("haralick")
	haralick_parser.add_argument("-d", "--distances", nargs="+", type=int, default=DISTANCES, help="frequencies used in GLCM, default: {}".format(DISTANCES))
	haralick_parser.add_argument("-a", "--angles", nargs="+", type=int, default=ANGLES, help="Angles used in GLCM, default: {}".format(ANGLES))

	#parameters for gabor
	FREQUENCIES=[0.2, 0.4, 0.8]
	GAB_ANGLES=[0,45,90,135]
	gabor_parser = subparsers.add_parser("gabor")
	gabor_parser.add_argument("-f", "--frequencies", nargs="+", type=float, default=FREQUENCIES, help="frequencies used in gabor filter, default: {}".format(FREQUENCIES))
	gabor_parser.add_argument("-a", "--angles", nargs="+", type=int, default=GAB_ANGLES, help="Angle used in gabor filter, default: {}".format(GAB_ANGLES))

	SAMPLE_DIM=120
	FFT_RANGE=[10,110]
	#parameters for FFT bins
	fft_bins_parser = subparsers.add_parser("fft_bins")
	fft_bins_parser.add_argument("-d", "--sample_dim", default=SAMPLE_DIM, type=int, help="size of the squared sample cropped from the image to calculate the fft, default: {}".format(SAMPLE_DIM))
	fft_bins_parser.add_argument("-f", "--fft_range", default=FFT_RANGE, type=int, nargs=2, help="minimum and maximum freq used for fft, default: {}".format(FFT_RANGE))
	#parameters for FFT slopes
	fft_slopes_parser = subparsers.add_parser("fft_slopes")
	fft_slopes_parser.add_argument("-d", "--sample_dim", default=SAMPLE_DIM, type=int, help="size of the squared sample cropped from the image to calculate the fft, default: {}".format(SAMPLE_DIM))
	fft_slopes_parser.add_argument("-f", "--fft_range", default=FFT_RANGE, type=int, nargs=2, help="minimum and maximum freq used for fft, default: {}".format(FFT_RANGE))
	#parameters for mean fft slopes
	mean_fft_slopes_parser = subparsers.add_parser("mean_fft_slopes")
	mean_fft_slopes_parser.add_argument("-d", "--sample_dim", default=SAMPLE_DIM, type=int, help="size of the squared sample cropped from the image to calculate the fft, default: {}".format(SAMPLE_DIM))
	mean_fft_slopes_parser.add_argument("-f", "--fft_range", default=FFT_RANGE, type=int, nargs=2, help="minimum and maximum freq used for fft, default: {}".format(FFT_RANGE))

	#parameters for DeepFeatures
	DEFAULT_NET="vgg16"
	df_parser = subparsers.add_parser("deep_features")
	df_parser.add_argument("-m", "--model", type=str, choices=["vgg16", "vgg19"], default=DEFAULT_NET, help="Neural network used to get the deep features, can be a path or a name like \"vgg16\" or \"vgg19\", default: {}".format(DEFAULT_NET))

	#parameters for autoencoder
	ae_parser = subparsers.add_parser("autoencoder")
	ae_parser.add_argument("model_path", help="path of the model used for inferrence")

	args = parser.parse_args()
	main(args)
