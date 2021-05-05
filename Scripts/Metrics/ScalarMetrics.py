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
CSV_STATS_METRICS="statistical_metrics.csv"
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
	'''
	Class used to calculate the ratio between 2 color channels of an image
	'''

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
	#default parameters
	RESIZE=(None, None)
	NORMALIZE=False
	STANDARDIZE=False
	IMG_TYPE=IMG.DARTER
	IMG_CHANNEL=CHANNEL.GRAY
	VERBOSE=1
	DISTANCES=[2,4]
	ANGLES=[0,45,90,135]

	#parsing parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("action", help="type of action needed", choices=["visu", "work"])
	parser.add_argument("metric", choices=["c_ratio", "moments"], help="metric to call,\"moments returns the statistical moments of the image\" and \"c_ratio\" gives the ratio between the two color channels (works only for channels=2)")
	parser.add_argument("input_path", help="path of the image file to open")
	parser.add_argument("-o", "--output_dir", default=DIR_RESULTS, help="directory where to put the csv output, default: {}".format(DIR_RESULTS))
	parser.add_argument("-d", "--distances", nargs="+", type=int, default=DISTANCES, help="frequencies used in GLCM, default: {}".format(DISTANCES))
	parser.add_argument("-a", "--angles", nargs="+", type=int, default=ANGLES, help="Angles used in GLCM, default: {}".format(ANGLES))
	parser.add_argument("-x", "--resize_X", default=RESIZE[0], type=int, help="shape to resize image x, default: {}".format(RESIZE[0]))
	parser.add_argument("-y", "--resize_Y", default=RESIZE[1], type=int, help="shape to resize image y, default: {}".format(RESIZE[1]))
	parser.add_argument("-n", "--normalize", default=NORMALIZE, type=lambda x: bool(eval(x)), help="if image should be normalized, default: {}".format(NORMALIZE))
	parser.add_argument("-s", "--standardize", default=STANDARDIZE, type=lambda x: bool(eval(x)), help="if image should be standardized, default: {}".format(STANDARDIZE))
	parser.add_argument("-v", "--verbose", default=VERBOSE, type=int, choices=[0,1,2], help="set the level of visualization, default: {}".format(VERBOSE))
	parser.add_argument("-t", "--type_img", default=IMG_TYPE.name, type=lambda x: IMG[x], choices=list(IMG), help="the type of image needed, default: {}".format(IMG_TYPE))
	parser.add_argument("-c", "--channel_img", default=IMG_CHANNEL.name, type=lambda x: CHANNEL[x], choices=list(CHANNEL), help="The channel used for the image, default: {}".format(IMG_CHANNEL))
	args = parser.parse_args()
	path = os.path.abspath(args.input_path)

	#prepare metric
	preprocess = Preprocess(resizeX=args.resize_X, resizeY=args.resize_Y, normalize=args.normalize, standardize=args.standardize, img_type=args.type_img, img_channel=args.channel_img)
	if args.metric=="c_ratio":
		file_path = os.path.join(args.output_dir, CSV_COLOR_RATIO)
		metric = ColorRatioMetrics(preprocess)
	elif args.metric=="moments":
		file_path = os.path.join(args.output_dir, CSV_STATS_METRICS)
		metric = StatMetrics(preprocess)

	if args.action == "visu":
		metric.load(file_path)
		metric.visualize()
	elif args.action=="work":
		metric.metric_from_csv(path)
		metric.save(file_path)
