import pandas as pd
import os
import argparse
from skimage.filters import gabor, gabor_kernel
import matplotlib.pyplot as plt
import seaborn as sns

from MotherMetric import MotherMetric
from Preprocess import *
from config import *

#GABOR
CSV_GABOR="gabor.csv"
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

	def visualize(self):
		data_image = pd.read_csv(os.path.join(DIR_RESULTS, CSV_IMAGE), index_col=0)
		merge_data = self.data.merge(data_image, on=COL_IMG_PATH)

		sns.relplot(data=merge_data, y=COL_GABOR_VALUES, x=COL_DIRECTORY, col=COL_GABOR_ANGLES, row=COL_GABOR_FREQ)
		plt.show()

		#plot the mean reaction per gabor filter
		i=1
		splitted_by_dir = merge_data.groupby([COL_DIRECTORY])
		for category, split_data in splitted_by_dir:
			mean_data = split_data.groupby([COL_GABOR_ANGLES, COL_GABOR_FREQ]).agg({COL_GABOR_VALUES:np.mean})
			mean_data.reset_index(level=(0,1), inplace=True)
			map = mean_data.pivot(COL_GABOR_FREQ, COL_GABOR_ANGLES, COL_GABOR_VALUES)

			ax = plt.subplot(1,len(splitted_by_dir),i)
			ax.set_title(category)
			sns.heatmap(ax=ax, data=map, cbar=True, square=True)
			i+=1
		plt.show()

		#plot the gabor filters
		frequencies = merge_data[COL_GABOR_FREQ].unique()
		angles = merge_data[COL_GABOR_ANGLES].unique()
		f, axs = plt.subplots(len(frequencies), len(angles), squeeze=False)
		for f, freq in enumerate(frequencies):
			for a, angle in enumerate(angles):
				gk = gabor_kernel(frequency=freq, theta=np.radians(angle))
				axs[f,a].imshow(gk.real)
				axs[f,a].set_title("frequency:{}, angle:{}".format(freq, angle))

		plt.show()



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
	#default parameters
	RESIZE=(None, None)
	NORMALIZE=False
	STANDARDIZE=False
	IMG_TYPE=IMG.DARTER
	IMG_CHANNEL=CHANNEL.GRAY
	VERBOSE=1
	FREQUENCIES=[0.2, 0.4, 0.8]
	ANGLES=[0,45,90,135]

	#parsing parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("action", help="type of action needed", choices=["visu", "work"])
	parser.add_argument("input_path", help="path of the image file to open")
	parser.add_argument("-o", "--output_dir", default=DIR_RESULTS, help="directory where to put the csv output, default: {}".format(DIR_RESULTS))
	parser.add_argument("-f", "--frequencies", nargs="+", type=float, default=FREQUENCIES, help="frequencies used in gabor filter, default: {}".format(FREQUENCIES))
	parser.add_argument("-a", "--angles", nargs="+", type=int, default=ANGLES, help="Angle used in gabor filter, default: {}".format(ANGLES))
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
	file_path = os.path.join(args.output_dir, CSV_GABOR)
	metric = GaborMetrics(angles=args.angles, frequencies=args.frequencies, preprocess=preprocess)

	if args.action == "visu":
		metric.load(file_path)
		metric.visualize()
	elif args.action=="work":
		metric.metric_from_csv(path)
		metric.save(file_path)
