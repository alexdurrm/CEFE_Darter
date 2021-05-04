import pandas as pd
import os
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

from MotherMetric import MotherMetric
from Preprocess import *
from PHOG.anna_phog import anna_phog
from config import *

#PHOG
CSV_PHOG_NAME="pyramidal hog"
CSV_PHOG="phog.csv"
COL_PHOG_LEVELS="phog_level"
COL_PHOG_ORIENTATIONS="phog_bins"
COL_PHOG_VALUE="phog_val"
COL_PHOG_BIN="phog_bin"
class PHOGMetrics(MotherMetric):
	def __init__(self, orientations=8, level=0, *args, **kwargs):
		self.orientations=orientations
		self.level=level
		super().__init__(*args, **kwargs)
		self.data.index.name=CSV_PHOG_NAME

	def function(self, image):
		df = pd.DataFrame()
		params = self.preprocess.get_params()

		roi = [0, image.shape[0], 0, image.shape[1]]
		phog = anna_phog(image.astype(np.uint8), self.orientations, 360, self.level, roi)

		for i, value in enumerate(phog):
			df.loc[i, params.columns] = params.loc[0]
			df.loc[i, [COL_PHOG_BIN, COL_PHOG_ORIENTATIONS, COL_PHOG_LEVELS, COL_PHOG_VALUE]] = [i, self.orientations, self.level, value]
		return df

	def visualize(self):
		data_image = pd.read_csv(os.path.join(DIR_RESULTS, CSV_IMAGE), index_col=0)
		merge_data = self.data.merge(data_image, on=COL_IMG_PATH)
		sns.relplot(data=merge_data, x=COL_PHOG_BIN, y=COL_PHOG_VALUE,
			hue=COL_DIRECTORY, col=COL_PHOG_LEVELS, units=COL_IMG_PATH, estimator=None)
		plt.show()


if __name__=='__main__':
	#default parameters
	RESIZE=(None, None)
	NORMALIZE=False
	STANDARDIZE=False
	IMG_TYPE=IMG.RGB
	IMG_CHANNEL=CHANNEL.GRAY
	VERBOSE=1
	ORIENTATIONS=8
	LEVEL=2

	#parsing parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("action", help="type of action needed", choices=["visu", "work"])
	parser.add_argument("input_path", help="path of the image file to open")
	parser.add_argument("-a", "--angles", default=ORIENTATIONS, type=int, help="Number of angles to use, default: {}".format(ORIENTATIONS))
	parser.add_argument("-l", "--level", default=LEVEL, type=int, help="Number of levels for the pyramid of histogram oriend gradients, default: {}".format(LEVEL))
	parser.add_argument("-o", "--output_dir", default=DIR_RESULTS, help="directory where to put the csv output, default: {}".format(DIR_RESULTS))
	parser.add_argument("-x", "--resize_X", default=RESIZE[0], type=int, help="shape to resize image x, default: {}".format(RESIZE[0]))
	parser.add_argument("-y", "--resize_Y", default=RESIZE[1], type=int, help="shape to resize image y, default: {}".format(RESIZE[1]))
	parser.add_argument("-n", "--normalize", default=NORMALIZE, type=lambda x: bool(eval(x)), help="if image should be normalized, default: {}".format(NORMALIZE))
	parser.add_argument("-s", "--standardize", default=STANDARDIZE, type=lambda x: bool(eval(x)), help="if image should be standardized, default: {}".format(STANDARDIZE))
	parser.add_argument("-v", "--verbose", default=VERBOSE, type=int, choices=[0,1,2], help="set the level of visualization, default: {}".format(VERBOSE))
	parser.add_argument("-t", "--type_img", default=IMG_TYPE.name, type=lambda x: IMG[x], choices=list(IMG), help="the type of image needed, default: {}".format(IMG_TYPE))
	parser.add_argument("-c", "--channel_img", default=IMG_CHANNEL.name, type=lambda x: CHANNEL[x], choices=list(CHANNEL), help="The channel used for the image, default: {}".format(IMG_CHANNEL))
	args = parser.parse_args()
	resize = (args.resize_X, args.resize_Y)
	path = os.path.abspath(args.input_path)

	#prepare metric
	preprocess = Preprocess(resizeX=args.resize_X, resizeY=args.resize_Y, normalize=args.normalize, standardize=args.standardize, img_type=args.type_img, img_channel=args.channel_img)
	file_path = os.path.join(args.output_dir, CSV_PHOG)
	metric = PHOGMetrics(orientations=args.angles, level=args.level, preprocess=preprocess)

	if args.action == "visu":
		metric.load(file_path)
		metric.visualize()
	elif args.action=="work":
		metric.metric_from_csv(path)
		metric.save(file_path)
