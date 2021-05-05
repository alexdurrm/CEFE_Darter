import tensorflow.keras as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
import pandas as pd
import numpy as np
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt

from Preprocess import *
from config import *
from MotherMetric import MotherMetric

from ScalarMetrics import get_gini


#deep features
CSV_DEEP_FEATURES="deep_features.csv"
COL_MODEL_NAME="model_name_deep_features"
COL_SPARSENESS_DF="gini_deep_features"
COL_ENTROPY_DF="deep_feature_entropy"
COL_KURTOSIS_DF="deep_feature_kurtosis"
COL_LAYER_DF="layer_deep_feature"
class DeepFeatureMetrics(MotherMetric):
	"""
	DeepFeatureMetrics is a class used to calculate and store
	different metrics derived from the neurons activation of a specific neural net
	to a given image 
	"""
	def __init__(self, base_model, input_shape, *args, **kwargs):
		self.base_model = base_model
		self.input_shape = input_shape
		input_tensor = K.Input(shape=self.input_shape)
		self.base_model.layers[0] = input_tensor
		self.deep_features = K.Model(inputs=self.base_model.input, outputs=[l.output for l in self.base_model.layers[1:]])
		super().__init__(*args, **kwargs)

	def get_deep_features(self, image, visu=False):
		'''
		get the feature space of an image propagated through the deep feature model
		return a list of np array, each element of the list represent an output of a layer, input layer is ignored
		'''
		image = (image - np.mean(image, axis=(0,1))) / np.std(image, axis=(0,1))
		#make the prediction
		pred = self.deep_features.predict(image[np.newaxis, ...])
		if visu:
			self.deep_features.summary()
			for p in pred:
				print(type(p))
				print(p.shape)
		return pred

	def get_layers_gini(self, image):
		deep_features = self.get_deep_features(image)
		sparseness=[get_gini(f[0]) for f in deep_features]
		return sparseness

	def get_layers_stats(self, image):
		deep_features = self.get_deep_features(image)
		sparseness=[get_gini(f[0]) for f in deep_features]
		kurtosis = [kurtosis(f[0], axis=None)]
		entropy = [entropy(f[0], axis=None)]
		return sparseness, kurtosis, entropy

	def function(self, image):
		gini = self.get_layers_gini(image)
		params = self.preprocess.get_params()

		df = pd.DataFrame()
		for layer_idx , gini in enumerate(gini):
			df.loc[layer_idx, params.columns] = params.iloc[0]
			df.loc[layer_idx, [COL_MODEL_NAME, COL_SPARSENESS_DF, COL_LAYER_DF]] = [self.base_model.name, gini, layer_idx]
		return df

	def visualize(self):
		'''
		plot for each image the gini coefficient of each network layer
		'''
		sns.set_palette(sns.color_palette(FLAT_UI))

		data_image = pd.read_csv(os.path.join(DIR_RESULTS, CSV_IMAGE), index_col=0)
		merge_data = self.data.merge(data_image, on=COL_IMG_PATH)

		sns.relplot(data=merge_data, x=COL_LAYER_DF, y=COL_SPARSENESS_DF, hue=COL_DIRECTORY, col=COL_MODEL_NAME, kind="line", units=COL_IMG_PATH, estimator=None, alpha=0.25)
		plt.show()

		sns.relplot(data=merge_data, x=COL_LAYER_DF, y=COL_ENTROPY_DF, hue=COL_DIRECTORY, col=COL_MODEL_NAME, kind="line", units=COL_IMG_PATH, estimator=None, alpha=0.25)
		plt.show()

		sns.relplot(data=merge_data, x=COL_LAYER_DF, y=COL_KURTOSIS_DF, hue=COL_DIRECTORY, col=COL_MODEL_NAME, kind="line", units=COL_IMG_PATH, estimator=None, alpha=0.25)
		plt.show()

if __name__ == '__main__':

	#DEFAULT PARAMETERS
	INPUT_SHAPE = (1500, 512)
	NORMALIZE=True
	STANDARDIZE=False
	IMG_TYPE=IMG.RGB
	IMG_CHANNEL=CHANNEL.ALL
	VERBOSE=1
	DEFAULT_NET="vgg16"

	#parameter parsing
	parser = argparse.ArgumentParser()
	parser.add_argument("action", help="type of action needed", choices=["visu", "work"])
	parser.add_argument("path_input", help="path of the image file to open")
	parser.add_argument("-m", "--model", type=str, choices=["vgg16", "vgg19"], default=DEFAULT_NET, help="Neural network used to get the deep features, can be a path or a name like \"vgg16\" or \"vgg19\", default: {}".format(DEFAULT_NET))
	parser.add_argument("-x", "--input_X", default=INPUT_SHAPE[0], type=int, help="shape to resize image X, default: {}".format(INPUT_SHAPE[0]))
	parser.add_argument("-y", "--input_Y", default=INPUT_SHAPE[0], type=int, help="shape to resize image y, default: {}".format(INPUT_SHAPE[1]))
	parser.add_argument("-o", "--output_dir", default=DIR_RESULTS, help="directory where to put the csv output, default: {}".format(DIR_RESULTS))
	parser.add_argument("-n", "--normalize", default=NORMALIZE, type=lambda x: bool(eval(x)), help="if image should be normalized, default: {}".format(NORMALIZE))
	parser.add_argument("-s", "--standardize", default=STANDARDIZE, type=lambda x: bool(eval(x)), help="if image should be standardized, default: {}".format(STANDARDIZE))
	parser.add_argument("-v", "--verbose", default=VERBOSE, type=int, choices=[0,1,2], help="set the level of visualization, default: {}".format(VERBOSE))
	parser.add_argument("-t", "--type_img", default=IMG_TYPE.name, type=lambda x: IMG[x], choices=list(IMG), help="the type of image needed, default: {}".format(IMG_TYPE))
	parser.add_argument("-c", "--channel_img", default=IMG_CHANNEL.name, type=lambda x: CHANNEL[x], choices=list(CHANNEL), help="The channel used for the image, default: {}".format(IMG_CHANNEL))
	args = parser.parse_args()

	input_shape = (args.input_X, args.input_Y)

	#prepare metric
	preprocess = Preprocess(resizeX=input_shape[0], resizeY=input_shape[1], normalize=args.normalize, standardize=args.standardize, img_type=args.type_img, img_channel=args.channel_img)
	filepath = os.path.join(args.output_dir, CSV_DEEP_FEATURES)

	if args.model=="vgg16":
		chosen_model = VGG16(weights='imagenet', include_top=False)
	elif args.model == "vgg19":
		chosen_model = VGG19(weights='imagenet', include_top=False)
	else:
		chosen_model = K.models.load_model(args.model)
	metric = DeepFeatureMetrics( chosen_model, input_shape, preprocess)


	if args.action == "visu":
		metric.load(filepath)
		metric.visualize()
	elif args.action=="work":
		metric.metric_from_csv(args.path_input)
		metric.save(filepath)
