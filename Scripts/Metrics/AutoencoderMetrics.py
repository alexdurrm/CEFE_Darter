import tensorflow.keras as K
from tensorflow.keras.applications.vgg16 import VGG16
import pandas as pd
import numpy as np
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

from MotherMetric import MotherMetric
from Preprocess import *
from config import *

PREDICTION_SIZE=128

CSV_AE="metricAE_"
COL_MODEL_NAME_AE="autoencoder_name"
COL_PRED_SIZE="prediction_size"

COL_ITERATION_AE="prediction_interation"
COL_MSE_AE="MSE_compared_to_start"
COL_MSE_PREV_AE="MSE_compared_to_previous_step"
COL_LATENT_DIST_AE="latent_distance_to_start"
COL_LATENT_DIST_PREV_AE="latent_distance_to_previous_step"
COL_SSIM_PREV_AE="SSIM_compared_to_previous_step"

class AutoencoderMetrics(MotherMetric):
	def __init__(self, model_path, *args, **kwargs):
		self.model = K.models.load_model(model_path)
		super().__init__(*args, **kwargs)

	def visualize(self):
		data_image = pd.read_csv(os.path.join(DIR_RESULTS, CSV_IMAGE), index_col=0)

		# #plot differences in quality prediction for each network
		# data_networks = self.data.loc[self.data[COL_ITERATION_AE]==0]
		# data_merged = data_networks.merge(data_image, on=COL_IMG_PATH)

		# sns.catplot(data=data_merged, x=COL_SPECIES, y=COL_MSE_AE,
		# col=COL_MODEL_NAME_AE, row=COL_TYPE,
		# hue=COL_FISH_SEX, split=True, kind='violin')
		# plt.show()
		# sns.catplot(data=data_merged, x=COL_SPECIES, y=COL_LATENT_DIST_AE,
		# col=COL_MODEL_NAME_AE, row=COL_TYPE,
		# hue=COL_FISH_SEX, split=True, kind='violin')
		# plt.show()

		# sns.relplot(data=data_merged, x=COL_MODEL_NAME_AE, y=COL_MSE_AE,
		# col=COL_FISH_SEX, hue=COL_SPECIES)
		# plt.show()
		# sns.relplot(data=data_merged, x=COL_MODEL_NAME_AE, y=COL_LATENT_DIST_AE,
		# col=COL_FISH_SEX, hue=COL_SPECIES)
		# plt.show()


		#specific plots for the network
		data_network = self.data.loc[self.data[COL_MODEL_NAME_AE]==self.model.name]
		merge = data_network.merge(data_image, on=COL_IMG_PATH)
		merge_data = merge.loc[merge[COL_TYPE]==FILE_TYPE.ORIG_FISH.value]

		#plot violin dist by species
		sns.catplot(data=merge_data, x=COL_SPECIES, y=COL_MSE_AE,
		col=COL_MODEL_NAME_AE, row=COL_TYPE,
		hue=COL_FISH_SEX, split=True, kind='violin')
		plt.show()
		sns.catplot(data=merge_data, x=COL_SPECIES, y=COL_LATENT_DIST_AE,
		col=COL_MODEL_NAME_AE, row=COL_TYPE,
		hue=COL_FISH_SEX, split=True, kind='violin')
		plt.show()

		# #plot individual divergences
		# sns.relplot(data=merge_data, x=COL_ITERATION_AE, y=COL_MSE_AE,
		# 	col=COL_TYPE, hue=COL_SPECIES,
		# 	kind="line", estimator=None, units=COL_IMG_PATH, alpha=0.5).set_titles("{}".format(self.model.name))
		# plt.show()
		# sns.relplot(data=merge_data, x=COL_ITERATION_AE, y=COL_LATENT_DIST_AE,
		# 	col=COL_TYPE, hue=COL_SPECIES,
		# 	kind="line", estimator=None, units=COL_IMG_PATH, alpha=0.5).set_titles("{}".format(self.model.name))
		# plt.show()

		#plot grouped divergences
		sns.relplot(data=merge_data, x=COL_ITERATION_AE, y=COL_MSE_AE,
			col=COL_TYPE, hue=COL_SPECIES,kind="line").set_titles("{}".format(self.model.name))
		plt.show()
		# plt.savefig("grp_mse_div_{}".format(self.model.name))
		sns.relplot(data=merge_data, x=COL_ITERATION_AE, y=COL_LATENT_DIST_AE,
			col=COL_TYPE, hue=COL_SPECIES, kind="line").set_titles("{}".format(self.model.name))
		plt.show()
		# plt.savefig("grp_lat_div_{}".format(self.model.name))

		# #plot individual diff with previous iteration
		# sns.relplot(data=merge_data, x=COL_ITERATION_AE, y=COL_MSE_PREV_AE,
		# 	col=COL_TYPE, hue=COL_SPECIES,
		# 	kind="line", estimator=None, units=COL_IMG_PATH, alpha=0.5).set_titles("{}".format(self.model.name))
		# plt.show()
		# sns.relplot(data=merge_data, x=COL_ITERATION_AE, y=COL_LATENT_DIST_PREV_AE,
		# 	col=COL_TYPE, hue=COL_SPECIES,
		# 	kind="line", estimator=None, units=COL_IMG_PATH, alpha=0.5).set_titles("{}".format(self.model.name))
		# plt.show()

		# #plot grouped diff with previous iteration
		# sns.relplot(data=merge_data, x=COL_ITERATION_AE, y=COL_MSE_PREV_AE,
		# 	col=COL_TYPE, hue=COL_SPECIES,kind="line").set_titles("{}".format(self.model.name))
		# plt.show()
		# sns.relplot(data=merge_data, x=COL_ITERATION_AE, y=COL_LATENT_DIST_PREV_AE,
		# 	col=COL_TYPE, hue=COL_SPECIES, kind="line").set_titles("{}".format(self.model.name))
		# plt.show()


	def function(self, image):
		df = pd.DataFrame()
		params = self.preprocess.get_params()

		image = image[(image.shape[0]-PREDICTION_SIZE)//2 : (image.shape[0]+PREDICTION_SIZE)//2,
		 			(image.shape[1]-PREDICTION_SIZE)//2 : (image.shape[1]+PREDICTION_SIZE)//2]	#crop in the middle

		shift_pxl, shift_latent, diff_pxl, diff_latent = divergence(self.model, image, 50)

		i=0
		for pxl_shift ,latent_shift, pxl_diff, latent_diff in zip(shift_pxl, shift_latent, diff_pxl, diff_latent):
			df.loc[i, params.columns] = params.iloc[0]
			df.loc[i, [COL_MODEL_NAME_AE, COL_ITERATION_AE, COL_PRED_SIZE, COL_MSE_AE, COL_LATENT_DIST_AE, COL_MSE_PREV_AE, COL_LATENT_DIST_PREV_AE]] = [self.model.name, i, PREDICTION_SIZE, pxl_shift, latent_shift, pxl_diff, latent_diff]
			i+=1
		return df


def fly_over_image(image, window, stride, return_coord=False):
	"""
	generate cropped images of the given image with a shape of the given window and stride given
	when return coord is true it returns the coordinates and not the crop
	"""
	img_Y = image.shape[-2]
	img_X = image.shape[-3]
	for start_x in range(0, img_X-window[0]+1, stride[0]):
		for start_y in range(0, img_Y-window[1]+1, stride[1]):
			if return_coord:
				yield (start_x, start_x+window[0], start_y, start_y+window[1])
			else:
				sample = image[start_x: start_x+window[0], start_y: start_y + window[1]]
				yield sample


def divergence(autoencoder, start, repetition, visu=True):
	if start.ndim<4:
		start=start[np.newaxis, ...]
	start_latent = autoencoder.encoder(start)
	prev_pxl = start
	shift_pxl = []

	prev_latent = start_latent
	shift_latent = []
	axis_latent = tuple(i for i in range(-1, -start_latent.ndim, -1))

	diff_pxl = []
	diff_latent = []

	if visu: plt.figure(figsize=(20, 4))
	for i in range(repetition):
		if visu:
			ax = plt.subplot(repetition//10, 10, i+1)
			plt.imshow(prev_pxl[0])

		new_pxl = autoencoder.decoder(prev_latent)
		new_latent = autoencoder.encoder(new_pxl)

		shift_pxl.append(np.mean(np.square(start - new_pxl), axis=(-1,-2,-3)))
		shift_latent.append(np.mean(np.square(start_latent - new_latent), axis=(axis_latent)))

		diff_pxl.append(np.mean(np.square(prev_pxl - new_pxl), axis=(-1,-2,-3)))
		diff_latent.append(np.mean(np.square(prev_latent - new_latent), axis=(axis_latent)))

		prev_pxl = new_pxl#.numpy()
		prev_latent = new_latent
	if visu:
		plt.show()
		#plt.savefig("divergence_{}_{}.png".format(autoencoder.name))
	return shift_pxl, shift_latent, diff_pxl, diff_latent


if __name__ == '__main__':

	#parameter parsing
	parser = argparse.ArgumentParser()
	parser.add_argument("action", help="type of action needed", choices=["visu", "work"])
	parser.add_argument("path_img", help="path of the image file to open")
	parser.add_argument("path_model", help="path of the model used to infer")
	parser.add_argument("-o", "--output_dir", default=DIR_RESULTS, help="directory where to put the csv output, default: {}".format(DIR_RESULTS))
	args = parser.parse_args()

	pr = Preprocess(resizeX=None, resizeY=None, normalize=True, standardize=True, img_type=IMG.RGB, img_channel=CHANNEL.ALL)
	model_name = os.path.split(args.path_model)[-1]
	metric = AutoencoderMetrics(args.path_model, pr, os.path.join(args.output_dir, CSV_AE+model_name+".csv"))

	if args.action == "visu":
		metric.load()
		metric.visualize()
	elif args.action=="work":
		metric.load()
		metric.metric_from_csv(args.path_img)
		metric.save()
