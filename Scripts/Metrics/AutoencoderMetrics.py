from scipy.stats import kurtosis, entropy
import tensorflow.keras as K
from tensorflow.image import ssim
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
from ScalarMetrics import get_gini

PREDICTION_SIZE=128

CSV_AE="metricAE_"
COL_MODEL_NAME_AE="autoencoder_name"
COL_PRED_SIZE="prediction_size"

COL_ITERATION_AE="prediction_interation"
COL_MSE_AE="MSE_compared_to_start"
COL_MSE_PREV_AE="MSE_compared_to_previous_step"
COL_LATENT_DIST_AE="latent_distance_to_start"
COL_LATENT_DIST_PREV_AE="latent_distance_to_previous_step"

COL_SSIM_AE="SSIM_compared_to_start"
COL_SSIM_PREV_AE="SSIM_compared_to_previous_step"

COL_GINI_AE="gini_pxl_space"
COL_KURTO_AE="kurtois_pxl_space"
COL_ENTRO_AE="entropy_pxl_space"
COL_GINI_LATENT_AE="gini_latent_space"
COL_KURTO_LATENT_AE="kurtois_latent_space"
COL_ENTRO_LATENT_AE="entropy_latent_space"

class AutoencoderMetrics(MotherMetric):
	def __init__(self, model_path, *args, **kwargs):
		self.model = K.models.load_model(model_path)
		super().__init__(*args, **kwargs)

	def visualize(self):
		data_image = pd.read_csv(os.path.join(DIR_RESULTS, CSV_IMAGE), index_col=0) 

		# #plot differences in quality prediction for each network
		data_networks_first_iter = self.data.loc[self.data[COL_ITERATION_AE]==0]
		data_merged = data_networks_first_iter.merge(data_image, on=COL_IMG_PATH)
		
		#plot metrics for iteration 1
		sns.relplot(data=data_merged, x=COL_SPECIES, y=COL_ENTRO_AE, hue=COL_MODEL_NAME_AE)
		plt.show()
		sns.relplot(data=data_merged, x=COL_SPECIES, y=COL_ENTRO_LATENT_AE, hue=COL_MODEL_NAME_AE)
		plt.show()
		sns.relplot(data=data_merged, x=COL_SPECIES, y=COL_KURTO_AE, hue=COL_MODEL_NAME_AE)
		plt.show()
		sns.relplot(data=data_merged, x=COL_SPECIES, y=COL_KURTO_LATENT_AE, hue=COL_MODEL_NAME_AE)
		plt.show()
		sns.relplot(data=data_merged, x=COL_SPECIES, y=COL_GINI_AE, hue=COL_MODEL_NAME_AE)
		plt.show()
		sns.relplot(data=data_merged, x=COL_SPECIES, y=COL_GINI_LATENT_AE, hue=COL_MODEL_NAME_AE)
		plt.show()

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
		data_my_network = self.data.loc[self.data[COL_MODEL_NAME_AE]==self.model.name]
		merge = data_my_network.merge(data_image, on=COL_IMG_PATH)
		merge_data = merge.loc[merge[COL_TYPE]==FILE_TYPE.ORIG_FISH.value]

		# #plot violin dist by species
		# sns.catplot(data=merge_data, x=COL_SPECIES, y=COL_MSE_AE,
		# col=COL_MODEL_NAME_AE, row=COL_TYPE,
		# hue=COL_FISH_SEX, split=True, kind='violin')
		# plt.show()
		# sns.catplot(data=merge_data, x=COL_SPECIES, y=COL_LATENT_DIST_AE,
		# col=COL_MODEL_NAME_AE, row=COL_TYPE,
		# hue=COL_FISH_SEX, split=True, kind='violin')
		# plt.show()

		#plot evolution of the metrics
		sns.relplot(data=merge_data, x=COL_ITERATION_AE, y=COL_ENTRO_AE, hue=COL_SPECIES, 
			kind='line', estimator=None, units=COL_IMG_PATH).set_titles("{}".format(self.model.name))
		plt.show()

		sns.relplot(data=merge_data, x=COL_ITERATION_AE, y=COL_GINI_AE, hue=COL_SPECIES, 
			kind='line', estimator=None, units=COL_IMG_PATH).set_titles("{}".format(self.model.name))
		plt.show()

		sns.relplot(data=merge_data, x=COL_ITERATION_AE, y=COL_KURTO_AE, hue=COL_SPECIES, 
			kind='line', estimator=None, units=COL_IMG_PATH).set_titles("{}".format(self.model.name))
		plt.show()

		sns.relplot(data=merge_data, x=COL_ITERATION_AE, y=COL_ENTRO_LATENT_AE, hue=COL_SPECIES, 
			kind='line', estimator=None, units=COL_IMG_PATH).set_titles("{}".format(self.model.name))
		plt.show()

		sns.relplot(data=merge_data, x=COL_ITERATION_AE, y=COL_GINI_LATENT_AE, hue=COL_SPECIES, 
			kind='line', estimator=None, units=COL_IMG_PATH).set_titles("{}".format(self.model.name))
		plt.show()

		sns.relplot(data=merge_data, x=COL_ITERATION_AE, y=COL_KURTO_LATENT_AE, hue=COL_SPECIES, 
			kind='line', estimator=None, units=COL_IMG_PATH).set_titles("{}".format(self.model.name))
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


	def function(self, image, verbose):
		df = pd.DataFrame()
		params = self.preprocess.get_params()

		image = image[(image.shape[0]-PREDICTION_SIZE)//2 : (image.shape[0]+PREDICTION_SIZE)//2,
		 			(image.shape[1]-PREDICTION_SIZE)//2 : (image.shape[1]+PREDICTION_SIZE)//2]	#crop in the middle

		values_stats , values_div = divergence(self.model, image, 50)
		

		i=0
		for value_stat, value_div in zip(values_stats , values_div):
			df.loc[i, params.columns] = params.iloc[0]
			df.loc[i, [COL_MODEL_NAME_AE, COL_ITERATION_AE, COL_PRED_SIZE]]=[self.model.name, i, PREDICTION_SIZE]
			df.loc[i, [COL_MSE_AE, COL_LATENT_DIST_AE, COL_SSIM_AE, COL_MSE_PREV_AE, COL_LATENT_DIST_PREV_AE, COL_SSIM_PREV_AE]] = [*value_div]
			df.loc[i, [COL_GINI_AE, COL_KURTO_AE, COL_ENTRO_AE, COL_GINI_LATENT_AE, COL_KURTO_LATENT_AE, COL_ENTRO_LATENT_AE]] = [*value_stat]
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


def divergence(autoencoder, start, repetition, visu=False):
	if start.ndim<4:
		start=start[np.newaxis, ...]
	start_latent = autoencoder.encoder(start)
	prev_pxl = start
	shift_pxl_mse = []
	shift_pxl_ssim = []

	prev_latent = start_latent
	shift_latent = []
	axis_latent = tuple(i for i in range(-1, -start_latent.ndim, -1))

	diff_pxl_mse = []
	diff_pxl_ssim = []
	diff_latent = []

	if visu: plt.figure(figsize=(20, 4))
	for i in range(repetition):
		if visu:
			ax = plt.subplot(repetition//10, 10, i+1)
			plt.imshow(prev_pxl[0])

		new_pxl = autoencoder.decoder(prev_latent)
		new_latent = autoencoder.encoder(new_pxl)
		#store the stat values
		gini_pxl_space.append(get_gini(prev_pxl))
		kurtois_pxl_space.append(kurtosis(prev_pxl, axis=None))
		entropy_pxl_space.append(entropy(prev_pxl, axis=None))
		gini_latent_space.append(get_gini(prev_latent))
		kurtois_latent_space.append(kurtosis(prev_latent, axis=None))
		entropy_latent_space.append(entropy(prev_latent, axis=None))
		#diff to start
		shift_pxl_mse.append(np.mean(np.square(start - new_pxl), axis=(-1,-2,-3)))
		shift_pxl_ssim.append(ssim(start, new_pxl, max_val=1))
		shift_latent.append(np.mean(np.square(start_latent - new_latent), axis=(axis_latent)))
		#diff to prev
		diff_pxl_mse.append(np.mean(np.square(prev_pxl - new_pxl), axis=(-1,-2,-3)))
		diff_pxl_ssim.append(ssim(prev_pxl, new_pxl, max_val=1))
		diff_latent.append(np.mean(np.square(prev_latent - new_latent), axis=(axis_latent)))

		prev_pxl = new_pxl
		prev_latent = new_latent
	if visu:
		plt.show()
		plt.savefig(os.path.join(DIR_RESULTS, "visu_divergence_{}.png".format(autoencoder.name)))
	values_div = (shift_pxl_mse, shift_latent, shift_pxl_ssim, diff_pxl_mse, diff_latent, diff_pxl_ssim)
	values_stat = (gini_pxl_space, kurtois_pxl_space, entropy_pxl_space, gini_latent_space, kurtois_latent_space, entropy_latent_space)
	return (values_stat, values_div)

def get_heat_prediction_fish(img, prediction_size, visu=False):
    heatmap_mse = np.zeros_like(img)
    ponderation = np.ones_like(img)
    batch = []
	stride = (prediction_size[0]//5, prediction_size[1]//5)
    for sample in fly_over_image(img, prediction_size, stride, False):
        batch += [sample]

    batch = np.array(batch)

    prediction = autoencoder.predict_on_batch(batch)
    mse = K.losses.MSE(batch, prediction)

    i = 0
    for x1, x2, y1, y2 in fly_over_image(img, prediction_size, stride, True):
        heatmap_mse[x1:x2, y1:y2, 0] += mse[i]
        ponderation[x1:x2, y1:y2, 0] += 1
        i+=1
	heatmap = np.divide(heatmap_mse, ponderation)
	if visu:
		plt.imshow(heatmap, cmap="")
		plt.title("mean mse heatmap")
		plt.colorbar()
    return heatmap


def save_heatmaps(images, filepath, pred_size, visu=False):
	heatmap = []
	for img in images:
		heatmap += [get_heat_prediction_fish(img, pred_size, visu)]
		heatmap = np.array(heatmap)
		np.save(filepath, heatmap)


if __name__ == '__main__':
	#DEFAULT PARAMETERS
	RESIZE_X=None
	RESIZE_Y=None
	NORMALIZE=False
	STANDARDIZE=False
	IMG_TYPE=IMG.RGB
	IMG_CHANNEL=CHANNEL.ALL
	VERBOSE=1

	#parameter parsing
	parser = argparse.ArgumentParser()
	parser.add_argument("action", help="type of action needed", choices=["visu", "work"])
	parser.add_argument("path_input", help="path of the image file to open")
	parser.add_argument("path_model", help="path of the model used to infer")
	parser.add_argument("-o", "--output_dir", default=DIR_RESULTS, help="directory where to put the csv output, default: {}".format(DIR_RESULTS))
	parser.add_argument("-x", "--resizeX", default=RESIZE_X, type=int, help="Resize X to this value, default: {}".format(RESIZE_X))
	parser.add_argument("-y", "--resizeY", default=RESIZE_Y, type=int, help="Resize Y to this value, default: {}".format(RESIZE_Y))
	parser.add_argument("-n", "--normalize", default=NORMALIZE, type=lambda x: bool(eval(x)), help="if image should be normalized, default: {}".format(NORMALIZE))
	parser.add_argument("-s", "--standardize", default=STANDARDIZE, type=lambda x: bool(eval(x)), help="if image should be standardized, default: {}".format(STANDARDIZE))
	parser.add_argument("-v", "--verbose", default=VERBOSE, type=int, choices=[0,1,2], help="set the level of visualization, default: {}".format(VERBOSE))
	parser.add_argument("-t", "--type_img", default=IMG_TYPE.name, type=lambda x: IMG[x], choices=list(IMG), help="the type of image needed, default: {}".format(IMG_TYPE))
	parser.add_argument("-c", "--channel_img", default=IMG_CHANNEL.name, type=lambda x: CHANNEL[x], choices=list(CHANNEL), help="The channel used for the image, default: {}".format(IMG_CHANNEL))
	args = parser.parse_args()

	#prepare metric
	pr = Preprocess(resizeX=args.resizeX, resizeY=args.resizeY, normalize=args.normalize, standardize=args.standardize, img_type=args.type_img, img_channel=args.channel_img)
	model_name = os.path.split(args.path_model)[-1]
	filepath = os.path.join(args.output_dir, CSV_AE+model_name+".csv")
	metric = AutoencoderMetrics(args.path_model, pr)

	if args.action == "visu":
		metric.load(filepath)
		metric.visualize()
	elif args.action=="work":
		metric.load(filepath)
		metric.metric_from_csv(args.path_input, verbose=args.verbose)
		metric.save(filepath)
