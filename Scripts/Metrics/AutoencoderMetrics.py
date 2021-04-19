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

COL_PRED_SIZE="prediciton_size"
CSV_AE="autoencoder.csv"
COL_ITERATION_AE="prediction_interation"
COL_MSE_AE="MSE_compared_to_start"
COL_LATENT_DIST_AE="latent_distance_to_start"
COL_MODEL_NAME_AE="autoencoder_name"

class AutoencoderMetrics(MotherMetric):
	def __init__(self, model_path, *args, **kwargs):
		self.model = K.models.load_model(model_path)
		super().__init__(*args, **kwargs)

	def visualize(self):
		data_image = pd.read_csv(os.path.join(DIR_RESULTS, CSV_IMAGE), index_col=0)
		merge = self.data.merge(data_image, on=COL_IMG_PATH)
		merge_data = merge.loc[merge[COL_TYPE]==FILE_TYPE.ORIG_FISH.value]

		sns.relplot(data=merge_data, x=COL_ITERATION_AE, y=COL_MSE_AE,
			col=COL_TYPE, hue=COL_SPECIES,
			kind="line", estimator=None, units=COL_IMG_PATH, alpha=0.5)
		plt.show()
		sns.relplot(data=merge_data, x=COL_ITERATION_AE, y=COL_LATENT_DIST_AE,
			col=COL_TYPE, hue=COL_SPECIES,
			kind="line", estimator=None, units=COL_IMG_PATH, alpha=0.5)
		plt.show()
		sns.relplot(data=merge_data, x=COL_ITERATION_AE, y=COL_MSE_AE,
			col=COL_TYPE, hue=COL_SPECIES,kind="line")
		plt.show()
		sns.relplot(data=merge_data, x=COL_ITERATION_AE, y=COL_LATENT_DIST_AE,
			col=COL_TYPE, hue=COL_SPECIES, kind="line")
		plt.show()

	def function(self, image):
		df = pd.DataFrame()
		params = self.preprocess.get_params()

		image = image[(image.shape[0]-PREDICTION_SIZE)//2 : (image.shape[0]+PREDICTION_SIZE)//2,
		 			(image.shape[1]-PREDICTION_SIZE)//2 : (image.shape[1]+PREDICTION_SIZE)//2]	#crop in the middle

		shift_pxl, shift_latent = divergence(self.model, image, 50)

		i=0
		for pxl , latent in zip(shift_pxl, shift_latent):
			df.loc[i, params.columns] = params.iloc[0]
			df.loc[i, [COL_MODEL_NAME_AE, COL_ITERATION_AE, COL_MSE_AE, COL_LATENT_DIST_AE, COL_PRED_SIZE]] = [self.model.name, i, pxl, latent, PREDICTION_SIZE]
			i+=1
		return df


def divergence(autoencoder, start, repetition):
	if start.ndim<4:
		start=start[np.newaxis, ...]
	start_latent = autoencoder.encoder(start)
	prev_pxl = start
	shift_pxl = []

	prev_latent = start_latent
	shift_latent = []
	axis_latent = tuple(i for i in range(-1, -start_latent.ndim, -1))

	for i in range(repetition):

		new_pxl = autoencoder.decoder(prev_latent)
		new_latent = autoencoder.encoder(new_pxl)

		shift_pxl.append(np.mean(np.square(start - new_pxl), axis=(-1,-2,-3)))

		shift_latent.append(np.mean(np.square(start_latent - new_latent), axis=(axis_latent)))

		prev_pxl = new_pxl#.numpy()
		prev_latent = new_latent

	return shift_pxl, shift_latent


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("action", help="type of action needed", choices=["visu", "work"])
	parser.add_argument("path_img", help="path of the image file to open")
	parser.add_argument("path_model", help="path of the model used to infer")
	args = parser.parse_args()

	pr = Preprocess(resize=None, normalize=True, standardize=True, img_type=IMG.RGB, img_channel=CHANNEL.ALL)
	model_name = os.path.split(args.path_model)[-1]
	metric = AutoencoderMetrics(args.path_model, pr, os.path.join(DIR_RESULTS, CSV_AE+model_name))

	if args.action == "visu":
		metric.load()
		metric.visualize()
	elif args.action=="work":
		metric.metric_from_csv(args.path_img)
		metric.save()
