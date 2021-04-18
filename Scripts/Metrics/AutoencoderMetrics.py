import tensorflow.keras as K
from tensorflow.keras.applications.vgg16 import VGG16
import pandas as pd
import numpy as np
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt

from MotherMetric import MotherMetric
from Preprocess import *
from config import *

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
		sns.lineplot(data=self.data, x=COL_ITERATION_AE, y=COL_MSE_AE, hue=COL_SPECIES)
		plt.plot()
		sns.lineplot(data=self.data, x=COL_ITERATION_AE, y=COL_LATENT_DIST_AE, hue=COL_SPECIES)
		plt.plot()


	def function(self, image):
		df = pd.DataFrame()
		params = self.preprocess.get_params()
		shift_pxl, shift_latent = divergence(self.model, image, 50)

		i=0
		for pxl , latent in zip(shift_pxl, shift_latent):
			df.loc[i, params.columns] = params.iloc[0]
			df.loc[i, [COL_MODEL_NAME_AE, COL_ITERATION_AE, COL_MSE_AE, COL_LATENT_DIST_AE]] = [self.model.name, i, pxl, latent]
			i+=1
		return df


def divergence(autoencoder, start, repetition):
	if start.ndim==3:
		start=start[np.newaxis, ...]
	sart_latent = autoencoder.encode(start)

	prev_pxl = start
	shift_pxl = []

	prev_latent = start_latent
	shift_latent = []

	for i in range(repetition):

		new_pxl = autoencoder.decode(prev_latent)
		new_latent = autoencoder.encode(new_pxl)

		shift_pxl.append(np.mean(np.square(start - new_pxl), axis=start.shape[1:]))
		shift_latent.append(np.mean(np.square(start_latent - new_latent)), axis=start_latent.shape[1:])

		prev_pxl = new_pxl#.numpy()
		prev_latent = new_latent

	return shift_pxl, shift_latent


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("action", help="type of action needed", choices=["visu", "work"])
	parser.add_argument("path_img", help="path of the image file to open")
	parser.add_argument("path_model", help="path of the model used to infer")
	args = parser.parse_args()

	pr = Preprocess(resize=(900, 300), normalize=True, standardize=True, img_type=IMG.DARTER, img_channel=CHANNEL.GRAY)

	metric = AutoencoderMetrics(args.path_model, pr, os.path.join(DIR_RESULTS, CSV_AE))

	if args.action == "visu":
		metric.load()
		metric.visualize()
	elif args.action=="work":
		metric.metric_from_csv(args.path_img)
		metric.save()
