import argparse
from sklearn.model_selection import train_test_split
import tensorflow.keras as K
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Softmax, Dense, Flatten, Reshape, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import imageio
import cv2

from CommonAE import *

class Autoencoder(Model):
	"""Convolutional variational autoencoder."""
	def __init__(self, name, latent_dim, prediction_shape):
		super(Autoencoder, self).__init__(name=name)
		self.latent_dim = latent_dim
		self.encoder = K.Sequential([
			Conv2D(latent_dim, kernel_size=(3,3), padding="same"),
			MaxPool2D(pool_size=(2,2), padding="same"),
			Conv2D(latent_dim*2, kernel_size=(3,3), padding="same"),
			MaxPool2D(pool_size=(2,2), padding="same"),
			Conv2D(latent_dim*2, kernel_size=(3,3), padding="same"),
		])
		self.decoder = K.Sequential([
			UpSampling2D(size=(2,2)),
			Conv2DTranspose(filters=latent_dim*2, kernel_size=(3,3), padding="same"),
			UpSampling2D(size=(2,2)),
			Conv2DTranspose(filters=prediction_shape[-1], kernel_size=(3,3), padding="same", activation='sigmoid')
		])


	@tf.function
	def sample(self, n_sample, img_shape, eps=None):
		if eps is None:
			eps = tf.random.normal(shape=(n_sample, img_shape[0]//4, img_shape[1]//4, self.latent_dim))
		return self.decode(eps, apply_sigmoid=True)

	def encode(self, x):
		mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=3)
		return mean, logvar

	def reparameterize(self, mean, logvar):
		eps = tf.random.normal(shape=tf.shape(mean))
		return eps * tf.exp(logvar * .5) + mean

	def decode(self, z, apply_sigmoid=False):
		logits = self.decoder(z)
		if apply_sigmoid:
			probs = tf.sigmoid(logits)
			return probs
		return logits

	def call(self, x):
		mean, logvar = self.encode(x)
		reparam = self.reparameterize(mean, logvar)
		decoded = self.decode(reparam)
		return decoded


if __name__ == '__main__':
	#default parameters
	DIR_SAVED_MODELS ="Results/AE_tif_mixed_bright"
	LATENT_DIM=20
	LATENT_DIM_SPACE = [2,5,10,20,40,80,160]
	BATCH_SIZE=50
	EPOCHS=30
	NETWORK_NAME="Convolutional"
	LOSS='mse'
	VERBOSE=5

	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(title="command", dest="command", help='action to perform')
	parser.add_argument("path_train", help="path of the training dataset to use")
	parser.add_argument("path_test", help="path of the testing dataset to use")
	parser.add_argument("--output_dir", default=DIR_SAVED_MODELS, help="path where to save the trained network, default {}".format(DIR_SAVED_MODELS))
	parser.add_argument("-n", "--name", type=str, default=NETWORK_NAME, help="network name, default {}".format(NETWORK_NAME))
	parser.add_argument("-b", "--batch", type=int, default=BATCH_SIZE, help="batch size for training, default {}".format(BATCH_SIZE))
	parser.add_argument("-e", "--epochs", type=int, default=EPOCHS, help="number of epochs max for training, default {}".format(EPOCHS))
	parser.add_argument("--loss", type=str, choices=['mse', 'ssim'], default=LOSS, help="network loss, default {}".format(LOSS))
	parser.add_argument("-v", "--verbose", default=VERBOSE, type=int, help="set number of predictions to show, if no result directory is given the graphs are not saved, default: {}".format(VERBOSE))
	#specific parameters for multiple trainings with diverse latent dims
	LD_parser = subparsers.add_parser("LD_selection")
	LD_parser.add_argument("-l", "--latent_dim", type=int, nargs="+", default=LATENT_DIM_SPACE, help="the latent dimension that will be tested, default {}".format(LATENT_DIM_SPACE))
	#specific parameters for a simple training
	training_parser = subparsers.add_parser("training")
	training_parser.add_argument("-l", "--latent_dim", type=int, default=LATENT_DIM, help="the latent dimention, default {}".format(LATENT_DIM))
	args = parser.parse_args()

	#prepare the data
	train = np.load(args.path_train)
	test = np.load(args.path_test)
	assert train.shape[1:]==test.shape[1:], "train and test should contain images of similar shape, here {} and {}".format(train.shape[1:], test.shape[1:])
	prediction_shape = train.shape[1:]
	dataset_descriptor, *_ = os.path.split(args.path_train)[-1].split('_') #get the descriptor of the dataset

	if args.command=="LD_selection":
		list_LD = args.latent_dim
	elif args.command=="training":
		list_LD = [args.latent_dim]

	#prepare directory output
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	output_dir = os.path.join(args.output_dir, args.name)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	losses = []
	val_losses = []
	for latent_dim in list_LD:
		K.backend.clear_session()

		#prepare the network
		network_name = "{}_{}_LD{}_pred{}x{}x{}".format(args.name, dataset_descriptor, latent_dim, *prediction_shape)
		autoencoder = Autoencoder(network_name, latent_dim, prediction_shape)
		autoencoder.compile(optimizer='adam', loss=args.loss)

		#prepare callbacks
		callbacks = [K.callbacks.EarlyStopping(monitor='val_loss', patience=4),
					K.callbacks.EarlyStopping(monitor='loss', patience=4),
					SavePredictionSample(n_samples=5, val_data=test[0:5*20:5], saving_dir=output_dir)]

		#train the network
		history = autoencoder.fit(x=train, y=train,
			batch_size=args.batch,
			validation_data=(test, test),
			epochs= args.epochs,
			callbacks= callbacks,
			shuffle= True
			)
		losses.append(history.history['loss'])
		val_losses.append(history.history['val_loss'])

		#plot examples samples
		samples = autoencoder.sample(args.verbose, prediction_shape)
		fig, axs = plt.subplots(nrows=1, ncols=args.verbose, sharex=True, sharey=True)
		fig.suptitle("VarAE sampling")
		for i in range(args.verbose):
			axs[i].imshow(samples[i], cmap='gray')
			axs[i].set_title("sample {}".format(i))
		plt.show()
		plt.savefig(os.path.join(output_dir, "{} sampling".format(autoencoder.name)))

	#plot the training losses
	plot_training_losses(losses, val_losses, list_LD, 
		"losses for different latent dims", 
		os.path.join(output_dir,"losses {}".format(autoencoder.name)))

	#plot the best validation for each latent dim
	best_losses=[]
	best_val_losses=[]
	for val_loss, loss in zip(val_losses, losses):
		best_losses.append(min(loss))
		best_val_losses.append(min(val_loss))
	plot_loss_per_ld(best_losses, best_val_losses, list_LD, 
		title="best losses per latent dim for {}".format(autoencoder.name),
		save_path=os.path.join(output_dir, "best losses {}".format(autoencoder.name))
		)

	#save the model
	if args.command=="training":
		autoencoder.save(os.path.join(output_dir, autoencoder.name), overwrite=True)
