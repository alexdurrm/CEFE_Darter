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


class Autoencoder(Model):
	"""Convolutional variational autoencoder."""
	def __init__(self, name, latent_dim, prediction_shape):
		super(Autoencoder, self).__init__(name=name)
		self.latent_dim = latent_dim
		self.encoder = K.Sequential([
			Conv2D(latent_dim, kernel_size=(3,3), padding="same"),
			MaxPool2D(pool_size=(3,3), padding="same"),
			Conv2D(latent_dim//2, kernel_size=(3,3), padding="same"),
			MaxPool2D(pool_size=(3,3), padding="same"),
			Conv2D(latent_dim//4, kernel_size=(3,3), padding="same"),
		])
		self.decoder = K.Sequential([
			UpSampling2D(size=(3,3)),
			Conv2DTranspose(filters=prediction_shape[-1]//2, kernel_size=(3,3), padding="same"),
			UpSampling2D(size=(3,3)),
			Conv2DTranspose(filters=prediction_shape[-1], kernel_size=(3,3), padding="same")
		])


	@tf.function
	def sample(self, eps=None):
		if eps is None:
			eps = tf.random.normal(shape=(100, self.latent_dim))
		return self.decode(eps, apply_sigmoid=True)

	def encode(self, x):
		mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
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

	def show_predictions(self, sample_test, n, saving_dir=None):
		"""
		plot test sample images and their reconstruction by the network
		"""
		prediction = self.call(sample_test)
		for i in range(n):
			# rdm = np.random.randint(0, len(sample_test))
			idx = 15*i
			fig=plt.figure()
			plt.title("{} reconstructions img {}".format(self.name, idx))
			# display original
			fig.add_subplot(1, 2, 1)
			plt.imshow(sample_test[idx], cmap='gray')
			plt.title("original")

			# display reconstruction
			fig.add_subplot(1, 2, 2)
			plt.imshow(prediction[idx], cmap='gray')
			plt.title("reconstructed")
			# plt.show()
			if saving_dir:
				plt.savefig(os.path.join(saving_dir,"{} reconstructions img {}".format(self.name, idx)))


if __name__ == '__main__':
	#default parameters
	DIR_SAVED_MODELS ="Scripts/AutoEncoders/TrainedModels"
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
	losses = []
	val_losses = []
	for latent_dim in list_LD:
		#prepare the network
		network_name = "{}_{}_LD{}_pred{}x{}x{}".format(args.name, dataset_descriptor, latent_dim, *prediction_shape)
		autoencoder = Autoencoder(network_name, latent_dim, prediction_shape)
		autoencoder.compile(optimizer='adam', loss=args.loss)

		#train the network
		callbacks = [K.callbacks.EarlyStopping(monitor='val_loss', patience=4),
					K.callbacks.EarlyStopping(monitor='loss', patience=4)]
		history = autoencoder.fit(x=train, y=train,
			batch_size=args.batch,
			validation_data=(test, test),
			epochs= args.epochs,
			callbacks= callbacks,
			shuffle= True
			)
		losses.append(history.history['loss'])
		val_losses.append(history.history['val_loss'])

		#plot the predictions
		autoencoder.show_predictions(sample_test=test, n=args.verbose, saving_dir=args.output_dir)

		#plot examples samples
		for i in range(args.verbose):
			plt.imshow(autoencoder.sample(), cmap='gray')
			plt.show()
			if graph_directory:
				plt.savefig(os.path.join(graph_directory, "{} sampling {}".format(autoencoder.name, i)))

	#plot the training
	fig=plt.figure()
	for loss in losses:
		plt.plot(loss)
	plt.legend(list_LD)
	plt.savefig(os.path.join(args.output_dir,"{} losses training".format(autoencoder.name)))
	fig=plt.figure()
	for v_loss in val_losses:
		plt.plot(v_loss)
	plt.legend(list_LD)
	plt.savefig(os.path.join(args.output_dir,"{} losses testing".format(autoencoder.name)))

	#save the model
	if args.command=="training":
		if not os.path.exists(args.output_dir):
			os.makedirs(args.output_dir)
		autoencoder.save(os.path.join(args.output_dir, autoencoder.name), overwrite=True)
