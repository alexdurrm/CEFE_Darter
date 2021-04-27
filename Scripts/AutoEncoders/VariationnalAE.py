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
		self.encoder = K.Sequential(
			[
				K.layers.Conv2D(
					filters=32, kernel_size=3, strides=(2, 2), activation='relu'),	#(pred_shape-3+2*0)/2+1
				K.layers.Conv2D(
					filters=64, kernel_size=3, strides=(2, 2), activation='relu'), #((pred_shape-3+2*0)/2+1-3+2*0)/2+1
				K.layers.Flatten(),
				# No activation
				K.layers.Dense(latent_dim*2),
			]
			#output shape of conv2D is [(W−K+2P)/S]+1
		)
		small_x = (prediction_shape[0]-3)//2 +1
		small_y = (prediction_shape[1]-3)//2 +1
		self.decoder = K.Sequential(
			[
				K.layers.InputLayer(input_shape=(latent_dim)),
				K.layers.Dense(units=small_x*small_y*32, activation=tf.nn.relu),
				K.layers.Reshape(target_shape=(7, 7, 32)),
				K.layers.Conv2DTranspose(
					filters=64, kernel_size=3, strides=2, padding='same',
					activation='relu'),
				K.layers.Conv2DTranspose(
					filters=32, kernel_size=3, strides=2, padding='same',
					activation='relu'),
				# No activation
				K.layers.Conv2DTranspose(
					filters=prediction_shape[-1], kernel_size=3, strides=1, padding='same'),
			]
		)

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

	def show_predictions(self, sample_test, n=10):
		"""
		plot test sample images and their reconstruction by the network
		"""
		prediction = self.call(sample_test)
		plt.figure(figsize=(20, 4))
		plt.title("{} reconstructions".format(self.name))
		for i in range(n):
			rdm = np.random.randint(0, len(sample_test))
			# display original
			ax = plt.subplot(3, n, i + 1)
			plt.imshow(sample_test[rdm], cmap='gray')
			plt.title("original")
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

			# display reconstruction
			ax = plt.subplot(2, n, i + 1 + n)
			plt.imshow(prediction[rdm], cmap='gray')
			plt.title("reconstructed")
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		plt.show()
		plt.savefig("{} predictions for test".format(self.name))


if __name__ == '__main__':
	#default parameters
	DIR_SAVED_MODELS ="Scripts/AutoEncoders/TrainedModels"
	LATENT_DIM=20
	BATCH_SIZE=50
	EPOCHS=30
	NETWORK_NAME="Convolutional"
	LOSS='mse'

	parser = argparse.ArgumentParser()
	parser.add_argument("path_train", help="path of the training dataset to use")
	parser.add_argument("path_test", help="path of the testing dataset to use")
	parser.add_argument("--output_dir", default=DIR_SAVED_MODELS, help="path where to save the trained network, default {}".format(DIR_SAVED_MODELS))
	parser.add_argument("-n", "--name", type=str, default=NETWORK_NAME, help="network name, default {}".format(NETWORK_NAME))
	parser.add_argument("-l", "--latent_dim", type=int, default=LATENT_DIM, help="the latent dimention, default {}".format(LATENT_DIM))
	parser.add_argument("-b", "--batch", type=int, default=BATCH_SIZE, help="batch size for training, default {}".format(BATCH_SIZE))
	parser.add_argument("-e", "--epochs", type=int, default=EPOCHS, help="number of epochs max for training, default {}".format(EPOCHS))
	parser.add_argument("--loss", type=str, choices=['mse', 'ssim'], default=LOSS, help="network loss, default {}".format(LOSS))
	args = parser.parse_args()

	#prepare the data
	train = np.load(args.path_train)
	test = np.load(args.path_test)
	assert train.shape[1:]==test.shape[1:], "train and test should contain images of similar shape, here {} and {}".format(train.shape[1:], test.shape[1:])

	prediction_shape = train.shape[1:]
	_, dataset_descriptor, presize, pred_shape = os.path.split(args.path_train)[-1].split('_') #get the descriptor of the dataset

	#prepare the network
	network_name = "{}_{}_LD{}_pred{}x{}x{}".format(args.name, dataset_descriptor, args.latent_dim, *prediction_shape)
	autoencoder = Autoencoder(network_name, args.latent_dim, prediction_shape)
	autoencoder.compile(optimizer='adam', loss=args.loss)

	#train the network
	callback = K.callbacks.EarlyStopping(monitor='val_loss', patience=5)
	history = autoencoder.fit(x=train, y=train,
		batch_size=args.batch,
		validation_data=(test, test),
		epochs= args.epochs,
		callbacks= callback,
		shuffle= True
		)

	#plot the training
	plt.plot(history.history['loss'], label="train")
	plt.plot(history.history['val_loss'], label="val")
	plt.legend()
	plt.show()
	plt.savefig(os.path.join( DIR_SAVED_MODELS, "{} training losses.png".format(autoencoder.name)))

	#save the model
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	autoencoder.save(os.path.join(args.output_dir, autoencoder.name), overwrite=True)

	#plot the predictions
	autoencoder.show_predictions(test)

	#plot examples samples
	for i in range(10):
		plt.imshow(autoencoder.sample(), cmap='gray')
		plt.show()
