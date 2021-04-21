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
	def __init__(self, name, latent_dim, pred_shape, color_channels):
		super(Autoencoder, self).__init__(name=name)
		self.latent_dim = latent_dim
		self.encoder = tf.keras.Sequential([
			VGG16(weights='imagenet', include_top=False,
							 input_shape=(*pred_shape, color_channels)),
		])

		self.decoder = tf.keras.Sequential([
			UpSampling2D(size = (2,2), name = 'upsp1'),
			Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'conv6_1'),
			Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'conv6_2'),
			Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'conv6_3'),

			UpSampling2D(size = (2,2), name = 'upsp2'),
			Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'conv7_1'),
			Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'conv7_2'),
			Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'conv7_3'),

			UpSampling2D(size = (2,2), name = 'upsp3'),
			Conv2D(256, 3, activation = 'relu', padding = 'same', name = 'conv8_1'),
			Conv2D(256, 3, activation = 'relu', padding = 'same', name = 'conv8_2'),
			Conv2D(256, 3, activation = 'relu', padding = 'same', name = 'conv8_3'),

			UpSampling2D(size = (2,2), name = 'upsp4'),
			Conv2D(128, 3, activation = 'relu', padding = 'same', name = 'conv9_1'),
			Conv2D(128, 3, activation = 'relu', padding = 'same', name = 'conv9_2'),

			UpSampling2D(size = (2,2), name = 'upsp5'),
			Conv2D(64, 3, activation = 'relu', padding = 'same', name = 'conv10_1'),
			Conv2D(64, 3, activation = 'relu', padding = 'same', name = 'conv10_2'),

			Conv2D(color_channels, 3, activation = 'relu', padding = 'same', name = 'conv11'),
		])

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
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
	parser.add_argument("-l", "--latent_dim", type=int, default=LATENT_DIM, help="the latent dimention, default {}".format(LATENT_DIM))
	parser.add_argument("-b", "--batch", type=int, default=BATCH_SIZE, help="batch size for training, default {}".format(BATCH_SIZE))
	parser.add_argument("-e", "--epochs", type=int, default=EPOCHS, help="number of epochs max for training, default {}".format(EPOCHS))
	parser.add_argument("-n", "--name", type=str, default=NETWORK_NAME, help="network name, default {}".format(NETWORK_NAME))
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
	autoencoder = Autoencoder(network_name, args.latent_dim, prediction_shape[-1])
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
	plt.savefig("{} training losses".format(autoencoder.name))

	#save the model
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	autoencoder.save(os.path.join(args.output_dir, autoencoder.name), overwrite=True)

	#plot the predictions
	autoencoder.show_predictions(test)
