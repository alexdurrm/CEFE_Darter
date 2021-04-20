import argparse
from sklearn.model_selection import train_test_split
import tensorflow.keras as K
from tensorflow.keras.layers import Conv2D, MaxPool2D, Softmax, Dense, Flatten, Reshape, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from tensorflow.keras.models import Model
import numpy as np
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import imageio
import cv2


class Autoencoder(Model):
	def __init__(self, name, latent_dim, pred_shape):
		super(Autoencoder, self).__init__(name=name)
		self.latent_dim = latent_dim
		self.encoder = K.Sequential([
			Flatten(),
			Dense(latent_dim, activation='relu')
			])
		self.decoder = K.Sequential([
			Dense(pred_shape[0]*pred_shape[1]*pred_shape[2], activation='sigmoid'),
			Reshape(pred_shape)
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


if __name__ == '__main__':
	#default parameters
	DIR_SAVED_MODELS ="Scripts/AutoEncoders/TrainedModels"
	LATENT_DIM=20
	BATCH_SIZE=50
	EPOCHS=30
	NETWORK_NAME="Perceptron"
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

	#save the model
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	autoencoder.save(os.path.join(args.output_dir, autoencoder.name), overwrite=True)

	#plot the predictions
	autoencoder.show_predictions(test)
