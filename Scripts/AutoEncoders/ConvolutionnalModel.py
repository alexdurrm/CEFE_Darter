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
	def __init__(self, name, latent_dim, color_channels):
		super(Autoencoder, self).__init__(name=name)
		self.latent_dim = latent_dim
		self.encoder = K.Sequential([
			Conv2D(latent_dim, kernel_size=(2,2), padding="same"),
			MaxPool2D(pool_size=(2,2), padding="same"),
			Conv2D(latent_dim, kernel_size=(2,2), padding="same"),
			MaxPool2D(pool_size=(2,2), padding="same"),
			Conv2D(latent_dim, kernel_size=(2,2), padding="same"),
		])
		self.decoder = K.Sequential([
			UpSampling2D(size=(2,2)),
			Conv2DTranspose(filters=latent_dim, kernel_size=(2,2), padding="same"),
			UpSampling2D(size=(2,2)),
			Conv2DTranspose(filters=color_channels, kernel_size=(2,2), padding="same")
		])

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded


class SavePredictionSample(K.callbacks.Callback):
	def __init__(self, n_samples, saving_dir):
		self.n_samples=n_samples
		self.saving_dir=saving_dir
		super().__init__()

	def on_epoch_end(self, epoch, logs=None):
		space = 15
		sample = self.validation_data[0 : space*self.n_samples : space]
		outputs = self.model.predict(sample)
		title = "reconstructions epoch {} model {}".format(epoch, self.name)
		show_predictions(sample, outputs, self.n_samples, title, self.saving_dir)


def show_predictions(sample_test, prediction, n, title, saving_dir=None):
	"""
	plot test sample images and their reconstruction by the network
	"""
	fig, axs = plt.subplots(nrows=2, ncols=n, sharex=True, sharey=True, squeeze=False)
	fig.suptitle(title)
	for i in range(n):
		axs[1][i+1].imshow(sample_test[i], cmap='gray')
		axs[1][i+1].set_title("original {}".format(i))

		axs[2][i+1].imshow(prediction[i], cmap='gray')
		axs[2][i+1].set_title("reconstructed {}".format(i))
	plt.show()
	if saving_dir:
		plt.savefig(os.path.join(saving_dir, title))


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
	parser.add_argument("-b", "--batch", type=int, default=BATCH_SIZE, help="batch size for training, default {}".format(BATCH_SIZE))
	parser.add_argument("-e", "--epochs", type=int, default=EPOCHS, help="number of epochs max for training, default {}".format(EPOCHS))
	parser.add_argument("-n", "--name", type=str, default=NETWORK_NAME, help="network name, default {}".format(NETWORK_NAME))
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

	losses = []
	val_losses = []
	for latent_dim in list_LD:
		#prepare the network
		network_name = "{}_{}_LD{}_pred{}x{}x{}".format(args.name, dataset_descriptor, latent_dim, *prediction_shape)
		autoencoder = Autoencoder(network_name, latent_dim, prediction_shape[-1])
		autoencoder.compile(optimizer='adam', loss=args.loss)

		#train the network
		callbacks = [K.callbacks.EarlyStopping(monitor='val_loss', patience=4),
					K.callbacks.EarlyStopping(monitor='loss', patience=4),
					SavePredictionSample(n_samples=5, saving_dir=args.output_dir)]

		history = autoencoder.fit(x=train, y=train,
			batch_size=args.batch,
			validation_data=(test, test),
			epochs= args.epochs,
			callbacks=callbacks,
			shuffle= True
			)
		losses.append(history.history['loss'])
		val_losses.append(history.history['val_loss'])


	#plot the training
	fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, squeeze=False)
	fig.suptitle("losses for different latent dims")
	axs[1][1].set_title("training losses")
	axs[2][1].set_title("validation losses")
	for ld, loss, val_loss in zip(list_LD, losses, val_losses):
		axs[1][1].plot(loss, labels=str(ld))
		axs[2][1].plot(val_loss, labels=str(ld))
	plt.legend()
	plt.savefig(os.path.join(args.output_dir,"{} losses ".format(autoencoder.name)))


	#save the model
	if args.command=="training":
		autoencoder.save(os.path.join(args.output_dir, autoencoder.name), overwrite=True)
