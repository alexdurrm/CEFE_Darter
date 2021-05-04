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

	def show_predictions(self, sample_test, n, saving_dir=None):
		"""
		plot test sample images and their reconstruction by the network
		"""
		prediction = self.call(sample_test)
		for i in range(n):
			rdm = np.random.randint(0, len(sample_test))
			plt.figure()
			plt.title("{} reconstructions img {}".format(self.name, rdm))
			# display original
			ax = plt.subplot(1, 2, 1)
			plt.imshow(sample_test[rdm], cmap='gray')
			plt.title("original")

			# display reconstruction
			ax = plt.subplot(1, 2, 2)
			plt.imshow(prediction[rdm], cmap='gray')
			plt.title("reconstructed")
			plt.show()
		if saving_dir:
			plt.savefig(os.path.join(saving_dir,"{} reconstructions img {}".format(self.name, rdm)))


if __name__ == '__main__':
	#default parameters
	DIR_SAVED_MODELS ="Scripts/AutoEncoders/TrainedModels"
	LATENT_DIM=20
	LATENT_DIM_SPACE = [2,5,10,20,40,80,160]
	BATCH_SIZE=50
	EPOCHS=30
	NETWORK_NAME="Perceptron"
	LOSS='mse'

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
	histories = []
	fig, ax = plt.subplots()
	for i, latent_dim in enumerate(list_LD):
		#prepare the network
		network_name = "{}_{}_LD{}_pred{}x{}x{}".format(args.name, dataset_descriptor, args.latent_dim, *prediction_shape)
		autoencoder = Autoencoder(network_name, args.latent_dim, prediction_shape)
		autoencoder.compile(optimizer='adam', loss=args.loss)

		#train the network
		callback = K.callbacks.EarlyStopping(monitor='val_loss', patience=5)
		histories.append(autoencoder.fit(x=train, y=train,
			batch_size=args.batch,
			validation_data=(test, test),
			epochs= args.epochs,
			callbacks= callback,
			shuffle= True
			))

		#save the model
		if not os.path.exists(args.output_dir):
			os.makedirs(args.output_dir)
			autoencoder.save(os.path.join(args.output_dir, autoencoder.name), overwrite=True)

		#plot the training
		plt.plot(histories[i].history['loss'], label="train")
		plt.plot(histories[i].history['val_loss'], label="val")

		#plot the predictions
		autoencoder.show_predictions(sample_test=test, n=args.verbose, saving_dir=args.output_dir)
		
	if args.command=="LD_selection":
		name_without_ld = '_'.join([*network_name.split("_")[:2], *network_name.split("_")[3:]])
		name_figure = "training losses per latent dim\n network {}".format(name_without_ld)
	elif args.command=="training":
		name_figure = "{} training losses".format(autoencoder.name)
	ax.legend()
	fig.savefig(os.path.join(args.output_dir, name_figure))
	plt.show()