import argparse
from sklearn.model_selection import train_test_split
import tensorflow.keras as K
from tensorflow.keras.layers import Conv2D, MaxPool2D, Softmax, Dense, Flatten, Reshape, Conv2DTranspose, UpSampling2D, ZeroPadding2D, ActivityRegularization
from tensorflow.keras.models import Model
import numpy as np
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import imageio
import cv2

# class MyRegularizer(regularizers.Regularizer):
#   # custom class regularizer
#     def __init__(self, strength):
#         self.strength = strength
#     def __call__(self, x):
#         return self.strength * tf.reduce_sum(tf.square(x))
#     def get_config(self):
#         return {'strength': self.strength}

class Autoencoder(Model):
	def __init__(self, name, latent_dim, color_channels, l1_regulizer):
		super(Autoencoder, self).__init__(name=name)
		self.latent_dim = latent_dim
		self.encoder = K.Sequential([
			Conv2D(latent_dim, kernel_size=(3,3), padding="same"),
			MaxPool2D(pool_size=(3,3), padding="same"),
			Conv2D(latent_dim//2, kernel_size=(3,3), padding="same"),
			MaxPool2D(pool_size=(3,3), padding="same"),
			Conv2D(latent_dim//4, kernel_size=(3,3), padding="same"),
			ActivityRegularization(l1=l1_regulizer)
		])
		self.decoder = K.Sequential([
			UpSampling2D(size=(3,3)),
			Conv2DTranspose(filters=color_channels//2, kernel_size=(3,3), padding="same"),
			UpSampling2D(size=(3,3)),
			Conv2DTranspose(filters=color_channels, kernel_size=(3,3), padding="same")
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
	BATCH_SIZE=50
	EPOCHS=30
	NETWORK_NAME="SparseConvolutionnal"
	LOSS='mse'
	VERBOSE=10
	L1_REGULIZER=0.001

	parser = argparse.ArgumentParser()
	parser.add_argument("path_train", help="path of the training dataset to use")
	parser.add_argument("path_test", help="path of the testing dataset to use")
	parser.add_argument("--output_dir", default=DIR_SAVED_MODELS, help="path where to save the trained network, default {}".format(DIR_SAVED_MODELS))
	parser.add_argument("-l", "--latent_dim", type=int, default=LATENT_DIM, help="the latent dimention, default {}".format(LATENT_DIM))
	parser.add_argument("-b", "--batch", type=int, default=BATCH_SIZE, help="batch size for training, default {}".format(BATCH_SIZE))
	parser.add_argument("-e", "--epochs", type=int, default=EPOCHS, help="number of epochs max for training, default {}".format(EPOCHS))
	parser.add_argument("-n", "--name", type=str, default=NETWORK_NAME, help="network name, default {}".format(NETWORK_NAME))
	parser.add_argument("-r", "--regulizer_l1", type=float, default=L1_REGULIZER, help="coefficient on the l1 regulizer, default {}".format(L1_REGULIZER))
	parser.add_argument("--loss", type=str, choices=['mse', 'ssim'], default=LOSS, help="network loss, default {}".format(LOSS))
	parser.add_argument("-v", "--verbose", default=VERBOSE, type=int, help="set number of predictions to show, if no result directory is given the graphs are not saved, default: {}".format(VERBOSE))
	args = parser.parse_args()
	assert  args.regulizer_l1>=0 and args.regulizer_l1 <= 1, "regulizer should be a float between 0 and 1, here: {}".format(args.regulizer_l1)

	#prepare the data
	train = np.load(args.path_train)
	test = np.load(args.path_test)
	assert train.shape[1:]==test.shape[1:], "train and test should contain images of similar shape, here {} and {}".format(train.shape[1:], test.shape[1:])

	prediction_shape = train.shape[1:]
	dataset_descriptor, *_ = os.path.split(args.path_train)[-1].split('_') #get the descriptor of the dataset

	#prepare the network
	network_name = "{}_{}_LD{}_pred{}x{}x{}".format(args.name, dataset_descriptor, args.latent_dim, *prediction_shape)
	autoencoder = Autoencoder(network_name, args.latent_dim, prediction_shape[-1], args.regulizer_l1)
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

	#save the model
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
		autoencoder.save(os.path.join(args.output_dir, autoencoder.name), overwrite=True)

	#plot the training
	plt.plot(history.history['loss'], label="train")
	plt.plot(history.history['val_loss'], label="val")
	plt.legend()
	plt.show()
	plt.savefig(os.path.join(args.output_dir, "{} training losses".format(autoencoder.name)))

	#plot the predictions
	autoencoder.show_predictions(sample_test=test, n=args.verbose, saving_dir=args.output_dir)
