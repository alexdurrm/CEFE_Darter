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

K.backend.clear_session()

class Autoencoder(Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim):
	super(Autoencoder, self).__init__()
	self.name = "VAE_LD{}".format(latent_dim)
	self.latent_dim = latent_dim
	self.encoder = K.Sequential(
		[
			K.layers.InputLayer(input_shape=(28, 28, 1)),
			K.layers.Conv2D(
				filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
			K.layers.Conv2D(
				filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
			K.layers.Flatten(),
			# No activation
			K.layers.Dense(latent_dim + latent_dim),
		]
	)

	self.decoder = K.Sequential(
		[
			K.layers.InputLayer(input_shape=(latent_dim,)),
			K.layers.Dense(units=7*7*32, activation=tf.nn.relu),
			K.layers.Reshape(target_shape=(7, 7, 32)),
			K.layers.Conv2DTranspose(
				filters=64, kernel_size=3, strides=2, padding='same',
				activation='relu'),
			K.layers.Conv2DTranspose(
				filters=32, kernel_size=3, strides=2, padding='same',
				activation='relu'),
			# No activation
			K.layers.Conv2DTranspose(
				filters=1, kernel_size=3, strides=1, padding='same'),
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
	eps = tf.random.normal(shape=mean.shape)
	return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
	logits = self.decoder(z)
	if apply_sigmoid:
	  probs = tf.sigmoid(logits)
	  return probs
	return logits



def fly_over_image(image, window, stride, return_coord=False):
	img_Y = image.shape[-2]
	img_X = image.shape[-3]
	for start_x in range(0, img_X-window[0]+1, stride[0]):
		for start_y in range(0, img_Y-window[1]+1, stride[1]):
			if return_coord:
				yield (start_x, start_x+window[0], start_y, start_y+window[1])
			else:
				sample = image[start_x: start_x+window[0], start_y: start_y + window[1]]
				yield sample

def rgb_2_darter(image):
	im_out = np.zeros([image.shape[0], image.shape[1], 3], dtype = np.float32)

	im_out[:, :, 1] = (140.7718694130528 +
		0.021721843447502408  * image[:, :, 0] +
		0.6777093385296341    * image[:, :, 1] +
		0.2718422677618606    * image[:, :, 2] +
		1.831294521246718E-8  * image[:, :, 0] * image[:, :, 1] +
		3.356941424659517E-7  * image[:, :, 0] * image[:, :, 2] +
		-1.181401963067949E-8 * image[:, :, 1] * image[:, :, 2])
	im_out[:, :, 0] = (329.4869869234302 +
		0.5254935133632187    * image[:, :, 0] +
		0.3540642397052902    * image[:, :, 1] +
		0.0907634883372674    * image[:, :, 2] +
		9.245344681241058E-7  * image[:, :, 0] * image[:, :, 1] +
		-6.975682782165032E-7 * image[:, :, 0] * image[:, :, 2] +
		5.828585657562557E-8  * image[:, :, 1] * image[:, :, 2])

	return im_out

def normalize(set_img):
	#set_img = (set_img - np.mean(set_img)) / np.std(set_img)
	return (set_img - np.min(set_img)) / (np.max(set_img) - np.min(set_img)).astype(np.float32)

def preprocess_habitat_image(image, new_shape, color_channels):
	image = cv2.resize(image, dsize=(new_shape[::-1]), interpolation=cv2.INTER_CUBIC)
	if color_channels==1:
		image = rgb_2_darter(image)
		image = image[..., 0]+image[..., 1]
		image = image[..., np.newaxis]
	elif color_channels==3:
		image = image/np.max(image)
	else:
		raise ValueError
	return image

def augment(set_img, prediction_shape):
	augmented = []
	for idx, img in enumerate(set_img):
		for sample in fly_over_image(img, prediction_shape, prediction_shape):
			augmented += [sample, np.flip(sample, axis=(-2))]
	return np.array(augmented)

def get_fitting_data(glob_imgs, resize_shape, pred_shape, color_channels):
	#list all image path
	habitat_path = glob(glob_imgs)
	print("number of images for training: {}".format(len(habitat_path)))
	#load all images and preprocess them
	habitat_img = np.empty(shape=(len(habitat_path), *resize_shape, color_channels))
	for i, path in enumerate(habitat_path):
		img = imageio.imread(path)
		habitat_img[i] = preprocess_habitat_image(img, resize_shape, color_channels)
	#split in train and test
	train, test = train_test_split(habitat_img, train_size=0.8, shuffle=True)
	#augment and normalize train and test
	test = augment(test, pred_shape)
	train = augment(train, pred_shape)
	test = normalize(test)
	train = normalize(train)
	#show one of these images
	plt.imshow(train[np.random.randint(0, len(train))], cmap='gray')
	plt.show()
	return (train, test)


def show_predictions(model, sample_test, n=10):
	prediction = model.predict(sample_test)
	plt.figure(figsize=(20, 4))
	for i in range(n):
		rdm=np.random.randint(0, len(sample_test))
		# display original
		ax = plt.subplot(3, n, i + 1)
		plt.imshow(sample_test[rdm])
		plt.title("original")
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# display reconstruction
		ax = plt.subplot(2, n, i + 1 + n)
		plt.imshow(prediction[rdm])
		plt.title("reconstructed")
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("glob", help="glob of the image directory to open")
	args = parser.parse_args()

	#parameters
	dir_saved_models ="Scripts/AutoEncoders/TrainedModels"
	color_channels=3
	pred_shape=(128,128)
	resize_shape=(900,300)#(1536, 512)
	latent_dim=200
	batch_size=70
	epochs=30

	#prepare the network
	autoencoder = Autoencoder(latent_dim, color_channels)
	autoencoder.compile(optimizer='adam', loss='mse')
	#prepare the data and train the network
	train, test = get_fitting_data(args.glob, resize_shape, pred_shape, color_channels)
	callback = K.callbacks.EarlyStopping(monitor='val_loss', patience=5)
	history = autoencoder.fit(x=train, y=train,
		batch_size=batch_size,
		validation_data=(test, test),
		epochs= epochs,
		callbacks= callback,
		shuffle= True
		)
	#plot the training
	plt.plot(history.history['loss'], label="train")
	plt.plot(history.history['val_loss'], label="val")
	plt.legend()
	plt.show()
	#plot the predictions
	show_predictions(autoencoder, test)
	#save the model
	if not os.path.exists(dir_saved_models):
		os.makedirs(dir_saved_models)
	autoencoder.save(os.path.join(dir_saved_models, autoencoder.name), overwrite=True)
