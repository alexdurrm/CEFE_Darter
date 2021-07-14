from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Softmax, Dense, Flatten, Reshape, Conv2DTranspose, Input,UpSampling2D, ZeroPadding2D, ActivityRegularization
import tensorflow as tf
import tensorflow.keras as K
import numpy as np

from CommonAE import *

#TODO: rename layers and check coherence with clusterize

#####################################################################################
#
#										CONVOLUTIONAL
#
#####################################################################################

class Convolutional(Model):
	def __init__(self, latent_dim, pred_shape, name="Convolutional"):
		super(Convolutional, self).__init__(name=name)
		self.latent_dim = latent_dim
		self.encoder = K.Sequential([
			Input(pred_shape, name="input_encoder"),
			Conv2D(32, kernel_size=(3,3), padding="same", activation="relu"),
			MaxPool2D(pool_size=(2,2), padding="same"),
			Conv2D(32, kernel_size=(3,3), padding="same", activation="relu"),
			MaxPool2D(pool_size=(2,2), padding="same"),
			Conv2D(64, kernel_size=(3,3), padding="same", activation="relu"),
			MaxPool2D(pool_size=(2,2), padding="same"),
			Conv2D(128, kernel_size=(3,3), padding="same", activation="relu"),
			MaxPool2D(pool_size=(2,2), padding="same", name="last_pool_encoder"),
			Flatten(),
			Dense(latent_dim, activation="relu", name="output_encoder"),
		])
		shape_conv = self.encoder.get_layer("last_pool_encoder").output_shape

		self.decoder = K.Sequential([
			Input(latent_dim, name="input_decoder"),
			Dense(np.prod(shape_conv[1:]), activation="relu"),
			Reshape(shape_conv[1:]),
			UpSampling2D(size=(2,2)),
			Conv2DTranspose(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
			UpSampling2D(size=(2,2)),
			Conv2DTranspose(filters=32, kernel_size=(3,3), padding="same", activation="relu"),
			UpSampling2D(size=(2,2)),
			Conv2DTranspose(filters=32, kernel_size=(3,3), padding="same", activation="relu"),
			UpSampling2D(size=(2,2)),
			Conv2DTranspose(filters=pred_shape[-1], kernel_size=(3,3), padding="same", activation='sigmoid', name="output_decoder")
		])

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded


#####################################################################################
#
#										PERCEPTRON
#
#####################################################################################

class Perceptron(Model):
	def __init__(self, latent_dim, pred_shape, name="Perceptron"):
		super(Perceptron, self).__init__(name=name)
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


#####################################################################################
#
#									SPARSE CONVOLUTIONAL
#
#####################################################################################

class SparseConvolutional(Model):
	def __init__(self, latent_dim, pred_shape, l1_regulizer, name="SparseConvolutional"):
		super(SparseConvolutional, self).__init__(name=name)
		self.latent_dim = latent_dim
		self.encoder = K.Sequential([
			Input(pred_shape, name="input_encoder"),
			Conv2D(32, kernel_size=(3,3), padding="same", activation="relu"),
			MaxPool2D(pool_size=(2,2), padding="same"),
			Conv2D(32, kernel_size=(3,3), padding="same", activation="relu"),
			MaxPool2D(pool_size=(2,2), padding="same"),
			Conv2D(64, kernel_size=(3,3), padding="same", activation="relu"),
			MaxPool2D(pool_size=(2,2), padding="same"),
			Conv2D(128, kernel_size=(3,3), padding="same", activation="relu"),
			MaxPool2D(pool_size=(2,2), padding="same", name="last_pool_encoder"),
			Flatten(),
			Dense(latent_dim, activation="relu", name="output_encoder"),
			ActivityRegularization(l1=l1_regulizer)
		])
		shape_conv = self.encoder.get_layer("last_pool_encoder").output_shape

		self.decoder = K.Sequential([
			Input(latent_dim, name="input_decoder"),
			Dense(np.prod(shape_conv[1:]), activation="relu"),
			Reshape(shape_conv[1:]),
			UpSampling2D(size=(2,2)),
			Conv2DTranspose(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
			UpSampling2D(size=(2,2)),
			Conv2DTranspose(filters=32, kernel_size=(3,3), padding="same", activation="relu"),
			UpSampling2D(size=(2,2)),
			Conv2DTranspose(filters=32, kernel_size=(3,3), padding="same", activation="relu"),
			UpSampling2D(size=(2,2)),
			Conv2DTranspose(filters=pred_shape[-1], kernel_size=(3,3), padding="same", activation='sigmoid', name="output_decoder")
		])

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded


#####################################################################################
#
#								VARIATIONAL AUTOENCODER
#
#####################################################################################

class VariationalAE(Model):
	"""Convolutional variational autoencoder."""
	def __init__(self, latent_dim, pred_shape, name="VAE"):
		super(VariationalAE, self).__init__(name=name)
		self.latent_dim = latent_dim
		self.encoder = K.Sequential([
			Conv2D(32, kernel_size=(3,3), padding="same"),
			MaxPool2D(pool_size=(2,2), padding="same"),
			Conv2D(64, kernel_size=(3,3), padding="same"),
			MaxPool2D(pool_size=(2,2), padding="same"),
			Conv2D(latent_dim*2, kernel_size=(3,3), padding="same"),
		])
		self.decoder = K.Sequential([
			UpSampling2D(size=(2,2)),
			Conv2DTranspose(filters=32, kernel_size=(3,3), padding="same"),
			UpSampling2D(size=(2,2)),
			Conv2DTranspose(filters=pred_shape[-1], kernel_size=(3,3), padding="same", activation='sigmoid')
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

#####################################################################################
#
#								VGG16 AUTOENCODER
#
#####################################################################################

class VGG16AE(Model):
	def __init__(self, latent_dim, pred_shape, name="VGG16AE"):
		super(VGG16AE, self).__init__(name=name)
		self.latent_dim = latent_dim
		self.encoder = tf.keras.Sequential([
			VGG16(weights='imagenet', include_top=False,
							 input_shape=pred_shape)
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

			Conv2D(pred_shape[-1], 3, activation = 'sigmoid', padding = 'same', name = 'conv11'),
		])

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

#####################################################################################
#
#								FUNCTIONS
#
#####################################################################################
def get_model(model_type, pred_shape, latent_dim, verbosity=0):
	K.backend.clear_session()

	if verbosity>=1: print(model_type)
	if model_type=="perceptron":
		model = Perceptron(latent_dim, pred_shape)
	elif model_type=="sparse_convolutional":
		model = SparseConvolutional(latent_dim, pred_shape, 0.00001)
	elif model_type=="variational_AE":
		model = VariationalAE(latent_dim, pred_shape)
	elif model_type=="VGG16AE":
		model = VGG16AE(latent_dim, pred_shape)
	elif model_type == "convolutional":
		model = Convolutional(latent_dim, pred_shape)
	else:
		raise ValueError("Unknown model type: {}".format(model_type))
	if verbosity>=1:
		print(model.encoder.summary())
		print(model.decoder.summary())
	return model
