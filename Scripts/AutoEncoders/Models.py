from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Conv2D, MaxPool2D, Softmax, Dense, Flatten, Reshape, Conv2DTranspose, Input,UpSampling2D, ZeroPadding2D, ActivityRegularization
import argparse
import numpy as np

from CommonAE import *


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

def load_model(model_type, model_name, pred_shape, latent_dim, loss ):
	K.backend.clear_session()

	if model_type=="perceptron":
		model = Perceptron(latent_dim, pred_shape, name=model_name)
	elif model_type=="sparse_convolutional":
		print("sparse_convolutional")
		model = SparseConvolutional(latent_dim, pred_shape, 0.1, name=model_name)
	elif model_type=="variational_AE":
		model = VariationalAE(latent_dim, pred_shape, name=model_name)
	elif model_type=="VGG16AE":
		model = VGG16AE(latent_dim, pred_shape, name=model_name)
	elif model_type == "convolutional":
		print("convolutional")
		model = Convolutional(latent_dim, pred_shape, name=model_name)
	else:
		raise ValueError("Unknown model type: {}".format(model_type))
	if loss=="ssim":
		loss = get_SSIM_Loss
	model.compile(optimizer='adam', loss=loss)
	print(model.encoder.summary())
	print(model.decoder.summary())
	return model

def get_callbacks(model_type, test, output_dir, verbosity):
	#prepare callbacks
	callbacks = [K.callbacks.EarlyStopping(monitor='val_loss', patience=6),
				K.callbacks.EarlyStopping(monitor='loss', patience=6),
				SavePredictionSample(n_samples=verbosity, val_data=test[0:5*20:5], saving_dir=output_dir),
				SaveActivations(val_img=test[0], saving_dir=output_dir)]
	if model_type=="variational_AE":
		callbacks.append(SaveSampling(test.shape[1:], n_samples=verbosity, saving_dir=output_dir))
	return callbacks

def train_model(model, train, test, epochs, batch_size, callbacks=[], output_dir=None):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	#train the network
	history = model.fit(x=train, y=train,
		batch_size=batch_size,
		validation_data=(test, test),
		epochs= epochs,
		callbacks= callbacks,
		shuffle= True
		)
	#plot the training losses
	plot_training_losses(history.history['loss'], history.history['val_loss'],
		title="losses train and test",
		save_path=os.path.join(output_dir,"losses {}".format(model.name)))

	#print("Final test: ", test_model(model, test, output_dir))
	if output_dir:
		model.save(os.path.join(output_dir, model.name), overwrite=True)
	return history


def test_model(model, test, output_dir=None, verbosity=5):
	predictions = model.predict(test)
	print(predictions.shape)
	print(test.shape)
	show_predictions(test, predictions, verbosity, "predictions {}".format(model.name), saving_dir=output_dir)
	if output_dir:
		np.save(os.path.join(output_dir,"predictions.npy"), predictions)
	print(model.encoder.summary())
	print(model.decoder.summary())
	return model.evaluate(test, test)


def LD_selection(model_type, model_name, train, test, epochs, batch_size, list_LD, loss, dir_results, verbosity):
	losses = []
	val_losses = []
	prediction_shape = test.shape[1:]
	for latent_dim in list_LD:
		#prepare the network
		network_name = model_name+"_LD{}_pred{}x{}x{}".format(latent_dim, *prediction_shape)
		autoencoder = load_model(model_type, network_name, prediction_shape, latent_dim, loss)

		#prepare the output path
		net_output_dir = os.path.join(dir_results, network_name)

		callbacks = get_callbacks(model_type, test, net_output_dir, verbosity)

		history = train_model(autoencoder, train, test, epochs, batch_size, callbacks, output_dir=net_output_dir)
		losses.append(history.history['loss'])
		val_losses.append(history.history['val_loss'])


	#plot the best validation for each latent dim
	best_losses=[]
	best_val_losses=[]
	for val_loss, loss in zip(val_losses, losses):
		best_losses.append(min(loss))
		best_val_losses.append(min(val_loss))
	plot_loss_per_ld(best_losses, best_val_losses, list_LD,
		title="best losses per latent dim for {}".format(autoencoder.name),
		save_path=os.path.join(dir_results, "best losses {}".format(autoencoder.name))
		)

def main(args):
	#prepare the data
	test = np.load(args.path_test)
	#prepare directory output
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	#TEST
	if args.command == "test":
		model = K.models.load_model(args.model_path)
		res = test_model(model, test, output_dir=args.output_dir, verbosity=args.verbose)
		print(res)
	else:
		train = np.load(args.path_train)
		assert train.shape[1:]==test.shape[1:], "train and test should contain images of similar shape, here {} and {}".format(train.shape[1:], test.shape[1:])
		try:
			_, dataset_descriptor, *_ = os.path.split(args.path_train)[-1].split('_') #get the descriptor of the dataset
		except ValueError:
			dataset_descriptor = ""
		#LATENT DIM SEARCH
		if args.command == "LD_selection":
			model_name = "{}_{}".format(args.name, dataset_descriptor)
			LD_selection(args.model_type, model_name, train, test, args.epochs, args.batch ,args.latent_dim, args.loss, args.output_dir, args.verbose)
		#TRAINING
		elif args.command == "train":
			model_name = "{}_{}_LD{}_pred{}x{}x{}".format(args.name, dataset_descriptor, args.latent_dim, *test.shape[1:])
			model = load_model(args.model_type, model_name, test.shape[1:], args.latent_dim, args.loss)
			callbacks = get_callbacks(args.model_type, test, args.output_dir, args.verbose)
			train_model(model, train, test, args.epochs, args.batch, callbacks, output_dir=args.output_dir)
		else:
			raise ValueError("Unknown command: {}".format(args.command))


if __name__=="__main__":
	LOSS="mse"
	DIR_SAVED_MODELS="Results"
	VERBOSE=0
	model_types = ["convolutional", "perceptron", "sparse_convolutional", "variational_AE", "VGG16AE"]

	parser = argparse.ArgumentParser(description="Script used to train, test and research optimal latend dim on different Autoencoders")
	parser.add_argument("path_test", help="path of the testing dataset to use")
	parser.add_argument("--loss", type=str, choices=['mse', 'ssim'], default=LOSS, help="network loss, default {}".format(LOSS))
	parser.add_argument("--output_dir", default=DIR_SAVED_MODELS, help="path where to save the trained network, default {}".format(DIR_SAVED_MODELS))
	parser.add_argument("-n", "--name", type=str, default=None, help="network name, default {}".format(None))
	parser.add_argument("-v", "--verbose", default=VERBOSE, type=int, help="set verbosity, default: {}".format(VERBOSE))
	subparsers = parser.add_subparsers(title="command", dest="command", help='action to perform')

	EPOCHS = 50
	BATCH_SIZE = 25
	#specific parameters for multiple trainings with diverse latent dims
	LATENT_DIM_SPACE = [4,8,16,32,64,128,256]
	LD_parser = subparsers.add_parser("LD_selection")
	LD_parser.add_argument("model_type", choices=model_types, help='The architecture of the model used')
	LD_parser.add_argument("path_train", help="path of the training dataset to use")
	LD_parser.add_argument("-l", "--latent_dim", type=int, nargs="+", default=LATENT_DIM_SPACE, help="the latent dimension that will be tested, default {}".format(LATENT_DIM_SPACE))
	LD_parser.add_argument("-b", "--batch", type=int, default=BATCH_SIZE, help="batch size for training, default {}".format(BATCH_SIZE))
	LD_parser.add_argument("-e", "--epochs", type=int, default=EPOCHS, help="number of epochs max for training, default {}".format(EPOCHS))

	#specific parameters for a simple training
	LATENT_DIM = 8
	training_parser = subparsers.add_parser("train")
	training_parser.add_argument("model_type", choices=model_types, help='The architecture of the model used')
	training_parser.add_argument("path_train", help="path of the training dataset to use")
	training_parser.add_argument("-l", "--latent_dim", type=int, default=LATENT_DIM, help="the latent dimention, default {}".format(LATENT_DIM))
	training_parser.add_argument("-b", "--batch", type=int, default=BATCH_SIZE, help="batch size for training, default {}".format(BATCH_SIZE))
	training_parser.add_argument("-e", "--epochs", type=int, default=EPOCHS, help="number of epochs max for training, default {}".format(EPOCHS))

	#specific parameters for a simple testing
	test_parser = subparsers.add_parser("test")
	test_parser.add_argument("model_path", help="path to the model to load")

	args = parser.parse_args()
	main(args)
