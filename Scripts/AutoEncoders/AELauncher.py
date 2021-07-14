import tensorflow as tf
import tensorflow.keras as K
from sklearn.model_selection import train_test_split
import argparse
import numpy as np


tf.random.set_seed(500)
np.random.seed(500)

from Models import *
from CommonAE import *

def get_callbacks(model_type, test, output_dir, save_activations, early_stopping, sample_preds, verbosity=0):
	#prepare callbacks
	if verbosity>=1: print("Retrieving callbacks save_activation:{}, early_stopping:{}, sample_preds:{}".format(save_activations, early_stopping, sample_preds))
	#select a sample of test
	#append callbacks
	callbacks = []
	if save_activations:
		# callbacks.append(SaveActivations(val_img=test[0], saving_dir=output_dir))
		print("No activation")
	if early_stopping:
		callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss', patience=6))
		callbacks.append(K.callbacks.EarlyStopping(monitor='loss', patience=6))
	if sample_preds: 
		sample = test[[i*10%len(test) for i in range(sample_preds)]]
		callbacks.append(SavePredictionSample(sample=sample, saving_dir=output_dir, verbosity=verbosity-1))
		if model_type=="variational_AE":
			callbacks.append(SaveSampling(sample_preds, test.shape[1:], saving_dir=output_dir))
	return callbacks

def get_augmentation(output_shape):
	return K.Sequential([
		K.layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
		K.layers.experimental.preprocessing.RandomRotation(0.2),
		K.layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
		K.layers.experimental.preprocessing.RandomCrop(height=output_shape[0]//2 , width=output_shape[1]//2),
		K.layers.experimental.preprocessing.Resizing(output_shape[0], output_shape[1])
		], name="data_augmentation")

def train_model(model, train, test, loss_func, output_dir, epochs, batch_size, callbacks, augmenter=None, verbosity=0):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	#wrap the network if augmentation needed
	wrap_model = K.Sequential([augmenter, model]) if augmenter else K.Sequential([model])
	wrap_model.compile("Adam", loss=loss_func)
	#train the network
	history = wrap_model.fit(x=train, y=train,
		batch_size=batch_size,
		validation_data=(test, test),
		epochs= epochs,
		callbacks= callbacks,
		shuffle= True
		)
	#plot the training losses
	plot_training_losses(history.history['loss'], history.history['val_loss'],
		title="losses train and test",
		save_path=os.path.join(output_dir,"losses {}".format(model.name)),
		verbosity=verbosity-1)
	#save the model
	if output_dir:
		model.save(os.path.join(output_dir, model.name), overwrite=True)
	return history

def load_train_test(train_path, test_path=None, descriptor=False, verbose=0):
	"""
	given at least one path for a numpy dataset
	return a train and test numpy array
	if test_path is None test is obtained by splitting 10% of the dataset
	if descriptor is True returns also a descripptor name from the given train_path
	"""
	train = np.load(train_path)
	if test_path:
		test = np.load(test_path)
		if verbose>=1: print("loading train and test from {} ad {}".format(train_path, test_path))
	else:
		train, test = train_test_split(data, test_size=0.1)
		if verbose>=1: print("splitting dataset {} in 90% train and 10% test".format(train_path))
	#do verification
	assert train.shape[1:]==test.shape[1:], "train and test should contain images of similar shape, here {} and {}".format(train.shape[1:], test.shape[1:])
	#if true return also a dataset descriptor
	if descriptor:
		try:
			_, dataset_descriptor, *_ = os.path.split(train_path)[-1].split('_') #get the descriptor of the dataset
		except ValueError:
			dataset_descriptor = ""
		return train, test, dataset_descriptor
	else:
		return train, test

def LD_selection(model_type, train, test, epochs, batch_size, list_LD, loss_func, output_dir, save_activations, early_stopping, sample_preds, augmenter=None, verbosity=0):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	losses, val_losses = [], []
	prediction_shape = train.shape[1:]
	augmenter = get_augmentation(prediction_shape)
	#for each latent dim train a network
	for latent_dim in list_LD:
		#prepare the network
		network_name = model_type+"_LD{}_pred{}x{}x{}".format(latent_dim, *prediction_shape)
		autoencoder = get_model(model_type, prediction_shape, latent_dim, verbosity-1)
		#train and save the network
		net_output_dir = os.path.join(output_dir, network_name)
		callbacks = get_callbacks(model_type, test, net_output_dir, save_activations, early_stopping, sample_preds)
		history = train_model(autoencoder, train, test, loss_func, net_output_dir, epochs, batch_size, callbacks, augmenter, verbosity-1)
		#store losses
		losses.append(history.history['loss'])
		val_losses.append(history.history['val_loss'])
	#plot the best validation for each latent dim
	best_losses, best_val_losses = [],[]
	for val_loss, loss in zip(val_losses, losses):
		best_losses.append(min(loss))
		best_val_losses.append(min(val_loss))
	plot_loss_per_ld(best_losses, best_val_losses, list_LD,
		title="best losses per latent dim for {}".format(autoencoder.name),
		save_path=os.path.join(output_dir, "best losses {}".format(autoencoder.name))
		)

def test_model(model_path, test, loss, sample_preds, output_dir=None, verbosity=0):
	"""
	given a model, a test dataset, and a loss function
	will output in output_dir the results of the model prediction and a sample of it
	"""
	#prepare directory output
	output_dir = output_dir if output_dir else model_path
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	# load model and save predictions
	model = tf.keras.models.load_model(model_path, compile=False)
	model.compile(optimizer="Adam", loss=loss)
	predictions = model.predict(test)
	np.save(os.path.join(output_dir,"predictions.npy"), predictions)
	#save a sample of these predictions
	selection = [i*10%len(test) for i in range(sample_preds)]
	sample_test, sample_preds = test[selection], predictions[selection]
	show_predictions(sample_test, sample_preds, "predictions {}".format(model.name), output_dir, verbosity-1)
	return model.evaluate(test, test)	#return an evaluation of the model with input test compared to expected test values

def main(args):
	"""
	handle the training, testing, or selection of the latent dimension for the model
	"""
	#prepare directory output
	output_dir = os.path.abspath(args.output_dir)
	
	#get loss
	loss = get_loss_from_name(args.loss)

	#TEST
	if args.command == "test":
		dataset = np.load(args.dataset)
		res = test_model(args.model_path, dataset, loss, args.sample_preds, output_dir, args.verbose)
		print(res)
	# TRAIN
	elif args.command == "train":
		train, test = load_train_test(train_path=args.dataset, test_path=args.test, descriptor=False, verbose=args.verbose)
		augmenter = get_augmentation(train.shape[1:]) if args.data_augment else None
		callbacks = get_callbacks(args.model_type, test, output_dir, args.save_activations, args.early_stopping, args.sample_preds, args.verbose)
		if isinstance(args.latent_dim, int):
			model = get_model(args.model_type, train.shape[1:], args.latent_dim, verbosity=args.verbose)
			train_model(model, train, test, loss, output_dir, args.epochs, args.batch_size, callbacks, augmenter, args.verbose)
		else:
			LD_selection(args.model_type, train, test, args.epochs, args.batch_size, args.latent_dim, loss, output_dir, 
				args.save_activations, args.early_stopping, args.sample_preds, 
				augmenter, args.verbose)

	else:
		raise ValueError("Unknown command: {}".format(args.command))
		

if __name__=="__main__":
	LOSS="mse"
	DIR_SAVED_MODELS="Results"
	VERBOSE=0
	model_types = ["convolutional", "perceptron", "sparse_convolutional", "variational_AE", "VGG16AE"]
	OPTIMIZER="Adam"
	SAMPLE_PREDS=4

	parser = argparse.ArgumentParser(description="Script used to train, test and research optimal latend dim on different Autoencoders")
	parser.add_argument("dataset", help="path of the numpy dataset dataset to use")
	parser.add_argument("--loss", type=str, choices=['mse', 'ssim'], default=LOSS, help="network loss, default {}".format(LOSS))
	parser.add_argument("--output_dir", default=DIR_SAVED_MODELS, help="path where to save the trained network, default {}".format(DIR_SAVED_MODELS))
	parser.add_argument("--sample_preds", type=int, default=SAMPLE_PREDS, help="number of predictions image sample to save per epoch, default {}".format(SAMPLE_PREDS))
	parser.add_argument("-v", "--verbose", default=VERBOSE, type=int, help="set verbosity, default: {}".format(VERBOSE))
	subparsers = parser.add_subparsers(title="command", dest="command", help='action to perform')

	#specific parameters for a simple training
	EPOCHS = 50
	BATCH_SIZE = 25
	LATENT_DIM = 8
	training_parser = subparsers.add_parser("train")
	training_parser.add_argument("model_type", choices=model_types, help='The architecture of the model used')
	training_parser.add_argument("--test", type=str, default=None, help="optionnal path to a numpy used as test, if None given split 10 percent of the dataset, default None")	
	training_parser.add_argument("-l", "--latent_dim", type=int, nargs='+', default=LATENT_DIM, help="the latent dimention, if multiple are given multiple networks will be trained, default {}".format(LATENT_DIM))
	training_parser.add_argument("-b", "--batch", type=int, default=BATCH_SIZE, help="batch size for training, default {}".format(BATCH_SIZE))
	training_parser.add_argument("-e", "--epochs", type=int, default=EPOCHS, help="number of epochs max for training, default {}".format(EPOCHS))
	# training_parser.add_argument("--optimizer", type=str, choices=['Adam'], default=OPTIMIZER, help="network optimizer, default {}".format(OPTIMIZER))
	training_parser.add_argument("--data_augment", action='store_true', default=False, help="option to use in-training data augmentation, default to False")
	training_parser.add_argument("--save_activations", action='store_true', default=False, help="option to save mean activation layers")
	training_parser.add_argument("--early_stopping", action='store_true', default=False, help="if model training should be stopped when loss do not progress")
	
	#specific parameters for a simple testing
	test_parser = subparsers.add_parser("test")
	test_parser.add_argument("model_path", help="path to the model to load")

	args = parser.parse_args()
	main(args)
