import tensorflow.keras as K
from tensorflow.data import Dataset
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import os

from AutoEncoders.Models import get_model, get_augmentation
from AutoEncoders.CommonAE import get_loss_from_name, show_predictions, plot_training_losses, get_callbacks
from Utils.FileManagement import save_args

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
	assert train.min()>=0 and test.min()>=0, "train and test should contain values between 0 and 1, here min {} and {}".format(train.min(), test.min())
	assert train.max()<=1 and test.max()<=1, "train and test should contain values between 0 and 1, here max {} and {}".format(train.max(), test.max())
	#if true return also a dataset descriptor
	if descriptor:
		try:
			_, dataset_descriptor, *_ = os.path.split(train_path)[-1].split('_') #get the descriptor of the dataset
		except ValueError:
			dataset_descriptor = ""
		return train, test, dataset_descriptor
	else:
		return train, test


def train_model(model_type, model_path, train, test, epochs, batch_size, loss_name, learning_rate, output_dir, save_activations, early_stopping, sample_preds, latent_dim, do_augment, lr_scheduler, verbosity=0):
	#prepare the network directory
	prediction_shape = train.shape[1:]
	network_name = "{}_{}_LD{}_pred{}x{}x{}".format(model_type, loss_name, latent_dim, *prediction_shape)
	net_output_dir = os.path.join(output_dir, network_name)
	if not os.path.exists(net_output_dir):
		os.makedirs(net_output_dir)
	#get the callbacks
	callbacks = get_callbacks(model_type, test, output_dir, save_activations, early_stopping, sample_preds, lr_scheduler, verbosity-1)

	#add augmentation if needed
	if do_augment:
		raise NotImplementedError("was too long to add correctly")
		train = Dataset.from_tensor_slices((train, train))
		test = Dataset.from_tensor_slices((test, test))
		train = get_augmentation(train, prediction_shape, batch_size, shuffle=True, augment=True, verbosity=verbosity-1)
		test = get_augmentation(test, prediction_shape, batch_size, verbosity=verbosity-1)

	#get the loss
	loss_func = get_loss_from_name(loss_name)

	#get the network
	if model_path or model_type=="transfer":
		print("loading model from {}".format(model_path))
		model = K.models.load_model(model_path, compile=False)
	else:
		model = get_model(model_type, prediction_shape, latent_dim, verbosity-1)

	#train the network
	optimizer = K.optimizers.Adam(learning_rate=learning_rate)
	model.compile(optimizer, loss_func)
	history = model.fit(x=train, y=train,
		batch_size= batch_size,
		validation_data= (test, test),
		epochs= epochs,
		callbacks= callbacks,
		shuffle= True
		)
	#plot the training losses
	plot_training_losses(history.history['loss'], history.history['val_loss'],
		title="losses train and test",
		save_path=os.path.join(output_dir,"losses "+network_name),
		verbosity=verbosity-1)
	#save the model
	model.save(net_output_dir, overwrite=True)
	K.backend.clear_session()
	return history


def test_model(model_path, test, loss_name, sample_preds, output_dir=None, verbosity=0):
	"""
	given a model, a test dataset, and a loss function
	will output in output_dir the results of the model prediction and a sample of it
	"""
	# prepare directory output
	output_dir = output_dir if output_dir else model_path
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	#get loss
	loss_func = get_loss_from_name(loss_name)
	# load model and save predictions
	model = K.models.load_model(model_path, compile=False)
	model.compile(optimizer="Adam", loss=loss_func)
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
	#TEST
	if args.command == "test":
		dataset = np.load(args.dataset)
		res = test_model(args.model_path, dataset, args.loss, args.sample_preds, output_dir, args.verbose)
		print(res)
	# TRAIN
	elif args.command == "train":
		train, test = load_train_test(train_path=args.dataset, test_path=args.test, descriptor=False, verbose=args.verbose)

		train_model(args.model_type, args.model_path, train, test, args.epochs, args.batch, args.loss, args.learning_rate, output_dir,
			args.save_activations, args.early_stopping, args.sample_preds,
			latent_dim=args.latent_dim, do_augment=args.data_augment, lr_scheduler=args.lr_scheduler,
			verbosity=args.verbose)
	else:
		raise ValueError("Unknown command: {}".format(args.command))


if __name__=="__main__":
	LOSS="mse"
	DIR_SAVED_MODELS="Results"
	VERBOSE=0
	MODEL_TYPES = ["convolutional", "perceptron", "sparse_convolutional", "variational_AE", "VGG16AE", "transfer"]
	OPTIMIZER="Adam"
	SAMPLE_PREDS=4
	EPOCHS = 50
	BATCH_SIZE = 25
	LATENT_DIM = 8
	LR = 0.001

	parser = argparse.ArgumentParser(description="Script used to train, test and research optimal latent dim on different Autoencoders")
	parser.add_argument("dataset",type=str, help="path of the numpy dataset to use as training (and 10percent used for testing if --test not given)")
	parser.add_argument("--loss", type=str, choices=['mse', 'ssim'], default=LOSS, help="network loss, default {}".format(LOSS))
	parser.add_argument("--output_dir", default=DIR_SAVED_MODELS, help="path where to save the trained network, default {}".format(DIR_SAVED_MODELS))
	parser.add_argument("--sample_preds", type=int, default=SAMPLE_PREDS, help="number of predictions image sample to save per epoch, default {}".format(SAMPLE_PREDS))
	parser.add_argument("-v", "--verbose", default=VERBOSE, type=int, help="set verbosity, default: {}".format(VERBOSE))
	subparsers = parser.add_subparsers(title="command", dest="command", help='action to perform')

	# specific parameters for a simple training
	training_parser = subparsers.add_parser("train")
	training_parser.add_argument("model_type", choices=MODEL_TYPES, help='The architecture name of the model used')
	training_parser.add_argument("--model_path", default=None, help="path to the model to load, if set will ignore model_type")
	training_parser.add_argument("--test", type=str, default=None, help="optionnal path to a numpy used as test, if None given split 10 percent of the dataset, default None")
	training_parser.add_argument("-l", "--latent_dim", type=int, default=LATENT_DIM, help="the latent dimention, if multiple are given multiple networks will be trained, default {}".format(LATENT_DIM))
	training_parser.add_argument("-b", "--batch", type=int, default=BATCH_SIZE, help="batch size for training, default {}".format(BATCH_SIZE))
	training_parser.add_argument("-e", "--epochs", type=int, default=EPOCHS, help="number of epochs max for training, default {}".format(EPOCHS))
	training_parser.add_argument("--learning_rate", type=float, default=LR, help="Starting learning rate, default {}".format(LR))
	training_parser.add_argument("--data_augment", action='store_true', default=False, help="option to use in-training data augmentation, default to False")
	training_parser.add_argument("--save_activations", action='store_true', default=False, help="option to save mean activation layers")
	training_parser.add_argument("--early_stopping", action='store_true', default=False, help="if model training should be stopped when loss do not progress")
	training_parser.add_argument("--lr_scheduler", action='store_true', default=False, help="Add a learning rate scheduler in callbacks")
	# training_parser.add_argument("--optimizer", type=str, choices=['Adam'], default=OPTIMIZER, help="network optimizer, default {}".format(OPTIMIZER))

	#specific parameters for a simple testing
	test_parser = subparsers.add_parser("test")
	test_parser.add_argument("model_path", help="path to the model to load")

	args = parser.parse_args()
	main(args)
	save_args(args, os.path.join(args.output_dir, "AE_"+args.command+"_params.txt"))
