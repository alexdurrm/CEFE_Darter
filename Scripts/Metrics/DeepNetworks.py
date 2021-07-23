import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

def visualize_layers_activation(function):
	"""
	wrapper for get_deep_features and others that allows to visualize
	the layers activation
	"""
	def wrap(*args, **kwargs):
		verbose = kwargs.get("verbose", False)
		output = function(*args, **kwargs)
		if verbose>=2:
			fig, axs = plt.subplots(nrows=len(output))
			plt.title("activations per layers")
			for i, layer_out in enumerate(output):
				#if dense layer
				axs[i].set_title("layer "+str(i+1))
				if layer_out.ndim==2:
					y=layer_out.flatten()
					axs[i].bar(range(len(y)) , height=y)
					axs[i].set_xlabel("index of neural activation")
					axs[i].set_ylabel("amplitude of activation")
				#if convolutionnal layer
				elif layer_out.ndim==4:
					activation = np.mean(layer_out, axis=(0, -1))[..., np.newaxis]
					axs[i].imshow(activation, cmap="hot")
			plt.show()
		return output
	return wrap


@visualize_layers_activation
def get_deep_features(model, test, verbose=0):
	'''
	get the feature space of an test tensor propagated through the deep feature model
	return a list of np array, each element of the list represent an output of a layer, input layer is ignored
	'''
	if test.ndim == K.ndim(model.input)-1:
		test=test[np.newaxis, ...]
	res = [K.function([model.input], layer.output)(test) for layer in model.layers]
	if verbose>=1:
		print("DeepNetwork::get_deep_features: DF shapes {}".format([x.shape for x in res]))
	return res


def autoencoder_generate_retro_prediction(autoencoder, start, repetition):
	"""
	yield the prediction of the autoencoder and its latent space when the input is the previous prediction
	for each iteration returns the latent space correpsonding to its output
	first iteration return start image and its corresponding latent space
	"""
	if start.ndim == len(autoencoder.input.shape)-1:
		start=start[np.newaxis, ...]

	input_pxl = start
	input_latent = autoencoder.encoder(start).numpy()

	for i in range(repetition):
		#get encoded and decoded values
		output_pxl = autoencoder.decoder(input_latent).numpy()
		output_latent = autoencoder.encoder(input_pxl).numpy()

		yield input_latent, input_pxl

		input_pxl = output_pxl
		input_latent = output_latent


def divergence(autoencoder, start, repetition, verbose=0):
	gini_pxl_space, kurtois_pxl_space, entropy_pxl_space = [],[],[]
	gini_latent_space, kurtois_latent_space, entropy_latent_space, mean_latent_space = [],[],[],[]

	if start.ndim<4:
		start=start[np.newaxis, ...]
	start_latent = autoencoder.encoder(start).numpy()
	prev_pxl = start
	shift_pxl_mse = []

	prev_latent = start_latent
	shift_latent = []
	axis_latent = tuple(i for i in range(-1, -start_latent.ndim, -1))

	diff_pxl_mse, diff_latent = [],[]

	if verbose>=0: plt.figure(figsize=(20, 4))
	for i in range(repetition):
		if verbose>=0:
			ax = plt.subplot(repetition//10, 10, i+1)
			plt.imshow(prev_pxl[0], vmin=0, vmax=1)

		new_pxl = autoencoder.decoder(prev_latent).numpy()
		new_latent = autoencoder.encoder(new_pxl).numpy()
		#store the stat values
		gini_pxl_space.append(get_gini(prev_pxl))
		kurtois_pxl_space.append(kurtosis(prev_pxl, axis=None))
		entropy_pxl_space.append(entropy(prev_pxl, axis=None))
		gini_latent_space.append(get_gini(prev_latent))
		kurtois_latent_space.append(kurtosis(prev_latent, axis=None))
		entropy_latent_space.append(entropy(prev_latent, axis=None))
		mean_latent_space.append(np.mean(prev_latent))
		#diff to start
		shift_pxl_mse.append(np.mean(np.square(start - new_pxl), axis=(-1,-2,-3)))
		shift_latent.append(np.mean(np.square(start_latent - new_latent), axis=(axis_latent)))
		#diff to prev
		diff_pxl_mse.append(np.mean(np.square(prev_pxl - new_pxl), axis=(-1,-2,-3)))
		diff_latent.append(np.mean(np.square(prev_latent - new_latent), axis=(axis_latent)))

		prev_pxl = new_pxl
		prev_latent = new_latent
	if verbose>=0:
		plt.show()
		plt.savefig(os.path.join(DIR_RESULTS, "visu_divergence_{}.png".format(autoencoder.name)))
	return (gini_pxl_space, kurtois_pxl_space, entropy_pxl_space, gini_latent_space, kurtois_latent_space, entropy_latent_space, mean_latent_space, shift_pxl_mse, shift_latent, shift_pxl_ssim, diff_pxl_mse, diff_latent, diff_pxl_ssim)
