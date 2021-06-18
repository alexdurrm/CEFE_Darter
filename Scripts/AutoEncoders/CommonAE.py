import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import numpy as np

class SavePredictionSample(Callback):
	def __init__(self, n_samples, val_data, saving_dir):
		super().__init__()
		self.n_samples=n_samples
		self.validation_data = val_data
		self.saving_dir = saving_dir

	def on_epoch_end(self, epoch, logs=None):
		outputs = self.model.predict(self.validation_data)
		title = "reconstructions epoch {} model {}".format(epoch, self.model.name)
		show_predictions(self.validation_data, outputs, self.n_samples, title, self.saving_dir)

class SaveActivations(Callback):
	def __init__(self, val_img, saving_dir):
		super().__init__()
		self.saving_dir = saving_dir
		self.validation_img = val_img

	def on_epoch_end(self, epoch, logs=None):
		if epoch%10!=0:
			return
		n_encoder_layers = len(self.model.encoder.layers)
		n_decoder_layers = len(self.model.decoder.layers)

		activations = [self.validation_img[np.newaxis,...]]
		layer_names = ["input"]

		print(self.model.encoder.summary())
		for i in range(n_encoder_layers):
			activations.append( tf.keras.backend.function([self.model.encoder.layers[i].input], self.model.encoder.layers[i].output)([activations[-1], 1]) )
			layer_names.append(self.model.encoder.layers[i].name)

		print(self.model.decoder.summary())
		for i in range(n_decoder_layers):
			activations.append( tf.keras.backend.function([self.model.decoder.layers[i].input], self.model.decoder.layers[i].output)([activations[-1], 1]) )
			layer_names.append(self.model.decoder.layers[i].name)

		fig = plt.figure(figsize=(24, 18), dpi=100)
		fig.suptitle("mean activations per layers")
		col=5
		n_layers = len(activations)
		for idx in range(n_layers):
			activation = np.mean(activations[idx], axis=0)	#remove batch dim
			if activation.ndim==3:	#if a conv layer
				if activation.shape[-1]!=1 and activation.shape[-1]!=3:
					activation = np.mean(activation, axis=(-1))[..., np.newaxis]
				ax = plt.subplot(n_layers//col+1, col, idx+1)
				ax.imshow(activation, cmap="hot")
				ax.set_title("n:{} m:{:.3f} M:{:.3f} \ns:{}".format(
													layer_names[idx],
													round(np.min(activation),3),
													round(np.max(activation),3),
													activations[idx].shape))
			else: #if another layer
				ax = plt.subplot(n_layers//col+1, col, idx+1)
				ax.bar(range(activation.size), activation.flatten())
				ax.set_title("n:{} m:{:.3f} M:{:.3f}".format(
													layer_names[idx],
													round(np.min(activation),3),
													round(np.max(activation),3)))
		plt.tight_layout()
		plt.savefig(os.path.join(self.saving_dir, "layer_activations_epoch{}".format(epoch)))
		# plt.show()
		plt.close()


class SaveSampling(Callback):
	def __init__(self, prediction_shape, n_samples, saving_dir):
		super().__init__()
		self.n_samples = n_samples
		self.prediction_shape = prediction_shape
		self.saving_dir = saving_dir

	def on_epoch_end(self, epoch, logs=None):
		#plot examples samples
		samples = self.model.sample(self.n_samples, self.prediction_shape)
		fig, axs = plt.subplots(nrows=1, ncols=self.n_samples, sharex=True, sharey=True)
		fig.suptitle("VarAE sampling")
		for i in range(self.n_samples):
			visu_sample = samples[i] if samples[i].shape[-1]!=2 else samples[i,...,0]+samples[i,...,1]
			axs[i].imshow(visu_sample, cmap='gray')
			axs[i].set_title("sample {}".format(i))
		plt.savefig(os.path.join(self.saving_dir, "{} sampling".format(self.model.name)))
		#plt.show()
		plt.close()


def show_predictions(sample_test, prediction, n, title, saving_dir=None):
	"""
	plot test sample images and their reconstruction by the network
	"""
	if n==0: return
	fig, axs = plt.subplots(nrows=2, ncols=n, sharex=True, sharey=True, squeeze=False)
	fig.suptitle(title)
	for i in range(n):
		visu_test = sample_test[i] if sample_test[i].shape[-1]!=2 else sample_test[i, ..., 0]+sample_test[i, ..., 1]
		axs[0][i].imshow(visu_test, cmap='gray')
		axs[0][i].set_title("original {}".format(i))

		visu_pred = prediction[i] if prediction[i].shape[-1]!=2 else prediction[i, ..., 0]+prediction[i, ..., 1]
		axs[1][i].imshow(visu_pred, cmap='gray')
		axs[1][i].set_title("reconstructed {}".format(i))
	if saving_dir:
		plt.savefig(os.path.join(saving_dir, title))
	# plt.show()
	plt.close()

def plot_training_losses(losses, val_losses, title="losses train and test", save_path=None):
	ax = plt.subplot(1, 1, 1)
	ax.set_title(title)
	ax.plot(losses, label="training loss")
	ax.plot(val_losses, label="validation loss")
	plt.legend()
	if save_path:
		plt.savefig(save_path)
	#plt.show()
	plt.close()

def plot_loss_per_ld(best_losses, best_val_losses, list_LD, title="", save_path=None):
	fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, squeeze=False)
	fig.suptitle(title)

	axs[0][0].set_title("training losses")
	axs[0][0].plot(list_LD, best_losses)
	axs[0][0].set_ylabel("best loss")

	axs[1][0].set_title("validation losses")
	axs[1][0].plot(list_LD, best_val_losses)
	axs[1][0].set_ylabel("best loss")
	axs[1][0].set_xlabel("latent_dim")

	if save_path:
		plt.savefig(save_path)
	#plt.show()
	plt.close()

def get_MSE(img1, img2):
	return np.mean(np.square(img1 - img2), axis=None)

def get_SSIM_Loss(y_true, y_pred):
	return 1- tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def visualize_conv_filters(model, layer):
	shape_input = tf.shape(model.inputs)
	# model = tf.keras.models.Model(inputs=model.inputs, outputs=layer)

	plt.suptitle("max activation image for layer {} filters".format(layer.name))
	rows, cols = shape_input[-1]//10+1, shape_input[-1]%10
	f, axs = plt.subplots(rows, cols)
	for i in range(layer.shape[-1]):
		ones = tf.ones(shape_input[:-1])
		input = tf.Variable(tf.random.uniform(shape_input))
		loss = tf.math.reduce_sum(tf.math.substract(ones ,layer[...,i]))
		opt = tf.keras.optimizers.Adam()
		for j in range(100):
			opt.minimize(loss, [input])
		axs[i//10+1, i%10+1].imshow(input.numpy())
	plt.show()

if __name__=='__main__':
	from tensorflow.keras.applications.vgg16 import VGG16
	model = VGG16(weights='imagenet', include_top=False,input_shape=pred_shape)
	visualize_conv_filters(model, model.get_layer("block1_pool"))
