import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import numpy as np

###################################################################################################
#
#	Store utility functions common to the autoencoders in Models, like callbacks, losses, ...
#
###################################################################################################

########################################
#
#				CALLBACKS
#
########################################
def scheduler(epoch, lr):
	"""
	update the learning rate at epoch 10  and 30
	"""
	if epoch==10 or epoch==30:
		return lr * 0.1
	else:
		return lr

def get_callbacks(model_type, test, output_dir, save_activations, early_stopping, sample_preds, lr_scheduler, verbosity=0):
	if verbosity>=1:
		print("get_callbacks:: Retrieving callbacks save_activation:{}, early_stopping:{}, sample_preds:{}, lr_scheduler:{}".format(save_activations, early_stopping, sample_preds, lr_scheduler))
	#append callbacks
	callbacks = []
	if lr_scheduler:
		callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
	if save_activations:
		callbacks.append(SaveActivations(val_img=test[0], saving_dir=output_dir, verbose=verbosity-1))
	if early_stopping:
		callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6))
		callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6))
	if sample_preds:
		sample = test[[i*10%len(test) for i in range(sample_preds)]]
		callbacks.append(SavePredictionSample(sample=sample, saving_dir=output_dir, verbosity=verbosity-1))
		if model_type=="variational_AE":
			callbacks.append(SaveSampling(sample_preds, test.shape[1:], saving_dir=output_dir))
	if verbosity>=1: print("get_callbacks:: Added {} callbacks".format(len(callbacks)))
	return callbacks


class SavePredictionSample(Callback):
	def __init__(self, sample, saving_dir, verbosity=0):
		super().__init__()
		self.sample = sample.copy()
		self.saving_dir = saving_dir
		self.verbosity = verbosity

	def on_epoch_end(self, epoch, logs=None):
		outputs = self.model.predict(self.sample)
		if self.verbosity>=1:
			print("SavePredictionSample: on epoch end")
			print("\nCommonAE.SavePredictionSample:: sample: {}/ {} / {} / {} / {}".format(type(self.sample), self.sample.shape, self.sample.min(), self.sample.max(), self.sample.dtype))
			print("\nCommonAE.SavePredictionSample:: output: {}/ {} / {} / {} / {}".format(type(outputs), outputs.shape, outputs.min(), outputs.max(), outputs.dtype))
		title = "reconstructions epoch {} model {}".format(epoch, self.model.name)
		show_predictions(self.sample, outputs, title, self.saving_dir, self.verbosity-1)


class SaveActivations(Callback):
	def __init__(self, val_img, saving_dir, verbose=0):
		super().__init__()
		self.saving_dir = saving_dir
		self.validation_img = val_img.copy()
		self.verbose = verbose

	def on_epoch_end(self, epoch, logs=None):
		if epoch%10!=0:
			return
		n_encoder_layers = len(self.model.encoder.layers)
		n_decoder_layers = len(self.model.decoder.layers)

		activations = [self.validation_img[np.newaxis,...]]
		layer_names = ["input"]

		if self.verbose>=1:
			print(self.model.encoder.summary())
			print(self.model.decoder.summary())

		for i in range(n_encoder_layers):
			print(self.model.encoder.layers[i])
			activations.append( tf.keras.backend.function([self.model.encoder.layers[i].input], self.model.encoder.layers[i].output.numpy())([activations[-1], 1]) )
			layer_names.append(self.model.encoder.layers[i].name)

		for i in range(n_decoder_layers):
			activations.append( tf.keras.backend.function([self.model.decoder.layers[i].input], self.model.decoder.layers[i].output.numpy())([activations[-1], 1]) )
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
		plt.savefig(os.path.join(self.saving_dir, "layer_activations_epoch{}.jpg".format(epoch)))
		if self.verbose>=2:
			plt.show()
		plt.close()


class SaveSampling(Callback):
	"""
	callback for a variationnal autoencoder that saves a sampling of generated images
	"""
	def __init__(self, n_sample, prediction_shape, saving_dir):
		super().__init__()
		self.n_sample = n_sample
		self.prediction_shape = prediction_shape
		self.saving_dir = saving_dir

	def on_epoch_end(self, epoch, logs=None):
		#plot examples samples
		samples = self.model.sample(self.n_sample, self.prediction_shape)
		fig, axs = plt.subplots(nrows=1, ncols=self.n_sample, sharex=True, sharey=True)
		fig.suptitle("VarAE sampling")
		for i in range(self.n_sample):
			visu_sample = samples[i] if samples[i].shape[-1]!=2 else samples[i,...,0]+samples[i,...,1]
			axs[i].imshow(visu_sample, cmap='gray', vmin=0, vmax=1)
			axs[i].set_title("sample {}".format(i))
		plt.savefig(os.path.join(self.saving_dir, "{} sampling.jpg".format(self.model.name)))
		plt.close()



########################################
#
#				LOSSES
#
########################################

def get_loss_from_name(name):
	if name=="mse":
		loss = get_MSE
	elif name=="ssim":
		loss = get_SSIM_Loss
	else:
		raise ValueError("Unknown loss name: {}".format(name))
	return loss

def get_MSE(img1, img2):
	return tf.reduce_mean(tf.math.squared_difference(img1, img2))

def get_SSIM_Loss(y_true, y_pred):
	return 1- tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

####################################################################
#
#							PLOT FUNCTIONS
#
####################################################################

def show_predictions(sample_test, prediction, title, saving_dir=None, verbosity=0):
	"""
	plot test sample images and their reconstruction by the network
	"""
	n_samples = len(sample_test)
	fig, axs = plt.subplots(nrows=2, ncols=n_samples, sharex=True, sharey=True, squeeze=False)
	fig.suptitle(title)
	for i in range(n_samples):
		visu_test = sample_test[i] if sample_test[i].shape[-1]!=2 else np.mean(sample_test[i], axis=-1)
		axs[0][i].set_title("original {}".format(i))
		axs[0][i].imshow(visu_test, cmap='gray', vmin=0, vmax=1)

		visu_pred = prediction[i] if prediction[i].shape[-1]!=2 else np.mean(prediction[i], axis=-1)
		mse = get_MSE(sample_test[i], prediction[i])
		axs[1][i].set_title("pred {}\nmse: {:.3f}".format(i, mse))
		axs[1][i].imshow(visu_pred, cmap='gray', vmin=0, vmax=1)
	if saving_dir:
		plt.savefig(os.path.join(saving_dir, title)+".jpg")
	if verbosity>=1:
		print("\nCommonAE.show_predictions: sample_test: {}/ {} / {} / {} / {}".format(type(visu_test), visu_test.shape, visu_test.min(), visu_test.max(), visu_test.dtype))
		print("\nCommonAE.show_predictions: visu_pred: {}/ {} / {} / {} / {}".format(type(visu_pred), visu_pred.shape, visu_pred.min(), visu_pred.max(), visu_pred.dtype))
		plt.show()
	plt.close()

def plot_training_losses(losses, val_losses, title="losses train and test", save_path=None, verbosity=0):
	"""
	plot a graph of training and validation losses at each epoch
	if save_path is given saves the result else show it
	"""
	ax = plt.subplot(1, 1, 1)
	ax.set_title(title)
	ax.plot(losses, label="training loss")
	ax.plot(val_losses, label="validation loss")
	plt.legend()
	if save_path:
		plt.savefig(save_path+".jpg")
	if verbosity>=1:
		plt.show()
	plt.close()

def plot_loss_per_ld(best_losses, best_val_losses, list_LD, title="", save_path=None, verbosity=0):
	"""
	plot a graph of best losses and best_val_losses for each latent dim
	if save_path is given save the result else show it
	"""
	ax = plt.subplot(1, 1, 1)
	ax.set_title(title)
	ax.plot(list_LD, best_losses, label="training loss")
	ax.plot(list_LD, best_val_losses, label="validation loss")
	ax.set_xlabel("latent_dim")
	ax.set_ylabel("loss")
	plt.legend()
	if save_path:
		plt.savefig(save_path+".jpg")
	if verbosity>=1:
		plt.show()
	plt.close()

# def visualize_conv_filters(model, layer, verbosity=0):
# 	"""
# 	given a model and a layer name
# 	return a list of input images that maximizes the activation of this layer
# 	"""
# 	## TODO
# 	shape_input = tf.shape(model.inputs)
# 	# model = tf.keras.models.Model(inputs=model.inputs, outputs=layer)
#
# 	plt.suptitle("max activation image for layer {} filters".format(layer.name))
# 	rows, cols = shape_input[-1]//10+1, shape_input[-1]%10
# 	f, axs = plt.subplots(rows, cols)
# 	for i in range(layer.shape[-1]):
# 		ones = tf.ones(shape_input[:-1])
# 		input = tf.Variable(tf.random.uniform(shape_input))
# 		loss = tf.math.reduce_sum(tf.math.substract(ones ,layer[...,i]))
# 		opt = tf.keras.optimizers.Adam()
# 		for j in range(100):
# 			opt.minimize(loss, [input])
# 		axs[i//10+1, i%10+1].imshow(input.numpy(), vmin=0, vmax=1, cmap='hot')
# 	if verbosity>=1:
# 		plt.show()
# 	plt.close()


if __name__=='__main__':
	from tensorflow.keras.applications.vgg16 import VGG16
	model = VGG16(weights='imagenet', include_top=False,input_shape=pred_shape)
	visualize_conv_filters(model, model.get_layer("block1_pool"))
