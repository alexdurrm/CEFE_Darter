import matplotlib.pyplot as plt 
from tensorflow.keras.Callback import Callback


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


def show_predictions(sample_test, prediction, n, title, saving_dir=None):
	"""
	plot test sample images and their reconstruction by the network
	"""
	fig, axs = plt.subplots(nrows=2, ncols=n, sharex=True, sharey=True, squeeze=False)
	fig.suptitle(title)
	for i in range(n):
		axs[0][i].imshow(sample_test[i], cmap='gray')
		axs[0][i].set_title("original {}".format(i))

		axs[1][i].imshow(prediction[i], cmap='gray')
		axs[1][i].set_title("reconstructed {}".format(i))
	plt.show()
	if saving_dir:
		plt.savefig(os.path.join(saving_dir, title))

def plot_training_losses(losses, val_losses, labels, title="losses train and test", save_path=None):
	fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, squeeze=False)
	fig.suptitle(title)
	axs[0][0].set_title("training losses")
	axs[1][0].set_title("validation losses")
	for loss, val_loss in zip(losses, val_losses):
		axs[0][0].plot(loss)
		axs[1][0].plot(val_loss)
	plt.legend(labels)
	plt.show()
	if save_path:
		plt.savefig(save_path)

def plot_loss_per_ld(best_losses, best_val_losses, title="", save_path=None)
	fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, squeeze=False)
	fig.suptitle(title)

	axs[0][0].set_title("training losses")
	axs[0][0].plot(list_LD, best_losses)
	axs[0][0].set_ylabel("best loss")
	
	axs[1][0].set_title("validation losses")
	axs[1][0].plot(list_LD, best_val_losses)
	axs[1][0].set_ylabel("best loss")
	axs[1][0].set_xlabel("latent_dim")

	plt.show()
	if save_path:
		plt.savefig(save_path)