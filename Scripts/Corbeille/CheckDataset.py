import numpy as np
import matplotlib.pyplot as plt
import argparse


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("input_file",type=str, help="numpy file to open")
	args = parser.parse_args()

	dataset = np.load(args.input_file)
	if dataset.ndim==3:
		dataset = dataset[..., np.newaxis]
	elif dataset.ndim==4:
		assert dataset.shape[-1]==3 or dataset.shape[-1]==1, "image channel should be either 1 or 3 dims, here :{}".format(dataset.shape[-1])
	else:
		raise Exception("Bad number of dimension for this data, should be 3 or 4, here {}".format(dataset.ndim))
	for i, img in enumerate(dataset):
		plt.imshow(img, cmap="gray")
		plt.title("img n°{}, min: {}, max: {}".format(i, np.min(img), np.max(img)))
		plt.show()
	print(dataset.shape, np.min(dataset), np.max(dataset))
