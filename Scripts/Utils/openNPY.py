import matplotlib.pyplot as plt
import numpy as np
import argparse
import random

def f_open(path, do_random=False):
	data = np.load(path)
	print(data.shape)
	assert data.ndim==4 and data.shape[-1]<=3, "wrong data shape {}".format(data.shape)
	indexes = [i for i in range(len(data))]
	if do_random:
		random.shuffle(indexes)
	for idx in indexes:
		plt.title(idx)
		img = data[idx] if data.shape[-1]!=2 else np.mean(data[idx], axis=-1)
		plt.imshow(img, vmin=0, vmax=1, cmap='gray')
		plt.show()




if __name__=="__main__":
	parser = argparse.ArgumentParser(description="Script used to train, test and research optimal latend dim on different Autoencoders")
	parser.add_argument("path_npy", help="path of the numpy to open")
	parser.add_argument("--random", default=False, action="store_true", help="show images in randomized way")
	args = parser.parse_args()
	f_open(args.path_npy, args.random)
