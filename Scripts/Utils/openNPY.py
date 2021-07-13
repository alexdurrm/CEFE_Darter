import matplotlib.pyplot as plt
import numpy as np
import argparse

def f_open(path):
	data = np.load(path)
	print(data.shape)
	assert data.ndim==4 and data.shape[-1]<=3, "wrong data shape {}".format(data.shape)
	for idx, img in enumerate(data):
		plt.title(idx)
		if data.shape[-1]==2:
			plt.imshow((img[...,0]+img[...,1])/2, cmap='gray')
		else:
			plt.imshow(img, cmap='gray')
		plt.show()




if __name__=="__main__":
	parser = argparse.ArgumentParser(description="Script used to train, test and research optimal latend dim on different Autoencoders")
	parser.add_argument("path_npy", help="path of the numpy to open")
	args = parser.parse_args()
	f_open(args.path_npy)
