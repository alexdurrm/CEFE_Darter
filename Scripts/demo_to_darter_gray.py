import numpy as np
import matplotlib.pyplot as plt
import argparse
import imageio
import cv2
from Utils.ImageManip import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("inputs", nargs="+", type=str, help="inputs")
	args = parser.parse_args()

	for path in args.inputs:
		print(path)
		f, axs = plt.subplots(1, 2)
		img = imageio.imread(path)

		darter_img = rgb_2_darter(img)
		img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)[...,np.newaxis]
		darter_img = (darter_img[...,0]+darter_img[...,1])[...,np.newaxis]
		darter_img = ((darter_img - np.max(darter_img))/ (np.max(darter_img)-np.min(darter_img))*256).astype(np.uint8)

		for channel_id in range(darter_img.shape[-1]):
			print(channel_id)
			histogram, bin_edges = np.histogram(darter_img[..., channel_id], bins=256, range=(0, 256))
			axs[0].plot(bin_edges[0:-1], histogram, color="gray")
			axs[0].set_title("image in darter space")

		for channel_id in range(img.shape[-1]):
			histogram, bin_edges = np.histogram(img[:, :, channel_id], bins=256, range=(0, 256))
			axs[1].plot(bin_edges[0:-1], histogram, color="gray")
		axs[1].set_title("image in RGB space")

		plt.show()
