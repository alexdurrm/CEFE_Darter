import argparse
from glob import glob
import imageio
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imageio
from skimage import feature

from Utils.ImageManip import normalize_img, standardize_img, resize_img_to_fit, rgb_2_darter

def prepare_fish_img(image, output_shape, padding_val=None, visu=False):

	if output_shape[-1]==1:
		image = rgb_2_darter(image, visu=visu)
		image = image[..., 0]+image[..., 1]
		image = image[..., np.newaxis]
	elif output_shape[-1]==2:
		image = rgb_2_darter(image)

	image = image.astype(np.float32)
	image = normalize_img(image, visu=visu)
	image = standardize_img(image, visu=visu)

	#prepare an output image of the shape expected with a backgroud value
	if not padding_val:
		padding_val = np.mean([image[0,0], image[-1,-1]])  #set the padding value to be the value at the top left corner
	output_img = np.full(output_shape, padding_val)

	#if the output image is small we shall resize the image
	if output_shape[0]<image.shape[0] or output_shape[1]<image.shape[1]:
		print("WARNING: output shape is smaller on one dimension than the image given, resizing. The resolution cm/pxl is not conserved anymore.")
		image = resize_img_to_fit(image, output_shape, keep_ratio=True, visu=visu)

	start_x, start_y = round((output_shape[0]-image.shape[0])/2), round((output_shape[1]-image.shape[1])/2)
	output_img[start_x:start_x+image.shape[0], start_y:start_y+image.shape[1], :] = image[...]
	return output_img


if __name__=='__main__':
	CHANNELS=3
	VERBOSE=0
	CROP_SHAPE=(128, 128)

	parser = argparse.ArgumentParser()
	parser.add_argument("glob", help="glob of the images to put in the dataset")
	parser.add_argument("output", help="output file where to save the dataset")
	parser.add_argument("-c", "--channels", type=int, choices=[1,2,3], default=CHANNELS,
		help="Number of channels to train on: 1 will be in darter gray luminance space,2 will be darter visualization, 3 will be in rgb, default is {}".format(CHANNELS))
	parser.add_argument("-s", "--shape", type=int, nargs=2, default=CROP_SHAPE, help="shape of the final image used as a network input/output, default is {}".format(CROP_SHAPE))
	parser.add_argument("-v", "--verbose", default=VERBOSE, type=int, choices=[0,1,2], help="Set the level of visualization, default: {}".format(VERBOSE))
	args = parser.parse_args()

	output_shape = (*args.shape, args.channels)
	path_images = glob(args.glob)
	image_list = np.empty((len(path_images), *output_shape), dtype=np.float32)
	for i, path in enumerate(path_images):
		if args.verbose>=1: print(path)
		img = imageio.imread(path)
		image_list[i] = prepare_fish_img(img, output_shape, visu=args.verbose>1)
	np.save(args.output, image_list)
