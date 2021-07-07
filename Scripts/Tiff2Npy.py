from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import imageio
from glob import glob
import os
import cv2
import matplotlib.pyplot as plt

from Utils.ImageManip import fly_over_image, rgb_2_darter, standardize_img, normalize_img

def preprocess_habitat_image(set_img, color_channels, visu=0):	## TODO: virer ca pour un Preprocess
	"""
	given an image and a specified number of color channels
	return the transformed image
	"""
	new_set = np.empty(shape=(*set_img.shape[:-1], color_channels))
	for i, image in enumerate(set_img):
		print("preprocess {}/{}".format(i+1, len(set_img)))
		if color_channels==1:
			image = rgb_2_darter(image)
			image = image[..., 0]+image[..., 1]
			image = image[..., np.newaxis]
		elif color_channels==2:
			image = rgb_2_darter(image)
		new_set[i] = image
	new_set = new_set.astype(np.float32)
	new_set = normalize_img(new_set)
	new_set = standardize_img(new_set)

	assert np.max(new_set)<=1 and np.min(new_set)>=0, "bad normalization,{} {} instead of {} {}".format(np.min(set_img), np.max(set_img), mini, maxi)
	for i in range(visu):
		plt.imshow(new_set[i], cmap='gray')
		plt.show()
	return new_set

def augment(set_img, prediction_shape, levels, visu=0):
	"""
	prediction shape is the shape of the output crops
	given a set of images returns an augmented set of images, by adding the symmetric version of the image and multiple crops
	"""
	augmented = []
	xy_ratio_pred = prediction_shape[0]/prediction_shape[1]
	for i, img in enumerate(set_img):
		print("\raugment {}/{}".format(i+1, len(set_img)), end='')
		for level in range(levels):
			#calculate the biggest window possible for this level
			crop_max = (img.shape[0]//(2**level), img.shape[1]//(2**level))
			xy_ratio_lvl = crop_max[0]/crop_max[1]
			if xy_ratio_pred == xy_ratio_lvl:
				level_crop = crop_max
			elif xy_ratio_pred < xy_ratio_lvl:
				level_crop = (xy_ratio_pred*crop_max[1], crop_max[1])
			else:
				level_crop = (crop_max[0], int(crop_max[0]/xy_ratio_pred))

			#for each image return crops from slinding window and corresponding mirror
			for sample in fly_over_image(img, level_crop, level_crop):
				sample = cv2.resize(sample, prediction_shape[::-1], interpolation=cv2.INTER_CUBIC)
				augmented += [sample, np.flip(sample, axis=(-2))]
	for i in range(visu*10):
		plt.imshow(augmented[i], cmap='gray')
		plt.show()
	print("augmentation gone from {} to {} images".format(len(set_img), len(augmented)))
	return np.array(augmented)

def get_datasets(glob_path, presize_shape, pred_shape, color_channels, levels, n_img_used, visu=0):
	"""
	given a glob for all images to include in the dataset
	and some parameters for preprocessing
	return train and test numpy arrays
	"""
	#list all image path
	habitat_path = glob(glob_path)
	if n_img_used is not None:
		habitat_path = habitat_path[:n_img_used]
	total = len(habitat_path)
	print("number of images for training: {}".format(total))
	#load all images and preprocess them
	habitat_img = []
	for i, path in enumerate(habitat_path):
		print("\r{}/{} {}".format(i+1, total, path), end='')
		image = imageio.imread(path)
		#if there is a presize givent
		if presize_shape[0] or presize_shape[1]:
			rX = presize_shape[0] if presize_shape[0] else image.shape[0]
			rY = presize_shape[1] if presize_shape[1] else image.shape[1]
			image = cv2.resize(image, dsize=(rY, rX), interpolation=cv2.INTER_CUBIC)
		habitat_img.append(image)

	for i in range(visu):
		plt.imshow(habitat_img[i], cmap='gray')
		plt.show()
	#split in train and test
	train, test = train_test_split(habitat_img, train_size=0.9, shuffle=True)
	#augment and normalize train and test
	test = augment(test, pred_shape, levels, visu)
	train = augment(train, pred_shape, levels, visu)
	print(train.shape, test.shape)
	test = preprocess_habitat_image(test, color_channels, visu)
	train = preprocess_habitat_image(train, color_channels, visu)
	print("end get dataset")
	return train, test


if __name__=='__main__':
	"""
	given a specified glob of tif images will load these images,
	transform them, split them into train and test dataset
	and save them into the specified output directory
	"""
	#default parameters
	PRESIZE_SHAPE = (None, None)
	CROP_SHAPE = (128, 128)
	CHANNELS = 3
	LEVELS = 3
	VERBOSE = 0
	NUMBER_IMG = None

	#parsing input parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("name", help="category name given to recognize the files used (ex: blennioides, careuleum, habitats, fish ...)")
	parser.add_argument("glob", help="glob of the images to put in the dataset")
	parser.add_argument("output", help="output directory where to save the dataset")
	parser.add_argument("-c", "--channels", type=int, choices=[1,2,3], default=CHANNELS,
		help="Number of channels to train on: 1 will be in darter gray luminance space,2 will be darter visualization, 3 will be in rgb, default is {}".format(CHANNELS))
	parser.add_argument("-s", "--shape", type=int, nargs=2, default=CROP_SHAPE, help="shape of the final image used as a network input/output, default is {}".format(CROP_SHAPE))
	parser.add_argument("-p", "--presize", type=int, nargs=2, default=PRESIZE_SHAPE, help="shape to which resize the original image before its croping and all, default is {}".format(PRESIZE_SHAPE))
	parser.add_argument("-l", "--levels", type=int, default=LEVELS, help="Number of levels to which zoom the data during augmentation, default is {}".format(LEVELS))
	parser.add_argument("-n", "--num_img", default=NUMBER_IMG, type=int, help="Number of images used to to the dataset, if none is given use all the images, default is {}".format(NUMBER_IMG))
	parser.add_argument("-v", "--verbose", type=int, default=VERBOSE, help="verbose, how much images we should display, square, default is {}".format(VERBOSE))
	args = parser.parse_args()

	#preparing data
	train, test = get_datasets(args.glob, args.presize, tuple(args.shape), args.channels, args.levels, args.num_img, visu=args.verbose)
	print("Shape train: {} \nShape test: {}".format(train.shape, test.shape))

	#saving
	output_dir = args.output
	name_data = "{}_presize{}x{}_L{}_pred{}x{}x{}".format(args.name, *args.presize, args.levels, *args.shape, args.channels)
	np.save(os.path.join(output_dir, "Train_"+name_data), train)
	np.save(os.path.join(output_dir, "Test_"+name_data), test)
