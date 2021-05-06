from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import imageio
from glob import glob
import os
import cv2
import matplotlib.pyplot as plt

def fly_over_image(image, window, stride, return_coord=False):
	"""
	generate cropped images of the given image with a shape of the given window and stride given
	when return coord is true it returns the coordinates and not the crop
	"""
	img_Y = image.shape[-2]
	img_X = image.shape[-3]
	for start_x in range(0, img_X-window[0]+1, stride[0]):
		for start_y in range(0, img_Y-window[1]+1, stride[1]):
			if return_coord:
				yield (start_x, start_x+window[0], start_y, start_y+window[1])
			else:
				sample = image[start_x: start_x+window[0], start_y: start_y + window[1]]
				yield sample

def rgb_2_darter(image):
	"""
	transfer the given image from the RGB domain space to the darter domain space
	"""
	im_out = np.zeros([image.shape[0], image.shape[1], 2], dtype = np.float32)

	im_out[:, :, 1] = (140.7718694130528 +
		0.021721843447502408  * image[:, :, 0] +
		0.6777093385296341    * image[:, :, 1] +
		0.2718422677618606    * image[:, :, 2] +
		1.831294521246718E-8  * image[:, :, 0] * image[:, :, 1] +
		3.356941424659517E-7  * image[:, :, 0] * image[:, :, 2] +
		-1.181401963067949E-8 * image[:, :, 1] * image[:, :, 2])
	im_out[:, :, 0] = (329.4869869234302 +
		0.5254935133632187    * image[:, :, 0] +
		0.3540642397052902    * image[:, :, 1] +
		0.0907634883372674    * image[:, :, 2] +
		9.245344681241058E-7  * image[:, :, 0] * image[:, :, 1] +
		-6.975682782165032E-7 * image[:, :, 0] * image[:, :, 2] +
		5.828585657562557E-8  * image[:, :, 1] * image[:, :, 2])

	return im_out

def preprocess_habitat_image(set_img, color_channels, visu=0):
	"""
	given an image and a specified number of color channels
	return the transformed image
	"""
	print("\n")
	for i, img in enumerate(set_img):
		print("\rpreprocess {}/{}".format(i+1, len(set_img)), end='')
		image = img.copy()
		if color_channels==1:
			image = rgb_2_darter(image)
			image = image[..., 0]+image[..., 1]
			image = image[..., np.newaxis]
		elif color_channels==2:
			image = rgb_2_darter(image)
		set_img[i] = image
	set_img = (set_img - np.mean(set_img)) / np.std(set_img)
	set_img = ((set_img - np.min(set_img)) / (np.max(set_img) - np.min(set_img))).astype(np.float32)
	print("\n")
	for i in range(visu):
		plt.imshow(set_img[i], cmap='gray')
		plt.show()
	print("end preprocess")
	return set_img

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
		print("\r{}/{}".format(i+1, total), end='')
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
	test = preprocess_habitat_image(test, color_channels, visu)
	train = preprocess_habitat_image(train, color_channels, visu)
	print("end get dataset")
	return (train, test)


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
	train, test = get_datasets(args.glob, args.presize, args.shape, args.channels, args.levels, args.num_img, visu=args.verbose)
	print("Shape train: {} \nShape test: {}".format(train.shape, test.shape))

	#saving
	output_dir = args.output
	name_data = "{}_presize{}x{}_L{}_pred{}x{}x{}".format(args.name, *args.presize, args.levels, *args.shape, args.channels)
	np.save(os.path.join(output_dir, "Train_"+name_data), train)
	np.save(os.path.join(output_dir, "Test_"+name_data), test)
