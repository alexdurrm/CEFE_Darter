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

def normalize(set_img):
	set_img = (set_img - np.mean(set_img)) / np.std(set_img)
	set_img = (set_img - np.min(set_img)) / (np.max(set_img) - np.min(set_img)).astype(np.float32)
	return set_img

def preprocess_habitat_image(image, new_shape, color_channels):
	"""
	given an image, its new shape, and a specified number of color channels
	return the transformed image
	"""
	image = cv2.resize(image, dsize=(new_shape[::-1]), interpolation=cv2.INTER_CUBIC)
	if color_channels==1:
		image = rgb_2_darter(image)
		image = image[..., 0]+image[..., 1]
		image = image[..., np.newaxis]
	elif color_channels==2:
		image = rgb_2_darter(image)
		image = image/np.max(image)
	elif color_channels==3:
		image = image/np.max(image)
	else:
		raise ValueError
	return image

def augment(set_img, prediction_shape):
	"""
	given a set of images returns an augmented set of images, by adding the symmetric version of the image
	"""
	augmented = []
	for img in set_img:
		for sample in fly_over_image(img, prediction_shape, prediction_shape):
			augmented += [sample, np.flip(sample, axis=(-2))]
	return np.array(augmented)

def get_datasets(glob_path, resize_shape, pred_shape, color_channels, visu=0):
	"""
	given a glob for all images to include in the dataset
	and some parameters for preprocessing
	return train and test numpy arrays
	"""
	#list all image path
	habitat_path = glob(glob_path)
	total = len(habitat_path)
	print("number of images for training: {}".format(total))
	#load all images and preprocess them
	habitat_img = np.empty(shape=(len(habitat_path), *resize_shape, color_channels))
	for i, path in enumerate(habitat_path):
		print("\r{}/{}".format(i, total), end='')
		img = imageio.imread(path)
		habitat_img[i] = preprocess_habitat_image(img, resize_shape, color_channels)
	if visu:
		for i in range(5):
			plt.imshow(habitat_img[i], cmap='gray')
			plt.show()
	#split in train and test
	train, test = train_test_split(habitat_img, train_size=0.9, shuffle=True)
	#augment and normalize train and test
	test = augment(test, pred_shape)
	train = augment(train, pred_shape)
	test = normalize(test)
	train = normalize(train)
	#show one of these images
	if visu:
		for i in range(5):
			plt.imshow(train[np.random.randint(0, len(train))], cmap='gray')
			plt.show()
	return (train, test)


if __name__=='__main__':
	"""
	given a specified glob of tif images will load these images,
	transform them, split them into train and test dataset
	and save them into the specified output directory
	"""
	#default parameters
	RESIZE_SHAPE = (600, 900)
	CROP_SHAPE = 128
	CHANNELS = 3

	#parsing input parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("name", help="category name given to recognize the files used (ex: blennioides, careuleum, habitats, fish ...)")
	parser.add_argument("glob", help="glob of the images to put in the dataset")
	parser.add_argument("output", help="output directory where to save the dataset")
	parser.add_argument("-c", "--channels", type=int, choices=[1,2,3], default=CHANNELS,
		help="Number of channels to train on: 1 will be in darter gray luminance space,2 will be darter visualization, 3 will be in rgb, default is {}".format(CHANNELS))
	parser.add_argument("-s", "--shape", type=int, default=CROP_SHAPE, help="shape of the final image used as a square network input/output, default is {}".format(CROP_SHAPE))
	parser.add_argument("-v", "--verbose", type=int, default=0, help="verbose, how much we should display, square, default is 0")
	args = parser.parse_args()

	#preparing data
	crop_shape = (args.shape, args.shape)
	train, test = get_datasets(args.glob, RESIZE_SHAPE, crop_shape, args.channels, visu=args.verbose)
	print("Shape train: {} \nShape test: {}".format(train.shape, test.shape))

	#saving
	output_dir = args.output
	name_data = "{}_presize{}x{}_pred{}x{}x{}".format(args.name, *RESIZE_SHAPE, *crop_shape, args.channels)
	np.save(os.path.join(output_dir, "Train_"+name_data), train)
	np.save(os.path.join(output_dir, "Test_"+name_data), test)
