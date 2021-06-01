import numpy as np
import matplotlib.pyplot as plt
import cv2

######################################################################
#																	 #
#	 File storing different functions used in image manipulation	 #
#																	 #
######################################################################


def img_manip_decorator(function):
	"""
	function to visualize the manipulated image of the manipulation functions
	"""
	def wrap(*args, **kwargs):
		input_img = kwargs.get("image", args[0])
		assert isinstance(input_img, np.ndarray), "given image should be a numpy array, here {}".format(type(input_img))
		assert input_img.ndim==3 or input_img.ndim==4, "image should have 3 dimensions, list of 4 dims are also accepted, here {}".format(input_img.ndim)
		visu = kwargs.pop("visu", False)
		if input_img.ndim==3:	#if given an image
			output = function(*args, **kwargs)
			if visu:
				fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
				visu_input = input_img if input_img.shape[-1]!=2 else input_img[...,0]+input_img[...,1]
				axs[0].imshow(visu_input, cmap='gray')
				visu_out = output if output.shape[-1]!=2 else output[..., 0]+output[...,1]
				axs[1].imshow(visu_out, cmap='gray')
				plt.title(function.__name__)
				plt.show()
				plt.close()
			if output.ndim==2:
				output = output_img[..., np.newaxis]
		else:	#if given a list of images
			if not kwargs.pop("image", None):
				args = args[1:]
			output = np.array([function(img, *args, **kwargs) for img in input_img])
		return output
	return wrap


def fly_over_image(image, window, stride, return_coord=False):
	"""
	generate cropped samples of the given image with a shape of the given window and stride given
	when return coord is true it returns the coordinates and not the crop
	"""
	assert (window[0]<=image.shape[0] and window[1]<=image.shape[1]), "Cropping window is too big for the given image: {} vs {}".format(window, image.shape)
	img_X, img_Y = image.shape[:2]
	for start_x in range(0, img_X-window[0]+1, stride[0]):
		for start_y in range(0, img_Y-window[1]+1, stride[1]):
			if return_coord:
				yield (start_x, start_x+window[0], start_y, start_y+window[1])
			else:
				sample = image[start_x: start_x+window[0], start_y: start_y + window[1]]
				yield sample

@img_manip_decorator
def resize_img(image, new_shape=(None, None)):
	"""
	given an image and a shape return a resized image
	different from cv2 resize because new_shape can have None values
	if None None is the new shape given the returned image is the same as the one in parameters
	"""
	if new_shape[0] or new_shape[1]:
		rX = new_shape[0] if new_shape[0] else image.shape[0]
		rY = new_shape[1] if new_shape[1] else image.shape[1]
		image = cv2.resize(image, dsize=(rY, rX), interpolation=cv2.INTER_CUBIC)
	return image

@img_manip_decorator
def resize_img_to_fit(image, new_shape, keep_ratio):
	"""
	given an image and an output shape (newX, newY), resize the image to fit the output_shape
	if keep_ratio is true the image is transformed to have smaller or equal dimensions to the new shape but will keep its original ratio
	output the resized image
	"""
	if keep_ratio:
		new_ratio = new_shape[1]/new_shape[0]
		old_ratio = image.shape[1]/image.shape[0]
		if new_ratio < old_ratio:   #nouveau est plus vertical on adapte le y
			new_shape = [round(image.shape[0]/image.shape[1]*new_shape[1]), new_shape[1]]
		elif new_ratio > old_ratio:   #nouveau est plus horizontal on adapte le x
			new_shape = [new_shape[0], round(image.shape[1]/image.shape[0]*new_shape[0])]
	image = cv2.resize(image, dsize=(new_shape[1], new_shape[0]), interpolation=cv2.INTER_CUBIC)
	return image

@img_manip_decorator
def standardize_img(image):
	"""
	bring the image to values between 0 and 1
	"""
	image = ((image - np.min(image)) / (np.max(image) - np.min(image)))
	assert np.max(image)<=1 and np.min(image)>=0, "bad normalization,{} {} instead of {} {}".format(np.min(image), np.max(image), mini, maxi)
	return image

@img_manip_decorator
def normalize_img(image):
	"""
	bring the distribution of the image to reach a mean of 0 and a standard deviation of 1
	"""
	image = (image - np.mean(image)) / np.std(image)	#image - np.mean(image, axis=(0,1))) / np.std(image, axis=(0,1)
	return image

@img_manip_decorator
def rgb_2_darter(image):
	"""
	transfer the given image from the RGB domain space to the darter domain space
	"""
	assert image.shape[-1]==3, "Image of wrong dimensions, should be NxMx3 but is {}".format(image.shape)
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
