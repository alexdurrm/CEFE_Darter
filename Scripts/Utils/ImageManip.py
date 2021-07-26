import numpy as np
import matplotlib.pyplot as plt
import cv2
import rawpy as rp
import imageio
import os

######################################################################
#																	 #
#	 File storing different functions used in image manipulation	 #
#																	 #
######################################################################
WORKING_TYPE="float32"

def img_manip_decorator(function):
	"""
	function to visualize the manipulated image of the manipulation functions
	also makes sure the input array corresponds to the correct expected dimensions (3 dims or a list of 3 dims)
	and the output corresponds to an array of 3 dims (or a list of 3 dims arrays)
	an optionnal argument "verbose" can be set to true if we want to visualize the manipulation done
	"""
	def wrap(*args, **kwargs):
		#take image and check its type and shape
		input_img = kwargs.get("image", args[0])
		assert isinstance(input_img, np.ndarray), "given image should be a numpy array, here {}".format(type(input_img))
		assert input_img.ndim==3 or input_img.ndim==4, "image should have 3 dimensions, list of 4 dims are also accepted, here {}".format(input_img.ndim)

		if input_img.ndim==3:	#if given an image, call the function
			output = function(*args, **kwargs)
			if output.ndim==2:
				output = output[..., np.newaxis]
			#if visu is true show the input and output images
			if kwargs.pop("verbose", 0)>=2:
				compare_images(input_img, output, "comparaison input/output", "input", "output")
		else:	#if given a list of images, call the function on each image and return a list
			#first remove the list form parameters
			if not kwargs.pop("image", None):
				args = args[1:]
			output = [function(img, *args, **kwargs) for img in input_img]
			if isinstance(input_img, np.ndarray):
				output = np.array(output)
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
	rX = new_shape[0] if new_shape[0] else image.shape[0]
	rY = new_shape[1] if new_shape[1] else image.shape[1]
	if rX!=image.shape[0] or rY!=image.shape[1]:
		image = cv2.resize(image, dsize=(rY, rX), interpolation=cv2.INTER_LINEAR)	#numpy and cv2 have inverted axes X and Y
	return image

@img_manip_decorator
def resize_img_to_fit(image, new_shape, keep_ratio, fit_method="padding"):
	"""
	given an image and an output shape (newX, newY), resize the image to fit the new shape
	if keep_ratio is true the image is transformed to keep its X to Y ratio
	fit_method can be "smaller" or "bigger":
		if "padding" the image returned will be smaller on both axis and added padding corresponding to the padidng method
		if "cropping" the returned image will be a centered crop of the image resized in the shape new_shape
	padding_method is only relevant when fit_method is "smaller" and indicate the padding strategy to use: "mean", "min", "max", or a value
	output the resized image of the shape new_shape
	"""
	if keep_ratio:
		#calculate a new shape that preserves the ratio
		new_ratio = new_shape[1]/new_shape[0]
		old_ratio = image.shape[1]/image.shape[0]
		#resize the image to be smaller or equal for all axis
		if fit_method=="padding":
			if new_ratio < old_ratio:   #nouveau est plus vertical on adapte le y
				temp_shape = (round(image.shape[0]/image.shape[1]*new_shape[1]), new_shape[1])
			elif new_ratio > old_ratio:   #nouveau est plus horizontal on adapte le x
				temp_shape = (new_shape[0], round(image.shape[1]/image.shape[0]*new_shape[0]))
			else:
				temp_shape = new_shape
			image = resize_img(image, new_shape=temp_shape)
			image = padd_img(image, padding_shape=new_shape)
		#resize the image to be bigger or equal for all axis
		elif fit_method=="cropping":
			if new_ratio > old_ratio:   #nouveau est plus vertical on adapte le y
				temp_shape = (round(image.shape[0]/image.shape[1]*new_shape[1]), new_shape[1])
			elif new_ratio < old_ratio:   #nouveau est plus horizontal on adapte le x
				temp_shape = (new_shape[0], round(image.shape[1]/image.shape[0]*new_shape[0]))
			else:
				temp_shape = new_shape
			image = resize_img(image, new_shape=temp_shape)
			image = crop_img(image, crop_shape=new_shape)
		else:
			ValueError("Unknown method: {}".format(fit_method))
	else:
		image = resize_img(image, new_shape=new_shape)
	return image

@img_manip_decorator
def padd_img(image, padding_shape, padding_method="mean"):
	"""
	image=numpy array with ndim>=2
	padding_shape= tuple of int shape of the output image
	padding_method: "mean", "min", "max", "zero" or value method to get the value of the padding
	return padded image to the shape padding_shape
	"""
	assert image.shape[0]<=padding_shape[0] and image.shape[1]<=padding_shape[1], "Can't do a padding on an image bigger than the padding shape, {} to {}".format(image.shape, padding_shape)
	if padding_method=="mean":
		padding = np.full(shape=(*padding_shape, *image.shape[2:]), fill_value=image.mean(), dtype=image.dtype)
	elif padding_method=="min":
		padding = np.full(shape=(*padding_shape, *image.shape[2:]), fill_value=image.min(), dtype=image.dtype)
	elif padding_method=="max":
		padding = np.full(shape=(*padding_shape, *image.shape[2:]), fill_value=image.max(), dtype=image.dtype)
	elif isinstance(padding_method, int) or isinstance(padding_method, float):
		padding = np.zeros(shape=(*padding_shape, *image.shape[2:]), fill_value=image.mean(), dtype=image.dtype)
	else:
		raise ValueError("Unrecognized padding method: {}".format(padding_method))
	anchor = [int((padding_shape[0]-image.shape[0])/2.0), int((padding_shape[1]-image.shape[1])/2.0)]
	padding[anchor[0]:anchor[0]+image.shape[0], anchor[1]:anchor[1]+image.shape[1]]=image[:,:]
	return padding

@img_manip_decorator
def crop_img(image, crop_shape):
	"""
	image: numpy array with ndim>=2
	crop_shape: a pair of int the size of the crop we want to make
	padding: if the crop shape is larger than the image given will padd the returned image with mean value of the original image
	"""
	#if no need to crop just return the image
	if image.shape==crop_shape:
		return image
	#place the anchor and do the cropping
	assert crop_shape[0]<=image.shape[0] and crop_shape[0]<=image.shape[0], "Cannot crop shape {} out of image {}".format(crop_shape, image.shape)
	anchor = [int((image.shape[0]-crop_shape[0])/2.0), int((image.shape[1]-crop_shape[1])/2.0)]
	return image[anchor[0]:anchor[0]+crop_shape[0], anchor[1]:anchor[1]+crop_shape[1]]


@img_manip_decorator
def standardize_img(image, type=WORKING_TYPE):
	"""
	bring the image to values between 0 and 1
	"""
	if image.dtype==np.uint8:
		image = image/255.0
	else:
		image = ((image - np.min(image)) / (np.max(image) - np.min(image)))
	assert np.max(image)<=1 and np.min(image)>=0, "bad normalization,{} {} instead of {} {}".format(np.min(image), np.max(image), mini, maxi)
	return image.astype(type)

@img_manip_decorator
def normalize_img(image, type=WORKING_TYPE):
	"""
	bring the distribution of the image to reach a mean of 0 and a standard deviation of 1
	"""
	image = (image - np.mean(image)) / np.std(image)	#image - np.mean(image, axis=(0,1))) / np.std(image, axis=(0,1)
	return image.astype(type)

@img_manip_decorator
def rgb_2_darter(image, type=WORKING_TYPE):	#TODO careful with the image type
	"""
	transfer the given image from the RGB domain space to the darter domain space
	"""
	assert image.shape[-1]==3, "Image of wrong dimensions, should be NxMx3 but is {}".format(image.shape)
	if image.dtype!="uint8":
		image = image*255.0
	im_out = np.zeros([image.shape[0], image.shape[1], 2], dtype = type)

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


#### LOADINGS
def openImage(path, verbosity=0):
	"""
	given a path for an image open it
	"""
	assert os.path.exists(path), "{} does not exist".format(path)
	if verbosity>=1:
		print("opening {}".format(path))
	ext = os.path.splitext(path)[1]
	if ext==".CR2":
		image = openCR2(path)
	elif ext==".npy":
		image = openNpy(path)
	elif ext in [".tif", ".tiff"]:
		image = openTiff(path)
	else:
		image = openImg(path)
	# check the image is of the right format and dimensions
	if image.ndim==2:
		image = image[..., np.newaxis]
	return image

def openTiff(path, type=WORKING_TYPE):
	"""
	open a .tiff image and return a float32 numpy array
	"""
	ext = os.path.splitext(path)[1]
	assert ext in [".tiff", ".tif"], "Wrong image format, expected \'.tiff\', or \'.tif\', got {}".format(ext)
	img = imageio.imread(path)
	return img.astype(type)

def openCR2(path, default_method=True, type=WORKING_TYPE):
	"""
	open a .CR2 image and return a float32 numpy array
	"""
	ext = os.path.splitext(path)[1]
	assert ext==".CR2", "Wrong image format, expected \'.CR2\', got {}".format(ext)
	raw = rp.imread(path)
	if default_method:
		rgb = raw.postprocess(use_camera_wb=True)
	else:
		rawParams = rp.Params(gamma=(1, 1),
							  no_auto_bright=True,
							  user_wb=(1, 1, 1, 1),
							  output_bps=16,
							  half_size=True)
		rgb = raw.postprocess(params = rawParams)
	return rgb.astype(type)

def openNpy(path, type=WORKING_TYPE):
	"""
	open a .npy image and return a float32 numpy array
	"""
	ext = os.path.splitext(path)[1]
	assert ext==".npy", "Wrong image format, expected \'.npy\', got {}".format(ext)
	return np.load(path).astype(type)

def openImg(path, type=WORKING_TYPE):
	"""
	open a .png or .jpg image and return a float32 numpy array
	"""
	ext = os.path.splitext(path)[1]
	assert ext in [".jpg", ".png"], "Wrong image format, expected \'.jpg\' or \'.png\', got {}".format(ext)
	img = imageio.imread(path)
	return (img/255.0).astype(type)

#### SAVINGS
def save_images(data, path, filenames=None, extension=None, verbose=0):	## TODO: get formats for each extension and prepare accordingly
	"""
	given a data of type numpy array will save it in the specified type
	if multiple images need to be saved a directory will be created at specified path
	"""
	#if not given explicit format infer from path
	path, ext = os.path.splitext(path)
	if not extension:
		extension = ext
	assert extension in [".jpg", ".png", ".tiff", ".tif", ".npy"], "wrong extention {}".format(extension)

	#if save to numpy
	if extension==".npy":
		np.save(path+extension, data)
		if verbose>=1:
			print("saving {}".format(path+extension))
	#if save to common format
	elif extension in [".jpg", ".png", ".tiff", ".tif"]:
		#if multiple images
		if isinstance(data, list) or data.ndim==4:
			if not os.path.exists(path):
				os.makedirs(path)
			for idx, img in enumerate(data):
				filepath = os.path.join(path, filenames[idx]+extension) if filenames else os.path.join(path, str(idx)+extension)
				saveImg(img, filepath, extension)
				if verbose>=2:
					print("saving {}".format(filepath))
		else:
			saveImg(data, path, extension)
			if verbose>=1:
				print("saving {}".format(path+extension))
	#if unknown format
	else:
		raise ValueError("extension must be one of the expected image format, received {}".format(extension))

#TODO: handle formats
def saveImg(image, path, ext):
	"""
	given a numpy array, a path and an extension will save it
	"""
	assert image.shape[-1]!=2 and image.ndim==3, "cannot save an image with shape {} in this format {}".format(image.shape, os.path.splitext(path)[-1])
	#append extension to filepath
	path = os.path.splitext(path)[0]+ext
	assert ext in [".jpg", ".png", ".tiff", ".tif"], "Wrong image format, expected \'.jpg\', \'.png\', \'.tiff\', or \'.tif\', got {}".format(ext)
	if ext in [".tiff", ".tif"]:
		imageio.imwrite(path, image)
	elif ext in [".jpg", ".png"]:
		image = (255*image).astype("uint8")
		imageio.imwrite(path, image)

################################################################################
#
#				AUGMENTATIONS
#
################################################################################

def crop_by_levels_augment(image, levels=1, verbose=0):
	"""
	given an image
	return a list of images cropped by level, each level is a division by 2
	the returned array of images are of the same shape as the inputs
	if level is 1 returns empty list
	if level is 2 returns list of 4 images
	if level is 3 returns list of 4+16=20 images
	"""
	augmented = []
	for level in range(1, levels):
		level_crop_shape = (round(image.shape[0]/2**level), round(image.shape[1]/2**level))
		if verbose >=1:
			print("crop_by_levels_augment:: level {}".format(level))
		#for each image return crops from slinding window and corresponding mirror
		gen_samples = fly_over_image(image, window=level_crop_shape, stride=level_crop_shape)
		augmented += [resize_img(sample, image.shape[:-1]) for sample in gen_samples]
	if verbose>=1:
		print("Crop_by_levels_augment: augmentation gone from 1 to {} images".format(len(augmented)))
	return augmented


################################################################################
#
#				VISUALISATIONS
#
################################################################################

def see_image(image, title="", cmap="gray"):
	"""
	just plot an image
	"""
	visu = np.mean(image, axis=-1) if image.shape[-1]==2 else image
	plt.imshow(visu, vmin=0, vmax=1, cmap=cmap)
	plt.title(title)
	plt.show()


def compare_images(image1, image2, title="", subtitle1="", subtitle2="", cmap="gray"):
	"""
	plot two images
	"""
	visu1 = np.mean(image1, axis=-1) if image1.shape[-1]==2 else image1
	visu2 = np.mean(image2, axis=-1) if image2.shape[-1]==2 else image2
	fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
	axs[0].imshow(visu1, vmin=0, vmax=1, cmap=cmap)
	axs[0].set_title(subtitle1)
	axs[1].imshow(visu2, vmin=0, vmax=1, cmap=cmap)
	axs[2].set_title(subtitle2)
	plt.title(title)
	plt.show()
	plt.close()
