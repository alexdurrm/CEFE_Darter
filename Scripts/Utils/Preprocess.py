import pandas as pd
import numpy as np
import cv2
import argparse
import imageio
import os
import matplotlib.pyplot as plt
from enum import Enum

#enable local imports
if __name__!='__main__':
	import sys
	new_path = os.path.dirname(os.path.realpath(__file__))
	if new_path not in sys.path:
		sys.path.append(new_path)

from ImageManip import *


#ENUMS
class IMG(Enum):
	DEFAULT=-1
	RGB="rgb",
	DARTER="darter"
	def __str__(self):
		return self.name

class CHANNEL(Enum):
	DEFAULT=-1
	GRAY="gray"
	C1=0    #R in RGB
	C2=1    #G in RGB
	C3=2    #B in RGB
	ALL="all"
	def __str__(self):
		return self.name


#PREPROCESSING COLUMN NAMES
COL_NORMALIZE="normalization"
COL_STANDARDIZE="standardization"
COL_IMG_TYPE="image_type"
COL_IMG_CHANNEL="channel_image"
COL_IMG_RESIZE_X="image_resize_x"
COL_IMG_RESIZE_Y="image_resize_y"
COL_IMG_KEEP_RATIO="image_keep_ratio"
COL_IMG_FIT_METHOD="image_resize_fitting_method"
COL_IMG_PATH="Image_path"


class Preprocess:
	'''
	A preprocess is a class that consistently preproces images the same way,
	it stores the preprocessed image so that we can fetch it multiple times without reprocessing everytime
	it also stores the parameters used to preprocess the image
	'''
	def __init__(self, resize, normalize, standardize, img_type, img_channel, keep_ratio, fit_method, verbose=0):
		'''
		initialise a process
		'''
		self.normalize = normalize
		self.standardize = standardize
		self.img_type = img_type
		self.img_channel = img_channel
		self.resizeX = resize[0]
		self.resizeY = resize[1]
		self.keep_ratio = keep_ratio
		self.fit_method = fit_method

		self.verbose=verbose
		self.image=None

		self.df_parameters = pd.DataFrame({COL_NORMALIZE:self.normalize, COL_STANDARDIZE:self.standardize,
			COL_IMG_TYPE:self.img_type.name,
			COL_IMG_CHANNEL:self.img_channel.name,
			COL_IMG_RESIZE_X:self.resizeX,
			COL_IMG_RESIZE_Y:self.resizeY,
			COL_IMG_KEEP_RATIO:self.keep_ratio,
			COL_IMG_FIT_METHOD:self.fit_method}, index=[0])

	def do_preprocess(self, input):
		"""
		given an image as a numpy array or as a string path,
		load it if needed and use the parameters of the class
		to preprocess the image
		"""
		#load image if needed and check for expected type
		if isinstance(input, str):
			image = openImage(input)
			if self.verbose>=1:print("Preprocess:do_preprocess:: input: ", input, image.shape, image.dtype)
		elif isinstance(input, np.ndarray):
			image = input
			if self.verbose>=1:print("Preprocess:do_preprocess:: input numpy img: ", image.shape, image.dtype)
		else:
			raise TypeError("Given type {}, expected type str or np.ndarray".format(type(input)))
		assert image.ndim==3 and image.shape[-1]==3, "wrong image dimension: {}".format(image.shape)

		#start preprocessing with a resize
		if self.resizeX or self.resizeY:
			image = resize_img_to_fit(image, (self.resizeX, self.resizeY), self.keep_ratio, self.fit_method)
		#convert the image type
		if self.img_type == IMG.DARTER:                     #darter
			image = rgb_2_darter(image)
			if self.img_channel == CHANNEL.GRAY:
				image = image[:, :, 0] + image[:, :, 1]
			elif self.img_channel == CHANNEL.C3:
				raise ValueError("channel 3 and darter type are not compatible parameters")
			elif self.img_channel in [CHANNEL.C1, CHANNEL.C2]:
				image = image[:, :, self.img_channel.value]
		elif self.img_type == IMG.RGB:                      #RGB
			if self.img_channel == CHANNEL.GRAY:
				image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			elif self.img_channel in [CHANNEL.C1, CHANNEL.C2, CHANNEL.C3]:
				image = image[:, :, self.img_channel.value]
		#ensure the image has 3 dimensions
		image = image if image.ndim==3 else image[..., np.newaxis]
		#normalize and standardize
		if self.normalize:
			image = normalize_img(image)
		if self.standardize:
			image = standardize_img(image)

		if self.verbose>=1:print("Preprocess:do_preprocess:: output numpy img: ", image.shape, image.dtype)
		return image

	def preprocess_list_img(self, img_list):
		"""
		preprocess a list of images (can be list of path or of numpy arrays)
		returns a list of preprocessed images
		"""
		return [self.do_preprocess(img) for img in img_list]

	def __call__(self, input, numpy_name=""):
		'''
		Take an image or an image path as input and update the preprocessed image
		img_p:can be string or numpy array
		numpy_name:name to store if img_p is a numpy array
		also return the preprocessed image
		'''
		#if list of images
		if isinstance(input, list) or (isinstance(input, np.ndarray) and input.ndim==4):
			res = self.preprocess_list_img(input)
			self.image=None
			self.df_parameters.loc[0, COL_IMG_PATH] = ""
		else:
			#store parameters and image
			res = self.do_preprocess(input)
			self.image = res.copy()
			self.df_parameters.loc[0, COL_IMG_PATH] = input if isinstance(input, str) else numpy_name
		return res

	def get_params(self):
		return self.df_parameters.copy()

	def get_image(self):
		return self.image.copy()


if __name__=='__main__':
	#DEFAULT PARAMETERS
	RESIZE=(None, None)
	NORMALIZE=False
	STANDARDIZE=False
	IMG_TYPE=IMG.DEFAULT
	IMG_CHANNEL=CHANNEL.DEFAULT
	VERBOSE=1
	DEF_FITTING=None
	#parsing parameters
	parser = argparse.ArgumentParser(description="Preprocess an image, apply transformations on a given image")
	parser.add_argument("input_path", help="path of the image to open")
	parser.add_argument("-o", "--output_path", default=None, help="path of the image to save")
	#parameters for preprocess
	parser.add_argument("-r", "--resize", type=int, nargs=2, default=RESIZE, help="resize image to this value, default is {}".format(RESIZE))
	parser.add_argument("-n", "--normalize", default=NORMALIZE, type=lambda x: bool(eval(x)), help="if image should be normalized, default: {}".format(NORMALIZE))
	parser.add_argument("-s", "--standardize", default=STANDARDIZE, type=lambda x: bool(eval(x)), help="if image should be standardized, default: {}".format(STANDARDIZE))
	parser.add_argument("-t", "--type_img", default=IMG_TYPE.name, type=lambda x: IMG[x], choices=list(IMG), help="the type of image needed, default: {}".format(IMG_TYPE))
	parser.add_argument("-c", "--channel_img", default=IMG_CHANNEL.name, type=lambda x: CHANNEL[x], choices=list(CHANNEL), help="The channel used for the image, default: {}".format(IMG_CHANNEL))
	parser.add_argument("-v", "--verbose", default=VERBOSE, type=int, choices=[0,1,2], help="set the level of visualization, default: {}".format(VERBOSE))
	parser.add_argument("--keep_ratio", default=False, action='store_true', help="If set, images resized keep the same X to Y ratio as originaly")
	parser.add_argument("-f", "--fit_method", default=DEF_FITTING, type=str, choices=["cropping","padding"], help="If keep_ratio is set, this is the method used to keep the original image ratio, default: {}".format(DEF_FITTING))

	args = parser.parse_args()

	params_preprocess={
		"resize":args.resize,
		"normalize":args.normalize,
		"standardize":args.standardize,
		"img_type":args.type_img,
		"img_channel":args.channel_img,
		"keep_ratio":args.keep_ratio,
		"fit_method":args.fit_method
	}
	preprocess = Preprocess(**params_preprocess)
	img_out = preprocess(args.input_path)
	if args.verbose>=1:
		if img_out.shape[-1]==2: img_out = img_out[...,0] + img_out[...,1]
		plt.title("{}".format(params_preprocess.values()))
		plt.imshow(img_out, cmap="gray")
		plt.show()
	if args.output_path:
		imageio.imwrite(args.output_path, img_out)
else:
	#enable local imports
	import sys
	new_path = os.path.dirname(os.path.realpath(__file__))
	if new_path not in sys.path:
		sys.path.append(new_path)
