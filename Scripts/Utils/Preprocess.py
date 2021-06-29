import pandas as pd
import numpy as np
import cv2
import argparse
import imageio
import os
import matplotlib.pyplot as plt
from enum import Enum

#enable local imports
import sys
new_path = os.path.dirname(os.path.realpath(__file__))
if new_path not in sys.path:
	sys.path.append(new_path)

from ImageManip import *


#ENUMS
class IMG(Enum):
	RGB="rgb",
	DARTER="darter"
	def __str__(self):
		return self.name

class CHANNEL(Enum):
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
COL_IMG_PATH="Image_path"



class Preprocess:
	'''
	A preprocess is a class that consistently preproces images the same way,
	it stores the preprocessed image so that we can fetch it multiple times without reprocessing everytime
	it also stores the parameters used to preprocess the image
	'''
	def __init__(self, resize, normalize, standardize, img_type, img_channel):
		'''
		initialise a process
		'''
		self.normalize = normalize
		self.standardize = standardize
		self.img_type = img_type
		self.img_channel = img_channel
		self.resizeX = resize[0]
		self.resizeY = resize[1]

		self.image=None

		self.df_parameters = pd.DataFrame({COL_NORMALIZE:self.normalize, COL_STANDARDIZE:self.standardize,
			COL_IMG_TYPE:self.img_type.name,
			COL_IMG_CHANNEL:self.img_channel.name,
			COL_IMG_RESIZE_X:self.resizeX,
			COL_IMG_RESIZE_Y:self.resizeY}, index=[0])

	def __call__(self, image_path):
		'''
		Take an image as input and update the preprocessed image
		also return the preprocessed image
		'''
		image = imageio.imread(image_path)
		print(image_path, image.dtype)
		assert image.ndim==3 and image.shape[-1]==3, "wrong image dimension: {}".format(image.shape)
		image = resize_img(image, [self.resizeX, self.resizeY])

		#convert the image type
		if self.img_type == IMG.DARTER:                     #darter
			image = rgb_2_darter(image)
			if self.img_channel == CHANNEL.GRAY:
				image = image[:, :, 0] + image[:, :, 1]
			elif self.img_channel == CHANNEL.ALL:
				image = image
			elif self.img_channel == CHANNEL.C3:
				raise ValueError("channel 3 and darter type are not compatible parameters")
			else:
				image = image[:, :, self.img_channel.value]
		elif self.img_type == IMG.RGB:                      #RGB
			if self.img_channel == CHANNEL.GRAY:
				image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			elif self.img_channel == CHANNEL.ALL:
				image = image
			else:
				image = image[:, :, self.img_channel.value]
		image = image if image.ndim==3 else image[..., np.newaxis]
		#normalize and standardize
		if self.normalize:
			image = normalize_img(image)
		if self.standardize:
			image = standardize_img(image)
		#store parameters and image
		self.df_parameters.loc[0, COL_IMG_PATH] = image_path
		self.image = image
		return image.copy()

	def get_params(self):
		return self.df_parameters.copy()

	def get_image(self):
		return self.image.copy()

if __name__=='__main__':
	#DEFAULT PARAMETERS
	RESIZE=(None, None)
	NORMALIZE=False
	STANDARDIZE=False
	IMG_TYPE=IMG.RGB
	IMG_CHANNEL=CHANNEL.ALL
	VERBOSE=1
	OVERRIDE=True
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
	args = parser.parse_args()

	params_preprocess={
		"resize":args.resize,
		"normalize":args.normalize,
		"standardize":args.standardize,
		"img_type":args.type_img,
		"img_channel":args.channel_img
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
