import pandas as pd
import numpy as np
import cv2
import argparse
import imageio
from enum import Enum
import os
import matplotlib.pyplot as plt

from FourierAnalysisMaster.pspec import rgb_2_darter

#PREPROCESSING COLUMN NAMES
COL_NORMALIZE="normalization"
COL_STANDARDIZE="standardization"
COL_IMG_TYPE="image_type"
COL_IMG_CHANNEL="channel_image"
COL_IMG_RESIZE_X="image_resize_x"
COL_IMG_RESIZE_Y="image_resize_y"
COL_IMG_PATH="Image_path"

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


class Preprocess:
	'''
	A preprocess is a class that consistently preproces images the same way,
	it stores the preprocessed image so that we can fetch it multiple times without reprocessing everytime
	it also stores the parameters used to preprocess the image
	'''
	def __init__(self, resizeX, resizeY, normalize, standardize, img_type, img_channel):
		'''
		initialise a process
		'''
		self.normalize = normalize
		self.standardize = standardize
		self.img_type = img_type
		self.img_channel = img_channel
		self.resizeX = resizeX
		self.resizeY = resizeY

		self.image=None

		self.df_parameters = pd.DataFrame({COL_NORMALIZE:self.normalize, COL_STANDARDIZE:self.standardize,
			COL_IMG_TYPE:self.img_type.name,
			COL_IMG_CHANNEL:self.img_channel.name,
			COL_IMG_RESIZE_X:resizeX,
			COL_IMG_RESIZE_Y:resizeY}, index=[0])

	def __call__(self, image_path):
		'''
		Take an image as input and update the preprocessed image
		also return the preprocessed image
		'''
		print(image_path)
		image = imageio.imread(image_path)
		#start with a resize if necessary
		if self.resizeX or self.resizeY:
			rX = self.resizeX if self.resizeX else image.shape[0]
			rY = self.resizeY if self.resizeY else image.shape[1]
			image = cv2.resize(image, dsize=(rY, rX), interpolation=cv2.INTER_CUBIC)
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
				image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			elif self.img_channel == CHANNEL.ALL:
				image = image
			else:
				image = image[:, :, self.img_channel.value]
		#normalize and standardize
		if self.normalize:
			image = (image - np.mean(image, axis=(0,1))) / np.std(image, axis=(0,1))
		if self.standardize:
			image = (image - np.min(image)) / (np.max(image)-np.min(image))

		self.df_parameters.loc[0, COL_IMG_PATH] = image_path

		self.image = image
		return self.image.copy()

	def get_params(self):
		return self.df_parameters.copy()

	def get_image(self):
		print(self.image.dtype)
		return self.image.copy()


if __name__=='__main__':
	#DEFAULT PARAMETERS
	RESIZE_X=None
	RESIZE_Y=None
	NORMALIZE=False
	STANDARDIZE=False
	IMG_TYPE=IMG.RGB
	IMG_CHANNEL=CHANNEL.ALL
	VERBOSE=1

	#parsing command
	parser = argparse.ArgumentParser()
	parser.add_argument("input_path", help="path of the file to open")
	parser.add_argument("-x", "--resizeX", default=RESIZE_X, type=int, help="Resize X to this value, default: {}".format(RESIZE_X))
	parser.add_argument("-y", "--resizeY", default=RESIZE_Y, type=int, help="Resize Y to this value, default: {}".format(RESIZE_Y))
	parser.add_argument("-n", "--normalize", default=NORMALIZE, type=lambda x: bool(eval(x)), help="if image should be normalized, default: {}".format(NORMALIZE))
	parser.add_argument("-s", "--standardize", default=STANDARDIZE, type=lambda x: bool(eval(x)), help="if image should be standardized, default: {}".format(STANDARDIZE))
	parser.add_argument("-v", "--verbose", default=VERBOSE, type=int, choices=[0,1,2], help="set the level of visualization, default: {}".format(VERBOSE))
	parser.add_argument("-t", "--type_img", default=IMG_TYPE.name, type=lambda x: IMG[x], choices=list(IMG), help="the type of image needed, default: {}".format(IMG_TYPE))
	parser.add_argument("-c", "--channel_img", default=IMG_CHANNEL.name, type=lambda x: CHANNEL[x], choices=list(CHANNEL), help="The channel used for the image, default: {}".format(IMG_CHANNEL))
	args = parser.parse_args()

	image = imageio.imread(args.input_path)
	pr = Preprocess(resizeX=args.resizeX, resizeY=args.resizeY, normalize=args.normalize, standardize=args.standardize,
		img_type=args.type_img, img_channel=args.channel_img)
	#1st way to do
	img_pr1 = pr(args.input_path)
	#2nd way to do
	img_pr2 = pr.get_image()

	if args.verbose >=1:
		f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)
		ax1.imshow(image, cmap='gray')
		ax2.imshow(img_pr1, cmap='gray')
		ax3.imshow(img_pr2, cmap='gray')
		plt.show()
