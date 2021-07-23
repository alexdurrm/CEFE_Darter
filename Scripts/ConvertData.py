import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import argparse
from glob import glob
import os
import random

import imageio

from Utils.Preprocess import *
from Utils.FileManagement import COL_IMG_PATH, save_args
from Utils.ImageManip import *

FORMATS_IN = [".jpg", ".npy", ".tiff", ".tif", ".CR2"]
FORMATS_OUT = [".jpg", ".npy", ".tiff", ".tif"]


def get_paths(input, expected_img_ext=None, verbose=0):
	"""
	given the input string "AAA.csv" "BBB.npy" "C**.jpg" "X*.tiff"
	check that the path given are coherent with the expected format
	return the list of paths to open
	"""
	ext = os.path.splitext(input)[-1]
	list_paths = glob(input)

	#if is the path of a file
	if len(list_paths)==1:
		if ext==".csv":
			list_paths = pd.read_csv(input[0])[COL_IMG_PATH]

	#check the image extentions
	for rev_idx, path in enumerate(list_paths[::-1]):
		if expected_img_ext and os.path.splitext(path)[-1] != expected_img_ext:
			del list_paths[-1-rev_idx]
			print("get_paths: Removed {} because of unexpected extension".format(path))
	#return if list of images valid
	if not list_paths:
		raise ValueError("get_paths: No valid path found in:\n{}".format(input))
	if verbose>=1: print("get_paths: loading paths\n{}".format(list_paths))
	return list_paths


def uniformize_shapes(list_images, policy, keep_ratio, fit_method):
	"""
	Given a list of images returns an array of those images resized at the same shape
	"""
	# get the reference shape
	ref_shape = list_images[0].shape
	for image in list_images[1:]:
		shape = image.shape
		if (shape[-1]!=ref_shape[-1]):
			raise ValueError("{} {} does not have the same number of channels as the other images: {}".format(path, shape, ref_shape))
		if shape != ref_shape:
			if policy=="strict":
				raise ValueError("{} {} is not of the expected shape: {}".format(path, shape, ref_shape))
			elif policy=="minimum":
				ref_shape = tuple(min(a,b) for a,b in zip(ref_shape, shape))
			elif policy=="maximum":
				ref_shape = tuple(max(a,b) for a,b in zip(ref_shape, shape))
	#resize if needed
	resized_list =  np.array([resize_img_to_fit(img, ref_shape, keep_ratio, fit_method) for img in list_images])
	return resized_list

def get_agg_numpies(list_paths):
	"""
	given a list of numpy arrays to load, aggregate them and return a numpy array
	"""
	data = np.load(list_paths[0])
	assert data.ndim==3 or data.ndim==4, "{} is not of the right dimension, expected 3D or 4D, got {}".format(list_paths[0], data.ndim)
	for path in list_paths[1:]:
		if data.ndim<4:
			data = np.expand_dims(data, 0)
		data = np.concatenate((data, np.load(path)), axis=0)
	return data

def augment(list_images, do_H_symetry, crop_level, randomize=True, verbose=0):
	"""
	given a list of images return an augmented list of these images
	"""
	#cropping by level
	len_a = len(list_images)
	if crop_level>1:
		augmented = np.array([crop_by_levels_augment(img, crop_level, verbose-1) for img in list_images]).reshape((-1, *list_images.shape[-3:]))
		list_images = np.concatenate((list_images, augmented))
		if verbose>=1: print("Augment: doing crop level {} from {} to {}".format(crop_level, len_a, len(list_images)))
	#horizontal symmetry
	len_a = len(list_images)
	if do_H_symetry:
		augmented = [np.flip(img, axis=(-2)) for img in list_images]
		list_images  = np.concatenate((list_images, augmented))
		if verbose>=1: print("Augment: doing horizontal symmetry from {} to {}".format(len_a, len(list_images)))
	#randomize images
	if randomize:
		np.random.shuffle(list_images)
	return list_images

def main(args):
	"""
	launch functions to convert images and given a namespace of args
	"""
	#get a list of path to load
	list_path = get_paths(args.input, args.input_format, args.verbose)
	#preprocess inputs
	preprocessor = Preprocess(args.resize, args.normalize, args.standardize,
								args.type_img, args.channel_img, args.keep_ratio,
								args.fit_method, verbose=args.verbose)
	if args.input_format==".npy":
		list_images = get_agg_numpies(list_path)
		list_images = preprocessor(list_images)
	else:
		list_images = preprocessor(list_path)
	#uniformely resize images if needed
	if args.resize_policy or args.output_format==".npy":
		resize_policy = args.resize_policy if args.resize_policy else "strict"
		list_images = uniformize_shapes(list_images, resize_policy, args.keep_ratio, args.fit_method)
	#augment and save
	if args.train_test_split:
		train, test = train_test_split(list_images, test_size=args.train_test_split, shuffle=True)
		train = augment(train, args.add_H_sym, args.crop_levels, not args.no_randomize, verbose=args.verbose)
		path, ext = os.path.splitext(args.output)
		save_images(train, path+"_train"+ext, extension=args.output_format, verbose=args.verbose)
		test = augment(test, args.add_H_sym, args.crop_levels, not args.no_randomize, verbose=args.verbose)
		save_images(test, path+"_test"+ext, extension=args.output_format, verbose=args.verbose)
	else:
		list_images = augment(list_images, args.add_H_sym, args.crop_levels, not args.no_randomize, verbose=args.verbose)
		save_images(list_images, args.output, extension=args.output_format, verbose=args.verbose)


if __name__=='__main__':
	RESIZE=(None, None)
	NORMALIZE=False
	STANDARDIZE=False
	IMG_TYPE=IMG.DEFAULT
	IMG_CHANNEL=CHANNEL.DEFAULT
	VERBOSE=0
	DEF_FITTING="cropping"

	parser = argparse.ArgumentParser(description="Script used to prepare datasets, convert images, basically centralize image manipulations into one script")
	#input params
	parser.add_argument("input", help="images to modify, can be the path of a .csv file containing a list of filepath under the column name \"{}\", regular expression (between quotes), or path".format(COL_IMG_PATH))
	parser.add_argument("input_format", choices=FORMATS_IN, help="the format of your input images, can be one of the supported formats: {}".format(FORMATS_IN))
	#output params
	parser.add_argument("output", help="path of the directory (if output_format is .jpg, .tiff ou .tif) or the file where to store the result (if output_format is .npy)")
	parser.add_argument("output_format", choices=FORMATS_OUT, help="the format of your output images, can be one of the supported formats: {}. !!!.npy requires all the images to be the same shape or to set a reshape parameter!!!".format(FORMATS_OUT))
	parser.add_argument("--train_test_split", default=None, type=int, help="if is set will split the data and save two different sets by the value given in percentage, default None")
	#Preprocessing
	parser.add_argument("-r", "--resize", type=int, nargs=2, default=RESIZE, help="resize image to this value, default is {}".format(RESIZE))
	parser.add_argument("-n", "--normalize", default=NORMALIZE, type=lambda x: bool(eval(x)), help="if image should be normalized, default: {}".format(NORMALIZE))
	parser.add_argument("-s", "--standardize", default=STANDARDIZE, type=lambda x: bool(eval(x)), help="if image should be standardized, default: {}".format(STANDARDIZE))
	parser.add_argument("-t", "--type_img", default=IMG_TYPE.name, type=lambda x: IMG[x], choices=list(IMG), help="the type of image needed, default: {}".format(IMG_TYPE))
	parser.add_argument("-c", "--channel_img", default=IMG_CHANNEL.name, type=lambda x: CHANNEL[x], choices=list(CHANNEL), help="The channel used for the image, default: {}".format(IMG_CHANNEL))
	parser.add_argument("-v", "--verbose", default=VERBOSE, type=int, help="set the level of visualization, default: {}".format(VERBOSE))
	parser.add_argument("--keep_ratio", default=False, action='store_true', help="If set, images resized keep the same X to Y ratio as originaly")
	parser.add_argument("-f", "--fit_method", default=DEF_FITTING, type=str, choices=["cropping","padding"], help="If keep_ratio is set, this is the method used to keep the original image ratio, default: {}".format(DEF_FITTING))

	#augmentation and other parameters
	parser.add_argument("--resize_policy", default=None, choices=["strict", "minimum", "maximum"], help="if images are of different sizes what policy should we adopt, strict fails, minimum takes the minimum shape, maximum takes the maximum shape, default None")
	parser.add_argument("--add_H_sym", default=False, action="store_true", help="add Horizontal symmetric images to the output data, default false")
	parser.add_argument("--crop_levels", default=0, type=int, choices=[1,2,3], help="augment the images with cropings, of the original image, 1 is no augmentation, 2 adds 4(2*2) quarter images, 3 adds 20=(4+16(2*2+4*4)) heights of the original image")
	parser.add_argument("--no_randomize", default=False, action="store_true", help="By default results are randomized, is set will not randomize")

	args = parser.parse_args()
	main(args)
	output_dir = os.path.split(args.output)[0]
	save_args(args, os.path.join(output_dir, "convertDataParams.txt"))
