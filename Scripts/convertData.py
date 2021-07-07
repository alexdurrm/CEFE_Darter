import numpy as np
import pandas as pd

import argparse
from glob import glob
import os

import rawpy as rp
import imageio

from Utils.FileManagement import COL_IMG_PATH

FORMATS_IN = [".jpg", ".npy", ".tiff", ".tif", ".CR2"]
FORMATS_OUT = [".jpg", ".npy", ".tiff", ".tif"]


def get_paths(input, expected_img_ext, force=False):
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
		if os.path.splitext(path)[-1] != expected_img_ext:
			if force:
				del list_paths[-1-idx]
				print("Removed {} because of unexpected extension".format(path))
			else:
				raise ValueError("One of the image is not of the expected extension {}: {}".format(expected_img_ext, path))
	return list_paths


def transform_images(preprocessor, list_paths):
	for p in list_paths:
		img = preprocessor(p)

def save_data(data, output_path, out_format):
	if out_format==".npy":
		np.save(output_path, data)
		print("Saved at {}".format(output_path))
	elif out_format in (".tif", ".tiff", ".jpg"):
		for img in data:
			imageio.imwrite(os.path.join(output_path, img_name)+out_format, data)

def open_data(input):
	"""
	given the input return the input data, either a list of paths or a numpy array
	"""
	ext = os.path.splitext(path)[-1]

	if ext==".npy":
		return np.load(path)
	else:
		return load_paths(input)

def check_shapes(list_paths, policy="strict"):
	"""
	check all images are of the same shape,
	if not
	"""
	ref_shape = open_img(list_paths[0]).shape
	for path in list_paths[1:]:
		shape = open_img(path)
		if shape != ref_shape and (shape[-1]==ref_shape[-1]):
			if policy=="strict":
				raise ValueError("{} {} is not of the expected shape: {}".format(path, shape, ref_shape))
			elif policy=="minimum":
				ref_shape = tuple(min(a,b) for a,b in zip(ref_shape, shape))
			elif policy=="maximum":
				ref_shape = tuple(max(a,b) for a,b in zip(ref_shape, shape))
		else:
			raise ValueError("{} {} does not have the same number of channels as the other images: {}".format(path, shape, ref_shape))
	return ref_shape[:-1]	#return new width and height

if __name__=='__main__':
	RESIZE=(None, None)
	NORMALIZE=False
	STANDARDIZE=False
	IMG_TYPE=IMG.DEFAULT
	IMG_CHANNEL=CHANNEL.DEFAULT
	VERBOSE=1

	parser = argparse.ArgumentParser(description="Script used to prepare datasets, convert images, basically centralize image manipulations into one script")
	#input params
	parser.add_argument("input", help="images to modify, can be the path of a .csv file containing a list of filepath under the column name \"{}\", regular expression (between quotes), or path".format(COL_IMG_PATH))
	parser.add_argument("input_format", choices=FORMATS_IN, help="the format of your input images, can be one of the supported formats: {}".format(FORMATS_IN))
	parser.add_argument("-f", "--force", action="store_true", default=False, help="if some of the given path are invalid, just ignore them and do not throw an exception")
	parser.add_argument("--resize-policy", default="strict", choices=["strict", "minimum", "maximum"], help="if images are of different sizes what policy should we adopt, strict fail, minimum takes the minimum shape, maximum takes the maximum shape")
	#output params
	parser.add_argument("output", help="path of the directory (if output_format is .jpg, .tiff ou .tif) or the file where to store the result (if output_format is .npy)")
	parser.add_argument("output_format", choices=FORMATS_OUT, help="the format of your output images, can be one of the supported formats: {}. !!!.npy requires all the images to be the same shape or to set a reshape parameter!!!".format(FORMATS_OUT))
	#Preprocessing
	parser.add_argument("-r", "--resize", type=int, nargs=2, default=RESIZE, help="resize image to this value, default is {}".format(RESIZE))
	parser.add_argument("-n", "--normalize", default=NORMALIZE, type=lambda x: bool(eval(x)), help="if image should be normalized, default: {}".format(NORMALIZE))
	parser.add_argument("-s", "--standardize", default=STANDARDIZE, type=lambda x: bool(eval(x)), help="if image should be standardized, default: {}".format(STANDARDIZE))
	parser.add_argument("-t", "--type_img", default=IMG_TYPE.name, type=lambda x: IMG[x], choices=list(IMG), help="the type of image needed, default: {}".format(IMG_TYPE))
	parser.add_argument("-c", "--channel_img", default=IMG_CHANNEL.name, type=lambda x: CHANNEL[x], choices=list(CHANNEL), help="The channel used for the image, default: {}".format(IMG_CHANNEL))
	parser.add_argument("-v", "--verbose", default=VERBOSE, type=int, choices=[0,1,2], help="set the level of visualization, default: {}".format(VERBOSE))
	#augmentation
	parser.add_argument("--add-H-sym", default=False, action="store_true", help="add Horizontal symmetric images to the output data")
	parser.add_argument("--crop-levels", default=0, type=int, choices=[1,2,3], help="augment the images with cropings, of the original image, 1 is no augmentation, 2 adds 4(2*2) quarter images, 3 adds 20=(4+16(2*2+4*4)) heights of the original image")
	parser.add_argument()

	args = parser.parse_args()

	list_paths = load_paths(args.input)
	if not list_paths:
		print("no valid path found")
	else:
		if args.input_format==".npy":
			pipeline_numpy(list_paths) ## TODO: numpy pipeline
		else:
			#si output est .npy verif que les images ont la meme shape ou qu'il y a un param reshape
			resize = args.resize
			if args.output_format==".npy" and not(args.resize[0] and args.resize[1]):
				resize = check_shapes(list_paths)	## TODO:  definir dans quel cas on peut choisir une nouvelle shape

			#prepare preprocessing of the image and tranform them
			preprocessor = Preprocess(resize, args.normalize, args.standardize, args.type_img, args.channel_img)
			images = transform_images(preprocessor, list_paths, args.output, args.output_format)
