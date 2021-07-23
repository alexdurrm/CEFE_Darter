import argparse
import numpy as np
import os
from glob import glob
import imageio
from Utils.Preprocess import *

def main(preprocessor, paths, file_out):
	#check shape on a sample
	shape = preprocessor(paths[0]).shape
	data = np.empty(shape=(len(paths), *shape), dtype='float32')
	#store the images
	for idx, path in enumerate(paths):
		data[idx] = preprocessor(path)
	np.save(file_out, data)

if __name__=="__main__":
	RESIZE=(None, None)
	NORMALIZE=False
	STANDARDIZE=False
	IMG_TYPE=IMG.DEFAULT
	IMG_CHANNEL=CHANNEL.DEFAULT
	DEF_FITTING=None
	parser = argparse.ArgumentParser(description="take images and return a numpy")
	parser.add_argument("glob_input")
	parser.add_argument("file_out")
	parser.add_argument("-r", "--resize", type=int, nargs=2, default=RESIZE, help="resize image to this value, default is {}".format(RESIZE))
	parser.add_argument("-n", "--normalize", default=NORMALIZE, type=lambda x: bool(eval(x)), help="if image should be normalized, default: {}".format(NORMALIZE))
	parser.add_argument("-s", "--standardize", default=STANDARDIZE, type=lambda x: bool(eval(x)), help="if image should be standardized, default: {}".format(STANDARDIZE))
	parser.add_argument("-t", "--type_img", default=IMG_TYPE.name, type=lambda x: IMG[x], choices=list(IMG), help="the type of image needed, default: {}".format(IMG_TYPE))
	parser.add_argument("-c", "--channel_img", default=IMG_CHANNEL.name, type=lambda x: CHANNEL[x], choices=list(CHANNEL), help="The channel used for the image, default: {}".format(IMG_CHANNEL))
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
	main(preprocess, glob(args.glob_input), args.file_out)
