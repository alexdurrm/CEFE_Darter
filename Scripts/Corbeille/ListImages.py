import numpy as np
import pandas as pd
import imageio
import os
import argparse
from functools import partial

from config import *

###############################################################
### File used to calculate metrics and save them in a csv file
###############################################################


if __name__ == '__main__':
	#DEFAULT PARAMETERS
	DEPTH = 2
	FORMATS = (".jpg",".png",".tif",".tiff")
	ONLY_ENDNODES = False
	VERBOSE=1

	#parsing parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("action", choices=["list", "group", "both"], help="type of action needed, \"list\" gives a csv containing informations about the image files, \"group\" groups them into experiment categories, \"both\" do both")
	parser.add_argument("input", help="The path of the main directory containing the subdirs or the path of the csv to use.")
	parser.add_argument("-d", "--depth", type=int, choices=[0, 1, 2, 3], default=DEPTH, help="Depth of the path searched, 0 is image, 1 is folder, 2 is subfolders ,etc. default: {}".format(DEPTH))
	parser.add_argument("-o", "--output_dir", default=DIR_RESULTS, help="Directory where to put the csv output, default: {}".format(DIR_RESULTS))
	parser.add_argument("-f", "--formats", default=FORMATS, nargs="+", type=str, help="Image extensions accepted, default: {}".format(FORMATS))
	parser.add_argument("-e", "--endnodes", default=ONLY_ENDNODES, type=lambda x: bool(eval(x)), help="If True accepts only images at endnode directories, else accept all encountered images, default: {}".format(ONLY_ENDNODES))
	parser.add_argument("-v", "--verbose", default=VERBOSE, type=int, choices=[0,1,2], help="Set the level of visualization, default: {}".format(VERBOSE))
	args = parser.parse_args()
	root_dir = os.path.abspath(args.input)
	formats = tuple(args.formats)

	#if the directory to store results do not exist create it
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	if args.action in ("list", "both"):
		#gather informations about the files and save them
		data = get_files(root_dir, args.depth, formats, [], only_endnodes=args.endnodes, visu=args.verbose>=1)
		data_path = os.path.join(args.output_dir, CSV_IMAGE)
		data.to_csv(data_path, index=True)
	if args.action in ("group", "both"):
		#group files by experiments and save experiments
		exp = group_files_by_experiments(data)
		exp_path = os.path.join(args.output_dir, CSV_EXPERIMENTS)
		exp.to_csv(exp_path, index=True)
