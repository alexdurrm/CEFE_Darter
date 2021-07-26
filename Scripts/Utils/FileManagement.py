import pandas as pd
import argparse
import imageio
from enum import Enum
import os


###Dictionnary fish to their corresponding habitat
DICT_HABITAT = {}
DICT_HABITAT['barrenense'] = 'bedrock'
DICT_HABITAT['blennioides'] = 'boulder'
DICT_HABITAT['caeruleum'] = 'gravel'
DICT_HABITAT['camurum'] = 'boulder'
DICT_HABITAT['chlorosomum'] = 'sand'
DICT_HABITAT['gracile'] = 'detritus'
DICT_HABITAT['olmstedi'] = 'sand'
DICT_HABITAT['pyrrhogaster'] = 'sand'
DICT_HABITAT['swaini'] = 'detritus'
DICT_HABITAT['zonale'] = 'gravel'
DICT_HABITAT['zonistium'] = 'sand'
DICT_HABITAT['punctulatum'] = 'bedrock'

#dict given by Tamra attributing to each specie a score of correspondance with each 3 clustered habitats
HABITAT_SCORE_3GRPS = {}
HABITAT_SCORE_3GRPS["name_grps"] = ["class 1", "class 2", "class 3"]
HABITAT_SCORE_3GRPS['caeruleum'] = [3,1,1]
HABITAT_SCORE_3GRPS['zonale'] = [2,1,2]
HABITAT_SCORE_3GRPS['zonistium'] = [1,2,1]
HABITAT_SCORE_3GRPS['gracile'] = [1,3,1]
HABITAT_SCORE_3GRPS['pyrrhogaster'] = [1,2,1]
HABITAT_SCORE_3GRPS['barrenense'] = [2,2,1]
HABITAT_SCORE_3GRPS['olmstedi'] = [1,1,1]
HABITAT_SCORE_3GRPS['chlorosomum'] = [1,2,1]
HABITAT_SCORE_3GRPS['camurum'] = [2,0,1]
HABITAT_SCORE_3GRPS['swaini'] = [2,2,1]
HABITAT_SCORE_3GRPS['punctulatum'] = [1,3,2]

#dict given by Tamra attributing to each specie a score of correspondance with each 6 clustered habitats
HABITAT_SCORE_6GRPS = {}
HABITAT_SCORE_6GRPS["name_grps"] = ["class 1", "class 2", "class 3", "class 4", "class 5", "class 6"]
HABITAT_SCORE_6GRPS['caeruleum'] = [3,1,1,1,1,2]
HABITAT_SCORE_6GRPS['zonale'] = [2,2,1,1,1,2]
HABITAT_SCORE_6GRPS['zonistium'] = [1,3,1,2,2,1]
HABITAT_SCORE_6GRPS['gracile'] = [1,3,1,1,2,1]
HABITAT_SCORE_6GRPS['pyrrhogaster'] = [1,3,1,2,2,1]
HABITAT_SCORE_6GRPS['barrenense'] = [2,2,2,0,0,1]
HABITAT_SCORE_6GRPS['olmstedi'] = [1,2,1,3,2,1]
HABITAT_SCORE_6GRPS['chlorosomum'] = [1,2,1,2,2,1]
HABITAT_SCORE_6GRPS['camurum'] = [2,0,3,0,1,3]
HABITAT_SCORE_6GRPS['swaini'] = [2,2,1,1,1,1]
HABITAT_SCORE_6GRPS['punctulatum'] = [1,2,1,1,2,1]


### COLUMN NAMES FOR THE IMAGE INFORMATIONS CSV FILE
COL_IMG_PATH="Image_path"
COL_FILENAME="filename"
COL_TYPE="type"
COL_DIRECTORY="folder"
COL_IMG_WIDTH="img_width"
COL_IMG_HEIGHT="img_height"
COL_IMG_EXT="image_extension"

LIST_COLUMNS_IMG=[COL_IMG_PATH, COL_FILENAME, COL_TYPE, COL_DIRECTORY, COL_IMG_WIDTH,COL_IMG_HEIGHT,COL_IMG_EXT]

class FILE_TYPE(Enum):
	ORIG_FISH="original fish"
	STYLIZED_FISH="fish stylized"
	HABITAT="habitat"
	ELSE="else"
	def __str__(self):
		return self.name

def save_args(args, textfile):
	"""
	given a namespace args and a name
	will save it in the specified textfile path
	"""
	if args.verbose>=1:
		print(vars(args))
	if os.path.exists(textfile):
		os.remove(textfile)
	with open(textfile, "w+") as file:
		for k,v in vars(args).items():
			file.write(str(k)+ " : "+str(v)+"\n")

def load_info_from_filepath(file_path):
	'''
	Given an absolute file path
	returns the informations about the file in the form of a dict
	'''
	long_head, filename = os.path.split(file_path)
	head, directory = os.path.split(long_head)
	_, ext = os.path.splitext(filename)
	image = imageio.imread(file_path)
	dict_info={COL_IMG_PATH:file_path, COL_FILENAME:filename, COL_DIRECTORY:directory,
		COL_IMG_HEIGHT:image.shape[0], COL_IMG_WIDTH:image.shape[1], COL_IMG_EXT:ext}
	return dict_info

def get_minimal_df(main_path, max_depth, types_allowed, ignored_folders, only_endnodes, verbose=0):
	'''
	Perform a width first search in the main directory and initiate a dataframe with minimum values in it
	look for files of the types allowed in the main path
	max_depth: is the depth of the search: 0 is for a file, 1 is to search one directory, 2 is dir and subdirs...
	only_endnodes: is whether or not to ignore files that are not located at the endnode of the directory tree
	ignored_folders: is a list of folders to ignore
	the return dataframe contains the path of the files selected and a minimal additional informations
	'''
	assert max_depth>=0, "You should not give a depth lower than 0"
	data = pd.DataFrame(columns=LIST_COLUMNS_IMG)
	to_visit=[(main_path, 0)]
	while to_visit:
		path, depth = to_visit.pop()
		#if path is a directory and we haven't reached max depth add it to the visit list
		if os.path.isdir(path) and depth<max_depth and (not [i for i in ignored_folders if i in path]):
			to_visit += [(os.path.join(path, next), depth+1) for next in os.listdir(path)]
		#if the path is a file and we can select it, retrieve its informations and store them in df
		elif os.path.isfile(path) and path.endswith(types_allowed):
			if depth==max_depth or not only_endnodes:
				if verbose>=1: print("ADDING {}".format(path))
				dict_info = load_info_from_filepath(path)
				data.loc[len(data), [*dict_info.keys()]] = [*dict_info.values()]
	return data
