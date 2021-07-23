import pandas as pd
import argparse
import imageio
from enum import Enum
import os

### DIRECTORY NAMES
DIR_STYLIZED_FISHES=["All_layers",
					"Layers_1and2",
					"Layers_3to5"]
DIR_ORIGINAL_FISHES=["FISH_images"]
DIR_ORIGINAL_HABITATS=["HABITAT_images"]
DIR_POISSONS=["Poissons/JPEG Cropped"]
DIR_SAMUEL=["crops", "Crops"]
DIR_PALETTES=["Palette", "CLUSTER_HABS_3", "CLUSTER_HABS_6"]
DIR_IGNORED=[]


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


### COLUMN NAMES FOR THE IMAGE CSV FILE
COL_IMG_PATH="Image_path"
COL_FILENAME="filename"
COL_TYPE="type"
COL_DIRECTORY="folder"
COL_HABITAT="habitat"
COL_COLOR_CONTROL="color_control"
COL_TV_LOSS="tv_loss"
COL_LAYERS="layers"
COL_FISH_SEX="sex"
COL_FISH_NUMBER="fish_n"
COL_SPECIES="species"
COL_IMG_WIDTH="img_width"
COL_IMG_HEIGHT="img_height"
COL_IMG_EXT="image_extension"

LIST_COLUMNS_IMG=[COL_IMG_PATH, COL_FILENAME, COL_TYPE, COL_DIRECTORY, COL_HABITAT,
	COL_COLOR_CONTROL, COL_TV_LOSS, COL_LAYERS, COL_FISH_SEX, COL_FISH_NUMBER,
	COL_SPECIES,COL_IMG_WIDTH,COL_IMG_HEIGHT,COL_IMG_EXT]

class FILE_TYPE(Enum):
	ORIG_FISH="original fish"
	STYLIZED_FISH="fish stylized"
	HABITAT="habitat"
	ELSE="else"
	def __str__(self):
		return self.name

### COLUMN NAMES FOR CSV EXPERIMENTS
COL_CONTENT_EXP_PATH="exp_fish_path"
COL_STYLE_EXP_PATH="exp_habitat_path"
COL_EXP_ID="exp_id"

LIST_COLUMNS_EXP=[COL_CONTENT_EXP_PATH, COL_STYLE_EXP_PATH, COL_EXP_ID]

### matplotlib parameters
FLAT_UI = ["#8c8c8c", "#5f9e6e", "#cc8963", "#5975a4", "#857aab", "#b55d60", "#c1b37f", "#8d7866", "#d095bf", "#71aec0"]


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

	#if the directory contains stylized fishes
	if [p for p in DIR_STYLIZED_FISHES if p == directory]:
		crossing, color_ctrl, _, tvloss, *_  = filename.split("_")
		middle, fish_n = crossing.rsplit("x", maxsplit=1)
		dict_info[COL_FISH_NUMBER] = fish_n[4:] #4 to remove FISH
		dict_info[COL_HABITAT] = middle
		dict_info[COL_COLOR_CONTROL] = color_ctrl[12:]
		dict_info[COL_TV_LOSS] = tvloss[6:]
		dict_info[COL_LAYERS] = directory
		dict_info[COL_TYPE] = FILE_TYPE.STYLIZED_FISH.value
	#if the directory contains original fishes
	elif [p for p in DIR_ORIGINAL_FISHES if p == directory]:
		fish_n, _, original_size, specie, *_, end = filename.split("_")
		dict_info[COL_SPECIES] = specie
		dict_info[COL_FISH_NUMBER] = end.split('.')[0][1:]
		dict_info[COL_FISH_SEX] = end[0]
		dict_info[COL_TYPE] = FILE_TYPE.ORIG_FISH.value
	#if the directory contains original habitats
	elif [p for p in DIR_ORIGINAL_HABITATS if p == directory]:
		middle, *_ = filename.split('_')
		dict_info[COL_HABITAT] = middle
		dict_info[COL_TYPE] = FILE_TYPE.HABITAT.value
	#if the folder is the samuel folder
	elif [p for p in DIR_SAMUEL if p == directory]:
		specie, *_, fish = filename.split("_")
		dict_info[COL_FISH_NUMBER] = fish.split('.')[0][1:]
		dict_info[COL_FISH_SEX] = fish.split('.')[0][0]
		dict_info[COL_SPECIES] = specie
		dict_info[COL_TYPE] = FILE_TYPE.ORIG_FISH.value
	elif [p for p in DIR_SAMUEL if p == os.path.split(head)[-1]]:
		specie = directory
		code_specie, *_, fish = filename.split("_")
		dict_info[COL_FISH_NUMBER] = fish.split('.')[0][1:]
		dict_info[COL_FISH_SEX] = fish.split('.')[0][0]
		dict_info[COL_SPECIES] = specie
		dict_info[COL_TYPE] = FILE_TYPE.ORIG_FISH.value
	elif [p in head for p in DIR_POISSONS]:
		specie = directory
		code_specie, *_, fish = filename.split("_")
		dict_info[COL_FISH_NUMBER] = fish.split('.')[0][1:]
		dict_info[COL_FISH_SEX] = fish.split('.')[0][0]
		dict_info[COL_SPECIES] = specie
		dict_info[COL_HABITAT] = DICT_HABITAT.get(specie, "not_listed")
		dict_info[COL_TYPE] = FILE_TYPE.ORIG_FISH.value
	elif [p in head for p in DIR_PALETTES]:
		dict_info[COL_HABITAT] = filename.split("_")[0]
		dict_info[COL_TYPE] = FILE_TYPE.HABITAT.value
	else:
		dict_info[COL_TYPE] = FILE_TYPE.ELSE.value
	return dict_info


def group_files_by_experiments(df_files):
	'''
	returns a dataframe containing for each image transfer output,
	a link between the path of the style image and the path of the content image
	'''
	experiments = pd.DataFrame(columns=LIST_COLUMNS_EXP)
	fishes = df_files[df_files[COL_TYPE]==FILE_TYPE.ORIG_FISH.value][[COL_IMG_PATH, COL_FISH_NUMBER, COL_FISH_SEX, COL_SPECIES]]
	habitat = df_files[df_files[COL_TYPE]==FILE_TYPE.HABITAT.value][[COL_IMG_PATH, COL_HABITAT]]

	output_net = df_files[df_files[COL_TYPE]==FILE_TYPE.STYLIZED_FISH.value]
	if not output_net.empty:
		output_net = output_net.drop_duplicates(subset=[COL_FISH_NUMBER, COL_FISH_SEX, COL_SPECIES, COL_HABITAT], ignore_index=True)

		experiments = output_net.merge(fishes, on=[COL_FISH_NUMBER], how="inner", suffixes=(None, "_fish"))
		experiments = experiments.merge(habitat, on=[COL_HABITAT], how="inner", suffixes=(None, "_hab"))

		fish_path = COL_IMG_PATH+"_fish"
		habitat_path = COL_IMG_PATH+"_hab"
		experiments = experiments[[fish_path, habitat_path]].reset_index().rename(columns={'index':COL_EXP_ID, fish_path:COL_CONTENT_EXP_PATH, habitat_path:COL_STYLE_EXP_PATH})
	return experiments


def get_files(main_path, max_depth, types_allowed, ignored_folders, only_endnodes, verbose=0):
	'''
	initiate a dataframe with minimum values in it
	max_depth is the depth of the search: 0 is for a file, 1 is to search one directory, 2 is dir and subdirs...
	only_endnodes is whether or not to ignore files that are not located at the endnode of the directory tree
	ignored_folders is a list of folders to ignore
	'''
	assert max_depth>=0, "You should not give a depth lower than 0"
	data = pd.DataFrame(columns=LIST_COLUMNS_IMG)
	to_visit=[(main_path, 0)]
	while to_visit:
		path, depth = to_visit.pop()
		if os.path.isdir(path) and depth<max_depth and (not [i for i in ignored_folders if i in path]):
			to_visit += [(os.path.join(path, next), depth+1) for next in os.listdir(path)]
		elif os.path.isfile(path) and path.endswith(types_allowed):
			if depth==max_depth or not only_endnodes:
				if verbose>=1: print("ADDING {}".format(path))
				dict_info = load_info_from_filepath(path)
				data.loc[len(data), [*dict_info.keys()]] = [*dict_info.values()]
	return data
