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
DIR_PALETTES=["Palette"]
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


def get_files(main_path, max_depth, types_allowed, ignored_folders, only_endnodes, visu=False):
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
				if visu: print("ADDING {}".format(path))
				dict_info = load_info_from_filepath(path)
				data.loc[len(data), [*dict_info.keys()]] = [*dict_info.values()]
	return data


def joinCSV(input_files):
	path_and_params_col = [COL_IMG_PATH, COL_NORMALIZE, COL_STANDARDIZE, COL_IMG_TYPE, COL_IMG_CHANNEL, COL_IMG_RESIZE_X, COL_IMG_RESIZE_Y]

	img_info_col=[COL_IMG_PATH,	COL_FILENAME, COL_TYPE,	COL_DIRECTORY, COL_HABITAT,	COL_COLOR_CONTROL, COL_TV_LOSS,
		COL_LAYERS, COL_FISH_SEX, COL_FISH_NUMBER, COL_SPECIES,	COL_IMG_WIDTH, COL_IMG_HEIGHT, COL_IMG_EXT]

	merged_df = pd.DataFrame(columns=path_and_params_col)

	img_info_there=False #become true if a file contains informations about the images
	#for each given file load it and if it recognize the file prepare it
	#so that every column correspond to a metric with its parameters
	#and every line corresponds to an image file
	for file_path in input_files:
		file = pd.read_csv(file_path, index_col=0)
		f_type = os.path.split(file_path)[-1]

		if f_type == CSV_IMAGE:
			img_info_there=True
			print("merging file {} as {} type".format(file_path, CSV_IMAGE))
			merged_df = merged_df.merge(file, how="outer", on=COL_IMG_PATH)
			continue	#add only w.r.t path image

		elif f_type == CSV_EXPERIMENTS:
			print("not merging this file {}".format(file_path))
			continue	#don't add to the df

		elif f_type == CSV_STATS_METRICS:
			print("merging file {} as {} type".format(file_path, CSV_STATS_METRICS))
			pass   		#no preparation needed

		elif f_type == CSV_COLOR_RATIO:
			print("merging file {} as {} type".format(file_path, CSV_COLOR_RATIO))
			pass 		#no preparation needed

		elif f_type == CSV_DEEP_FEATURES:
			print("merging file {} as {} type".format(file_path, CSV_DEEP_FEATURES))
			file = file.pivot(index=path_and_params_col, columns=[COL_MODEL_NAME, COL_LAYER_DF], values=[COL_SPARSENESS_DF, COL_ENTROPY_DF, COL_KURTOSIS_DF, COL_MEAN_DF])
			merged_df.columns = ["_".join(map(str, col )) if isinstance(col, tuple) else col for col in merged_df.columns]
			file.reset_index(inplace=True)

		elif f_type == CSV_FFT_SLOPE:
			print("merging file {} as {} type".format(file_path, CSV_FFT_SLOPE))
			file = file.pivot(index=path_and_params_col, columns=[COL_F_WIN_SIZE, COL_FFT_RANGE_MIN, COL_FFT_RANGE_MAX, COL_F_SAMPLE_IDX], values=[COL_F_SLOPE_SAMPLE])
			merged_df.columns = ["_".join(map(str, col )) if isinstance(col, tuple) else col for col in merged_df.columns]
			file.reset_index(inplace=True)

		elif f_type == CSV_MEAN_FFT_SLOPE:
			print("merging file {} as {} type".format(file_path, CSV_MEAN_FFT_SLOPE))
			file = file.pivot(index=path_and_params_col, columns=[COL_F_WIN_SIZE, COL_FFT_RANGE_MIN, COL_FFT_RANGE_MAX], values=[COL_F_MEAN_SLOPE])
			merged_df.columns = ["_".join(map(str, col )) if isinstance(col, tuple) else col for col in merged_df.columns]
			file.reset_index(inplace=True)

		elif f_type == CSV_FFT_BINS:
			print("merging file {} as {} type".format(file_path, CSV_FFT_BINS))
			file = file.pivot(index=path_and_params_col, columns=[COL_F_WIN_SIZE, COL_FFT_RANGE_MIN, COL_FFT_RANGE_MAX, COL_F_SAMPLE_IDX, COL_FREQ_F], values=[COL_AMPL_F])
			merged_df.columns = ["_".join(map(str, col )) if isinstance(col, tuple) else col for col in merged_df.columns]
			file.reset_index(inplace=True)

		elif f_type == CSV_GABOR:
			print("merging file {} as {} type".format(file_path, CSV_GABOR))
			file = file.pivot(index=path_and_params_col, columns=[COL_GABOR_ANGLES, COL_GABOR_FREQ], values=[COL_GABOR_VALUES])
			merged_df.columns = ["_".join(map(str, col )) if isinstance(col, tuple) else col for col in merged_df.columns]
			file.reset_index(inplace=True)

		elif f_type == CSV_HARALICK:
			print("merging file {} as {} type".format(file_path, CSV_HARALICK))
			haralick_descriptors = [COL_GLCM_MEAN, COL_GLCM_VAR, COL_GLCM_CORR, COL_GLCM_CONTRAST,
			COL_GLCM_DISSIMIL, COL_GLCM_HOMO, COL_GLCM_ASM, COL_GLCM_ENERGY, COL_GLCM_MAXP, COL_GLCM_ENTROPY]
			file = file.pivot(index=path_and_params_col, columns=[COL_GLCM_ANGLE, COL_GLCM_DIST], values=haralick_descriptors)
			merged_df.columns = ["_".join(map(str, col )) if isinstance(col, tuple) else col for col in merged_df.columns]
			file.reset_index(inplace=True)

		elif f_type == CSV_LBP:
			print("merging file {} as {} type".format(file_path, CSV_LBP))
			file = file.pivot(index=path_and_params_col, columns=[COL_POINTS_LBP, COL_RADIUS_LBP, COL_BIN_LBP], values=[COL_COUNT_LBP])
			merged_df.columns = ["_".join(map(str, col )) if isinstance(col, tuple) else col for col in merged_df.columns]
			file.reset_index(inplace=True)

		elif f_type == CSV_BEST_LBP:
			print("merging file {} as {} type".format(file_path, CSV_BEST_LBP))
			file = file.pivot(index=path_and_params_col, columns=[COL_POINTS_LBP, COL_RADIUS_LBP, COL_RANK_LBP], values=[COL_VALUE_LBP, COL_COUNT_LBP])
			merged_df.columns = ["_".join(map(str, col )) if isinstance(col, tuple) else col for col in merged_df.columns]
			file.reset_index(inplace=True)

		elif f_type == CSV_PHOG:
			print("merging file {} as {} type".format(file_path, CSV_PHOG))
			file = file.pivot(index=path_and_params_col, columns=[COL_PHOG_BIN, COL_PHOG_ORIENTATIONS, COL_PHOG_LEVELS], values=[COL_PHOG_VALUE])
			merged_df.columns = ["_".join(map(str, col )) if isinstance(col, tuple) else col for col in merged_df.columns]
			file.reset_index(inplace=True)

		elif CSV_AE in f_type:
			print("merging file {} as {} type".format(file_path, CSV_AE))
			file = file.pivot(index=path_and_params_col, columns=[], values=[])
			merged_df.columns = ["_".join(map(str, col)) if isinstance(col, tuple) else col for col in merged_df.columns]
			file.reset_index(inplace=True)

		else:
			print("Unrecognized file {}".format(file_path))
			continue

		merged_df = merged_df.merge(file, how="outer", on=path_and_params_col)

	if img_info_there:
		merged_df = merged_df.pivot(index=img_info_col, columns=[COL_NORMALIZE, COL_STANDARDIZE, COL_IMG_TYPE, COL_IMG_CHANNEL, COL_IMG_RESIZE_X, COL_IMG_RESIZE_Y])
	else:
		merged_df = merged_df.pivot(index=COL_IMG_PATH, columns=[COL_NORMALIZE, COL_STANDARDIZE, COL_IMG_TYPE, COL_IMG_CHANNEL, COL_IMG_RESIZE_X, COL_IMG_RESIZE_Y])

	merged_df.columns = ["_".join(map(str, col)) if isinstance(col, tuple) else col for col in merged_df.columns]
	merged_df.dropna(axis=1, how="all", inplace=True)
	# print(merged_df)
	print(merged_df.columns)
	merged_df.reset_index(inplace=True)
	return merged_df

if __name__ == '__main__':
	#parsing parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("input_files", nargs="+", type=str, help="Files to join")
	parser.add_argument("-o", "--output_dir", type=str, default=DIR_RESULTS, help="output directory, default: {}".format(DIR_RESULTS))
	args = parser.parse_args()

	merged_df = joinCSV(args.input_files)
	merged_df.to_csv(os.path.join(args.output_dir, "merged.csv"))
