import os
import pandas as pd
import argparse

from config import *
from Preprocess import *
from AutoencoderMetrics import *
from DeepFeatureMetrics import *
from FFTMetrics import *
from GaborMetrics import *
from GLCMMetrics import *
from LBPMetrics import *
from PHOGMetrics import *
from ScalarMetrics import *

if __name__ == '__main__':
	#parsing parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("input_files", nargs="+", type=str, help="Files to join")
	parser.add_argument("-o", "--output_dir", type=str, default=DIR_RESULTS, help="output directory, default: {}".format(DIR_RESULTS))
	args = parser.parse_args()

	path_and_params_col = [COL_IMG_PATH, COL_NORMALIZE, COL_STANDARDIZE, COL_IMG_TYPE, COL_IMG_CHANNEL, COL_IMG_RESIZE_X, COL_IMG_RESIZE_Y]

	img_info_col=[COL_IMG_PATH,	COL_FILENAME, COL_TYPE,	COL_DIRECTORY, COL_HABITAT,	COL_COLOR_CONTROL, COL_TV_LOSS,
		COL_LAYERS, COL_FISH_SEX, COL_FISH_NUMBER, COL_SPECIES,	COL_IMG_WIDTH, COL_IMG_HEIGHT, COL_IMG_EXT]

	merged_df = pd.DataFrame(columns=path_and_params_col)

	img_info_there=False #become true if a file contains informations about the images

	#for each given file load it and if it recognize the file prepare it
	#so that every column correspond to a metric with its parameters
	#and every line corresponds to an image file
	for file_path in args.input_files:
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
	merged_df.to_csv(os.path.join(args.output_dir, "merged.csv"))
