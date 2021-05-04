import pandas as pd
import argparse

from config import *
from AutoencoderMetrics import CSV_AE_NAME


if __name__ == '__main__':
	#parsing parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input_files", nargs="+", type=str, help="Files to join")
	args = parser.parse_args()

	merged_df = pd.DataFrame(columns=[COL_IMG_PATH])
	for file_path in args.input_files:
		file = pd.read_csv(file_path, index_col=0)
		type = file.index.name

		if type == CSV_IMAGE_NAME:
			pass	#nothing to do
		elif type == CSV_EXPERIMENTS_NAME:
			continue	#don't add to the df
		elif type == CSV_AE_NAME:

		elif type == CSV_DF_NAME:

		elif type == CSV_FFT_SLOPE_NAME:

		elif type == CSV_MEAN_FFT_SLOPE_NAME:

		elif type == CSV_FFT_BINS_NAME:

		elif type == CSV_GABOR_NAME:

		elif type == CSV_HARALICK_NAME:

		elif type == CSV_LBP_NAME:

		elif type == CSV_BEST_LBP_NAME:

		elif type == CSV_PHOG_NAME:

		elif type == CSV_STATS_NAME:

		elif type == CSV_C_RATIO_NAME:
				
		merged_df = merged_df.merge(file, how="outer", on=COL_IMG_PATH)
