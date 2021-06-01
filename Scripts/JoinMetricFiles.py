import os
from config import *

if __name__ == '__main__':
	#parsing parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("input_files", nargs="+", type=str, help="Files to join")
	parser.add_argument("-o", "--output_dir", type=str, default=DIR_RESULTS, help="output directory, default: {}".format(DIR_RESULTS))
	args = parser.parse_args()

	merged_df = joinCSV(args.input_files)
	merged_df.to_csv(os.path.join(args.output_dir, "merged.csv"))
