import pandas as pd
import argparse

from config import *



if __name__ == '__main__':
	#parsing parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input_files", nargs="+", type=str, help="Files to join")
	args = parser.parse_args()
	
	for file_path in args.input_files:
		file = pd.read_csv(file_path, index_col=0)
		type = file.index.name