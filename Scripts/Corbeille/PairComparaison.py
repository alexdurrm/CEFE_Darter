import numpy as np
import pandas as pd
import argparse
import imageio

from config import *



def metrics_from_csv(df_path):
	list_path = pd.read_csv(df_path, index_col=0)[COL_IMG_PATH]
	data = pd.DataFrame()
	for i, path_x in enumerate(list_path):
		gram_x = GramMatrix(imageio.imread(path_x))
		for j , path_y in enumerate(list_path):
			idx_data = len(data)
			data.loc[idx_data, ["path_img_1", "path_img_2"]] = [path_x, path_y]
			gram_y = GramMatrix(imageio.imread(path_y))
			dist_gram = EuclideanDist(gram_x, gram_y) 
			print(" {}/{} dist {}".format(idx_data, len(list_path), dist))
			data.loc[idx_data, []] = [dist_gram]
	return data


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("input_path", help="path of the image file to open")
	parser.add_argument("output_path", help="where to save the file")
	parser.add_argument("model_path", help="path to the autoencoder to load")
	args = parser.parse_args()

	data = metrics_from_csv(args.input_path)
	data.to_csv(args.output_path, index=True)
