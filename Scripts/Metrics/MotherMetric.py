import pandas as pd
import os

from Preprocess import *


class MotherMetric:
	'''
	Parent class for each metric, takes care of the data processing
	'''
	def __init__(self, preprocess=None, load_from=None):
		if preprocess:
			self.preprocess = preprocess
		else:
			self.preprocess = Preprocess()
		#if given a valid path load the data
		self.data = pd.DataFrame()
		if load_from:
			self.load(load_from)

	def __call__(self, path=None):
		'''
		prepare the image and call the function on it, then store the result
		'''
		if not path:
			image = self.preprocess.get_image()
		else:
			image = self.preprocess(path)
		value = self.function(image)
		self.data = self.data.append(value, ignore_index=True)
		return value

	def metric_from_csv(self, df_path, *args, **kwargs):
		df = pd.read_csv(df_path, index_col=0)
		for path in df[COL_IMG_PATH]:
			image = self.preprocess(path)
			value = self.function(image, *args, **kwargs)
			self.data = self.data.append(value, ignore_index=True)

	def load(self, data_path):
		'''
		load a csv_file as class data
		'''
		try:
			self.data = pd.read_csv(data_path, index_col=0)
		except FileNotFoundError:
			print("nothing to load, continuing with current data")

	def save(self, output_path):
		'''
		save the data as csv, will erase previous
		'''
		#if the directory to store results do not exist create it
		directory = os.path.dirname(output_path)
		if not os.path.exists(directory):
			os.makedirs(directory)
		self.data.to_csv(output_path, index=True)
		print("saved at {}".format(output_path))

	def clear(self):
		'''
		clear the data
		'''
		del self.data
		self.data = pd.DataFrame()

	def function(self, image):
		'''
		perform the metric on the image
		return the value consisting of a df with the parameters and result of the metric
		'''
		raise NotImplementedError("Hey, Don't forget to implement the function!")

	def visualize(self):
		'''
		plot a visualization of the metric
		'''
		raise NotImplementedError("Hey, Don't forget to implement the function!")
