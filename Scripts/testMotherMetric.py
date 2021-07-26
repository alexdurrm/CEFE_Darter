import unittest
import pandas as pd

from Utils.Preprocess import *
from MotherMetric import *


class test_AE(unittest.TestCase):
	def test(self):
		params_preprocess={
			"resize":(120, 120),
			"normalize":True,
			"standardize":True,
			"img_type":IMG["DARTER"],
			"img_channel":CHANNEL["GRAY"],
			"keep_ratio":True,
			"fit_method":"mean"
		}
		preprocess = Preprocess(**params_preprocess)
		metric = AutoencoderMetrics(args.model, prediction_shape, preprocess=preprocess)
		metric.metric_from_path_list(LIST_PATH_TEST)

class test_DF(unittest.TestCase):
	def test(self):
		params_preprocess={
			"resize":(120, 120),
			"normalize":True,
			"standardize":True,
			"img_type":IMG["DARTER"],
			"img_channel":CHANNEL["GRAY"],
			"keep_ratio":True,
			"fit_method":"min"
		}
		preprocess = Preprocess(**params_preprocess)
		metric = DeepFeatureMetrics("vgg16", (120,120,1), preprocess=preprocess)
		metric.metric_from_path_list(LIST_PATH_TEST)

class test_FFT(unittest.TestCase):
	def test(self):
		params_preprocess={
			"resize":(120, 120),
			"normalize":True,
			"standardize":True,
			"img_type":IMG["DARTER"],
			"img_channel":CHANNEL["GRAY"],
			"keep_ratio":True,
			"fit_method":"max"
		}
		fft_range=[10, 80]
		preprocess = Preprocess(**params_preprocess)
		metric = FFT_bins(fft_range=fft_range, sample_dim=50, preprocess=preprocess)
		metric.metric_from_path_list(LIST_PATH_TEST)

		metric = FFTSlopes(fft_range=fft_range, sample_dim=50, preprocess=preprocess)
		metric.metric_from_path_list(LIST_PATH_TEST)

		metric = MeanFFTSlope(fft_range=fft_range, sample_dim=50, preprocess=preprocess)
		metric.metric_from_path_list(LIST_PATH_TEST)

class test_GABOR(unittest.TestCase):
	def test(self):
		params_preprocess={
			"resize":(120, 120),
			"normalize":True,
			"standardize":True,
			"img_type":IMG["DARTER"],
			"img_channel":CHANNEL["GRAY"],
			"keep_ratio":True,
			"fit_method":0.5
		}
		preprocess = Preprocess(**params_preprocess)
		metric = GaborMetrics(angles=[12, 25], frequencies=[1,2], preprocess=preprocess)
		metric.metric_from_path_list(LIST_PATH_TEST)

class test_HARALICK(unittest.TestCase):
	def test(self):
		params_preprocess={
			"resize":(120, 120),
			"normalize":True,
			"standardize":True,
			"img_type":IMG["DARTER"],
			"img_channel":CHANNEL["GRAY"],
			"keep_ratio":True,
			"fit_method":"mean"
		}
		preprocess = Preprocess(**params_preprocess)
		metric = HaralickMetrics(distances=[2, 5], angles=[90, 45], preprocess=preprocess)
		metric.metric_from_path_list(LIST_PATH_TEST)

class test_LBP(unittest.TestCase):
	def test(self):
		params_preprocess={
			"resize":(120, 120),
			"normalize":True,
			"standardize":True,
			"img_type":IMG["DARTER"],
			"img_channel":CHANNEL["GRAY"],
			"keep_ratio":True,
			"fit_method":"mean"
		}
		preprocess = Preprocess(**params_preprocess)
		metric = BestLBPMetrics(points=[4,8], radius=[2,4], n_best=20, preprocess=preprocess)
		metric.metric_from_path_list(LIST_PATH_TEST)

		metric = LBPHistMetrics(points=[4,8], radius=[2,4], nbins=20, preprocess=preprocess)
		metric.metric_from_path_list(LIST_PATH_TEST)

class test_PHOG(unittest.TestCase):
	def test(self):
		params_preprocess={
			"resize":(120, 120),
			"normalize":True,
			"standardize":True,
			"img_type":IMG["DARTER"],
			"img_channel":CHANNEL["GRAY"],
			"keep_ratio":True,
			"fit_method":"mean"
		}
		preprocess = Preprocess(**params_preprocess)
		metric = PHOGMetrics(orientations=[4, 8], level=3, preprocess=preprocess)
		metric.metric_from_path_list(LIST_PATH_TEST)

class test_stats(unittest.TestCase):
	def test(self):
		params_preprocess={
			"resize":(120, 120),
			"normalize":True,
			"standardize":True,
			"img_type":IMG["DARTER"],
			"img_channel":CHANNEL["GRAY"],
			"keep_ratio":True,
			"fit_method":"mean"
		}
		preprocess = Preprocess(**params_preprocess)
		metric = StatMetrics(preprocess, load_from=load_from)
		metric.metric_from_path_list(LIST_PATH_TEST)

class test_color_ratio(unittest.TestCase):
	def test(self):
		params_preprocess={
			"resize":(120, 120),
			"normalize":True,
			"standardize":True,
			"img_type":IMG["DARTER"],
			"img_channel":CHANNEL["ALL"],
			"keep_ratio":True,
			"fit_method":"mean"
		}
		preprocess = Preprocess(**params_preprocess)
		metric = ColorRatioMetrics(preprocess)
		metric.metric_from_path_list(LIST_PATH_TEST)


if __name__=='__main__':
	PATH_TEST = "Images/Test/image_list_test.csv"
	metric = get_minimal_df("TestImages/Fish/", 2, types_allowed=(".jpg"), ignored_folders=[], only_endnodes=True)
	metric.to_csv(PATH_TEST, index=True)
	path_list = pd.read_csv(PATH_TEST)[COL_IMG_PATH]

	unittest.main()
