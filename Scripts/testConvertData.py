import unittest
import os
import shutil
from glob import glob

from ConvertData import *
from Utils.Preprocess import *

TEST_DIRECTORY="TestImages"
TEST_DIR_OUT="TestImages/test_out"

# TEST_FORMATS_IN = [".jpg", ".npy", ".tiff", ".tif", ".CR2"]
# TEST_FORMATS_OUT = [".jpg", ".npy", ".tiff", ".tif"]
class Params:
	def __init__(self):
		self.input = os.path.join(TEST_DIRECTORY,"*.jpg")
		self.input_format = '.jpg'
		self.output = TEST_DIR_OUT
		self.output_format = ".jpg"
		self.train_test_split = None
		self.resize_policy = None
		self.resize = (None, None)
		self.normalize = False
		self.standardize = False
		self.type_img = IMG["DEFAULT"]
		self.channel_img = CHANNEL["DEFAULT"]
		self.verbose = 0
		self.add_H_sym = False
		self.keep_ratio = False
		self.crop_levels = 0
		self.no_randomize = False

class test_main_npy(unittest.TestCase):
	def test(self):
		#test with default params
		def_args = Params()
		#test for each input format
		for ext_in in FORMATS_IN:
			def_args.input_format = ext_in
			#for each output format
			for ext_out in FORMATS_OUT:
				main(def_args)
				if ext_out==".npy":
					#check there is at least one image
					img_out = TEST_DIR_OUT+ext_out
					self.assertTrue(os.path.exists(img_out))
				else:
					imgs_out = glob(os.path.join(TEST_DIR_OUT,"*"+ext_out))
					for p in imgs_out:
						self.assertTrue(os.path.exists(p))
					if ext_in!=".npy":
						img_in = glob(os.path.join(TEST_DIRECTORY, "*"+ext_in))
						self.assertEqual(len(imgs_out), len(img_in))
				# shutil.rmtree(TEST_DIR_OUT)
if __name__=='__main__':
	unittest.main()
