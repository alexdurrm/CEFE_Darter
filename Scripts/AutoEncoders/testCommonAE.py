import unittest
import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
import os
import shutil

from CommonAE import *
import Models

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

TEST_DIR="test_dir123456_tmp"

class test_MSE(unittest.TestCase):
	def test(self):
		a = np.random.rand(10,100,100,3)
		b = np.random.rand(10,100,100,3)
		npMSE = np.square(a - b).mean()

		tensor1 = tf.constant(a)
		tensor2 = tf.constant(b)
		tensorMSE = get_MSE(tensor1, tensor2).numpy()

		self.assertAlmostEqual(npMSE, tensorMSE, 9)

class test_SSIM(unittest.TestCase):
	def test(self):
		a = np.random.rand(10,100,100,3)
		b = np.random.rand(10,100,100,3)
		npSSIM = 1-ssim(a, b, data_range=1, sigma=1.5, channel_axis=3, multichannel=True) # win_size=11,

		tensor1 = tf.constant(a)
		tensor2 = tf.constant(b)
		tensorSSIM = get_SSIM_Loss(tensor1, tensor2).numpy()

		#skimage fails to calculate with a win_size 11 as tf so lets accept an approximate
		self.assertAlmostEqual(npSSIM, tensorSSIM, 2)

class test_SavePredictionSample(unittest.TestCase):
	def test(self):
		if not os.path.exists(TEST_DIR):
			os.makedirs(TEST_DIR)
		epochs=5
		data = np.random.rand(10,112,112,3)
		model = Models.Convolutional(10, data.shape[1:], name="Convolutional")
		model.compile("Adam", "mse")
		callback = SavePredictionSample(data[0:3], saving_dir=TEST_DIR)

		model.fit(data, data, batch_size=5,
		epochs= epochs,
		callbacks= [callback])
		#check it created files
		for epoch in range(epochs):
			path = os.path.join(TEST_DIR, "reconstructions epoch {} model Convolutional.jpg".format(epoch))
			self.assertTrue(os.path.exists(path))
		shutil.rmtree(TEST_DIR)

class test_get_loss_from_name(unittest.TestCase):
	def test(self):
		"""
		test different loss names
		"""
		l = get_loss_from_name("mse")
		l = get_loss_from_name("ssim")
		with self.assertRaises(ValueError):
			l = get_loss_from_name("fake name")

# class test_XXXX(unittest.TestCase):
# 	def test(self):
# 		pass


if __name__=='__main__':
	unittest.main()
