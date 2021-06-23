from ImageManip import *
import numpy as np
import unittest
import random
import math

######################################################################
#																	 #
#	 			Unit tests for Image Manip file 				 	 #
#																	 #
######################################################################


class test_fly_over_image(unittest.TestCase):
	def test(self):
		channel_img1 = np.random.randint(256, size=(200,200,1))
		channel_img2 = np.random.randint(256, size=(200,200,2))
		channel_img3 = np.random.randint(256, size=(100,200,3))
		channel_img4 = np.random.randint(256, size=(256,200,4))

		#test assertion error when window is too big for the image
		with self.assertRaises(AssertionError):
			_ = next(fly_over_image(channel_img1, window=(400,100), stride=(10,10), return_coord=False))
		with self.assertRaises(AssertionError):
			_ = next(fly_over_image(channel_img2, window=(100,400), stride=(10,10), return_coord=False))

		#test the number of outputs given different strides and windows
		for image in [channel_img1, channel_img2, channel_img3, channel_img4]:
			for _ in range(10):

				stride = (random.randint(1, image.shape[0]), random.randint(1, image.shape[1]))
				window = (random.randint(1, image.shape[0]), random.randint(1, image.shape[1]))

				n_x = (image.shape[0]-window[0])//stride[0]+1
				n_y = (image.shape[1]-window[1])//stride[1]+1

				n=0
				for sample in fly_over_image(image, window, stride):
					self.assertEqual(sample.ndim, image.ndim)
					self.assertEqual(sample.shape[:-1], window)
					n+=1
				self.assertEqual(n, n_x*n_y)

				n=0
				for xmin,xmax,ymin,ymax in fly_over_image(image, window, stride, return_coord=True):
					self.assertEqual((xmax-xmin,ymax-ymin), window)
					if n>0:
						self.assertTrue((xmin-prev_xmin) in stride or (ymin-prev_ymin) in stride)
					prev_xmin, prev_ymin = xmin, ymin
					n+=1
				self.assertEqual(n, n_x*n_y)


class test_resize_img(unittest.TestCase):
	def test(self):
		channel_img1 = np.random.randint(256, size=(200,200,1), dtype=np.uint8)
		channel_img2 = np.random.randint(256, size=(200,200,2), dtype=np.uint8)
		channel_img3 = np.random.randint(256, size=(100,200,3), dtype=np.uint8)
		channel_img4 = np.random.randint(256, size=(256,200,4), dtype=np.uint8)

		for image in [channel_img1, channel_img2, channel_img3, channel_img4]:
			for i in range(20):
				new_shape = tuple(np.random.randint(1, 400, size=2))#(int(np.random.randint(1, 400, size=1)), int(np.random.randint(1, 400, size=1)))

				output = resize_img(image, new_shape=new_shape)
				self.assertEqual(output.shape, (*new_shape, image.shape[-1]))

				output = resize_img(image, new_shape=(new_shape[0],None))
				self.assertEqual(output.shape, (new_shape[0], *image.shape[1:]))

				output = resize_img(image, new_shape=(None, new_shape[1]))
				self.assertEqual(output.shape, (image.shape[0], new_shape[1], image.shape[-1]))

class test_resize_to_fit(unittest.TestCase):
	def test(self):
		new_shape=(122, 10)

		channel_img1 = np.random.randint(256, size=(200,200,1), dtype=np.uint8)
		channel_img2 = np.random.randint(256, size=(200,200,2), dtype=np.uint8)
		channel_img3 = np.random.randint(256, size=(100,200,3), dtype=np.uint8)
		channel_img4 = np.random.randint(256, size=(256,200,4), dtype=np.uint8)

		#checks with keep ratio on
		#square images
		output = resize_img_to_fit(channel_img1, new_shape, True)
		self.assertEqual(output.shape[0], new_shape[1])
		self.assertEqual(output.shape[1], new_shape[1])

		output = resize_img_to_fit(channel_img2, new_shape, True)
		self.assertEqual(output.shape[0], new_shape[1])
		self.assertEqual(output.shape[1], new_shape[1])

		#horizontal image
		output = resize_img_to_fit(channel_img3, new_shape, True)
		self.assertEqual(output.shape[0], 5)
		self.assertEqual(output.shape[1], 10)

		#vertival images
		output = resize_img_to_fit(channel_img4, new_shape, True)
		self.assertLessEqual(output.shape[0], 13)
		self.assertGreaterEqual(output.shape[0], 12)
		self.assertEqual(output.shape[1], 10)

		#test different shapes
		for i in range(50):
			shape = tuple(np.random.randint(1, 400, size=2))
			for image in [channel_img1, channel_img2, channel_img3, channel_img4]:
				#checks with keep ratio off
				output = resize_img_to_fit(image, shape, False)
				self.assertEqual(output.shape[:-1], shape)
				self.assertEqual(output.ndim, image.ndim)

				#checks with keep ratio on
				output = resize_img_to_fit(image, shape, True)
				self.assertLessEqual(output.shape[0], shape[0])
				self.assertLessEqual(output.shape[1], shape[1])
				self.assertEqual(output.ndim, image.ndim)

				self.assertLessEqual(output.shape[0]/output.shape[1], math.ceil(image.shape[0]/image.shape[1]))
				self.assertGreaterEqual(output.shape[0]/output.shape[1], math.floor(image.shape[0]/image.shape[1]))

				# self.assertAlmostEqual(output.shape[0]/output.shape[1], image.shape[0]/image.shape[1], 1)


class test_standardize_img(unittest.TestCase):
	def test(self):
		channel_img1 = np.random.randint(256, size=(200,200,1))
		channel_img2 = np.random.randint(256, size=(200,200,2))
		channel_img3 = np.random.randint(256, size=(100,200,3))
		channel_img4 = np.random.randint(256, size=(256,200,4))
		for image in [channel_img1, channel_img2, channel_img3, channel_img4]:
			output = standardize_img(image)
			self.assertEqual(output.shape, image.shape)
			self.assertGreaterEqual(np.min(output), 0)
			self.assertLessEqual(np.max(output), 1)


class test_normalize_img(unittest.TestCase):
	def test(self):
		channel_img1 = np.random.randint(256, size=(200,200,1))
		channel_img2 = np.random.randint(256, size=(200,200,2))
		channel_img3 = np.random.randint(256, size=(100,200,3))
		channel_img4 = np.random.randint(256, size=(256,200,4))
		for image in [channel_img1, channel_img2, channel_img3, channel_img4]:
			output = normalize_img(image)
			self.assertEqual(output.shape, image.shape)
			self.assertAlmostEqual(np.mean(output), 0, places=3)
			self.assertAlmostEqual(np.std(output), 1, places=3)

class test_rgb_2_darter(unittest.TestCase):
	def test(self):
		image = np.random.rand(100,200,3)
		output = rgb_2_darter(image)
		self.assertEqual(3, output.ndim)
		self.assertEqual(2, output.shape[-1])
		self.assertEqual(image.shape[:-1], output.shape[:-1])

		channel_img1 = np.random.randint(256, size=(200,200,1))
		channel_img2 = np.random.randint(256, size=(200,200,2))
		channel_img4 = np.random.randint(256, size=(256,200,4))
		with self.assertRaises(AssertionError):
			_ = rgb_2_darter(channel_img1)
		with self.assertRaises(AssertionError):
			_ = rgb_2_darter(channel_img2)
		with self.assertRaises(AssertionError):
			_ = rgb_2_darter(channel_img4)

class test_decorator(unittest.TestCase):
	def test(self):
		#create arrays of values between 0 and 1
		liste = np.random.randint(256, size=100)
		img2D = np.random.randint(256, size=(120, 128))
		tensor5D = np.random.randint(256, size=(20,52,90,39,99))

		for image in [liste, img2D, tensor5D]:
			with self.assertRaises(AssertionError):
				_ = rgb_2_darter(image)
			with self.assertRaises(AssertionError):
				_ = normalize_img(image)
			with self.assertRaises(AssertionError):
				_ = standardize_img(image)
			with self.assertRaises(AssertionError):
				_ = resize_img_to_fit(image, (42,42))
			with self.assertRaises(AssertionError):
				_ = resize_img(image, (42,42))

if __name__=='__main__':
	unittest.main()
