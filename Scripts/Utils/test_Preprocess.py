import unittest
import os
import numpy as np
import imageio

from Preprocess import *

PATH_TEST="15161img_test.png"

class test_wrong_img(unittest.TestCase):
    def test(self):
        #test raise error for 2D image
        im0 = np.random.randint(0,256, size=(100,200), dtype="uint8")
        imageio.imwrite(PATH_TEST, im0)
        with self.assertRaises(AssertionError):
            _ = Preprocess(resize=(None,None), normalize=False, standardize=False, img_type=IMG.RGB, img_channel=CHANNEL.ALL)(PATH_TEST)

        #test raise error for 1D image
        im1 = np.random.randint(0,256, size=(100,200,1), dtype="uint8")
        imageio.imwrite(PATH_TEST, im1)
        with self.assertRaises(AssertionError):
            _ = Preprocess(resize=(None,None), normalize=False, standardize=False, img_type=IMG.RGB, img_channel=CHANNEL.ALL)(PATH_TEST)

        #test raise error for 4D images
        im4 = np.random.randint(0,256, size=(100,200,4), dtype="uint8")
        imageio.imwrite(PATH_TEST, im4)
        with self.assertRaises(AssertionError):
            _ = Preprocess(resize=(None,None), normalize=False, standardize=False, img_type=IMG.RGB, img_channel=CHANNEL.ALL)(PATH_TEST)

        os.remove(PATH_TEST)

class test_preprocess(unittest.TestCase):
    def test(self):
        im = np.random.randint(0,256, size=(100,200,3), dtype="uint8")
        imageio.imwrite(PATH_TEST, im)

        nothing_done = Preprocess(resize=(None,None), normalize=False, standardize=False, img_type=IMG.RGB, img_channel=CHANNEL.ALL)(PATH_TEST)
        self.assertTrue(np.array_equal(nothing_done, im))

        resized = Preprocess(resize=(20,20), normalize=False, standardize=False, img_type=IMG.RGB, img_channel=CHANNEL.ALL)(PATH_TEST)
        self.assertEqual(resized.shape, (20,20,3))
        self.assertEqual(resized.dtype, "uint8")

        normalized = Preprocess(resize=(None,None), normalize=True, standardize=False, img_type=IMG.RGB, img_channel=CHANNEL.ALL)(PATH_TEST)
        self.assertEqual(normalized.shape, im.shape)
        self.assertAlmostEqual(np.mean(normalized), 0, places=3)
        self.assertAlmostEqual(np.std(normalized), 1, places=3)
        self.assertEqual(normalized.dtype, "float32")

        standardized = Preprocess(resize=(None,None), normalize=False, standardize=True, img_type=IMG.RGB, img_channel=CHANNEL.ALL)(PATH_TEST)
        self.assertEqual(normalized.shape, im.shape)
        self.assertGreaterEqual(np.min(standardized), 0)
        self.assertLessEqual(np.max(standardized), 1)
        self.assertEqual(standardized.dtype, "float32")

        to_darter = Preprocess(resize=(None,None), normalize=False, standardize=False, img_type=IMG.DARTER, img_channel=CHANNEL.ALL)(PATH_TEST)
        self.assertEqual(to_darter.shape, (100,200,2))

        to_darter_gray = Preprocess(resize=(None,None), normalize=False, standardize=False, img_type=IMG.DARTER, img_channel=CHANNEL.GRAY)(PATH_TEST)
        self.assertEqual(to_darter_gray.shape, (100,200,1))
        self.assertEqual(to_darter_gray.dtype, "float32")

        with self.assertRaises(ValueError):
            _ = Preprocess(resize=(None,None), normalize=False, standardize=False, img_type=IMG.DARTER, img_channel=CHANNEL.C3)(PATH_TEST)

        to_c1 = Preprocess(resize=(None,None), normalize=False, standardize=False, img_type=IMG.RGB, img_channel=CHANNEL.C1)(PATH_TEST)
        self.assertEqual(to_c1.shape, (100,200,1))
        self.assertTrue(np.array_equal(to_c1, im[..., 0][...,np.newaxis]))
        self.assertEqual(to_c1.dtype, "uint8")

        to_c2 = Preprocess(resize=(None,None), normalize=False, standardize=False, img_type=IMG.RGB, img_channel=CHANNEL.C2)(PATH_TEST)
        self.assertEqual(to_c2.shape, (100,200,1))
        self.assertTrue(np.array_equal(to_c2, im[..., 1][...,np.newaxis]))
        self.assertEqual(to_c2.dtype, "uint8")

        to_c3 = Preprocess(resize=(None,None), normalize=False, standardize=False, img_type=IMG.RGB, img_channel=CHANNEL.C3)(PATH_TEST)
        self.assertEqual(to_c3.shape, (100,200,1))
        self.assertTrue(np.array_equal(to_c3, im[..., 2][...,np.newaxis]))
        self.assertEqual(to_c3.dtype, "uint8")

        to_gray = Preprocess(resize=(None,None), normalize=False, standardize=False, img_type=IMG.RGB, img_channel=CHANNEL.GRAY)(PATH_TEST)
        self.assertEqual(to_gray.shape, (100,200,1))

        os.remove(PATH_TEST)

if __name__=='__main__':
	unittest.main()
