import numpy as np
import unittest

from Models import *

class test_augmentation(unittest.TestCase):
    def test(self):
        #without reshape
        a=np.random.randint(100, size=(60,50,50,1))
        aug=get_augmentation((50,50))
        b=aug(a).numpy()
        self.assertNotEqual((a!=b).sum() , 0)
        self.assertEqual(a.shape, b.shape)  #check the output is not a
        #with reshape
        foo=get_augmentation((78, 22))
        b=foo(a).numpy()
        self.assertEqual(len(a), len(b))
        self.assertNotEqual(a.shape, b.shape)
        self.assertEqual(a.shape[-1], b.shape[-1]) #same channels


if __name__=='__main__':
	unittest.main()
