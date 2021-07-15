import numpy as np
import unittest
import shutil
import os

from AELauncher import *

TEST_DATA_PATH="test123456_temp.npy"
TEST_NETWORK_DIR="testNetwork123456_temp"

class test_train(unittest.TestCase):
    def test(self):
        data = np.random.rand(10,112,112,3) #10 random RGB images between 0 and 1
        for model_type in ["VGG16AE", "convolutional", "perceptron", "sparse_convolutional", "variational_AE"]:
            train_model(model_type, data, data, 2, 5, "ssim", TEST_NETWORK_DIR,
                        save_activations=True, early_stopping=True, sample_preds=4,
                        latent_dim=8, do_augment=True, verbosity=0)
            self.assertTrue(os.path.exists(TEST_NETWORK_DIR))
            shutil.rmtree(TEST_NETWORK_DIR)

class test_test(unittest.TestCase):
    def test(self):
        """
        just check if the function do not fail when called with ssim or mse
        """
        #construct data
        data = np.random.rand(10,112,112,3) #10 random RGB images between 0 and 1
        path = os.path.join(TEST_NETWORK_DIR, "predictions.npy")
        for model_type in ["VGG16AE", "convolutional", "perceptron", "sparse_convolutional", "variational_AE"]:
            model = get_model(model_type, data.shape[1:], 8)
            model.compile("Adam", "mse")
            model.fit(x=data, y=data, batch_size=5, epochs= 2)
            model.save(TEST_NETWORK_DIR)
            del model

            #train model with MSE
            test_model(TEST_NETWORK_DIR, data, "mse", 3, TEST_NETWORK_DIR)
            self.assertTrue(os.path.exists(path))

            #train model with SSIM
            test_model(TEST_NETWORK_DIR, data, "ssim", 3, TEST_NETWORK_DIR)
            self.assertTrue(os.path.exists(path))
            #delete the network folder and data
            shutil.rmtree(TEST_NETWORK_DIR)


class test_latent_search(unittest.TestCase):
    def test(self):
        data = np.random.rand(10,112,112,3)
        for model_type in ["VGG16AE", "convolutional", "perceptron", "sparse_convolutional", "variational_AE"]:
            LD_selection(model_type, data, data, 2, 5, 'mse', TEST_NETWORK_DIR,
                        False, True, 3, [8,16], True, 0)
            shutil.rmtree(TEST_NETWORK_DIR)

# class test_main(unittest.TestCase):
#     def test(self):
#         pass
#TODO test de classe main


if __name__=='__main__':
	unittest.main()
