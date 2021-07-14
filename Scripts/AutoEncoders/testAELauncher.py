import numpy as np
import unittest
import shutil
import os

from AELauncher import *
from CommonAE import get_loss_from_name

TEST_DATA_PATH="test123456_temp.npy"
TEST_NETWORK_DIR="testNetwork123456_temp"

class test_augmentation(unittest.TestCase):
    def test(self):
        """
        check if the function do not fail when called 
        and if augmentation produces a different result for input
        """
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
        #check result is always different
        boo=get_augmentation((60,50))
        x = boo(a).numpy()
        y = boo(a).numpy()
        self.assertTrue((x!=y).any())


class test_train(unittest.TestCase):
    def test(self):
        #train_model(model, train, test, loss_func, output_dir, epochs, batch_size, callbacks, augmenter=None, verbosity=0)
        data = np.random.rand(10,112,112,3) #10 random RGB images between 0 and 1
        model = Convolutional(8, data.shape[1:])
        loss = get_loss_from_name("ssim")
        for model_type in ["convolutional", "perceptron", "sparse_convolutional", "variational_AE"]: #  "VGG16AE", TODO:debug VGGAE
            callbacks = get_callbacks(model_type, data, TEST_NETWORK_DIR, True, True, 3)
            model = get_model(model_type, data.shape[1:], 8)
            train_model(model, data, data, loss, TEST_NETWORK_DIR, 2, 5, callbacks, augmenter=None)
            self.assertTrue(os.path.exists(TEST_NETWORK_DIR))
            shutil.rmtree(TEST_NETWORK_DIR)

class test_test(unittest.TestCase):
    def test(self):
        """
        just check if the function do not fail when called with ssim or mse
        """
        #test_model(model_path, test, loss, output_dir, sample_preds, verbosity=0)
        #construct data
        data = np.random.rand(10,112,112,3) #10 random RGB images between 0 and 1
        path = os.path.join(TEST_NETWORK_DIR, "predictions.npy")
        for model_type in ["convolutional"]:#, "perceptron", "sparse_convolutional", "variational_AE", "VGG16AE"]:
            model = get_model(model_type, data.shape[1:], 8)
            model.compile("Adam", "mse")
            model.fit(x=data, y=data, batch_size=5, epochs= 2)
            model.save(TEST_NETWORK_DIR)
            del model

            #train model with MSE
            loss = get_loss_from_name("mse")
            test_model(TEST_NETWORK_DIR, data, loss, 3, TEST_NETWORK_DIR)
            self.assertTrue(os.path.exists(path))

            #train model with SSIM
            loss = get_loss_from_name("ssim")
            test_model(TEST_NETWORK_DIR, data, loss, 3, TEST_NETWORK_DIR)
            self.assertTrue(os.path.exists(path))
            #delete the network folder and data
            shutil.rmtree(TEST_NETWORK_DIR)

        
class test_latent_search(unittest.TestCase):
    def test(self):
        # LD_selection(model_type, train, test, epochs, batch_size, list_LD, loss, dir_results, sample_preds, augmenter=None, verbosity=0)
        data = np.random.rand(10,112,112,3)
        loss = get_loss_from_name("mse")
        aug = get_augmentation(data.shape[1:])
        for model_type in ["convolutional", "perceptron", "sparse_convolutional", "variational_AE"]:#, , "VGG16AE"]:
            LD_selection(model_type, data, data, 2, 5, [2,4], loss, TEST_NETWORK_DIR, True, True, 3, augmenter=aug)
            shutil.rmtree(TEST_NETWORK_DIR)

# class test_main(unittest.TestCase):
#     def test(self):
#         pass
#TODO test de classe main


if __name__=='__main__':
	unittest.main()
