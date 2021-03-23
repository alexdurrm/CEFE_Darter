import pandas as pd
import numpy as np
import cv2
import argparse
import imageio
from enum import Enum
import os

from FourierAnalysisMaster.pspec import rgb_2_darter


#PREPROCESSING
COL_NORMALIZE="normalization"
COL_STANDARDIZE="standardization"
COL_IMG_TYPE="image_type"
COL_IMG_CHANNEL="channel_image"
COL_IMG_RESIZE_X="image_resize_x"
COL_IMG_RESIZE_Y="image_resize_y"
COL_IMG_PATH="Image_path"


class IMG(Enum):
    RGB="rgb",
    DARTER="darter"
class CHANNEL(Enum):
    GRAY="gray"
    C1=0    #R in RGB
    C2=1    #G in RGB
    C3=2    #B in RGB
    ALL="all"


class Preprocess:
    '''
    A preprocess is a class that consistently preproces images the same way,
    it stores the preprocessed image so that we can fetch it multiple times without reprocess everytime
    it also stores the parameters used to preprocess the image
    '''
    def __init__(self, resize=None, normalize=False, standardize=False, img_type=IMG.RGB, img_channel=CHANNEL.ALL):
        '''
        initialise a process
        '''
        self.normalize = normalize
        self.standardize = standardize
        self.img_type = img_type
        self.img_channel = img_channel
        self.resize = resize

        self.image=None

        resizeX = resize[0] if resize else None
        resizeY = resize[1] if resize else None
        self.df_parameters = pd.DataFrame({COL_NORMALIZE:self.normalize, COL_STANDARDIZE:self.standardize,
            COL_IMG_TYPE:self.img_type.name,
            COL_IMG_CHANNEL:self.img_channel.name,
            COL_IMG_RESIZE_X:resizeX,
            COL_IMG_RESIZE_Y:resizeY}, index=[0])

    def __call__(self, image_path):
        '''
        Take an image as input and update the preprocessed image
        also return the preprocessed image
        '''
        print(image_path)
        image = imageio.imread(image_path)
        #start with a resize if necessary
        if self.resize:
            image = cv2.resize(image, dsize=(self.resize[1], self.resize[0]), interpolation=cv2.INTER_CUBIC)
        #convert the image type
        if self.img_type == IMG.DARTER:                     #darter
            image = rgb_2_darter(image)
            if self.img_channel == CHANNEL.GRAY:
                image = image[:, :, 0] + image[:, :, 1]
            elif self.img_channel == CHANNEL.ALL:
                image = image[:,:,0:2]
            elif self.img_channel == CHANNEL.C3:
                raise ValueError("channel 3 and darter type are not compatible parameters")
            else:
                image = image[:, :, self.img_channel.value]
        elif self.img_type == IMG.RGB:                      #RGB
            if self.img_channel == CHANNEL.GRAY:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif self.img_channel == CHANNEL.ALL:
                image = image
            else:
                image = image[:, :, self.img_channel.value]
        #normalize and standardize
        if self.normalize:
            image = (image - np.mean(image, axis=(0,1))) / np.std(image, axis=(0,1))
        if self.standardize:
            image = (image - np.min(image)) / (np.max(image)-np.min(image))

        self.df_parameters.loc[0, COL_IMG_PATH] = image_path

        self.image = image
        return self.image.copy()

    def get_params(self):
        return self.df_parameters.copy()

    def get_image(self):
        print(self.image.dtype)
        return self.image.copy()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path of the image file to open")
    args = parser.parse_args()
    image = imageio.imread(args.image)

    pr = Preprocess()
    processed_img = pr(image)
    assert (processed_img==pr.get_image()).all(), "1) should be equal"
    assert (image==pr.get_image()).all(), "2) should be equal if no parameters"
    print(pr.get_params())

    pr2 = Preprocess(resize=(120, 150), normalize=True, standardize=True, img_type=IMG.DARTER, img_channel=CHANNEL.GRAY)
    processed_img = pr2(image)
    assert (processed_img==pr2.get_image()).all(), "3) should be equal"
    assert not np.array_equiv(image, pr2.get_image()), "4) should not be equal if parameters"
    print(pr2.get_params())
