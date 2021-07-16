import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="take images and return a numpy")
parser.add_argument("input")
parser.add_argument("output")
args = parser.parse_args()
data = np.load(args.input)

for idx, img in enumerate(data):
    img = (img*255).astype("uint8")
    # plt.imshow(img)
    # plt.show()
    cv2.imwrite(os.path.join(args.output, "img_hab_{}.jpg".format(idx)), img[:,:,::-1]) # -1 because cv2 channels are BGR
