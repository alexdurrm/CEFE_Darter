from pspec import get_pspec
import argparse
import os
import matplotlib.pyplot as plt
import imageio
import numpy as np
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="the path of the dir to open")
args = parser.parse_args()
path = args.input_dir


types_allowed=(".png", ".jpg", ".tiff")
resize=True
sample_dim=512

for element in os.listdir(path):
    element = os.path.join(path, element)

    if os.path.isfile(element) and element.endswith(types_allowed):
        print(element)
        image = imageio.imread(element)
        if resize:
            resize_ratio = sample_dim/np.min(image.shape[0:2])
            new_x, new_y = (round(resize_ratio*dim) for dim in image.shape[0:2])
            image = cv2.resize(image, dsize=(new_y, new_x),
                interpolation=cv2.INTER_CUBIC)  #cv2 (x,y) are numpy (y,x)
        else: assert image.shape[0]>=sample_dim<=image.shape[1], "sample dim should be less or equal than image shapes"
        # Define a sliding square window to iterate on the image
        stride = int(sample_dim/2)
        slopes = []
        tot_samples = 0
        for start_x in range(0, image.shape[0]-sample_dim+1, stride):
            for start_y in range(0, image.shape[1]-sample_dim+1, stride):
                tot_samples+=1 
                sample = image[start_x: start_x+sample_dim, start_y: start_y + sample_dim]
                x, y = get_pspec(sample, return_bins=True)
                slopes.append(y)
        mean_slopes = np.mean(slopes, axis=0)
        plt.loglog(x, mean_slopes)
plt.show()
