import argparse
import numpy as np
import os
from glob import glob
import imageio

parser = argparse.ArgumentParser(description="take images and return a numpy")
parser.add_argument("glob_input")
parser.add_argument("file_out")
args = parser.parse_args()

paths = glob(args.glob_input)
file_out = args.file_out

shape = imageio.imread(paths[0]).shape
data = np.empty(shape=(len(paths), *shape), dtype='float32')

for idx, path in enumerate(paths):
    data[idx] = imageio.imread(path)/255.0
np.save(file_out, data)
