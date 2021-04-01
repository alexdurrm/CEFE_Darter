import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import imageio
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", help="path of the file to open")
args = parser.parse_args()

path = args.input
files = os.listdir(path)
n_files = len(files)
print(n_files)


for f in files:
    if os.path.isdir(f):
        continue
       
    img_path = os.path.join(path, f)
    image = Image.open(img_path)
    
    # Setting the points for the cropped image 
    left = 1700 #to adapt: fish are not centred the same way on the different images
    right = left + 3000
    top = 1700  #to adapt: fish are not centred the same way on the different images
    bottom = top + 1000
    
    # Cropped image of above dimension 
    # (It will not change the original image) 
    im1 = image.crop((left, top, right, bottom)) 
  
    # Display the cropped image
    # plt.imshow(im1) 
    # plt.show()
    
    output_dir = os.path.join(path, "Cropped")
    if not(os.path.exists(output_dir) and os.path.isdir(output_dir)):
        os.makedirs(output_dir)
    im1.save(os.path.join(output_dir, f), quality=100)