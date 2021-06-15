import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
data = np.load(input("chemin numpy:"))
dir_out = input("dossier de sortie:")
for idx, img in enumerate(data):
    img = (img*255).astype("uint8")
    # plt.imshow(img)
    # plt.show()
    cv2.imwrite(os.path.join(dir_out, "img_hab_{}.jpg".format(idx)), img[:,:,::-1])
