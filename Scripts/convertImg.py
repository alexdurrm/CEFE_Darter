import rawpy as rp
import numpy as np
import imageio
import os

def openCR2(path):
    ext = os.path.splitext(path)[1]
    assert ext==".CR2", "Wrong image format, expected \'.CR2\', got {}".format(ext)
	raw = rp.imread(path)
	rgb = raw.postprocess(use_camera_wb=True)
	return rgb

def openNpy(path):
    ext = os.path.splitext(path)[1]
    assert ext==".npy", "Wrong image format, expected \'.npy\', got {}".format(ext)
    return np.load(path)

def openImg(path):
    ext = os.path.splitext(path)[1]
    assert ext in [".jpg", ".png", ".tiff", ".tif"], "Wrong image format, expected \'.jpg\', \'.png\', \'.tiff\', or \'.tif\', got {}".format(ext)
    return imageio.imread(path)


def save_images(data, path, extension):
    """
    given a data of type numpy array will save
    """
    assert extension in [".jpg", ".png", ".tiff", ".npy"], "wrong extention {}".format(extension)
    #format the path
    if not os.path.splitext(path)[1]:
        path+=extension
    else:
        assert extention == ext, "extention and path parameters are contradictory formats: {} and {}".format(extention, ext)
        path=os.path.splitext(path)[0]+extention
    #
