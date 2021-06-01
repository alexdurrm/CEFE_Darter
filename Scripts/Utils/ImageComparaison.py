import numpy as np

def getEuclideanDist(img1, img2):
	return np.sqrt(np.sum(np.square(img1-img2)))
