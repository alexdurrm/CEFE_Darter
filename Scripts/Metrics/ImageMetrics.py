from collections import Counter

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from skimage.filters import gabor, gabor_kernel
from skimage.feature import greycomatrix, greycoprops
from skimage.feature import local_binary_pattern
from scipy.stats import kurtosis, skew, entropy
from skimage.metrics import structural_similarity as ssim


def get_L0(vector):
	return 1- np.count_nonzero(vector)/vector.size

def get_SSIM(img1, img2):
	return ssim(img1, img2, multichannel=(img1.ndim==3))

def get_gini(array, visu=False):
	'''
	Calculate the Gini coefficient of a numpy array.
	Author: Olivia Guest (oliviaguest)
	Original publication of this code available at https://github.com/oliviaguest/gini/blob/master/gini.py
	'''
	# All values are treated equally, arrays must be 1d:
	array = array.flatten()
	if np.amin(array) < 0:
		# Values cannot be negative:
		array -= np.amin(array)
	# Values cannot be 0:
	array += 0.0000001
	# Values must be sorted:
	array = np.sort(array)
	# Index per array element:
	index = np.arange(1,array.shape[0]+1)
	# Number of array elements:
	n = array.shape[0]
	# Gini coefficient:
	gini_val = (np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))
	if visu: print("GINI: {}".format(gini_val))
	return gini_val


def get_color_ratio(image, visu=False):
	'''
	return the color ratio slope between the two color channel
	'''
	assert image.ndim==3, "Image should have 3 dims, here {}".format(image.ndim)
	assert image.shape[2]==2, "Image should have two channels, here image is shape{}".format(image.shape)

	size_sample = np.min([image.size, 1000])
	selection = np.random.choice(np.arange(size_sample), size=size_sample, replace=False)
	X = image[..., 0].flatten()[selection]
	Y = image[..., 1].flatten()[selection]

	slope, b = np.polyfit(X, Y, 1)
	print(slope, b)
	if visu:
		x=np.arange(0, np.max(X), 16)
		y=slope*x+b
		plt.plot(x, y)
		plt.scatter(X, Y)
		plt.show()
	return slope


def get_statistical_features(array, axis=None, visu=False):
	'''
	get an array and return the statistical features like
	mean value, standard deviation, skewness, kurtosis, and entropy
	(calculated on flattened image)
	'''
	mean = np.mean(array, axis=axis)
	std = np.std(array, axis=axis)
	skewness = skew(array, axis=axis)
	kurto = kurtosis(array, axis=axis)
	entro = entropy(array, axis=axis)
	return (mean, std, skewness, kurto, entro)


def get_gabor_filters(image, angles, frequencies, visu=False):
	'''
	produces a set of gabor filters and
	angles is the angles of the gabor filters, given in degrees
	return a map of the mean activation of each gabor filter
	'''
	assert image.ndim==2 , "Should be a 2D array"

	activation_map = np.empty(shape=[len(angles), len(frequencies)])
	rad_angles = np.radians(angles)
	for t, theta in enumerate(rad_angles):
		for f, freq in enumerate(frequencies):
			real, _ = gabor(image, freq, theta)
			if visu:
				plt.imshow(real, cmap="gray")
				plt.title("gabor theta:{}  frequency:{}".format(t, f))
				plt.colorbar()
				plt.show()
			activation_map[t, f] = np.mean(real)
	if visu:
		ax = sns.heatmap(activation_map, annot=True, center=1, xticklabels=frequencies, yticklabels=angles)
		plt.show()
	return activation_map


def get_Haralick_descriptors(image, distances, angles):
	'''
	get an image and calculates its grey level co-occurence matrix
	calculate it along different angles and distances
	returns a few characteristics about this GLCM
	'''
	assert image.ndim==2, "Image should be 2D"

	glcm = greycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False)

	mean = np.mean(glcm, axis=(0,1))
	var = np.var(glcm, axis=(0,1))
	corr = greycoprops(glcm, 'correlation')

	contrast = greycoprops(glcm, 'contrast')
	dissimil = greycoprops(glcm, 'dissimilarity')
	homo = greycoprops(glcm, 'homogeneity')

	asm = greycoprops(glcm, 'ASM')
	energy = greycoprops(glcm, 'energy')

	maxi = np.max(glcm, axis=(0,1))
	entro = entropy(glcm, axis=(0,1))

	return (mean, var, corr, contrast, dissimil, homo, asm, energy, maxi, entro)

def get_most_common_lbp(image, point, radius, n_best):
	"""
	return the n most common binary patterns
	"""
	assert image.shape[-1]==1, "image given for LBP should have only one channel"
	lbp_image = local_binary_pattern(image[:,:,0], point, radius)
	cnt = Counter(lbp_image.flatten())
	best_lbp = cnt.most_common(n_best)
	return best_lbp

def GramMatrix(img):
	img = np.reshape(img, (-1, img.shape[-1]))
	gram = img.T.dot(img)
	return gram
