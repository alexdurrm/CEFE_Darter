#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This file contains methods to be used for the computation of the computation of
the Fourier power spectrum of images.
'''

import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt

__author__ = 'Samuel Hulse'
__email__ = 'hsamuel1@umbc.edu'

def get_Fourier_slope(image,
	bin_range=(10,110),
	kaiser=True,
	n_bins=20):
	"""
	given an image returns its fourier slope
	"""
	x, y = get_pspec(image, bin_range, kaiser, n_bins)
	slope = get_slope(x, y)
	return slope

def get_pspec(image,
	bin_range=(10, 110),
	kaiser=True,
	n_bins=20):
	if kaiser:
		image = kaiser2D(image, 2)

	pspec = imfft(image)
	x, y = bin_pspec(pspec, n_bins, bin_range)
	return (x, y)

def bin_pspec(data, n_bins, bin_range):
	bins = np.logspace(np.log(bin_range[0]),
		np.log(bin_range[1]),
		n_bins,
		base = np.e)
	x = np.linspace(1, len(data), len(data))

	bins_x = []
	bins_y = []

	for i in range(len(bins[0:-1])):
		bin_x = np.mean(x[np.logical_and(
			x >= bins[i],
			x < bins[i+1])])
		bins_x.append(bin_x)

		bin_y = np.mean(data[np.logical_and(
			x >= bins[i],
			x < bins[i+1])])
		bins_y.append(bin_y)

	return (bins_x, bins_y)

def imfft(image):
	imfft = fftshift(fft2(image))
	impfft = np.absolute(imfft) ** 2
	pspec = rotavg(impfft)

	return pspec

def get_slope(x, y):
	"""returns a coefficient for the given slope passed to log scale"""
	x, y = np.log((x, y))
	slope = np.polyfit(x, y, 1)[0]

	return slope

def rotavg(data):
	center = np.divide(data.shape, 2)
	x_sample = np.linspace(0, data.shape[0], data.shape[0])
	y_sample = np.linspace(0, data.shape[1], data.shape[1])
	x, y = np.meshgrid(y_sample, x_sample)

	x = np.absolute(x - center[1])
	y = np.absolute(y - center[0])
	dist_matrix = np.sqrt(x**2 + y**2)

	max_dist = np.sqrt(np.sum(np.square(center)))
	n_bins = int(np.ceil(max_dist))
	bins = np.linspace(0, max_dist, n_bins)

	radialprofile = np.zeros(n_bins - 1)

	for i in range(len(bins[0:-1])):
		filter = np.zeros(data.shape)
		filter[np.logical_and(
			dist_matrix >= bins[i],
			dist_matrix < bins[i+1])] = 1

		radialprofile[i] = np.sum(filter * data) / np.sum(filter)

	return radialprofile

def kaiser2D(img, alpha):
	#Calculate the 2D Kaiser-Bessel window as the outer product of the 1D window with itself
	kaiser = np.kaiser(img.shape[0], alpha*np.pi)
	A = np.outer(kaiser, kaiser)

	#Normalize by the sum of squared weights
	w = np.sum(A*A)
	A = A / w

	#Apply the window by performing elementwise multiplication
	imout = np.multiply(img, A)
	return imout
