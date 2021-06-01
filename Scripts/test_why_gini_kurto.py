import matplotlib.pyplot as plt
import numpy as np
import argparse
from glob import glob
import imageio

from Metrics.ImageMetrics import get_gini
from scipy.stats import kurtosis


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("glob", help="glob of images to test to test")
	args = parser.parse_args()

	for path in glob(args.glob):
		image = imageio.imread(path)
		data = image / image.max() #normalizes data in range 0 - 255
		data = 255 * data
		img = data.astype(np.uint8)
		gini = get_gini(data, visu=False)
		kurto = kurtosis(image, axis=None)
		plt.hist(image.flatten(), bins=50)
		plt.title("gini: {}, kurto: {}".format(round(gini, 3), round(kurto, 3)))
		plt.show()
