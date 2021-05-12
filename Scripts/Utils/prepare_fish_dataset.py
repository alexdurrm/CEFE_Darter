import argparse
from glob import glob
import imageio
import matplotlib.pyplot as plt
import cv2
import numpy as np

def rgb_2_darter(image):
	"""
	transfer the given image from the RGB domain space to the darter domain space
	"""
	im_out = np.zeros([image.shape[0], image.shape[1], 2], dtype = np.float32)

	im_out[:, :, 1] = (140.7718694130528 +
		0.021721843447502408  * image[:, :, 0] +
		0.6777093385296341    * image[:, :, 1] +
		0.2718422677618606    * image[:, :, 2] +
		1.831294521246718E-8  * image[:, :, 0] * image[:, :, 1] +
		3.356941424659517E-7  * image[:, :, 0] * image[:, :, 2] +
		-1.181401963067949E-8 * image[:, :, 1] * image[:, :, 2])
	im_out[:, :, 0] = (329.4869869234302 +
		0.5254935133632187    * image[:, :, 0] +
		0.3540642397052902    * image[:, :, 1] +
		0.0907634883372674    * image[:, :, 2] +
		9.245344681241058E-7  * image[:, :, 0] * image[:, :, 1] +
		-6.975682782165032E-7 * image[:, :, 0] * image[:, :, 2] +
		5.828585657562557E-8  * image[:, :, 1] * image[:, :, 2])
	return im_out


def prepare_fish_img(image, output_shape):
    if color_channels==1:
        image = rgb_2_darter(image)
        image = image[..., 0]+image[..., 1]
        image = image[..., np.newaxis]
    elif color_channels==2:
        image = rgb_2_darter(image)
    set_img = set_img.astype(np.float32)
    mean = np.mean(set_img)
    std = np.std(set_img)
    set_img = (set_img - mean) / std
    mini = np.min(set_img)
    maxi = np.max(set_img)
    set_img = ((set_img - mini) / (maxi - mini))
    print("\n")
    assert np.max(set_img)<=1 and np.min(set_img)>=0, "bad normalization,{} {} instead of {} {}".format(np.min(set_img), np.max(set_img), mini, maxi)

    #resize the image to the output_shape by adding a padding

def get_image_background_val(img, visu=False):
    #blur the image with a gaussian filter
    blurred_img = cv2.blur(img, (5,5),0)
    #convert to grayscale
    im_gray = cv.cvtColor(im_color, cv.COLOR_BGR2GRAY)
    #find the value of the background
    counts, bins = np.histogram(im_gray, bins=255)

    if visu:
        fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, squeeze=True)
        axs[0].imshow(img)
        axs[0].set_title("original")
        axs[1].imshow(blurred_img)
        axs[1].set_title("blurred")
        axs[2].imshow(im_gray)
        axs[2].set_title("gray blurred")
        axs[3].hist(bins[:-1], bins, weights=counts)
        axs[3].set_title("bins gray blurred")
        plt.show()
        plt.close()

def get_fishes_masks(array_fishes, visu=False):
    masks = np.empty_like(array_fishes)
    for i, fish in enumerate(array_fishes):
        threshold = get_image_background_val(fish)
        #mask given a threshold
        mask = 
        masks[i] = mask
        if visu:
            fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, squeeze=True)
            axs[0].imshow(fish)
            axs[0].set_title("original")
            axs[1].imshow(blurred_img)
            axs[1].set_title("blurred")
            axs[2].imshow(im_gray)
            axs[2].set_title("gray blurred")
            axs[3].hist(bins[:-1], bins, weights=counts)
            axs[3].set_title("bins gray blurred")
            plt.show()
            plt.close()



if __name__=='__main__':
    CHANNELS=3
    VERBOSE=0
    CROP_SHAPE=(128, 128)

    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="category name given to recognize the files used (ex: blennioides, careuleum, habitats, fish ...)")
	parser.add_argument("glob", help="glob of the images to put in the dataset")
	parser.add_argument("output", help="output directory where to save the dataset")
    parser.add_argument("-c", "--channels", type=int, choices=[1,2,3], default=CHANNELS,
		help="Number of channels to train on: 1 will be in darter gray luminance space,2 will be darter visualization, 3 will be in rgb, default is {}".format(CHANNELS))
	parser.add_argument("-s", "--shape", type=int, nargs=2, default=CROP_SHAPE, help="shape of the final image used as a network input/output, default is {}".format(CROP_SHAPE))
    parser.add_argument("-v", "--verbose", default=VERBOSE, type=int, choices=[0,1,2], help="Set the level of visualization, default: {}".format(VERBOSE))
	args = parser.parse_args()

    output_shape = (*args.shape, args.channels)
    path_images = glob(args.glob)
    image_list = np.zeros((len(path_images, output_shape), dtype=np.float32))
    for i, path in enumerate(path_images):
        if args.verbose>=1: print(path)
        img = imageio.imread(path)
        image_list[i] = prepare_fish_img(img, output_shape)
    masks = get_fishes_masks(image_list, args.verbose>=1)
