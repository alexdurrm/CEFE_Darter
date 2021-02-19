from skimage.feature import local_binary_pattern
import argparse
import imageio
import os
import matplotlib.pyplot as plt
from collections import Counter
from numpy import save as save_numpy
from pspec import rgb_2_darter
import cv2


def get_LBP(image, P, R, visu=False):
    '''
    calculates the Local Binary Pattern of a given image
    P is the number of neighbors points to use
    R is the radius of the circle around the central pixel
    visu is to visualise the result
    return the path to the saved image
    '''
    image = rgb_2_darter(image)
    image = image[:, :, 0] + image[:, :, 1]
    lbp_image = local_binary_pattern(image, P, R)

    if visu:
        fig, (ax0, ax1, ax2) = plt.subplots(figsize=(6, 12), nrows=3)
        ax0.imshow(image, cmap='gray')
        ax0.set_title("original image")

        bar = ax1.imshow(lbp_image, cmap='gray')
        fig.colorbar(bar, ax=ax1, orientation="vertical")
        ax1.set_title("LBP with params P={} and R={}".format(P, R))

        ax2.hist(lbp_image.flatten(), bins=P)
        ax2.set_title("lbp values histogram")
        plt.show()
    return lbp_image


def do_LBP_metric(img_path, P, R, verbosity=1, resize=None):
    '''
    calls get_LBP and saves its result
    return a dict of informations about the LBP obtained
    '''
    image = imageio.imread(img_path)
    if resize:
        image = cv2.resize(image, dsize=resize, interpolation=cv2.INTER_CUBIC)
    LBP_image = get_LBP(image, P, R, verbosity>=2)

    head, image_name = os.path.split(img_path)
    image_name, ext = os.path.splitext(image_name)
    output_dir = os.path.join(head, "LocalBinaryPattern_P{}_R{}".format(P,R))

    if not( os.path.exists(output_dir) and os.path.isdir(output_dir) ):
        os.mkdir(output_dir)
    output_filepath = os.path.join(output_dir, image_name+"_LBP.npy")
    save_numpy(output_filepath, LBP_image)
    if verbosity>=1: print("LBP image saved at {}".format(output_filepath))
    return {"path_LBP": output_filepath, "radius_LBP":R , "points_LBP":P, "resize_LBP":bool(resize)}


# def do_network_deep_features(img_path, model):
    # '''
    # get an image path and a DNN model
    # calculate the deep features of the model when infering on the image
    # returns a dict containing the path of the deep features
    # '''
    # return {}




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path of the image file to open")
    parser.add_argument("-r", "--radius", default=1, type=int,
                        help="radius of the circle. Default 1.")
    parser.add_argument("-p", "--points", default=8, type=int,
                        help="Number of points on the circle. Default 8.")
    args = parser.parse_args()
    image = imageio.imread(os.path.abspath(args.image))
    get_LBP(image, args.points, args.radius, True)
