import numpy as np
import pandas as pd
import imageio
import os
import argparse
from functools import partial
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19

from config import *
from Preprocess import *
from DeepFeatureMetrics import *
from FFTMetrics import *
from GaborMetrics import *
from GLCMMetrics import *
from HOGMetrics import *
from LBPMetrics import *
from ScalarMetrics import *

###############################################################
### File used to calculate metrics and save them in a csv file
###############################################################

def get_files(main_path, max_depth, types_allowed, ignored_folders, only_endnodes, visu=False):
    '''
    initiate a dataframe with minimum values in it
    max_depth is the depth of the search: 0 is for a file, 1 is to search one directory, 2 is dir and subdirs...
    only_endnodes is whether or not to ignore files that are not located at the endnode of the directory tree
    ignored_folders is a list of folders to ignore
    '''
    if main_path.endswith(".csv"):  #load the csv file if there is one
        return pd.read_csv(main_path, index_col=0)
    assert max_depth>=0, "You should not give a depth lower than 0"
    data = pd.DataFrame(columns=LIST_COLUMNS_IMG)
    to_visit=[(main_path, 0)]
    while to_visit:
        path, depth = to_visit.pop()
        if os.path.isdir(path) and depth<max_depth and (not [i for i in ignored_folders if i in path]):
            to_visit += [(os.path.join(path, next), depth+1) for next in os.listdir(path)]
        elif os.path.isfile(path) and path.endswith(types_allowed):
            if depth==max_depth or not only_endnodes:
                if visu: print("ADDING {}".format(path))
                dict_info = load_info_from_filepath(path)
                data.loc[os.path.abspath(path), [*dict_info.keys()]] = [*dict_info.values()]
    return data


def load_info_from_filepath(file_path):
    '''
    Given an absolute file path
    returns the informations about the file in the form of a dict
    '''
    long_head, filename = os.path.split(file_path)
    head, directory = os.path.split(long_head)
    _, ext = os.path.splitext(filename)
    image = imageio.imread(file_path)
    dict_info={COL_IMG_PATH:file_path, COL_FILENAME:filename, COL_DIRECTORY:directory,
        COL_IMG_HEIGHT:image.shape[0], COL_IMG_WIDTH:image.shape[1], COL_IMG_EXT:ext}

    #if the directory contains stylized fishes
    if [p for p in DIR_STYLIZED_FISHES if p == directory]:
        crossing, color_ctrl, _, tvloss, *_  = filename.split("_")
        middle, fish_n = crossing.rsplit("x", maxsplit=1)
        dict_info[COL_FISH_NUMBER] = fish_n[4:] #4 to remove FISH
        dict_info[COL_HABITAT] = middle
        dict_info[COL_COLOR_CONTROL] = color_ctrl[12:]
        dict_info[COL_TV_LOSS] = tvloss[6:]
        dict_info[COL_LAYERS] = directory
        dict_info[COL_TYPE] = FILE_TYPE.STYLIZED_FISH.value
    #if the directory contains original fishes
    elif [p for p in DIR_ORIGINAL_FISHES if p == directory]:
        fish_n, _, original_size, specie, *_, end = filename.split("_")
        dict_info[COL_SPECIES] = specie
        dict_info[COL_FISH_NUMBER] = end.split('.')[0][1:]
        dict_info[COL_FISH_SEX] = end[0]
        dict_info[COL_TYPE] = FILE_TYPE.ORIG_FISH.value
    #if the directory contains original habitats
    elif [p for p in DIR_ORIGINAL_HABITATS if p == directory]:
        middle, *_ = filename.split('_')
        dict_info[COL_HABITAT] = middle
        dict_info[COL_TYPE] = FILE_TYPE.HABITAT.value
    #if the folder is the samuel folder
    elif [p for p in DIR_SAMUEL if p == directory]:
        specie, *_, fish = filename.split("_")
        dict_info[COL_FISH_NUMBER] = fish.split('.')[0][1:]
        dict_info[COL_FISH_SEX] = fish.split('.')[0][0]
        dict_info[COL_SPECIES] = specie
        dict_info[COL_TYPE] = FILE_TYPE.ORIG_FISH.value
    else:
        dict_info[COL_TYPE] = FILE_TYPE.ELSE.value
    return dict_info


def main(data_path, verbosity=1):
    '''
    create a csv file or load an existing one
    update the values in this csv by running new metrics
    save and return the data
    '''
    fft_range=(10, 110) #110 pour des fenetres 200x200!!!
    # GLCM_DISTANCES=[1]
    gabor_angles=[0, 45, 90, 135]
    gabor_freq=[0.2, 0.4, 0.8]
    resize = (1536, 512)

    #initialize preprocessor used
    process_darter_gray = Preprocess(resize=resize, normalize=False, standardize=False, img_type=IMG.DARTER, img_channel=CHANNEL.GRAY)
    process_darter_all = Preprocess(resize=resize, normalize=False, standardize=False, img_type=IMG.DARTER, img_channel=CHANNEL.ALL)
    process_RGB_all = Preprocess(resize=resize, normalize=True, standardize=False, img_type=IMG.RGB, img_channel=CHANNEL.ALL)

    #initialize metrics used
    vgg16_model = DeepFeatureMetrics( VGG16(weights='imagenet', include_top=False), resize, process_RGB_all, os.path.join(DIR_RESULTS, CSV_DEEP_FEATURES))
    vgg19_model = DeepFeatureMetrics( VGG19(weights='imagenet', include_top=False), resize, process_RGB_all, os.path.join(DIR_RESULTS, CSV_DEEP_FEATURES))
    fft_slope = FFTSlopes(fft_range, 512, process_darter_gray, os.path.join(DIR_RESULTS, CSV_FFT_SLOPE))
    fft_bins = FFT_bins(fft_range, 512, process_darter_gray, os.path.join(DIR_RESULTS, CSV_FFT_BINS))
    gabor_metric = GaborMetrics(gabor_angles, gabor_freq, process_darter_gray, os.path.join(DIR_RESULTS, CSV_GABOR))
    glcm_metric = HaralickMetrics([2,4], gabor_angles, process_darter_gray, os.path.join(DIR_RESULTS, CSV_HARALICK))
    phog_metric = PHOGMetrics(40, 2, process_darter_gray, os.path.join(DIR_RESULTS, CSV_PHOG))
    lbp_metric = LBPHistMetrics([8, 16], [2,4], 100, process_darter_gray, os.path.join(DIR_RESULTS, CSV_LBP))
    best_lbp_metric = BestLBPMetrics([8, 16], [2,4], 100, process_darter_gray, os.path.join(DIR_RESULTS, CSV_BEST_LBP))
    stats_metric = StatMetrics(process_darter_gray, os.path.join(DIR_RESULTS, CSV_STATS_METRICS))
    color_ratio = ColorRatioMetrics(process_darter_all, os.path.join(DIR_RESULTS, CSV_COLOR_RATIO))

    # call the metrics
    vgg16_model.metric_from_csv(data_path)
    vgg16_model.save()

    vgg19_model.load()
    vgg19_model.metric_from_csv(data_path)
    vgg19_model.save()

    fft_slope.metric_from_csv(data_path)
    fft_slope.save()

    fft_bins.metric_from_csv(data_path)
    fft_bins.save()

    gabor_metric.metric_from_csv(data_path)
    gabor_metric.save()

    glcm_metric.metric_from_csv(data_path)
    glcm_metric.save()

    phog_metric.metric_from_csv(data_path)
    phog_metric.save()

    lbp_metric.metric_from_csv(data_path)
    lbp_metric.save()

    best_lbp_metric.metric_from_csv(data_path)
    best_lbp_metric.save()

    stats_metric.metric_from_csv(data_path)
    stats_metric.save()

    color_ratio.metric_from_csv(data_path)
    color_ratio.save()

    print("DONE")


def group_files_by_experiments(df_files):
    '''
    returns a dataframe containing for each image transfer output,
    a link between the path of the style image and the path of the content image
    '''
    experiments = pd.DataFrame(columns=LIST_COLUMNS_EXP)
    fishes = df_files[df_files[COL_TYPE]==FILE_TYPE.ORIG_FISH.value][[COL_IMG_PATH, COL_FISH_NUMBER, COL_FISH_SEX, COL_SPECIES]]
    habitat = df_files[df_files[COL_TYPE]==FILE_TYPE.HABITAT.value][[COL_IMG_PATH, COL_HABITAT]]

    output_net = df_files[df_files[COL_TYPE]==FILE_TYPE.STYLIZED_FISH.value]
    if not output_net.empty:
        output_net = output_net.drop_duplicates(subset=[COL_FISH_NUMBER, COL_FISH_SEX, COL_SPECIES, COL_HABITAT], ignore_index=True)

        experiments = output_net.merge(fishes, on=[COL_FISH_NUMBER], how="inner", suffixes=(None, "_fish"))
        experiments = experiments.merge(habitat, on=[COL_HABITAT], how="inner", suffixes=(None, "_hab"))

        fish_path = COL_IMG_PATH+"_fish"
        habitat_path = COL_IMG_PATH+"_hab"
        experiments = experiments[[fish_path, habitat_path]].reset_index().rename(columns={'index':COL_EXP_ID, fish_path:COL_CONTENT_EXP_PATH, habitat_path:COL_STYLE_EXP_PATH})
    return experiments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="the path of the main directory containing the subdirs or the path of the csv to use.")
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=1,
                    help="increase output verbosity, default is 1.")
    parser.add_argument("-d", "--depth", type=int, choices=[0, 1, 2, 3], default=2,
                    help="depth of the path searched, 0 is image, 1 is folder, 2:subfolders... not used if input is a csv file")
    args = parser.parse_args()

    main_file = os.path.abspath(args.input)

    #if the directory to store results do not exist create it
    if not os.path.exists(DIR_RESULTS):
        os.makedirs(DIR_RESULTS)

    #gather informations about the files and save them
    data = get_files(main_file, args.depth, (".jpg",".png",".tiff"), [], only_endnodes=True, visu=False)
    data_path = os.path.join(DIR_RESULTS, CSV_IMAGE)
    data.to_csv(data_path, index=False)

    #group files by experiments and save experiements
    exp = group_files_by_experiments(data)
    exp_path = os.path.join(DIR_RESULTS, CSV_EXPERIMENTS)
    exp.to_csv(exp_path, index=False)

    #execute the metrics
    #main(data_path)
