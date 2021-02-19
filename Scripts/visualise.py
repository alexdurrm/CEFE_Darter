import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns
import os
import argparse
from collections import Counter
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("function", help="name of the function to use", choices=["fourier_per_folder", "transfer_effect_slope", "publication_plot", "LBP_hist_per_folder"])
parser.add_argument("input_file", help="the path of the csv to open")
args = parser.parse_args()


def fourier_per_folder(filepath):
    '''
    used to plot the violin graph the fourier slopes categorized per folder
    '''
    data = pd.read_csv(filepath, sep=',')
    matplotlib.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    matplotlib.rcParams['axes.linewidth'] = 2
    sns.set_palette(sns.color_palette(FLAT_UI))

    #ax = sns.catplot(x=COL_DIRECTORY, y=COL_F_SLOPE, data=data.where(data.folder == "caeruleum"),
    ax = sns.catplot(x=COL_DIRECTORY, y=COL_F_SLOPE, data=data,
                    #col=COL_COLOR_CONTROL,
                    #hue=COL_FISH_SEX,
                    #split=True,
                    kind="violin")

    ax.set_ylabels('Slope of Fourier Power Spectrum ')
    ax.set_yticklabels(fontstyle='italic')
    plt.xticks(rotation=45)
    plt.title("mean Fourrier slopes per folder")
    plt.show()


def transfer_effect_slope(filepath):
    '''
    used to compare the slopes of fish, environment, and different modified fishes
    to measure the effects after a style transfer
    '''
    data = pd.read_csv(filepath, sep=',')

    #plot a naive but complete scatter of the fourier slopes
    plt.scatter(data[COL_DIRECTORY], data[COL_F_SLOPE])
    plt.title("mean Fourrier slope of each images per folder")
    plt.show()

    # split the data in databases for the middle, the fishes, the network predictions
    # and rename the column name F_slope to a more appropriate value
    slope_names=["HABITAT_images","FISH_images","All_layers",
                "Layers_1and2","Layers_3to5"]

    middles = data.loc[data[COL_DIRECTORY] == slope_names[0]][[COL_HABITAT, COL_F_SLOPE]]
    middles = middles.rename(columns={COL_F_SLOPE: slope_names[0]})

    fishes = data.loc[data[COL_DIRECTORY] == slope_names[1]][[COL_FISH_NUMBER, COL_F_SLOPE]]
    fishes = fishes.rename(columns={COL_F_SLOPE: slope_names[1]})

    all_layers = data.loc[data[COL_DIRECTORY] == slope_names[2]][[COL_HABITAT, COL_FISH_NUMBER, COL_F_SLOPE,COL_COLOR_CONTROL]]
    all_layers = all_layers.rename(columns={COL_F_SLOPE: slope_names[2]})

    layers1_2 = data.loc[data[COL_DIRECTORY] == slope_names[3]][[COL_HABITAT, COL_FISH_NUMBER, COL_F_SLOPE,COL_COLOR_CONTROL]]
    layers1_2 = layers1_2.rename(columns={COL_F_SLOPE: slope_names[3]})

    layers3_5 = data.loc[data[COL_DIRECTORY] == slope_names[4]][[COL_HABITAT, COL_FISH_NUMBER, COL_F_SLOPE,COL_COLOR_CONTROL]]
    layers3_5 = layers3_5.rename(columns={COL_F_SLOPE: slope_names[4]})

    # merge the dataframe so a line correspond to an experiment: a fish + a middle + some corresponding network transfer
    exp = all_layers.merge(layers1_2, on=[COL_FISH_NUMBER,COL_HABITAT,COL_COLOR_CONTROL], how="outer")
    exp = exp.merge(layers3_5, on=[COL_FISH_NUMBER,COL_HABITAT,COL_COLOR_CONTROL], how="outer")
    exp = exp.merge(middles, on=[COL_HABITAT], how="outer")
    exp = exp.merge(fishes, on=[COL_FISH_NUMBER], how="outer")

    # split the experiments between controlled and uncontrolled color
    Y_controled=exp.loc[exp.color_control=="ON"]
    Y_uncontroled=exp.loc[exp.color_control=="OFF"]

    #sort is used to group experiments in color groups
    for sort in [COL_HABITAT, COL_FISH_NUMBER]:
        for name_middle, grp in data.groupby(sort):
            plt.plot(grp[COL_DIRECTORY], grp[COL_F_SLOPE], linestyle='',  marker='o', label=name_middle)
        plt.legend(title="type of {}".format(sort), loc='best')
        plt.title("mean Fourrier slope of each images per folder")

        groups_Y = Y_controled.groupby(sort, dropna=False)
        groups_uncontrolled_Y = Y_uncontroled.groupby(sort, dropna=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey="row")
        i=0
        for name , grp in groups_Y:
            color = FLAT_UI[i%len(groups_Y)]
            i+=1
            for _ , row in grp.iterrows():
                ax1.plot(slope_names, row[slope_names], marker='o', c=color, label=name)
        ax1.set_xlabel("images")
        ax1.set_title("Fourier slopes per experiment with color control")
        i=0
        for name , grp in groups_uncontrolled_Y:
            color = FLAT_UI[i%len(groups_uncontrolled_Y)]
            i+=1
            for _ , row in grp.iterrows():
                ax2.plot(slope_names, row[slope_names], marker='o', c=color, label=name)
        ax2.set_xlabel("images")
        ax2.set_title("Fourier slopes per experiment without color control")
        # ax2.legend(title="type of {}".format(sort), loc='best')
        plt.show()


def publication_plot(filepath):
    '''
    reproduces the plot seen in the Hulse nature paper
    '''
    data = pd.read_csv(filepath, sep=',')
    matplotlib.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    matplotlib.rcParams['axes.linewidth'] = 2
    sns.set_palette(sns.color_palette(FLAT_UI))

    ax = sns.catplot(x=COL_DIRECTORY, y=COL_F_SLOPE, data=data,
                    hue=COL_FISH_SEX,
                    split=True,
                    kind="violin")

    ax.set_ylabels('Slope of Fourier Power Spectrum ')
    ax.set_yticklabels(fontstyle='italic')
    plt.xticks(rotation=45)
    plt.title("mean Fourrier slopes per folder")
    plt.show()


def LBP_hist_per_folder(filepath):
    data = pd.read_csv(filepath, sep=',')

    split_by = [COL_COLOR_CONTROL, COL_DIRECTORY]
    params_lbp = [COL_RADIUS_LBP, COL_POINTS_LBP]

    param_groups = data.groupby(params_lbp)[[COL_PATH_LBP, *params_lbp, *split_by]]
    col=0
    for param_val, param_grp in param_groups:   #plot a graph for each parameters
        R, P = map(int, param_val)
        for split in split_by:  #group data on different categories
            groups = param_grp.groupby(split)
            # if len(groups)>1:
            n_row = len(groups)
            n_col = len(param_groups)
            fig1, axs = plt.subplots(n_row, n_col, sharex='col', squeeze=False)
            fig2, ax2 = plt.subplots(1, 1)
            fig1.suptitle("splitted by {}, params R:{}, P:{}".format(split, R, P))
            row=0
            for name, grp in groups:   #plot a resuming picture for each group
                axs[row,col].set_title("group {}".format(name))
                axs[row,col].set_xlabel("LBP value")
                axs[row,col].set_ylabel("image")
                visu_LBP = np.zeros(shape=(len(grp.index), 2**P))
                i=0
                for _ , line in grp.iterrows():  #sum lbp for each lbp image
                    LBP_values = np.load(line[COL_PATH_LBP]).astype(int).flatten()
                    counter_lbp = Counter(LBP_values)
                    percent_appearance = np.array(list(counter_lbp.values()))/LBP_values.size*100
                    visu_LBP[i, list(counter_lbp.keys())] = percent_appearance
                    i+=1
                axs[row,col].imshow(visu_LBP, aspect='auto')
                #a bar graph
                ax2.bar(np.arange(0, 2**P), visu_LBP.mean(axis=0), alpha=0.5,label=name)
                ax2.legend()
                row+=1
            plt.show()
        col+=1


if __name__ == '__main__':
    globals()[args.function](args.input_file)
