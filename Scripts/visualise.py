import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns
import os
import argparse
from collections import Counter
import imageio

from config import *


def group_by_experiment(data):
    '''
    group data by experiments, each experiment consisting of a group (FISH, HABITAT, OUTPUT_NETWORK...)
    '''
    slope_names=["HABITAT_images","FISH_images","All_layers", "Layers_1and2","Layers_3to5"]

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
    return exp


def transfer_effect_slope(data):
    '''
    used to compare the slopes of fish, environment, and different modified fishes
    to measure the effects after a style transfer
    '''

    #plot a naive but complete scatter of the fourier slopes
    plt.scatter(data[COL_DIRECTORY], data[COL_F_SLOPE])
    plt.title("mean Fourrier slope of each images per folder")
    plt.show()

    # split the data in databases for the middle, the fishes, the network predictions
    # and rename the column name F_slope to a more appropriate value
    slope_names=["HABITAT_images","FISH_images","All_layers",
                "Layers_1and2","Layers_3to5"]

    exp = group_by_experiment(data)

    # split the experiments between controlled and uncontrolled color
    Y_controled=exp.loc[exp[COL_COLOR_CONTROL]=="ON"]
    Y_uncontroled=exp.loc[exp[COL_COLOR_CONTROL]=="OFF"]

    #sort is used to group experiments in color groups
    for sort in [COL_HABITAT, COL_FISH_NUMBER]:
        for name_middle, grp in data.groupby(sort):
            plt.plot(grp[COL_DIRECTORY], grp[COL_F_SLOPE], linestyle='',  marker='o', label=name_middle)
        plt.legend(title="type of {}".format(sort), loc='best')
        plt.title("mean Fourrier slope of each images per folder")
        plt.show()

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


def publication_plot(data):
    '''
    reproduces the plot seen in the Hulse nature paper
    '''
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


def LBP_hist_per_folder(data):
    '''
    Compare the mean local binary patterns histograms for each of the folders
    '''

    split_by = [COL_COLOR_CONTROL, COL_DIRECTORY]
    params_lbp = [COL_RADIUS_LBP, COL_POINTS_LBP]

    param_groups = data.groupby(params_lbp)[[COL_PATH_LBP, *params_lbp, *split_by]]
    col=0
    for param_val, param_grp in param_groups:   #plot a graph for each parameters
        R, P = map(int, param_val)
        for split in split_by:  #group data on different categories
            groups = param_grp.groupby(split)
            n_row = len(groups)
            n_col = len(param_groups)
            fig1, axs = plt.subplots(n_row, n_col, sharex='col', squeeze=False)
            fig2, ax2 = plt.subplots(1, 1)
            fig1.suptitle("mean LBP histograms splitted by {}, params R:{}, P:{}".format(split, R, P))
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
                    percent_appearance = [lbp/LBP_values.size*100 for lbp in counter_lbp.values()]
                    visu_LBP[i, list(counter_lbp.keys())] = percent_appearance
                    i+=1
                axs[row,col].imshow(visu_LBP, aspect='auto')
                #a bar graph
                ax2.bar(np.arange(0, 2**P), visu_LBP.mean(axis=0), alpha=0.5,label=name)
                ax2.legend()
                row+=1
            plt.show()
        col+=1


def Gini_compare_folders(data):
    '''
    compare the gini values of each images
    '''

    matplotlib.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    matplotlib.rcParams['axes.linewidth'] = 2
    sns.set_palette(sns.color_palette(FLAT_UI))

    ax = sns.catplot(x=COL_DIRECTORY, y=COL_GINI_VALUE, data=data)
    ax.set_ylabels(COL_GINI_VALUE)
    ax.set_yticklabels(fontstyle='italic')
    plt.xticks(rotation=45)
    plt.title("GINI value")
    plt.show()

    ax = sns.catplot(x=COL_DIRECTORY, y=COL_GINI_VALUE, data=data, hue=COL_HABITAT)
    plt.xticks(rotation=45)
    plt.show()


def network_sparseness(data):
    '''
    plot for each image the gini coefficient of each network layer
    '''
    df = pd.DataFrame()
    x=0
    for i, row in data.iterrows():
        ginis = eval(row[COL_SPARSENESS_DF])
        layers= ["layer_"+str(x) for x in range(len(ginis))]
        for j in range(len(ginis)):
            df.loc[x, ['layer_network', COL_SPARSENESS_DF, COL_DIRECTORY, COL_FILENAME]] = [j, ginis[j], row[COL_DIRECTORY], row[COL_FILENAME]]
            x+=1
    sns.relplot(data=df, x='layer_network', y=COL_SPARSENESS_DF, hue=COL_DIRECTORY, units=COL_FILENAME, kind="line", estimator=None, alpha=0.5)
    plt.show()
    print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("function", help="name of the function to use", choices=["fourier_per_folder", "transfer_effect_slope", "publication_plot",
        "LBP_hist_per_folder", "Haralick_compare_folders", "Gini_compare_folders", "Stat_compare_folders", "network_sparseness"])
    parser.add_argument("input_file", help="the path of the csv to open")
    args = parser.parse_args()

    data = pd.read_csv(args.input_file, sep=',')
    globals()[args.function](data)
