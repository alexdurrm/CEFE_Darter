import tensorflow.keras as K
from tensorflow.keras.applications.vgg16 import VGG16
import pandas as pd
import numpy as np
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt

from Preprocess import *
from config import *
from MotherMetric import MotherMetric

from ScalarMetrics import get_gini


#deep features
CSV_DEEP_FEATURES="deep_features.csv"
COL_MODEL_NAME="model_name_deep_features"
COL_PATH_DEEP_FEATURES="path_deep_features"
COL_SPARSENESS_DF="sparseness_deep_features"
COL_LAYER_DF="layer_deep_feature"
class DeepFeatureMetrics(MotherMetric):
    def __init__(self, base_model, input_shape, *args, **kwargs):
        self.base_model = base_model
        self.input_shape = input_shape
        input_tensor = K.Input(shape=self.input_shape)
        self.base_model.layers[0] = input_tensor
        self.deep_features = K.Model(inputs=self.base_model.input, outputs=[l.output for l in self.base_model.layers[1:]])

        super().__init__(*args, **kwargs)

    def get_deep_features(self, image, visu=False):
        '''
        get the feature space of an image propagated through the deep feature model
        return a list of np array, each element of the list represent an output of a layer, input layer is ignored
        '''
        image = (image - np.mean(image, axis=(0,1))) / np.std(image, axis=(0,1))
        #make the prediction
        pred = self.deep_features.predict(image[np.newaxis, ...])
        if visu:
            self.deep_features.summary()
            for p in pred:
                print(type(p))
                print(p.shape)
        return pred

    def get_layers_gini(self, image):
        deep_features = self.get_deep_features(image)
        sparseness=[get_gini(f[0]) for f in deep_features]
        return sparseness

    def function(self, image):
        gini = self.get_layers_gini(image)
        params = self.preprocess.get_params()

        df = pd.DataFrame()
        for layer_idx , gini in enumerate(gini):
            df.loc[layer_idx, params.columns] = params.iloc[0]
            df.loc[layer_idx, [COL_MODEL_NAME, COL_SPARSENESS_DF, COL_LAYER_DF]] = [self.base_model.name, gini, layer_idx]
        return df

    def visualize(self):
        '''
        plot for each image the gini coefficient of each network layer
        '''
        sns.set_palette(sns.color_palette(FLAT_UI))

        data_image = pd.read_csv(os.path.join(DIR_RESULTS, CSV_IMAGE), index_col=0)
        merge_data = self.data.merge(data_image, on=COL_IMG_PATH)

        sns.relplot(data=merge_data, x=COL_LAYER_DF, y=COL_SPARSENESS_DF, hue=COL_DIRECTORY, col=COL_MODEL_NAME, kind="line", units=COL_IMG_PATH, estimator=None, alpha=0.25)
        plt.show()

if __name__ == '__main__':
    #default parameters
    INPUT_SHAPE = (1500, 512)

    #parsing input
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help="type of action needed", choices=["visu", "work"])
    parser.add_argument("path", help="path of the image file to open")
    parser.add_argument("-o", "--output_dir", default=DIR_RESULTS, help="directory where to put the csv output, default: {}".format(DIR_RESULTS))
    parser.add_argument("-x", "--input_X", default=INPUT_SHAPE[0], type=int, help="shape to resize image X, default: {}".format(INPUT_SHAPE[0]))
    parser.add_argument("-y", "--input_Y", default=INPUT_SHAPE[0], type=int, help="shape to resize image y, default: {}".format(INPUT_SHAPE[1]))

    args = parser.parse_args()
    input_shape = (args.input_X, args.input_Y)

    pr = Preprocess(resizeX=input_shape[0], resizeY=input_shape[1], normalize=True)
    vgg16_model = DeepFeatureMetrics( VGG16(weights='imagenet', include_top=False), input_shape, pr, os.path.join(args.output_dir, CSV_DEEP_FEATURES))

    if args.action == "visu":
        vgg16_model.load()
        vgg16_model.visualize()
    elif args.action=="work":
        vgg16_model.metric_from_csv(args.path)
        vgg16_model.save()
