import pandas as pd
import os
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

from MotherMetric import MotherMetric
from Preprocess import *
from PHOG.anna_phog import anna_phog
from config import *

#PHOG
CSV_PHOG="phog.csv"
COL_PHOG_LEVELS="phog_level"
COL_PHOG_ORIENTATIONS="phog_bins"
COL_PHOG_VALUE="phog_val"
COL_PHOG_BIN="phog_bin"
class PHOGMetrics(MotherMetric):
    def __init__(self, orientations=8, level=0, *args, **kwargs):
        self.orientations=orientations
        self.level=level
        super().__init__(*args, **kwargs)

    def function(self, image):
        df = pd.DataFrame()
        params = self.preprocess.get_params()

        roi = [0, image.shape[0], 0, image.shape[1]]
        phog = anna_phog(image.astype(np.uint8), self.orientations, 360, self.level, roi)

        for i, value in enumerate(phog):
            df.loc[i, params.columns] = params.loc[0]
            df.loc[i, [COL_PHOG_BIN, COL_PHOG_ORIENTATIONS, COL_PHOG_LEVELS, COL_PHOG_VALUE]] = [i, self.orientations, self.level, value]
        return df

    def visualize(self):
        data_image = pd.read_csv(os.path.join(DIR_RESULTS, CSV_IMAGE), index_col=0)
        merge_data = self.data.merge(data_image, on=COL_IMG_PATH)
        sns.relplot(data=merge_data, x=COL_PHOG_BIN, y=COL_PHOG_VALUE,
            hue=COL_DIRECTORY, col=COL_PHOG_LEVELS, units=COL_IMG_PATH, estimator=None)
        plt.show()


if __name__=='__main__':
    #default parameters
    RESIZE=(None, None)
    CHANNELS=CHANNEL.GRAY
    IMG_TYPE=IMG.RGB

    ORIENTATIONS=8
    LEVEL=2

    #parsing input
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path of the image file to open")
    parser.add_argument("action", help="type of action needed", choices=["visu", "work"])
    parser.add_argument("-o", "--output_dir", default=DIR_RESULTS, help="directory where to put the csv output, default: {}".format(DIR_RESULTS))
    parser.add_argument("-x", "--resize_X", default=RESIZE[0], type=int, help="shape to resize image x, default: {}".format(RESIZE[0]))
    parser.add_argument("-y", "--resize_Y", default=RESIZE[1], type=int, help="shape to resize image y, default: {}".format(RESIZE[1]))
    parser.add_argument("-n""--n_orientation", default=ORIENTATIONS, type=int, help="number of orientations to classify gradients, default: {}".format(ORIENTATIONS))
    parser.add_argument("-l", "--level", default=LEVEL, type=int, help="Levels for the pyramidal hog, default: {}".format(LEVEL))
    args = parser.parse_args()
    resize = (args.resize_X, args.resize_Y)
    path = os.path.abspath(args.path)

    pr = Preprocess(img_type=IMG_TYPE, img_channel=CHANNELS, resize=resize)
    metric = PHOGMetrics(orientations=args.n_orientation, level=args.level, preprocess=pr, path=os.path.join(args.output_dir, CSV_PHOG))

    if args.action == "visu":
        metric.load()
        metric.visualize()
    elif args.action=="work":
        metric.metric_from_csv(path)
        metric.save()
