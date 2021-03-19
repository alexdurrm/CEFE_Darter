import pandas as pd
import os
import argparse

from MotherMetric import MotherMetric
from Preprocess import *
from PHOG.anna_phog import anna_phog

#PHOG
COL_PHOG_LEVELS="phog_level"
COL_PHOG_BINS="phog_bins"
COL_PHOG_VALUE="phog_val"
class PHOGMetrics(MotherMetric):
    def __init__(self, orientations=8, level=0, preprocess=None):
        self.orientations=orientations
        self.level=level
        if not preprocess:
            preprocess = Preprocess(img_type=IMG.RGB, img_channel=CHANNEL.GRAY)
        super().__init__(preprocess)
        
    def function(self, image):
        df = pd.DataFrame()
        params = self.preprocess.get_params()
        
        roi = [0, image.shape[0], 0, image.shape[1]]
        phog = anna_phog(image, self.orientations, 360, self.level, roi)

        for i, value in enumerate(phog):
            df.loc[i, params.columns] = params.loc[0]
            df.loc[i, [COL_PHOG_BINS, COL_PHOG_LEVELS, COL_PHOG_VALUE]] = [self.orientations, self.level, value]
        return df


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path of the image file to open")
    args = parser.parse_args()
    image_path = os.path.abspath(args.image)

    metric = PHOGMetrics(orientations=40, level=2)
    print(metric(image_path))