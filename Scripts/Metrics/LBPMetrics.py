from skimage.feature import local_binary_pattern
import argparse
import os
from collections import Counter

from MotherMetric import MotherMetric
from Preprocess import *

#local binary Pattern
CSV_LBP="lbp.csv"

COL_POINTS_LBP="points_LBP"
COL_RADIUS_LBP="radius_LBP"
COL_VALUE_LBP="value_LBP"
COL_COUNT_LBP="count_LBP_value"
class LBPHistMetrics(MotherMetric):
    def __init__(self, points, radius, nbins, *args, **kwargs):
        assert len(points)==len(radius), "points and radius are used zipped, should be the same length"
        self.points = points
        self.radius = radius
        self.nbins = nbins

        super().__init__(*args, **kwargs)

    def function(self, image, visu=False):
        '''
        calculates the Local Binary Pattern of a given image
        P is the number of neighbors points to use
        R is the radius of the circle around the central pixel
        visu is to visualise the result
        '''
        assert image.ndim==2, "image should be 2 dimensions: H,W, here{}".format(image.shape)
        df = pd.DataFrame()
        params = self.preprocess.get_params()
        idx=0
        for P, R in zip(self.points, self.radius):
            lbp_image = local_binary_pattern(image, P, R)
            vals, bins = np.histogram(lbp_image, bins=self.nbins)
            for bin, val in zip(bins, vals):
                df.loc[idx, params.columns] = params.iloc[0]
                df.loc[idx, [COL_POINTS_LBP, COL_RADIUS_LBP, COL_VALUE_LBP, COL_COUNT_LBP]] = [P, R, bin, val]
                idx+=1
            if visu:
                fig, (ax0, ax1, ax2) = plt.subplots(figsize=(6, 12), nrows=3)
                ax0.imshow(image, cmap='gray')
                ax0.set_title("original image")

                bar = ax1.imshow(lbp_image, cmap='gray')
                fig.colorbar(bar, ax=ax1, orientation="vertical")
                ax1.set_title("LBP with params P={} and R={}".format(P, R))

                ax2.hist(lbp_image.flatten(), bins=self.bins)
                ax2.set_title("lbp values histogram")
                plt.show()

        return df


CSV_BEST_LBP="best_lbp.csv"
class BestLBPMetrics(MotherMetric):
    def __init__(self, points, radius, n_best=20, *args, **kwargs):
        assert len(points)==len(radius), "points and radius are used zipped, should be the same length"
        self.points = points
        self.radius = radius
        self.n_best = n_best
        super().__init__(*args, **kwargs)

    def function(self, image, visu=False):
        '''
        calculates the Local Binary Pattern of a given image
        P is the number of neighbors points to use
        R is the radius of the circle around the central pixel
        visu is to visualise the result
        return the path to the saved image
        '''
        assert image.ndim==2, "image should be 2 dimensions: H,W, here{}".format(image.shape)
        df = pd.DataFrame()
        params = self.preprocess.get_params()
        idx=0
        for P, R in zip(self.points, self.radius):
            lbp_image = local_binary_pattern(image, P, R)
            cnt = Counter(lbp_image.flatten())
            best_lbp = cnt.most_common(self.n_best)
            for rank, lbp in enumerate(best_lbp):
                lbp_val, lbp_count = lbp
                df.loc[idx, params.columns] = params.iloc[0]
                df.loc[idx, [COL_POINTS_LBP, COL_RADIUS_LBP, COL_VALUE_LBP, COL_COUNT_LBP]] = [P, R, lbp_val, lbp_count]
                idx+=1
        return df




if __name__=='__main__':
    #default parameters
    RESIZE=(None, None)
    CHANNELS=CHANNEL.GRAY
    IMG_TYPE=IMG.DARTER
    POINTS=[8,16]
    RADIUS=[2,4]
    N_BINS=100

    #parsing input
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path of the image file to open")
    parser.add_argument("action", help="type of action needed", choices=["visu", "work"])
    parser.add_argument("metric", help="metric to call, lbp calculates the full, best_lbp only remember the most comons", choices=["lbp", "best_lbp"])
    parser.add_argument("-o", "--output_dir", default=DIR_RESULTS, help="directory where to put the csv output, default: {}".format(DIR_RESULTS))
    parser.add_argument("-x", "--resize_X", default=RESIZE[0], type=int, help="shape to resize image x, default: {}".format(RESIZE[0]))
    parser.add_argument("-y", "--resize_Y", default=RESIZE[1], type=int, help="shape to resize image y, default: {}".format(RESIZE[1]))
    parser.add_argument("-n", "--n_bins", default=N_BINS, type=int, help="Number of bins into which cast the lbp values, or number of best lbp values to remember, default: {}".format(N_BINS))
    args = parser.parse_args()
    resize = (args.resize_X, args.resize_Y)
    path = os.path.abspath(args.path)

    #prepare the metric
    preprocess = Preprocess(resizeX=resize[0], resizeY=resize[1], img_type=IMG_TYPE, img_channel=CHANNELS)

    if args.metric=="lbp":
        metric = LBPHistMetrics(points=POINTS, radius=RADIUS, nbins=args.nbins, preprocess=preprocess)
    elif args.metric=="best_lbp":
        metric = BestLBPMetrics(points=POINTS, radius=RADIUS, n_best=args.nbins, preprocess=preprocess)

    if args.action == "visu":
        metric.load()
        metric.visualize()
    elif args.action=="work":
        metric.metric_from_csv(path)
        metric.save()
