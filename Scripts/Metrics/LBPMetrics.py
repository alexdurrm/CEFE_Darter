from skimage.feature import local_binary_pattern
import argparse
import os
from collections import Counter

from MotherMetric import MotherMetric
from Preprocess import *

#local binary Pattern
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
        return the path to the saved image
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
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path of the image file to open")
    args = parser.parse_args()
    image_path = os.path.abspath(args.image)
    
    preprocess = Preprocess(img_type=IMG.DARTER, img_channel=CHANNEL.GRAY)
    
    metric = LBPHistMetrics(points=[8, 16], radius=[2,4], nbins=20, preprocess=preprocess)
    print(metric(image_path))
    
    metric = BestLBPMetrics(points=[8, 16], radius=[2,4], n_best=20, preprocess=preprocess)
    print(metric(image_path))