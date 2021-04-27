import rawpy as rp
import numpy as np
import imageio
import glob

from matplotlib import pyplot as plt

def process_linear_raw(fname):
    rawParams = rp.Params(gamma = (1, 1),
        no_auto_bright = True,
        user_wb = (1, 1, 1, 1),
        output_bps = 16,
        half_size = True)

    raw = rp.imread(fname)
    rgb = raw.postprocess(params = rawParams)

    return rgb

if (__name__ == '__main__'):
    outdir = './tif/'
    fnames = glob.glob('./*.CR2')

    for i in fnames:
        print('Processing file ' + str(i) + ' of ' + str(len(fnames)))
        rgb = process_linear_raw(i)
        imageio.imwrite((outdir + i[0:-4] + '.tif'), rgb)
