"""
Methods to process Canon .CR2 raw files, and save them as tiffs without any
gamma correction, or compression; and with/without white balance.
"""

import rawpy as rp
import imageio
import os

# path = 'C:/Users/preinstalled/PycharmProjects/pythonProject/fish_images/TrammelCreek/RAW/'
path = 'C:/Users/preinstalled/PycharmProjects/pythonProject/fish_images/e_gracile_BrushxUnknownCreeks/RAW/'

files = os.listdir(path)
print(path, files)
# outdir = 'C:/Users/preinstalled/PycharmProjects/pythonProject/fish_images/TrammelCreek/test_defWB/'
outdir = 'C:/Users/preinstalled/PycharmProjects/pythonProject/fish_images/e_gracile_BrushxUnknownCreeks/whitebalance/'

count = 0
for i in files:
    count += 1
    fnames = os.path.join(path, i)
    print(fnames)
    print('Processing file ' + str(count) + ' of ' + str(len(files)))
    raw = rp.imread(fnames)
    print(raw)
    # rawParams = rp.Params(gamma=(1, 1),
    #                       no_auto_bright=False,
    #                       user_wb=(1, 1, 1, 1),
    #                       output_bps=16,
    #                       half_size=True)
    # print(rawParams)
    # rgb = raw.postprocess(params = rawParams)
    rgb = raw.postprocess(use_camera_wb=True)           # waiting for Sam to send the proper parameters
    print(rgb)
    imageio.imwrite((outdir + i[1:-4] + 'defWB' + '.tif'), rgb)