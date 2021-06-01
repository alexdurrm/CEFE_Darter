"""
Methods to process Canon .CR2 raw files, and save them as tiffs without any
gamma correction, or compression; and with/without white balance.
"""

import rawpy as rp
import imageio
import os
import argparse

# path = 'C:/Users/preinstalled/PycharmProjects/pythonProject/fish_images/TrammelCreek/RAW/'

parser = argparse.ArgumentParser()
parser.add_argument("input", help="path of the file to open")
parser.add_argument("--delete", type=lambda x:bool(eval(x)), default=False, help="say True if should delete .CR2 afterward")
args = parser.parse_args()

path = args.input
files = os.listdir(path)
# print(path, files)
# outdir = 'C:/Users/preinstalled/PycharmProjects/pythonProject/fish_images/TrammelCreek/test_defWB/'
outdir = os.path.join(path, "TIFF")
if not(os.path.exists(outdir) and os.path.isdir(outdir)):
    os.makedirs(outdir)

count = 0
for i in files:
    count += 1
    fnames = os.path.join(path, i)
    if os.path.isdir(fnames):
        continue
    print('Processing file ' + str(count) + ' of ' + str(len(files)))
    raw = rp.imread(fnames)
    # print(raw)
    # rawParams = rp.Params(gamma=(1, 1),
    #                       no_auto_bright=True,
    #                       user_wb=(1, 1, 1, 1),
    #                       output_bps=16,
    #                       half_size=True)
    # print(rawParams)
    # rgb = raw.postprocess(params = rawParams)
    rgb = raw.postprocess(use_camera_wb=True)           # waiting for Sam to send the proper parameters
    # print(rgb)
    imageio.imwrite(os.path.join(outdir, i[1:-4] + "_WB"+ '.tif'), rgb)
    if args.delete:
        print("deleting {}".format(fnames))
        os.remove(fnames)
