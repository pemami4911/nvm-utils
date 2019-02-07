##
# Author: Patrick Emami, based on convert.py by Adam King
#

import sys
import os.path
import pickle
import numpy as np
from tqdm import tqdm


### REPLACE WITH YOUR FILES HERE
data_dir = ""
inputFile = data_dir + "reconstruction.nvm"
trainSet = data_dir + "train/dataset_train.txt"
testSet = data_dir + "test/dataset_test.txt"

###################################################################
# Parse the .nvm file

if inputFile.endswith(".nvm"):
    from readNvm import *
    print("parsing {}".format(inputFile))
    nvmObj = readNvm(inputFile)


###################################################################
print("parsing train/test files")

tr_imgs = []
te_imgs = []
for istest, set_ in enumerate([trainSet, testSet]):
    with open(set_, "r") as f:
        # throw away first 3 lines
        for _ in range(3): f.readline()
        for line in f:
            data = line.split(" ")[0]
            if istest:
                te_imgs.append(data)
            else:
                tr_imgs.append(data)

###################################################################
# store the focal length

tr_focals = {}
te_focals = {}
for i in tqdm(range(nvmObj.modelArray[0].numCameras)):
    fileName, focal_len = nvmObj.modelArray[0].cameraArray[i].focalLength.split('\t')
    fileName = fileName.replace(".jpg", ".png")
    if fileName in tr_imgs:
        tr_focals[fileName] = float(focal_len)
    else:
        te_focals[fileName] = float(focal_len)

###################################################################
#
print("saving camera stuff into files")

for i, set_ in enumerate([trainSet, testSet]):
    foc = None
    if i == 0:
        set_dir = os.path.join(data_dir, "train")
        foc = tr_focals
    else:
        set_dir = os.path.join(data_dir, "test")
        foc = te_focals
    if not os.path.exists(set_dir):
        os.makedirs(set_dir)
    vis_dir = os.path.join(set_dir, "calibration")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    with open(os.path.join(set_dir, "calibration.txt"), "w+") as f:
        for k, v in foc.items():
            # extract the frame number
            seq, frame = os.path.splitext(k)[0].split("/")
            vis_dir_ = os.path.join(vis_dir, seq)
            if not os.path.exists(vis_dir_):
                os.makedirs(vis_dir_)
            vis_fname = os.path.join(vis_dir_, "calib_{}.txt".format(frame))
            f.write(os.path.join("calibration", seq, "calib_{}.txt".format(frame)) + "\n")
            with open(vis_fname, "w") as vv:
                vv.write(str(v) + "\n")
