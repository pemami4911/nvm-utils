##
# Author: Patrick Emami, based on convert.py by Adam King
#

import sys
import os.path
import pickle
import numpy as np
from tqdm import tqdm


### REPLACE WITH YOUR FILES HERE
data_dir = "/data/pemami/iccv2019/KingsCollege/"
inputFile = data_dir + "reconstruction.nvm"
trainSet = data_dir + "train/dataset_train.txt"
testSet = data_dir + "train/dataset_test.txt"

###################################################################
# Parse the .nvm file

if inputFile.endswith(".nvm"):
    from readNvm import *
    print("parsing {}".format(inputFile))
    nvmObj = readNvm(inputFile)


if not os.path.exists("vis_matrix.npy"):
###################################################################
# Compute overlapping 3D points for all image pairs. Kinda slow
# so we save the intermediate result.

    from itertools import combinations_with_replacement

    print("computing 3D point overlap")

    num_imgs = len(nvmObj.modelArray[0].cameraArray)
    vis_matrix = np.zeros((num_imgs,num_imgs), dtype=np.int32)
    # for each 3D point, record all image pairs that can view it
    for pt in tqdm(nvmObj.modelArray[0].pointArray):
        img_idxs = []
        for m in pt.measurementArray:
            img_idxs.append(int(m.imageIndex))
        all_pairs = combinations_with_replacement(img_idxs, 2)
        for p in all_pairs:
            if p[0] == p[1]:
                continue
            vis_matrix[p[0],p[1]] += 1

    np.save("vis_matrix", vis_matrix)

else:
    print("loading vis_matrix")
    vis_matrix = np.load("vis_matrix.npy")
    num_imgs, _ = vis_matrix.shape

###################################################################
# grab camera poses to use for matching with the 3D reconstruction.
# This is because you may have a list of image files that
# are part of your training set, which you only want to 
# compute visibility against other training images.

print("parsing train/test files")

set_Imgs = {}
for set_ in [trainSet, testSet]:
    set_Imgs[set_] = {}
    with open(set_, "r") as f:
        # throw away first 3 lines
        for _ in range(3): f.readline()
        for line in f:
            data = line.split(" ")
            qarr = np.zeros(4)
            tarr = np.zeros(3)
            for i in range(3):
                tarr[i] = float(data[1+i])
            for i in range(4):
                qarr[i] = float(data[4+i])
            set_Imgs[set_][data[0]] = {}
            set_Imgs[set_][data[0]]['q'] = qarr
            set_Imgs[set_][data[0]]['t'] = tarr

###################################################################
# Compute the VSfM indices corresponding to the train/test file names
# by matching with the closest camera pose

img_idx_dict = {}
for i in tqdm(range(nvmObj.modelArray[0].numCameras)):
    qarr = np.zeros(4)
    tarr = np.zeros(3)
    for j in range(3):
        tarr[j] = float(nvmObj.modelArray[0].cameraArray[i].cameraCenter[j])
    for j in range(4):
        qarr[j] = float(nvmObj.modelArray[0].cameraArray[i].quaternionArray[j])
    img_idx_dict[i] = {'name': "", 'is_test': -1}
    found = False
    for is_test, set_ in enumerate([trainSet, testSet]):
        if found:
            break
        for k, v in set_Imgs[set_].items():
            if np.allclose(qarr, v['q'], 1e-3) and \
                    np.allclose(tarr, v['t'], 1e-3):
                img_idx_dict[i]['name'] = k
                img_idx_dict[i]['is_test'] = is_test
                found = True
                break
    if not found:
        print("!!! couldn't match image idx {}".format(i))

###################################################################
#
print("saving viz into .txt files")

for i, set_ in enumerate([trainSet, testSet]):
    if i == 0:
        set_dir = os.path.join(data_dir, "train")
    else:
        set_dir = os.path.join(data_dir, "test")
    if not os.path.exists(set_dir):
        os.makedirs(set_dir)
    vis_dir = os.path.join(set_dir, "visibility")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    with open(os.path.join(set_dir, "visibility.txt"), "w+") as f:
        for k, _ in set_Imgs[set_].items():
            # extract the frame number
            seq, frame = os.path.splitext(k)[0].split("/")
            vis_dir_ = os.path.join(vis_dir, seq)
            if not os.path.exists(vis_dir_):
                os.makedirs(vis_dir_)
            vis_fname = os.path.join(vis_dir_, "vis_{}.txt".format(frame))
            f.write(os.path.join("visibility", seq, "vis_{}.txt".format(frame)) + "\n")
            with open(vis_fname, "w") as v:
                # find match
                for kk, vv in img_idx_dict.items():
                    if vv['name'] == k:
                        break
                for jj in range(num_imgs):
                    if img_idx_dict[jj]['is_test'] == i:
                        v.write(str(vis_matrix[kk,jj]) + "\n")
 
