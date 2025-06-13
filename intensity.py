#!/usr/bin/env python3
import numpy as np
from glob import glob
import cv2
import h5py
import os


class FeatureAbstraction:
    def __init__(self, trainroot, testroot, calibrationroot, dstroot):
        # Specify output (aka destination) root
        self.dstroot = dstroot
        # Resize inputs
        self.newdim = (320, 240)

        for phase in ["train", "in", "out", "calibration"]:
            # List file for train and test loaders
            self.store = []

            if "train" in phase:
                path = trainroot
                locs = glob(path + "*")
                names = locs
                phase_type = "train"
            elif "in" in phase:
                path = testroot + phase + "/"
                locs = glob(path + "*")
                names = locs
                phase_type = "test"
            elif "out" in phase:
                path = testroot + phase + "/"
                locs = glob(path + "*")
                names = locs
                phase_type = "test"
            elif "calibration" in phase:
                path = calibrationroot
                locs = glob(path + "*")
                names = locs
                phase_type = "calibration"

            for idx, scenefolder in enumerate(locs):
                features_x = []
                features_y = []
                im1 = None
                prev_pts = None  # Initialize prev_pts

                for imagefile in sorted(glob(scenefolder + "/*.png")):
                    im2 = cv2.cvtColor(cv2.resize(cv2.imread(imagefile), self.newdim), cv2.COLOR_BGR2GRAY)

                    if im1 is not None:
                        # print("Image 1 shape:", im1.shape)
                        # print("Image 2 shape:", im2.shape)

                        # Detect feature points in the first frame using Shi-Tomasi corner detection
                        if prev_pts is None:
                            feature_params = dict(maxCorners=50000, qualityLevel=0.3, minDistance=1, blockSize=3)
                            prev_pts = cv2.goodFeaturesToTrack(im1, mask=None, **feature_params).reshape(-1, 1, 2)

                        # Calculate sparse optical flow using Lucas-Kanade with prev_pts
                        lk_params = dict(winSize=(3,3), maxLevel=5,
                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 0.0001))
                        flow, status, _ = cv2.calcOpticalFlowPyrLK(im1, im2, prev_pts, None, **lk_params)

                        # Extract the sparse flow vectors
                        flow_x = flow[:, 0, 0]
                        flow_y = flow[:, 0, 1]

                        features_x.append(flow_x)
                        features_y.append(flow_y)

                        # Update prev_pts for the next frame
                        prev_pts = flow.reshape(-1, 1, 2)

                    im1 = im2

                # Write optic flow fields of one video episode to a h5 file
                file = os.path.join(self.dstroot, phase_type + "." +names[idx].split(os.path.sep)[-1] + ".h5")

                with h5py.File(file, "w") as f:
                    f.create_dataset("x", data=features_x)
                    f.create_dataset("y", data=features_y)
                self.store.append(file)

            h5fillist = os.path.join(self.dstroot, phase + "." + phase_type)
            with open(h5fillist, "w") as f:
                for scene in self.store:
                    f.write(scene + "\n")

            print("See feature extraction results for phase={} in {}".format(phase, h5fillist))


if __name__ == "__main__":
    trainroot = 'F:/pycharmworkspace/time-series-OOD-main/data/data/snowy_dataset/training/'
    testroot = 'F:/pycharmworkspace/time-series-OOD-main/data/data/snowy_dataset/testing/'
    dstroot = 'F:/pycharmworkspace/time-series-OOD-main/data/data/snowy_dataset/xin_feature/'
    calibrationroot = 'F:/pycharmworkspace/time-series-OOD-main/data/data/snowy_dataset/calibration/'

    try:
        os.mkdir(dstroot)
    except:
        pass
    FeatureAbstraction(trainroot, testroot, calibrationroot, dstroot)
