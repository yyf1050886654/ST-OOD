import cv2
import numpy as np
import random
import os
import h5py
import torch
from torch.utils.data import Dataset

_seed = 20201205


def seed():
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Bi3DOFDataset(Dataset):

    def __init__(self, args):
        self.training = args.training
        self.nd = args.nd
        self.n_seq =106# 106 41
        self.group = args.group
        self.image_size = args.input_size[2:]  # input_size [g,d,h,w]
        self.transform_size = args.transform_size  # (h,w)
        self.episodes = []

        with open(args.data_file, "r") as f:
            for l in f:
                self.episodes.append(l.strip("\n"))
                # print("episodes: ", self.episodes)

    def transform(self, idx):
        episodes_filepath = self.episodes[idx]
        # print("episodes_filepath: ", episodes_filepath)
        # print("self.episodes[idx]: ", self.episodes[idx])
        # print("idx: ", idx)
        with h5py.File(self.episodes[idx], "r") as f:
            keys = list(f.keys())
            data_x = list(f[keys[0]])
            data_y = list(f[keys[1]])

        episode_n_frames = len(data_x)
        output = np.zeros((self.n_seq, self.group, self.nd, self.transform_size[0], self.transform_size[1]))

        seq_idx = []
        # print("episode_n_frames: ",episode_n_frames)
        if self.training:
            # print("cal random!")
            # random sample
            for i in range(self.n_seq):
                seq_idx.append(np.random.randint(0, episode_n_frames - self.nd))
        else:
            # sequentially advance 1 frame per step
            for i in range(self.n_seq):
                seq_idx.append(i)
        # print("self.n_seq:",self.n_seq)
        # print("self.nd:",self.nd)
        for i in range(self.n_seq):
            for j in range(self.nd):
                X = data_x[seq_idx[i] + j]
                Y = data_y[seq_idx[i] + j]
                # print("len(data_x):",len(data_x))
                # print("len(data_y):", len(data_y))
                # print("X.shape[0]:",X.shape[0])
                # print("self.image_size[0]:", self.image_size[0])
                # print("X.shape[1]:", X.shape[1])
                # print("self.image_size[1]:", self.image_size[1])
                # # Resize or crop frames to match network input size
                # X = cv2.resize(X, (152, 113))  # Or self.transform_size[1], self.transform_size[0]
                # Y = cv2.resize(Y, (152, 113))  # Or self.transform_size[1], self.transform_size[0]
                #
                # # Use the frames as they are
                # output[i, 0, j, :, :] = X
                # output[i, 1, j, :, :] = Y
                assert X.shape[0] == self.image_size[0] and X.shape[1] == self.image_size[1], \
                    "Mismatch between network input dimension and data dimension."
                # random crop
                h_margin = random.randint(0, int((X.shape[0] - self.transform_size[0] - 2) / 2))
                w_margin = random.randint(0, int((X.shape[1] - self.transform_size[1] - 2) / 2))
                output[i, 0, j, :, :] = X[h_margin:h_margin + self.transform_size[0],
                                        w_margin:w_margin + self.transform_size[1]]
                output[i, 1, j, :, :] = Y[h_margin:h_margin + self.transform_size[0],
                                        w_margin:w_margin + self.transform_size[1]]

        return output

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        output = torch.Tensor(self.transform(idx))
        return output

    def __len__(self):
        return len(self.episodes)