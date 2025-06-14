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
        self.n_seq = 41 if args.task == 'carla' and args.carla_task == 'replay' else 106
        self.group = args.group
        self.image_size = args.input_size[2:]  # input_size [g,d,h,w]
        self.transform_size = args.transform_size  # (h,w)
        self.episodes = []

        with open(args.data_file, "r") as f:
            for l in f:
                self.episodes.append(l.strip("\n"))

    def transform(self, idx):
        with h5py.File(self.episodes[idx], "r") as f:
            keys = list(f.keys())
            data_x = list(f[keys[0]])
            data_y = list(f[keys[1]])

        episode_n_frames = len(data_x)
        output = np.zeros((self.n_seq, self.group, self.nd, self.transform_size[0], self.transform_size[1]))

        seq_idx = []
        if self.training:
            # random sample
            for i in range(self.n_seq):
                seq_idx.append(np.random.randint(0, episode_n_frames - self.nd))
        else:
            # sequentially advance 1 frame per step
            for i in range(self.n_seq):
                seq_idx.append(i)
        for i in range(self.n_seq):
            for j in range(self.nd):
                X = data_x[seq_idx[i] + j]
                Y = data_y[seq_idx[i] + j]
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