#!/usr/bin/env python3
import argparse
import os

import torch

from data_provider.feature_abstraction_carla import FeatureAbstraction
from exp.ood_detection_carla import train

'''

Train routine for bi3dof and bi3dofopt methods in the paper "Improving Variational Autoencoder 
based Out-of-Distribution Detection for Embedded Real-time Applications" 

A bi3dof detector is trained by the default parameter values. 

To train a bi3dof-optprior detector, first compute the mu and var of optic flow fields of the training data set. 
Then run the python script with arguments --latentprior optimal and mu&var values as described in help of init_param().

More technical details please see in paper link: https://arxiv.org/abs/2107.11750

'''

if __name__ == '__main__':
    # obtain below from feature_abstraction
    frame_lens = {
        'train': [148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148],
        'in': [122, 121, 121, 122, 122, 121, 122, 123, 123, 121, 123, 123],
        # 'out': [123, 122, 123, 121, 124, 123, 121, 121, 122, 123, 123, 121, 123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122],
        'out': [50,50,50, 50, 50, 50, 50,50, 50, 50,50, 50,50, 50, 50,50, 50,50, 50, 50,50, 50,50, 50, 50,50,50],
        # 'out':[141, 122, 112, 114, 141, 140, 141, 140, 111, 112, 111, 113, 111, 114, 111, 114, 141, 114, 111, 116, 111, 114, 141, 111, 111, 141, 142],
        # 'out':[123, 122, 123, 121, 124, 123, 121, 121, 122, 123, 123, 121, 123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122],
        # 'validate': [123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122]}
        'validate': [130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130]}
    parser = argparse.ArgumentParser()
    # Basic Config
    parser.add_argument("--task", type=str, default="carla", help="carla or drift")
    parser.add_argument("--carla_task", type=str, default="rainy", help="Carla dataset subtask")
    # Dataset
    parser.add_argument("--data_path_prefix", type=str, default="F:/data/data", help="Dataset path prefix")
    # Model Parameters
    parser.add_argument("--episode_size", type=int, default=12, help="number of videos in one mini-batch")
    parser.add_argument("--n_seq", type=int, default=141, help="number of sequence/window to sample from one video")
    parser.add_argument("--nd", type=int, default=4,
                        help="number of frames in a 3D input cube")  # prev: default=6; changed to 16 based on Ramneet's choice of frames in a window/sequence
    parser.add_argument("--group", type=int, default=2,
                        help="2 refer to the two latent subspaces, in horizontal and vertical direction respectively.")
    parser.add_argument("--nz", type=int, default=12, help="Dimension of one latent subspace")
    parser.add_argument("--latentprior", type=str, default='simple',
                        help="If 'optimal' the follows mu and var values will be used to compute distribution descrepancy in latent space.")
    parser.add_argument("--mu1", type=float, default=0.,
                        help="Mean optic flow fields in the horizontal direction. Need to be computed standalone from a specific trainig set.")
    parser.add_argument("--mu2", type=float, default=0., help="Mean optic flow value in the vertical direction")
    parser.add_argument("--var1", type=float, default=1., help="Varaince in the horizontal direction")
    parser.add_argument("--var2", type=float, default=1., help="Varaince in the horizontal? (vertical) direction")
    parser.add_argument("--kl_weight", type=int, default=1, help="KL with weight")
    # Run Parameters
    parser.add_argument("--seed", type=int, default=2, help="Model random seed")
    parser.add_argument("--epochs", type=int, default=600, help="Model training epochs")
    parser.add_argument("--lr_base", type=int, default=0.00001, help="Basic learning rate")
    args = parser.parse_args()

    args.n_seqs = [args.n_seq for _ in range(len(frame_lens["train"]))]
    args.input_size = [args.group, args.nd, 120, 160]  # input dimension as in # [g,d,h,w]
    args.transform_size = [113, 152]
    # device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.data_path = os.path.join(args.data_path_prefix, "{}_dataset".format(args.carla_task))

    # data pre_process
    args.features_folder = os.path.join(args.data_path, "st-vae-icad-feature")
    if not os.path.exists(args.features_folder):
        os.mkdir(args.features_folder)
        FeatureAbstraction(os.path.join(args.data_path, "training"),
                           os.path.join(args.data_path, "testing"),
                           os.path.join(args.data_path, "calibration"), args.features_folder)

    args.training = True
    args.data_file = os.path.join(args.features_folder, "train.train")
    args.model_save_folder = "{}_models".format(args.carla_task)
    train(args)