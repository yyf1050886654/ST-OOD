import random

import numpy as np
import argparse
import torch
from torch.cuda import current_blas_handle
from network import Bi3DOF, Encoder, Decoder
from datasets import seed, Bi3DOFDataset
from more_utils import make2D, OOD_score_to_iD_score, min_of_each_row, compute_epsilon_on_iD_traces_only, get_det_delay_for_detected_traces, scan_iD_scores_of_windows_and_print_list, collapse_to_1D, getTNR
import os
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

'''
    Inputs are test clips are hdf5 files prodiced byfeature_abstraction.py.
    Place all test clips in folder data/nuscenes-v1.0-mini.test.

    Outputs are scores for OoD detection.
'''
def load_model(test_config):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Default value
    args.training = False
    args.kl_weight = 1
    args.group = 2
    args.nz = 12
    args.nd = 4
    args.mu1, args.mu2 =0 , 0
    # test config
    args.latentprior = test_config["network"]
    args.var1, args.var2 = 1, 1

    # Input dimension as in # [grp,nd,h,w]
    args.input_size = [args.group, args.nd, 120,160]
    args.transform_size = [113,152]

    # Assifn test clips
    args.data_file = test_config["test_clips"]
    # print("args.data_file: ",args.data_file)
    # print("test_config[""]: ",test_config["frames_per_clip"])
    args.n_seq = 106
    # args.n_seq = 114
    args.n_seqs = [args.n_seq for f in test_config["frames_per_clip"]]
    # args.number_windows = [105 for f in test_config["frames_per_clip"]]
    args.number_windows = [94 for f in test_config["frames_per_clip"]]
    # args.n_seqs = [f - args.nd for f in test_config["frames_per_clip"]]
    # New: changed to list # Changed to test all sequences(aka window) in video(aka trace). Number of sequences(aka windows) is total frames-frame size(aka nd = 16)
    # print("args.n_seqs: ",args.n_seqs)
    # Load  weights
    encoder = Encoder(args)
    decoder = Decoder(args)
    model = Bi3DOF(encoder, decoder, args).to(args.device)
    model.load_state_dict(torch.load(test_config["model_file"], map_location = args.device))
    model.eval()
    return model, args
seed = 2
random.seed(seed)

# 随机生成一个数
random_number = random.random()
def compute_score(model, args):
    args.batch_size = 1
    testset = Bi3DOFDataset(args)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)
    d_horizontal = []
    d_vertical = []
    model.eval()
    with torch.no_grad():

        for b_idx, (batch_data) in enumerate(test_loader):
            data = batch_data.to(args.device)
            (b1, b2, g, d, h, w) = data.shape
            data = data.view((b1 * b2, g, d, h, w))
            # for j in range(14):
            #     _, (d_grp1, d_grp2) = model.encode(data)
            #     # print("len(d_grp1): ",len(d_grp1))
            #     for i in range(len(d_grp1)):
            #         d_horizontal.append(d_grp1[i].cpu().numpy())
            #         d_vertical.append(d_grp2[i].cpu().numpy())
            _, (d_grp1, d_grp2) = model.encode(data)
            for i in range(len(d_grp1)):
                d_horizontal.append(d_grp1[i].cpu().numpy())
                d_vertical.append(d_grp2[i].cpu().numpy())
    return d_horizontal, d_vertical

# obtain below from feature_abstraction
frame_lens = {
    'train': [149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149],
    'in': [122, 121, 121, 122, 122, 121, 122, 123, 123, 121,123,123,124],
    'out': [123, 122, 123, 121, 124, 123, 121, 121, 122, 123, 123, 121, 123,122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122],
    # 'out': [50, 50, 50, 50, 50, 51, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,50, 50, 50, 50, 50],
    'validate': [130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130]}
bi3dof_simple_test_validate = {
    # "model_file" : lambda model_save_folder: "{}/bi3dof-simple-600epoch-6seq-seed{}.pt".format(model_save_folder, SEED), # "model/nuscenes-mini/bi3dof-simple-600epoch.pt",
    "model_file": "./replay_models/bi3dof-simple-600epoch-6seq.pt",
    "network": "simple",
    "test_clips": "F:/pycharmworkspace/time-series-OOD-main/data/data/replay_dataset/carla_features_all/validate.test",
    # "data/nuscenes-v1.0-mini.test",
    "frames_per_clip": frame_lens['validate']
}
bi3dof_simple_test_in = {
    # "model_file" : lambda model_save_folder: "{}/bi3dof-simple-600epoch-6seq-seed{}.pt".format(model_save_folder, SEED), # "model/nuscenes-mini/bi3dof-simple-600epoch.pt",
    "model_file": "./replay_models/bi3dof-simple-600epoch-6seq.pt",
    "network": "simple",
    "test_clips": "F:/pycharmworkspace/time-series-OOD-main/data/data/replay_dataset/carla_features_all/in.test",
    # "data/nuscenes-v1.0-mini.test",
    "frames_per_clip": frame_lens['in']
}


def getOutBi3DOF(type_of_OOD):
    features_folder = "F:/pycharmworkspace/time-series-OOD-main/data/data/replay_dataset/carla_features_all/"  # Change to "../NTU_features_rainy_only/" for rainy
    bi3dof_simple_test_out = {
        # "model_file" : lambda model_save_folder: "{}/bi3dof-simple-600epoch-6seq-seed{}.pt".format(model_save_folder, SEED), # "model/nuscenes-mini/bi3dof-simple-600epoch.pt",
        "model_file": "./replay_models/bi3dof-simple-600epoch-6seq.pt",
        "network": "simple",
        "test_clips": features_folder + "{}.test".format(type_of_OOD),  # "data/nuscenes-v1.0-mini.test",
        "frames_per_clip": frame_lens[type_of_OOD]
    }
    return bi3dof_simple_test_out


def run(type_of_OOD):
    print('\n', type_of_OOD, '\n')
    # ROC curve calculation
    iD_scores_all = []; GTs_all = []
    scores_of_only_in_points = []
    scores_of_only_out_points = []

    for idx, bi3dof_simple in enumerate([bi3dof_simple_test_in, getOutBi3DOF(type_of_OOD)]):

        # i.e. for traces in [iD traces, OOD traces]
        print("idx: ",idx)
        print("bi3dof_simple: ",bi3dof_simple)
        model, args = load_model(bi3dof_simple)
        h,v = compute_score(model, args)
        OOD_scores_flattened = [h[i]+v[i] for i in range(len(h))]
        print("len(h): ",len(h))
        print("OOD_scores_flattened: ",len(OOD_scores_flattened))
        print("args.n_seqs: ",sum(args.n_seqs))
        OOD_scores_2D_list = make2D(OOD_scores_flattened, args.n_seqs)
        iD_scores_2D_list = OOD_score_to_iD_score(OOD_scores_2D_list)
        #iD_scores_for_each_trace = min_of_each_row(iD_scores_2D_list)
        iD_scores_windows = collapse_to_1D(iD_scores_2D_list)
        if ('in' in bi3dof_simple["test_clips"]) and ('rainy' not in bi3dof_simple["test_clips"]):
            GT = 1
            scores_of_only_in_points.extend(iD_scores_windows)
        if ('out' in bi3dof_simple["test_clips"]) or ('rainy' in bi3dof_simple["test_clips"]):
            GT = 0
            scores_of_only_out_points.extend(iD_scores_windows)
        #GTs_for_each_trace = [GT for _ in range(len(iD_scores_for_each_trace))]
        GTs_for_each_window = [GT for _ in range(len(iD_scores_windows))]

        if GT==0:
            iD_scores_2D_list_of_OOD_traces_only = iD_scores_2D_list

        # iD_scores_all.extend(iD_scores_for_each_trace)
        iD_scores_all.extend(iD_scores_windows)
        GTs_all.extend(GTs_for_each_window)

    fpr, tpr, threshs = roc_curve(GTs_all, iD_scores_all)
    unique, counts = np.unique(GTs_all, return_counts=True)
    print(dict(zip(unique, counts)))
    print("GTs_all: ",GTs_all)
    print("iD_scores_all: ",iD_scores_all)
    auroc = roc_auc_score(GTs_all, iD_scores_all)
    # plt.figure()
    # plt.plot(fpr, tpr)
    # plt.legend(['ROC curve (AUROC: {})'.format(auroc)])
    # try:
    # 	os.mkdir('./plots_carla/')
    # except:
    # 	pass
    # plt.savefig('./plots_carla/plot_{}.png'.format(type_of_OOD))

    try:
        os.mkdir('./npz_saved/')
    except:
        pass
    second_half_of_type_of_OOD = type_of_OOD.split('_')[-1]
    if second_half_of_type_of_OOD == "replay":
        np.save(f'./npz_saved/{second_half_of_type_of_OOD}_win_in_NTU', scores_of_only_in_points)
        np.save(f'./npz_saved/{second_half_of_type_of_OOD}_win_out_NTU', scores_of_only_out_points)

    TNR, tau = getTNR(scores_of_only_in_points, scores_of_only_out_points)
    det_delay = get_det_delay_for_detected_traces(iD_scores_2D_list_of_OOD_traces_only, tau)

    print(f'(AUROC, TNR, Avg Det Delay): ({auroc}, {TNR}, {det_delay})')

if __name__ == "__main__":
    for type_of_OOD in ['out_rainy', 'out_snowy', 'out_foggy', 'out_night']:
        run(type_of_OOD)