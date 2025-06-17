import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score
from torch import optim
from torch.utils.data import DataLoader

from check_OOD_carla import calc_cal_ce_loss, calc_p_value
from data_provider.datasets import Bi3DOFDataset
from model.network import Encoder, Decoder, Bi3DOF
from scripts.martingales import SMM
from test_carla import frame_lens, compute_score, load_model
from utils.more_utils import getTNR, get_det_delay_for_detected_traces, make2D, OOD_score_to_iD_score, collapse_to_1D, \
    getPrecisionRecallF1
from utils.utils import progress_bar


def train(args):
    # load data
    train_dataset = Bi3DOFDataset(args)
    train_size = len(train_dataset)
    print("train_size: ",train_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.episode_size, shuffle=True, num_workers=0)

    # initialize network and optimizor
    encoder = Encoder(args)
    decoder = Decoder(args)
    vae = Bi3DOF(encoder, decoder, args).to(args.device)
    enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=args.lr_base)
    dec_optimizer = optim.Adam(vae.decoder.parameters(), lr=args.lr_base)

    for epoch in range(args.epochs):
        loss_reconstruction, loss_latent, num_examples = 0, 0, 0
        vae.train()
        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(args.device)
            (b1, b2, g, d, h, w) = batch_data.shape
            batch_data = batch_data.view((b1 * b2, g, d, h, w))
            num_examples += b1 * b2

            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            loss, loss_rec, (loss_latent_grp1, loss_latent_grp2) = vae.loss(batch_data, args.kl_weight)
            loss = loss.mean(dim=-1)
            loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()

            loss_reconstruction += loss_rec.view(-1).mean()
            loss_latent += (loss_latent_grp1.view(-1).mean() + loss_latent_grp2.view(-1).mean())

            progress_bar(batch_idx, len(train_loader), 'Epoch%3d  Recontruction/Loss: %.6f  Latent/Loss: %.6f'
                         % (epoch, loss_reconstruction / num_examples, loss_latent / num_examples))
    # save
    try:
        os.mkdir("./{}_models".format(args.carla_task))
    except:
        pass

    torch.save(vae.state_dict(),
               "{}/bi3dof-{}-{}epoch-{}seq-seed{}-{}nd.pt".format(args.model_save_folder, args.latentprior, args.epochs,
                                                             args.n_seqs[0], args.seed,args.nd))

def getOutBi3DOF(args, in_flag):
    prefix = "in" if in_flag else "out_replay" if (args.task == 'carla' and args.carla_task == 'replay') else "out"
    return {
        "model_file" : "{}/bi3dof-{}-{}epoch-{}seq-seed{}-{}nd.pt".format(args.model_save_folder, args.latentprior, args.epochs,
                                                             args.n_seqs[0], args.seed,args.nd),
        "network": "simple",
        "test_clips": os.path.join(args.features_folder, "{0}.{0}".format("in" if in_flag else "out")),
        "frames_per_clip": frame_lens[prefix]
    }

def test(args):
    # ROC curve calculation
    iD_scores_all = []; GTs_all = []
    scores_of_only_in_points = []
    scores_of_only_out_points = []

    for idx, bi3dof_simple in enumerate([getOutBi3DOF(args, in_flag=True), getOutBi3DOF(args, in_flag=False)]):
        # i.e. for traces in [iD traces, OOD traces]
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
    second_half_of_type_of_OOD = "carla_ood".split('_')[-1]
    # if second_half_of_type_of_OOD == "replay":
    np.save(f'./npz_saved/{second_half_of_type_of_OOD}_win_in_NTU', scores_of_only_in_points)
    np.save(f'./npz_saved/{second_half_of_type_of_OOD}_win_out_NTU', scores_of_only_out_points)

    TNR, tau = getTNR(scores_of_only_in_points, scores_of_only_out_points)
    det_delay = get_det_delay_for_detected_traces(iD_scores_2D_list_of_OOD_traces_only, tau)

    print(f'(AUROC, TNR, Avg Det Delay): ({auroc}, {TNR}, {det_delay})')



def check_ood_carla(args):

    calc_cal_ce_loss()
    scores_of_only_in_points = []
    scores_of_only_out_points = []
    iD_scores_all = []
    GTs_all = []

    for idx, bi3dof_simple in enumerate([getOutBi3DOF(args, in_flag=True), getOutBi3DOF(args, in_flag=False)]):
        time_start = time.time()
        calc_cal_ce_loss()
        calc_save_path = "./{}_models/nc_calibration_vae_600epoch.npy".format(args.task)
        model, args = load_model(bi3dof_simple)
        h, v = compute_score(model, args)
        test_loss_list = [h[i] + v[i]  for i in range(len(h))]
        OOD_scores_flattened = []
        i = 0
        m_array = np.array([]).reshape(0, 2)
        p_array = np.array([]).reshape(0, 2)
        nd = 4
        while i < (len(bi3dof_simple['frames_per_clip'])):#13
            j = 0
            while j < 94:
                z = 0
                smm = SMM(4)
                while z < 4:
                    index = i * 106 + j + z*nd

                    test_loss = test_loss_list[index]
                    calc_loss = np.load(calc_save_path)
                    p = calc_p_value(test_loss, calc_loss)
                    m = smm(p)
                    m_array = np.vstack([m_array, [i * 94 + j, m]])
                    p_array = np.vstack([p_array, [i * 94 + j, p]])
                    z = z + 1
                OOD_scores_flattened.append(m)
                j = j + 1
            i = i + 1
        OOD_scores_2D_list = make2D(OOD_scores_flattened, args.number_windows)
        # OOD_scores_2D_list = make2D(OOD_scores_flattened, args.n_seqs)
        iD_scores_2D_list = OOD_score_to_iD_score(OOD_scores_2D_list)
        iD_scores_windows = collapse_to_1D(iD_scores_2D_list)
        # if ('in' in bi3dof_simple["test_clips"]):
        if ('in.in' in args.data_file):
            GT = 0
            scores_of_only_in_points.extend(iD_scores_windows)

        else:
            GT = 1
            scores_of_only_out_points.extend(iD_scores_windows)
        GTs_for_each_window = [GT for _ in range(len(iD_scores_windows))]

        if GT == 1:
            iD_scores_2D_list_of_OOD_traces_only = iD_scores_2D_list

        iD_scores_all.extend(iD_scores_windows)
        GTs_all.extend(GTs_for_each_window)

    auroc = roc_auc_score(GTs_all, iD_scores_all)
    # TNR, tau = getTNR(scores_of_only_in_points, scores_of_only_out_points)
    precision, recall, f1, fpr, fnr = getPrecisionRecallF1(scores_of_only_in_points, scores_of_only_out_points)
    time_end = time.time();
    print("time_sum: ",time_end-time_start)
    print(f'(AUROC, fnr): ({auroc}, {fnr})')
    print(f'(precision, recall, f1, Fpr): ({precision}, {recall}, {f1},{fpr})')
