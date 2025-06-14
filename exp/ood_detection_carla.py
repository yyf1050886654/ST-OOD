import os

import torch
from torch import optim
from torch.utils.data import DataLoader

from data_provider.datasets import Bi3DOFDataset
from model.network import Encoder, Decoder, Bi3DOF
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
               "{}/bi3dof-{}-{}epoch-{}seq-seed{}-{}nd.pt".format(args.model_save_folder, args.latentprior, epoch + 1,
                                                             args.n_seqs[0], args.seed,args.nd))


def test(type_of_OOD):
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
    # if second_half_of_type_of_OOD == "replay":
    np.save(f'./npz_saved/{second_half_of_type_of_OOD}_win_in_NTU', scores_of_only_in_points)
    np.save(f'./npz_saved/{second_half_of_type_of_OOD}_win_out_NTU', scores_of_only_out_points)

    TNR, tau = getTNR(scores_of_only_in_points, scores_of_only_out_points)
    det_delay = get_det_delay_for_detected_traces(iD_scores_2D_list_of_OOD_traces_only, tau)

    print(f'(AUROC, TNR, Avg Det Delay): ({auroc}, {TNR}, {det_delay})')
