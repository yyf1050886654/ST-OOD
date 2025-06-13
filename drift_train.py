import argparse
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from network import Bi3DOF, Encoder, Decoder
from datasets import *
from utils import progress_bar
from test_drift import compute_score, load_model
from more_utils import make2D, OOD_score_to_iD_score, min_of_each_row, compute_epsilon_on_iD_traces_only, \
    get_det_delay_for_detected_traces, scan_iD_scores_of_windows_and_print_list, collapse_to_1D, getTNR, \
    getPrecisionRecallF1

SEED = 2
frame_lens = {
    'train': [49, 74, 49, 49, 49, 49, 59, 59, 59, 59, 59, 49, 59, 59, 59, 59, 59, 49, 49, 49, 49, 48, 49, 49],
    'in': [49, 89, 59, 59, 59, 59, 59, 59, 59, 59, 59, 49, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 49, 59, 59, 59,
           59, 59, 49, 49, 49, 49, 49, 49],
    'out': [59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 89, 89, 59, 59, 59, 47, 47, 71, 47, 47, 47, 47, 49, 47, 47, 47,
            47, 47, 47, 47, 47, 47, 71, 49, 47, 47, 47, 47, 47, 47, 47, 47, 59, 59, 49, 59, 89, 59, 59, 59, 59, 59,
            59, 59, 59, 49, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 49, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 49,
            59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 49, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59],
    # 'out': [59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 89, 89, 59, 59, 59, 47, 47, 71, 47]
    'validate': [50, 50, 50, 50, 50, 50, 50, 50, 50, 60, 60, 60, 60, 60]
}


def getOutBi3DOF(type_of_OOD):
    features_folder = "D:/pycharm-workspace/time-series-OOD-main/data/data/drift_dataset/carla_features_all/"  # Change to "../NTU_features_rainy_only/" for rainy
    bi3dof_simple_test_out = {
        # "model_file" : lambda model_save_folder: "{}/bi3dof-simple-600epoch-6seq-seed{}.pt".format(model_save_folder, SEED), # "model/nuscenes-mini/bi3dof-simple-600epoch.pt",
        "model_file": "./drift_models/bi3dof-simple-600epoch-6seq-seed2.pt",
        "network": "simple",
        "test_clips": features_folder + "{}.test".format(type_of_OOD),  # "data/nuscenes-v1.0-mini.test",
        "frames_per_clip": frame_lens[type_of_OOD]
    }
    return bi3dof_simple_test_out


bi3dof_simple_test_validate = {
    # "model_file" : lambda model_save_folder: "{}/bi3dof-simple-600epoch-6seq-seed{}.pt".format(model_save_folder, SEED), # "model/nuscenes-mini/bi3dof-simple-600epoch.pt",
    "model_file": "./drift_models/bi3dof-simple-600epoch-6seq-seed2.pt",
    "network": "simple",
    "test_clips": "D:/pycharm-workspace/time-series-OOD-main/data/data/drift_dataset/carla_features_all/validate.test",
    # "data/nuscenes-v1.0-mini.test",
    "frames_per_clip": frame_lens['validate']
}
bi3dof_simple_test_in = {
    # "model_file" : lambda model_save_folder: "{}/bi3dof-simple-600epoch-6seq-seed{}.pt".format(model_save_folder, SEED), # "model/nuscenes-mini/bi3dof-simple-600epoch.pt",
    "model_file": "./drift_models/bi3dof-simple-600epoch-6seq-seed2.pt",
    "network": "simple",
    "test_clips": "D:/pycharm-workspace/time-series-OOD-main/data/data/drift_dataset/carla_features_all/in.test",
    # "data/nuscenes-v1.0-mini.test",
    "frames_per_clip": frame_lens['in']
}
type_of_OOD = 'out'


def train(args):
    # load data
    train_id = Bi3DOFDataset(args)
    train_size = len(train_id)
    train_loader = torch.utils.data.DataLoader(train_id, batch_size=args.episode_size, shuffle=True, num_workers=0)

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
        os.mkdir(args.model_save_folder)
    except:
        pass

    torch.save(vae.state_dict(),
               "{}/bi3dof-{}-{}epoch-{}seq-seed{}.pt".format(args.model_save_folder, args.latentprior, epoch + 1,
                                                             args.n_seqs[0], SEED))


def validate(args, writer):
    validate_id = Bi3DOFDataset(args)
    validate_size = len(validate_id)
    validate_loader = torch.utils.data.DataLoader(validate_id, batch_size=args.episode_size, shuffle=True,
                                                  num_workers=0)

    # initialize network and optimizor
    # encoder = Encoder(args)
    # decoder = Decoder(args)
    # vae = Bi3DOF(encoder, decoder, args).to(args.device)
    bi3dof_simple = [bi3dof_simple_test_in, bi3dof_simple_test_validate]
    model, args = load_model(bi3dof_simple)
    for epoch in range(args.epochs):
        loss_reconstruction, loss_latent, num_examples = 0, 0, 0
        model.eval()
        for batch_idx, batch_data in enumerate(validate_loader):
            batch_data = batch_data.to(args.device)
            (b1, b2, g, d, h, w) = batch_data.shape
            batch_data = batch_data.view((b1 * b2, g, d, h, w))
            num_examples += b1 * b2
            loss, loss_rec, (loss_latent_grp1, loss_latent_grp2) = model.loss(batch_data, args.kl_weight)
            loss = loss.mean(dim=-1)

            loss_reconstruction += loss_rec.view(-1).mean()
            loss_latent += (loss_latent_grp1.view(-1).mean() + loss_latent_grp2.view(-1).mean())

            avg_loss_reconstruction = loss_reconstruction / num_examples
            avg_loss_latent = loss_latent / num_examples

            progress_bar(batch_idx, len(validate_loader),
                         'Epoch%3d   avg_loss_reconstruction: %.6f  avg_loss_latent: %.6f'
                         % (epoch, avg_loss_reconstruction, avg_loss_latent))
        # save
    try:
        os.mkdir(args.model_save_folder)
    except:
        pass
    writer.add_scalar('val/avg_loss_reconstruction', avg_loss_reconstruction, epoch)
    writer.add_scalar('val/avg_loss_latent', avg_loss_latent, epoch)
    print('[VAL] loss: {:.3f}, acc: {:.3f}'.format(avg_loss_reconstruction, avg_loss_latent))
    return avg_loss_reconstruction


def test(args):
    test_id = Bi3DOFDataset(args)
    test_size = len(test_id)
    test_loader = torch.utils.data.DataLoader(test_id, batch_size=args.episode_size, shuffle=True,
                                                  num_workers=0)

    # initialize network and optimizor
    # encoder = Encoder(args)
    # decoder = Decoder(args)
    # vae = Bi3DOF(encoder, decoder, args).to(args.device)
    bi3dof_simple = [bi3dof_simple_test_in, bi3dof_simple_test_validate]
    model, args = load_model(bi3dof_simple)
    for epoch in range(args.epochs):
        loss_reconstruction, loss_latent, num_examples = 0, 0, 0
        model.eval()
        for batch_idx, batch_data in enumerate(test_loader):
            batch_data = batch_data.to(args.device)
            (b1, b2, g, d, h, w) = batch_data.shape
            batch_data = batch_data.view((b1 * b2, g, d, h, w))
            num_examples += b1 * b2
            loss, loss_rec, (loss_latent_grp1, loss_latent_grp2) = model.loss(batch_data, args.kl_weight)
            loss = loss.mean(dim=-1)

            loss_reconstruction += loss_rec.view(-1).mean()
            loss_latent += (loss_latent_grp1.view(-1).mean() + loss_latent_grp2.view(-1).mean())

            avg_loss_reconstruction = loss_reconstruction / num_examples
            avg_loss_latent = loss_latent / num_examples

            progress_bar(batch_idx, len(test_loader),
                         'Epoch%3d   avg_loss_reconstruction: %.6f  avg_loss_latent: %.6f'
                         % (epoch, avg_loss_reconstruction, avg_loss_latent))
    print('[VAL] loss: {:.3f}, acc: {:.3f}'.format(avg_loss_reconstruction, avg_loss_latent))
    return avg_loss_reconstruction


def init_param():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_size", type=int, default=5, help="number of videos in one mini-batch")
    # parser.add_argument("--n_seq", type=int, default = 6, help="number of sequence/window to sample from one video")
    # Changed above to use fixed number of random sequences (aka windows) in each video (aka trace) every epoch to train (by changing datasets file to use list of n_seq values ==> helpful to test variable number of sequences at test time)
    parser.add_argument("--nd", type=int, default=16,
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
    parser.add_argument("--model_save_folder", type=str, default="./drift_models",
                        help="set different folder names only for drift clip length ablations.")
    args = parser.parse_args()

    # min_of_n_seqs = min([f - args.nd for f in frame_lens["train"]]) # 114
    # args.n_seqs = [min_of_n_seqs for _ in range(len(frame_lens["train"]))] # New: changed to list
    args.n_seqs = [6 for _ in range(len(frame_lens["train"]))]

    # input dimension as in # [g,d,h,w]
    args.input_size = [args.group, args.nd, 120, 160]
    args.transform_size = [113, 152]

    # default values
    args.epochs = 600
    args.lr_base = 0.0001
    args.kl_weight = 1
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args


if __name__ == '__main__':
    args = init_param()
    args.training = True
    args.data_file = "F:/pycharmworkspace/time-series-OOD-main/data/data/drift_dataset/carla_features_all/train.train"
    train(args)