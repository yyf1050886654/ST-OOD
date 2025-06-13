from matplotlib import pyplot as plt
import time
from datasets import *

from test_drift import compute_score, load_model
from scripts.martingales import SMM, RPM
from more_utils import make2D, OOD_score_to_iD_score, \
    get_det_delay_for_detected_traces, collapse_to_1D, getTNR,get_binary_labels_with_95_TPR
from more_utils import getPrecisionRecallF1
from sklearn.metrics import roc_curve, roc_auc_score

frame_lens = {
    'train': [49, 74, 49, 49, 49, 49, 59, 59, 59, 59, 59, 49, 59, 59, 59, 59, 59, 49, 49, 49, 49, 48, 49, 49],
    'in': [49, 59, 59, 59, 59, 59,
           59, 59, 59, 59, 49, 59,
           59, 59, 59, 59, 59, 59,
           59, 59, 59, 49, 59, 59,
           59, 59, 59, 49, 49, 49,
           49, 49, 49, 49],
    'out': [ 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 89, 89, 59, 59, 59, 47, 47, 71, 47, 47, 47, 47, 49, 47, 47, 47,
            47, 47, 47, 47, 47, 47, 71, 49, 47, 47, 47, 47, 47, 47, 47, 47, 59, 59, 49, 59, 89, 59, 59, 59, 59, 59,
            59, 59, 59, 49, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 49, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 49,
            59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 49, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59],
    # 'out': [59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 89, 89, 59, 59, 59, 47, 47, 71, 47]
    'validate': [49,50, 50, 50, 50, 50, 50, 50, 50, 50, 60, 60, 60, 60, 60,49]
}
bi3dof_simple_test_validate = {
    # "model_file" : lambda model_save_folder: "{}/bi3dof-simple-600epoch-6seq-seed{}.pt".format(model_save_folder, SEED), # "model/nuscenes-mini/bi3dof-simple-600epoch.pt",
    # "model_file": "./drift_models_new/bi3dof-simple-600epoch-42seq-seed2-6nd.pt",
    "model_file":"F:./drift_models/bi3dof-simple-600epoch-41seq-seed2-6nd.pt",
    "network": "simple",
    "test_clips": "F:/lsm/time-series-OOD-main/data/data/drift_dataset/ntu_feature/calibration.calibration",
    # "data/nuscenes-v1.0-mini.test",
    "frames_per_clip": frame_lens['validate']
}
bi3dof_simple_test_in = {
    # "model_file" : lambda model_save_folder: "{}/bi3dof-simple-600epoch-6seq-seed{}.pt".format(model_save_folder, SEED), # "model/nuscenes-mini/bi3dof-simple-600epoch.pt",
    # "model_file": "./drift_models_new/bi3dof-simple-600epoch-42seq-seed2-6nd.pt",
    "model_file":"F:./drift_models/bi3dof-simple-600epoch-41seq-seed2-6nd.pt",
    "network": "simple",
    "test_clips": "F:/lsm/time-series-OOD-main/data/data/drift_dataset/ntu_feature/in.test",
    # "data/nuscenes-v1.0-mini.test",
    "frames_per_clip": frame_lens['in']
}


def getOutBi3DOF(type_of_OOD):
    features_folder = "F:/lsm/time-series-OOD-main/data/data/drift_dataset/ntu_feature/"  # Change to "../NTU_features_rainy_only/" for rainy
    bi3dof_simple_test_out = {
        # "model_file" : lambda model_save_folder: "{}/bi3dof-simple-600epoch-6seq-seed{}.pt".format(model_save_folder, SEED), # "model/nuscenes-mini/bi3dof-simple-600epoch.pt",
        # "model_file": "./drift_models_new/bi3dof-simple-600epoch-42seq-seed2-6nd.pt",
        "model_file":"F:./drift_models/bi3dof-simple-600epoch-41seq-seed2-6nd.pt",
        "network": "simple",
        "test_clips": features_folder + "{}.test".format(type_of_OOD),  # "data/nuscenes-v1.0-mini.test",
        "frames_per_clip": frame_lens[type_of_OOD]
    }
    return bi3dof_simple_test_out


def calc_cal_ce_loss():  # for calibration datapoint, we want one randomly sampled window for 1 datapoint

    ce_loss_list = []
    calc_save_path = f"./drift_models/nc_calibration_vae_600epoch.npy"
    for idx, bi3dof_simple in enumerate([bi3dof_simple_test_validate]):  # i.e. for traces in [iD traces, OOD traces]
        model, args = load_model(bi3dof_simple)
        args.training = True
        h, v = compute_score(model, args)
        # ce_loss_list = [h[i] + v[i] for i in range(len(h))]
        ce_loss_list = [h[i] + v[i] for i in range(0, len(h), 8)]#8
    print("calc_nc.size:", len(ce_loss_list))
    # 保存到文件
    np.save(calc_save_path, ce_loss_list)


def calc_p_value(test_ce_loss_value, cal_set_ce_loss):
    # 计算百分位数
    compare = 0;
    for i in range(len(cal_set_ce_loss)):
        if test_ce_loss_value <= (cal_set_ce_loss[i]):
            compare = compare+1;
    p_value = (compare+1) / ((len(cal_set_ce_loss))+1)
    # print("len(cal_set_ce_loss: ",len(cal_set_ce_loss))
    return p_value


SEED = 2
# seed(SEED)
seed()

def run(type_of_OOD):
    time_start = time.time();
    calc_cal_ce_loss()
    calc_save_path = f"./drift_models/nc_calibration_vae_600epoch.npy"
    print('\n', type_of_OOD, '\n')
    scores_of_only_in_points = []
    scores_of_only_out_points = []
    iD_scores_all = []
    GTs_all = []

    # 记录 in 和 out 的执行时间
    in_start_time = None
    in_end_time = None
    out_start_time = None
    out_end_time = None

    for idx, bi3dof_simple in enumerate([bi3dof_simple_test_in, getOutBi3DOF(type_of_OOD)]):
        # 记录 in 的开始时间
        if 'in.test' in bi3dof_simple["test_clips"]:
            in_start_time = time.time()

        # 记录 out 的开始时间
        if 'out.test' in bi3dof_simple["test_clips"]:
            out_start_time = time.time()

        model, args = load_model(bi3dof_simple)
        h, v = compute_score(model, args)
        test_loss_list = [h[i] + v[i]  for i in range(len(h))]
        # print("test_loss_list: ",test_loss_list)
        OOD_scores_flattened = []
        m_array = np.array([]).reshape(0, 2)
        p_array = np.array([]).reshape(0, 2)
        print("len(test_loss_list)", len(test_loss_list))
        i=0
        nd=6
        while i<(len(bi3dof_simple['frames_per_clip'])):
            j = 0
            while j<22:
                test_all = 0;
                z = 0
                smm = SMM(4)
                while z<4:
                    test_loss = test_loss_list[i*41+j+z*nd]
                    calc_loss = np.load(calc_save_path)
                    p = calc_p_value(test_loss, calc_loss)
                    m = smm(p)
                    m_array = np.vstack([m_array, [i*22+j, m]])
                    p_array = np.vstack([p_array, [i*22+j, p]])
                    z = z+1
                OOD_scores_flattened.append(m)
                j = j+1
            i = i+1
        OOD_scores_2D_list = make2D(OOD_scores_flattened, args.number_windows)

        iD_scores_2D_list = OOD_score_to_iD_score(OOD_scores_2D_list)
        iD_scores_windows = collapse_to_1D(iD_scores_2D_list)
        if ('in.test' in args.data_file):
            GT = 0
            scores_of_only_in_points.extend(iD_scores_windows)
            # np.savetxt('m_array_in.txt', m_array, fmt='%d %.6f', delimiter=' ',newline='\n')
            # np.savetxt('p_array_in.txt', p_array, fmt='%d %.6f', delimiter=' ', newline='\n')
            in_end_time = time.time()
        else:
            GT = 1
            scores_of_only_out_points.extend(iD_scores_windows)
            # np.savetxt('m_array_out.txt', m_array, fmt='%d %.6f', delimiter=' ',newline='\n')
            # np.savetxt('p_array_out.txt', p_array, fmt='%d %.6f', delimiter=' ', newline='\n')
            out_end_time = time.time()
        GTs_for_each_window = [GT for _ in range(len(iD_scores_windows))]

        if GT == 1:
            iD_scores_2D_list_of_OOD_traces_only = iD_scores_2D_list

        iD_scores_all.extend(iD_scores_windows)
        GTs_all.extend(GTs_for_each_window)
    # if in_start_time and in_end_time:
    #     in_elapsed_time = in_end_time - in_start_time
    #     print(f"Time taken for 'in': {in_elapsed_time:.2f} seconds")

    # if out_start_time and out_end_time:
    #     out_elapsed_time = out_end_time - out_start_time
    #     print(f"Time taken for 'out': {out_elapsed_time:.2f} seconds")
    auroc = roc_auc_score(GTs_all, iD_scores_all)
    TNR, tau = getTNR(scores_of_only_in_points, scores_of_only_out_points)

    precision, recall, f1, fpr, fnr = getPrecisionRecallF1(scores_of_only_in_points, scores_of_only_out_points)
    time_end = time.time()-time_start
    print("time_end: ",time_end)
    print(f'(recall, fnr): ({recall}, {fnr})')
    print(f'(precision, auroc, f1,Fpr): ({precision}, {auroc}, {f1},{fpr})')

if __name__ == "__main__":
    for type_of_OOD in ['out']:
        run(type_of_OOD)
