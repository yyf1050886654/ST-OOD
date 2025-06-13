import time

from datasets import *

from test_carla import compute_score, load_model
from scripts.martingales import SMM, RPM
from more_utils import make2D, OOD_score_to_iD_score, \
    get_det_delay_for_detected_traces, collapse_to_1D, getTNR,get_binary_labels_with_95_TPR
from more_utils import getPrecisionRecallF1
from sklearn.metrics import roc_curve, roc_auc_score
seed = 42
random.seed(seed)

# 随机生成一个数
random_number = random.random()

frame_lens = {
    'train': [148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148],
    'in': [123, 122, 123, 123, 122, 122, 123, 122, 122, 122, 123, 121, 124],
    # 'out': [141, 122, 112, 114, 141, 140, 141, 140, 111, 112, 111, 113, 111, 114, 111, 114, 141, 114, 111, 116, 111, 114, 141, 111, 111, 141, 142],
   'out_foggy': [123, 122, 123, 121, 124, 123, 121, 121, 122, 123, 123, 121, 123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122],
	'out_night': [123, 122, 123, 121, 124, 123, 121, 121, 122, 123, 123, 121, 123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122],
	'out_snowy': [123, 122, 123, 121, 124, 123, 121, 121, 122, 123, 123, 121, 123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122, 122],
	#'out_rainy_old': [111, 141, 142, 112, 114, 141, 140, 141, 141, 140, 111, 112, 111, 113, 114, 111, 114, 141, 114, 111, 116, 141, 112, 122, 112, 141, 111, 112, 141, 141, 112, 112, 111, 116, 142, 140, 111, 116, 111, 116, 116, 114, 113, 111, 142, 115, 114, 111, 141, 116, 122, 114, 114, 141, 112, 141, 114, 141, 111, 111, 111, 113, 111, 114, 111, 141, 116, 111, 122, 117, 111, 111, 111],
	'out_rainy': [141, 122, 112, 114, 141, 140, 141, 140, 111, 112, 111, 113, 111, 114, 111, 114, 141, 114, 111, 116, 111, 114, 141, 111, 111, 141, 142],
	# 'out_replay': [50, 50, 50, 50, 50, 51, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
    'validate': [123, 122, 121, 121, 122, 122, 123, 123, 123, 122, 122, 123, 122, 122]}
bi3dof_simple_test_validate = {
    "model_file": "./snowy_models/bi3dof-simple-600epoch-141seq-seed2-4nd.pt",
    # "model_file": "./carla_models/bi3dof-simple-600epoch-6seq.pt",
    "network": "simple",
    "test_clips": "F:/lsm/time-series-OOD-main/data/data/carla_data/carla_features/calibration.calibration",
    "frames_per_clip": frame_lens['validate']
}
bi3dof_simple_test_in = {
    "model_file": "./snowy_models/bi3dof-simple-600epoch-141seq-seed2-4nd.pt",
    # "model_file": "./carla_models/bi3dof-simple-600epoch-6seq.pt",
    "network": "simple",
    "test_clips": "F:/lsm/data/carla_features/in.test",
    "frames_per_clip": frame_lens['in']
}


def getOutBi3DOF(type_of_OOD):
    features_folder = "F:/lsm/data/carla_features/"  # Change to "../NTU_features_rainy_only/" for rainy
    bi3dof_simple_test_out = {
        # "model_file" : "{}/bi3dof-simple-600epoch-6seq-seed{}.pt".format(model_save_folder, SEED), # "model/nuscenes-mini/bi3dof-simple-600epoch.pt",
        "model_file": "./snowy_models/bi3dof-simple-600epoch-141seq-seed2-4nd.pt",
        # "model_file": "./carla_models/bi3dof-simple-600epoch-6seq.pt",
        "network": "simple",
        "test_clips": features_folder + "{}.test".format(type_of_OOD),  # "data/nuscenes-v1.0-mini.test",
        "frames_per_clip": frame_lens[type_of_OOD]
    }
    return bi3dof_simple_test_out


def calc_cal_ce_loss():  # for calibration datapoint, we want one randomly sampled window for 1 datapoint

    ce_loss_list = []
    calc_save_path = f"./snowy_models/nc_calibration_vae_600epoch.npy"
    for idx, bi3dof_simple in enumerate([bi3dof_simple_test_validate]):  # i.e. for traces in [iD traces, OOD traces]
        # print("bi3dof_simple: ",bi3dof_simple
        model, args = load_model(bi3dof_simple)
        args.training=True
        h, v = compute_score(model, args)
        ce_loss_list = [h[i] + v[i] for i in range(0,len(h),8)]
    print("calc_nc.size:", len(ce_loss_list))
    np.save(calc_save_path, ce_loss_list)

def calc_p_value(test_ce_loss_value, cal_set_ce_loss):
    # 计算百分位数
    compare = 0;
    for i in range(len(cal_set_ce_loss)):
        if test_ce_loss_value <= (cal_set_ce_loss[i]):
            compare = compare+1;
    p_value = (compare+1) / ((len(cal_set_ce_loss))+1)
    return p_value



def run(type_of_OOD):

    calc_cal_ce_loss()
    calc_save_path = f"./snowy_models/nc_calibration_vae_600epoch.npy"
    # [print(item) for item in calc_loss]
    print('\n', type_of_OOD, '\n')
    scores_of_only_in_points = []
    scores_of_only_out_points = []
    iD_scores_all = []
    GTs_all = []

    for idx, bi3dof_simple in enumerate([bi3dof_simple_test_in, getOutBi3DOF(type_of_OOD)]):
        time_start = time.time()
        calc_cal_ce_loss()
        calc_save_path = f"./snowy_models/nc_calibration_vae_600epoch.npy"
        model, args = load_model(bi3dof_simple)
        h, v = compute_score(model, args)
        test_loss_list = [h[i] + v[i]  for i in range(len(h))]
        # print("test_loss_list:",len(test_loss_list))
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
        if ('in.test' in args.data_file):
            GT = 0
            scores_of_only_in_points.extend(iD_scores_windows)
            # np.savetxt('s_m_array_in.txt', m_array, fmt='%d %.6f', delimiter=' ', newline='\n')
            # np.savetxt('s_p_array_in.txt', p_array, fmt='%d %.6f', delimiter=' ', newline='\n')

        else:
            GT = 1
            scores_of_only_out_points.extend(iD_scores_windows)
            # np.savetxt('s_m_array_out.txt', m_array, fmt='%d %.6f', delimiter=' ', newline='\n')
            # np.savetxt('s_p_array_out.txt', p_array, fmt='%d %.6f', delimiter=' ', newline='\n')
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
    print(f'(precision, recall, f1,Fpr): ({precision}, {recall}, {f1},{fpr})')


if __name__ == "__main__":
    for type_of_OOD in ['out_rainy']:
        run(type_of_OOD)

