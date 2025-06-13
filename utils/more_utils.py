import numpy as np


# def make2D(scores_flattened, list_of_no_of_windows_in_traces):
# 	assert len(scores_flattened) == sum(list_of_no_of_windows_in_traces)
# 	curr = 0
# 	scores_2D = []
# 	for no_windows_in_trace in list_of_no_of_windows_in_traces:
# 		current_trace_scores = scores_flattened[curr:curr+no_windows_in_trace]
# 		scores_2D.append(current_trace_scores)
# 		curr += no_windows_in_trace
# 	return scores_2D

def make2D(scores_flattened, list_of_no_of_windows_in_traces):
    print("len(scores_flattened)", len(scores_flattened))
    print("sum(list_of_no_of_windows_in_traces)", sum(list_of_no_of_windows_in_traces))
    assert len(scores_flattened) == sum(list_of_no_of_windows_in_traces)
    curr = 0
    scores_2D = []
    for no_windows_in_trace in list_of_no_of_windows_in_traces:
        current_trace_scores = scores_flattened[curr:curr + no_windows_in_trace]
        scores_2D.append(current_trace_scores)
        curr += no_windows_in_trace
    return scores_2D


def OOD_score_to_iD_score(list_2D):
    return [[1 * i for i in row] for row in list_2D]


def OOD_to_iD(float_list):
    return [-1 * x for x in float_list]


def min_of_each_row(list_2D):
    return [min(row) for row in list_2D]


def collapse_to_1D(list_2D):
    list_1D = []
    for row in list_2D:
        list_1D.extend(row)
    return list_1D


def compute_epsilon_on_iD_traces_only(iD_scores_all, GTs_all, TPR=0.95):
    only_iD_traces_scores = []
    for trace_idx, trace_GT in enumerate(GTs_all):
        if trace_GT == 1:  # i.e if iD
            only_iD_traces_scores.append(iD_scores_all[trace_idx])
    # sort in descending order
    only_iD_traces_scores = np.array(sorted(only_iD_traces_scores, reverse=True))
    n_traces = len(only_iD_traces_scores)
    epsilon = only_iD_traces_scores[int(TPR * n_traces) - 1]
    print('No of iD points: {} | TPR {} | location = int(TPR x n_pts) - 1: {} | epsilon: {}'.format(n_traces, TPR,
                                                                                                    int(TPR * n_traces) - 1,
                                                                                                    epsilon))
    return epsilon


def getTNR(in_scores, out_scores):
    in_scores, out_scores = np.array(in_scores), np.array(out_scores)
    # print("in_scores, out_scores: ",in_scores, out_scores)
    # in_fisher = np.sort(out_scores)[::-1]  # sorting in descending order
    tau = np.sort(out_scores)[::-1][int(0.95 * len(out_scores))]
    # tau =  20.35117639512095
    tnr = 100 * (len(in_scores[in_scores < tau]) / len(in_scores))
    return tnr, tau

def get_det_delay_for_detected_traces(scores_2D_list, tau):
   det_delays = []
   det_delays_test = []
   for trace_idx, row in enumerate(scores_2D_list):
       # print("row:",row)
       for window_idx, val in enumerate(row):
           det_delays.append(window_idx)
           if val>tau:
               det_delays.append(window_idx)
               break
   print("det_delays", sum(det_delays))
   print("scores_2D_list.length: ",len(scores_2D_list)*6)
   # avg_det_delay = sum(det_delays)/(len(scores_2D_list)*6)
   avg_det_delay = sum(det_delays) / (len(det_delays))
   # print("det_delays: ", det_delays)
   return avg_det_delay
# def get_det_delay_for_detected_traces(iD_scores_2D_list_of_OOD_traces_only, tau):
#     det_delay = []
#     for iD_scores_list in iD_scores_2D_list_of_OOD_traces_only:
#         detected = False
#         delay = None
#         for t, score in enumerate(iD_scores_list):
#             if score >= tau:
#                 detected = True
#                 delay = t
#                 break
#         if detected:
#             det_delay.append(delay)
#     avg_det_delay = np.mean(det_delay)
#     return avg_det_delay


def scan_iD_scores_of_windows_and_print_list(iD_scores_2D_list, epsilon):
    det_delays = []
    not_detected_as_OOD = 0
    detected_as_OOD = 0
    for row in iD_scores_2D_list:
        for window_idx, val in enumerate(row):
            if val < epsilon:
                detected_as_OOD += 1
                det_delays.append(window_idx)
                break
            if window_idx == len(row) - 1:
                not_detected_as_OOD += 1
                det_delays.append(-1)
    TN = detected_as_OOD
    FP = not_detected_as_OOD
    TNR = TN / (TN + FP)
    print('Window detected idx: ', det_delays)
    return -1, TNR


def getPrecisionRecallF1(in_scores, out_scores):
    in_scores, out_scores = np.array(in_scores), np.array(out_scores)
    # print("in_scores:",in_scores)
    # print("out_scores:",out_scores)

    # print("len(in_scores):",len(in_scores))
    # print("len(out_scores): ",len(out_scores))
    # Calculate precision
    # threshold = np.sort(in_scores)[::-1][int(0.95 * len(in_scores))]  # Threshold at 95% TPR
    # threshold = np.sort(out_scores)[::-1][int(0.95 * len(out_scores))]  #
    threshold = np.sort(out_scores)[int(0.05 * len(out_scores))]
    # threshold = 10.32919068
    print("threshold:",threshold)
    # threshold = 32.41698047
    false_positives = len(in_scores[in_scores >= threshold])
    true_positives = len(out_scores[out_scores >= threshold])
    precision = true_positives / (true_positives + false_positives)
    false_negatives = len(out_scores)-true_positives
    true_negatives = len(in_scores)-false_positives
    fnr = false_negatives / len(out_scores)
    fpr = false_positives / len(in_scores)


    # Calculate recall (sensitivity)
    total_positives = len(out_scores)
    recall = true_positives / total_positives

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    true_negatives = len(out_scores[out_scores < threshold])
    total_samples = len(in_scores) + len(out_scores)
    accuracy = (true_positives + true_negatives) / total_samples

    return precision, recall, f1, fpr, fnr




# def getPrecisionRecallF1(in_scores, out_scores):
#     in_scores, out_scores = np.array(in_scores), np.array(out_scores)
#     print("in_scores:",in_scores)
#     print("out_scores:",out_scores)
#
#     # print("len(in_scores):",len(in_scores))
#     # print("len(out_scores): ",len(out_scores))
#     # Calculate precision
#     # threshold = np.sort(in_scores)[::-1][int(0.95 * len(in_scores))]  # Threshold at 95% TPR
#     threshold = np.sort(in_scores)[int(0.95 * len(in_scores))]  #
#     print("threshold:",threshold)
#     true_positives = len(in_scores[in_scores < threshold])
#     false_positives = len(out_scores[out_scores < threshold])
#     precision = true_positives / (true_positives + false_positives)
#     false_negatives = len(in_scores[in_scores > threshold])
#     true_negatives = len(in_scores[in_scores < threshold])
#     fpr = false_positives / len(out_scores)
#     fnr = false_negatives/ len(in_scores)
#
#     # Calculate recall (sensitivity)
#     total_positives = len(out_scores)
#     recall = true_positives / total_positives
#
#     # Calculate F1 score
#     f1 = 2 * (precision * recall) / (precision + recall)
#     true_negatives = len(out_scores[out_scores < threshold])
#     total_samples = len(in_scores) + len(out_scores)
#     accuracy = (true_positives + true_negatives) / total_samples
#
#     return precision, recall, f1, fpr,fnr



def get_binary_labels_with_95_TPR(scores, true_labels):
    # Assuming scores are anomaly scores (higher score means more likely to be an anomaly)
    sorted_indices = np.argsort(scores)[::-1]
    num_anomalies = np.sum(true_labels)  # Count the number of actual anomalies

    # Find the threshold that corresponds to 95% TPR
    threshold_index = int(0.95 * num_anomalies)
    threshold_score = scores[sorted_indices[threshold_index]]

    # Convert anomaly scores to binary labels based on the threshold
    binary_labels = np.zeros_like(true_labels)
    binary_labels[scores > threshold_score] = 1

    return binary_labels
