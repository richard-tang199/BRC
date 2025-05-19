# -*-coding:utf-8-*-
import numpy as np
from toolkit.result_plot import get_segments
from torch.utils.data import Dataset


def get_retrain_windows(test_data,
                        test_anomaly_scores,
                        window_length,
                        test_labels):
    """
    Parameters
    ----------
    test_data
    test_anomaly_scores
    window_length
    test_labels
    Returns
    -------
    retrain_windows
    window_labels
    """
    test_threshold = np.mean(test_anomaly_scores) + 3 * np.std(test_anomaly_scores)
    all_alarms = (test_anomaly_scores > test_threshold).astype(int)
    true_alarms = get_segments(test_labels)
    test_anomaly_segments = get_segments(all_alarms)

    false_alarms = []
    for seg in test_anomaly_segments:
        flag = 0
        for true_seg in true_alarms:
            if seg[1] <= true_seg[0] or seg[0] >= true_seg[1]:
                continue
            else:
                flag = 1

        if flag == 0:
            false_alarms.append(seg)

    false_negatives = []
    for seg in true_alarms:
        flag = 0
        for anomaly_seg in test_anomaly_segments:
            if seg[1] <= anomaly_seg[0] or seg[0] >= anomaly_seg[1]:
                continue
            else:
                flag = 1

        if flag == 0:
            false_negatives.append(seg)

    retrain_windows = []
    window_labels = []

    for false_alarm in false_alarms:
        center = (false_alarm[0] + false_alarm[1]) // 2

        if center - window_length // 2 < 0:
            center = window_length // 2
        elif center + window_length // 2 > len(test_data):
            center = len(test_data) - window_length // 2 - 1

        train_window = test_data[center - window_length // 2: center + window_length // 2]
        train_label = test_labels[center - window_length // 2: center + window_length // 2]
        window_label = 1 if np.sum(train_label) > 0 else 0
        retrain_windows.append(train_window)
        window_labels.append(train_label)

    # for false_negative in false_negatives:
    #     center = (false_negative[0] + false_negative[1]) // 2
    #
    #     if center - window_length // 2 < 0:
    #         center = window_length // 2
    #     elif center + window_length // 2 > len(test_data):
    #         center = len(test_data) - window_length // 2 - 1
    #
    #     train_window = test_data[center - window_length // 2: center + window_length // 2]
    #     train_label = test_labels[center - window_length // 2: center + window_length // 2]
    #     window_label = 1 if np.sum(train_label) > 0 else 0
    #     retrain_windows.append(train_window)
    #     window_labels.append(train_label)

    retrain_windows = np.array(retrain_windows, dtype=np.float32)

    return retrain_windows, window_labels


class RetrainDataset(Dataset):
    def __init__(self, retrain_windows, labels):
        self.retrain_windows = retrain_windows
        self.labels = labels

    def __len__(self):
        return len(self.retrain_windows)

    def __getitem__(self, idx):
        return self.retrain_windows[idx], self.labels[idx]








