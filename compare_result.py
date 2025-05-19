# -*-coding:utf-8-*-
import os
import numpy as np
import json
import pandas as pd


def read_data(data_dir):
    data_group = os.listdir(data_dir)
    data_group.sort()
    # data_group = data_group[:121]
    result_list = []

    for data in data_group:
        try:
            # path = os.path.join("new_results", data, "result.json")
            result = json.load(open(os.path.join(data_dir, data), "r"))
            result_list.append(result)
        except FileNotFoundError:
            print(f"{data} not found")
            continue

    ucr_1 = [result["ucr"]["ucr_01"] for result in result_list if result["ucr"] is not None]
    ucr_2 = [result["ucr"]["ucr_03"] for result in result_list if result["ucr"] is not None]
    ucr_3 = [result["ucr"]["ucr_05"] for result in result_list if result["ucr"] is not None]

    return np.array(ucr_1), np.array(ucr_2), np.array(ucr_3)


data_path_a = "mask_results//UCR//freq_tf_v2//classify//data"
data_path_b = "mask_results//UCR//freq_tf_v2//recon//data"

ucr_1_a, ucr_2_a, ucr_3_a = read_data(data_path_a)
ucr_1_b, ucr_2_b, ucr_3_b = read_data(data_path_b)

minus_1 = ucr_1_a - ucr_1_b
indices = np.where(minus_1 < 0)[0]
indices += 1
print(indices)

