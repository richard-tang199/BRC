from toolkit.utils import *
import pickle

dataset_name = "TSB-AD-U"

if "TSB" in dataset_name:
    group_name = list(range(225, 257)) + list(range(260, 277)) + list(range(287, 302))
elif dataset_name == "UCR":
    group_name = list(range(1, 251))
else:
    raiser("Invalid dataset name")

train_data_list, test_data_list, test_label_list, subsequence_list = load_dataset(data_name=dataset_name,
                                                                                  group_name=group_name)

train_data_list_new = [data[: int(0.9 * len(data))] for data in train_data_list]
valid_data_list_new = [data[int(0.9 * len(data)):] for data in train_data_list]


train_data_dict = {str(i + 1).zfill(3): data for i, data in enumerate(train_data_list_new)}
valid_data_dict = {str(i + 1).zfill(3): data for i, data in enumerate(valid_data_list_new)}
test_data_dict = {str(i + 1).zfill(3): data for i, data in enumerate(test_data_list)}
test_label_dict = {str(i + 1).zfill(3): label for i, label in enumerate(test_label_list)}
subsequence_dict = {str(i + 1).zfill(3): subsequence for i, subsequence in enumerate(subsequence_list)}

data_set = {
    "train_data": train_data_dict,
    "valid_data": valid_data_dict,
    "test_data": test_data_dict,
    "test_labels": test_label_dict,
    "subsequence_length": subsequence_dict
}

if "TSB" in dataset_name:
    pickle.dump(data_set, open("tsb_data.pt", "wb"))
elif dataset_name == "UCR":
    pickle.dump(data_set, open("ucr_data.pt", "wb"))
else:
    raiser("Invalid dataset name")

