import argparse
from evaluation.evaluator import evaluate
# from toolkit.result_plot import recon_plot, get_segments
from toolkit.utils import *
import json
from dataclasses import dataclass, asdict
# from toolkit.get_anomaly_score import AnomalyScoreCalculator
from baseline.norma import NORMA
from baseline.series2graph import Series2Graph
from sklearn.preprocessing import MinMaxScaler
from baseline.ano_kmeans import KMeansAD
from baseline.damp import DAMP
from baseline.pca import PCA
from baseline.sand import SAND
import math
import stumpy
from numba import cuda
from evaluation.evaluator import EfficiencyResult
import time

seed = 42

# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='PCA', help='Name of the model')
parser.add_argument('--dataset_name', type=str, default='UCR', help='Name of the dataset')
parser.add_argument("--group_name", type=str, default="001", help="group in the dataset")
parser.add_argument("--plot", type=str, default="False", help="whether to plot the results")

if __name__ == '__main__':
    args = parser.parse_args()
    args.group_name = [args.group_name]
    save_dir = f"baseline_result/{args.dataset_name}/{args.model_name}"

    if args.plot.lower() == "true":
        args.plot = True
    elif args.plot.lower() == "false":
        args.plot = False
    else:
        raise ValueError("Invalid value for plot")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/figure", exist_ok=True)
    os.makedirs(f"{save_dir}/data", exist_ok=True)
    os.makedirs(f"{save_dir}/anomaly_score", exist_ok=True)
    os.makedirs(f"{save_dir}/efficiency", exist_ok=True)

    # args.group_name = [args.group_name]
    print(f"Start training: {args.group_name}, model_name: {args.model_name}")
    train_data_list, test_data_list, test_label_list, subsequence_list = load_dataset(data_name=args.dataset_name,
                                                                                      group_name=args.group_name)

    train_data = train_data_list[0]
    subsequence_length = subsequence_list[0]
    test_data = test_data_list[0]
    test_label = test_label_list[0]

    if args.model_name == 'KshapeAD':
        model = NORMA(pattern_length=subsequence_list[0], nm_size=3 * subsequence_list[0], clustering="kshape")
        new_test_data = test_data.copy()
        start_time = time.time()
        model.fit(new_test_data)
        duration = time.time() - start_time
        efficiency_result = EfficiencyResult(duration=duration)
        test_anomaly_score = model.decision_scores_
        test_anomaly_score = np.array([test_anomaly_score[0]] * ((subsequence_length - 1) // 2)
                                      + list(test_anomaly_score) + [test_anomaly_score[-1]] * (
                                              (subsequence_length - 1) // 2))
        test_anomaly_score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
            test_anomaly_score.reshape(-1, 1)).ravel()

    elif args.model_name == 'Series2Graph':
        model = Series2Graph(pattern_length=subsequence_list[0])
        new_test_data = test_data.copy()
        start_time = time.time()
        model.fit(new_test_data)
        query_length = 2 * subsequence_length
        model.score(query_length=2 * subsequence_length, dataset=new_test_data)
        test_anomaly_score = model.decision_scores_
        duration = time.time() - start_time
        efficiency_result = EfficiencyResult(duration=duration)
        test_anomaly_score = np.array(
            [test_anomaly_score[0]] * math.ceil(query_length // 2) + list(test_anomaly_score) + [
                test_anomaly_score[-1]] * (query_length // 2))
        test_anomaly_score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
            test_anomaly_score.reshape(-1, 1)).ravel()

    elif args.model_name == "Kmeans":
        model = KMeansAD(k=20, window_size=subsequence_length, stride=1)
        start_time = time.time()
        test_anomaly_score = model.fit_predict(test_data)
        duration = time.time() - start_time
        efficiency_result = EfficiencyResult(duration=duration)
        test_anomaly_score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
            test_anomaly_score.reshape(-1, 1)).ravel()

    elif args.model_name == "damp":
        model = DAMP(m=subsequence_length, sp_index=subsequence_length + 1)
        start_time = time.time()
        model.fit(test_data)
        duration = time.time() - start_time
        efficiency_result = EfficiencyResult(duration=duration)
        test_anomaly_score = model.decision_scores_
        score = test_anomaly_score
        slidingWindow = subsequence_length
        test_anomaly_score = np.array(
            [score[0]] * math.ceil((slidingWindow - 1) / 2) + list(score) + [score[-1]] * ((slidingWindow - 1) // 2))
        test_anomaly_score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
            test_anomaly_score.reshape(-1, 1)).ravel()

    elif args.model_name == "MP":
        all_gpu_devices = [device.id for device in cuda.list_devices()]
        test_data = np.squeeze(test_data)
        start_time = time.time()
        test_anomaly_score = stumpy.gpu_stump(test_data, m=subsequence_length, device_id=all_gpu_devices)[:, 0]
        duration = time.time() - start_time
        efficiency_result = EfficiencyResult(duration=duration)
        test_anomaly_score = np.array(
            [test_anomaly_score[0]] * math.ceil((subsequence_length - 1) / 2) + list(test_anomaly_score) + [
                test_anomaly_score[-1]] * ((subsequence_length - 1) // 2))

    elif args.model_name == "PCA":
        model = PCA(
            slidingWindow=subsequence_length,
            n_components=None
        )
        test_data = test_data[:, np.newaxis]
        start_time = time.time()
        model.fit(test_data)
        duration = time.time() - start_time
        efficiency_result = EfficiencyResult(duration=duration)
        test_anomaly_score = model.decision_scores_
        test_anomaly_score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
            test_anomaly_score.reshape(-1, 1)).ravel()

    elif args.model_name == "SAND":
        clf = SAND(pattern_length=subsequence_length, subsequence_length=4 * subsequence_length)
        start_time = time.time()
        clf.fit(test_data, overlaping_rate=int(1.5 * subsequence_length),
                init_length=30 * subsequence_length, alpha=0.5,
                batch_size=10 * subsequence_length)
        duration = time.time() - start_time
        efficiency_result = EfficiencyResult(duration=duration)
        test_anomaly_score = clf.decision_scores_
        test_anomaly_score = MinMaxScaler(feature_range=(0, 1)).fit_transform(test_anomaly_score.reshape(-1, 1)).ravel()

    else:
        raise ValueError("Invalid model name")

    if args.dataset_name == "TSB-AD-U":
        subsequence_length = None

    use_ucr = False
    if args.dataset_name == "UCR":
        use_ucr = True

    eval_result = evaluate(ground_truth=test_label,
                           anomaly_scores=test_anomaly_score,
                           subsequence=subsequence_length,
                           use_ucr=use_ucr,
                           mode="test")

    # test_threshold = np.mean(test_anomaly_score) + 3 * np.std(test_anomaly_score)
    # anomaly_index = test_anomaly_score > test_threshold
    # test_anomaly_segments = get_segments(anomaly_index)
    group_name = str(args.group_name[0]).zfill(3)

    result_path = os.path.join(save_dir, f"data/group_{group_name}.json")
    efficiency_path = os.path.join(save_dir, f"efficiency/group_{group_name}.json")

    with open(efficiency_path, "w", encoding="utf-8") as f:
        json.dump(asdict(efficiency_result), f, indent=4)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(asdict(eval_result), f, indent=4)

    # np.save(os.path.join(save_dir, f"anomaly_score/group_{group_name}.npy"), test_anomaly_score)
    # save_path = os.path.join(save_dir, f"figure/group_{group_name}.png")
    #
    # if args.plot:
    #     recon_plot(
    #         save_path=save_path,
    #         gap=subsequence_list[0],
    #         figure_width=10,
    #         figure_length=120,
    #         train_data=train_data,
    #         test_data=test_data,
    #         test_label=test_label,
    #         threshold=test_threshold,
    #         test_anomaly_score=test_anomaly_score
    #     )
