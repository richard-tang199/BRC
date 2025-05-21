# -*-coding:utf-8-*-
import pandas as pd
import argparse
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
import torch

from evaluation.evaluator import evaluate, EfficiencyResult
from toolkit.result_plot import recon_plot, get_segments
from toolkit.utils import *
import time
import json
from dataclasses import dataclass, asdict
from toolkit.get_anomaly_score import AnomalyScoreCalculator
from baseline.MTFAE import MTFA
# from baseline.CutAddPaste import CutAddPaste
from baseline.COUTA import COUTA
# from baseline.FCVAE import MyVAE
# from baseline.ts_benchmark.baselines.catch.CATCH import CATCH
# from baseline.ts_benchmark.baselines.self_impl.TFAD.TFAD import TFAD
from baseline.DADA import DADA
from baseline.M2N2 import M2N2
import thop
import time



torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='moment', help='Name of the model')
parser.add_argument('--dataset_name', type=str, default='UCR', help='Name of the dataset')
parser.add_argument("--group_name", type=str, default="004", help="group in the dataset")
parser.add_argument("--plot", type=str, default="True", help="whether to plot the results")
parser.add_argument("--device", type=str, default='cuda:0', help="device for training")
parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproducibility")

if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.dataset_name == "UCR":
        use_ucr = True
    else:
        use_ucr = False

    if args.plot.lower() == "true":
        args.plot = True
    elif args.plot.lower() == "false":
        args.plot = False
    else:
        raise ValueError("Invalid value for plot")

    args.group_name = [args.group_name]
    save_dir = f"baseline_result/{args.dataset_name}/{args.model_name}/{args.random_seed}"
    print(f"\n model name: {args.model_name}, data name: {args.dataset_name}_{args.group_name}, device: {device}\n")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/figure", exist_ok=True)
    os.makedirs(f"{save_dir}/data", exist_ok=True)
    os.makedirs(f"{save_dir}/anomaly_score", exist_ok=True)

    train_data_list, test_data_list, test_label_list, subsequence_list = load_dataset(data_name=args.dataset_name,
                                                                                      group_name=args.group_name)

    train_data = train_data_list[0]
    subsequence_length = subsequence_list[0]
    test_data = test_data_list[0]
    test_label = test_label_list[0]

    if len(train_data) < 1000:
        train_stride = 1
    elif 1000 <= len(train_data) < 10000:
        train_stride = 2
    elif 10000 <= len(train_data) < 60000:
        train_stride = 5
    elif 60000 <= len(train_data) < 100000:
        train_stride = args.patch_size
    else:
        train_stride = subsequence_length

    if len(train_data.shape) == 1:
        train_data = train_data[:, np.newaxis]
        test_data = test_data[:, np.newaxis]

    if args.model_name == "cutaddpaste":
        batch_size = 256
        window_length = 64
        stride = 1
        model = CutAddPaste(device=args.device)
        model.to(device)
        flops, params = model.fit(train_data, epochs=1)
        test_loader, _ = get_dataloader(
            data=test_data,
            batch_size=batch_size,
            window_length=window_length,
            mode="test",
            test_stride=stride
        )
        start_time = time.time()
        test_anomaly_score = model.predict(test_loader)
        duration = time.time() - start_time
        test_label = test_label[:len(test_anomaly_score)]
        test_anomaly_score = test_anomaly_score[:len(test_label)]
        test_data = test_data[:len(test_anomaly_score)]
        eval_result = evaluate(ground_truth=test_label,
                               anomaly_scores=test_anomaly_score,
                               subsequence=subsequence_length,
                               use_ucr=use_ucr,
                               mode="test")

    elif args.model_name == "couta":
        model_configs = {'sequence_length': 64, 'stride': train_stride, 'num_epochs':1}
        model = COUTA(**model_configs)
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
        model.fit(train_data)
        start_time = time.time()
        score_dict = model.predict(test_data)
        flops, params = score_dict["flops"], score_dict["params"]
        duration = time.time() - start_time
        test_anomaly_score = score_dict["score_t"]
        eval_result = evaluate(ground_truth=test_label,
                               anomaly_scores=test_anomaly_score,
                               subsequence=subsequence_length,
                               use_ucr=use_ucr,
                               mode="test")

    elif args.model_name == "fcvae":
        model = MyVAE()
        model.to(device)
        flops, params = model.fit(train_data, device=device, num_epoch=1)
        start_time = time.time()
        test_anomaly_score, test_label = model.predict(test_data, test_label, device=device)
        duration = time.time() - start_time
        eval_result = evaluate(ground_truth=test_label,
                               anomaly_scores=test_anomaly_score,
                               subsequence=subsequence_length,
                               use_ucr=use_ucr,
                               mode="test")

    elif args.model_name == "catch":
        model = CATCH()
        model.config.train_stride = train_stride
        train_data = np.concatenate([train_data, train_data], axis=1)
        test_data = np.concatenate([test_data, test_data], axis=1)
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)

        start_time = time.time()
        flops, params = model.detect_fit(train_data, test_data, epochs = 1)
        test_anomaly_score = model.detect_score(test_data)
        duration = time.time() - start_time
        test_label = test_label[:len(test_anomaly_score)]
        eval_result = evaluate(ground_truth=test_label,
                               anomaly_scores=test_anomaly_score,
                               subsequence=subsequence_length,
                               use_ucr=use_ucr,
                               mode="test")
        train_data = train_data.iloc[:, 0].values
        test_data = test_data.iloc[:, 0].values
        test_data = test_data[:len(test_anomaly_score)]

    elif args.model_name == "tfad":
        model = TFAD()
        model.config.train_stride = train_stride
        model.config.num_epochs = 1
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
        model.detect_fit(train_data)
        start_time = time.time()
        test_anomaly_score = model.detect_score(test_data)
        flops, params = None, None
        duration = time.time() - start_time
        test_label = test_label[:len(test_anomaly_score)]
        eval_result = evaluate(ground_truth=test_label,
                               anomaly_scores=test_anomaly_score,
                               subsequence=subsequence_length,
                               use_ucr=use_ucr,
                               mode="test")
        train_data = train_data.values
        test_data = test_data.values
        test_data = test_data[:len(test_anomaly_score)]

    elif args.model_name == "dada":
        model = DADA()
        test_loader, test_window_converter = get_dataloader(data=test_data,
                                                            batch_size=model.args.batch_size,
                                                            window_length=model.args.seq_len,
                                                            test_stride=model.args.seq_len)
        start_time = time.time()
        test_anomaly_score= model.zero_shot(test_loader)
        duration = time.time() - start_time
        test_anomaly_score = test_anomaly_score[:, :, np.newaxis]
        test_anomaly_score = test_window_converter.windows_to_sequence(test_anomaly_score)
        test_anomaly_score = test_anomaly_score.squeeze()

        eval_result = evaluate(ground_truth=test_label,
                               anomaly_scores=test_anomaly_score,
                               subsequence=subsequence_length,
                               use_ucr=use_ucr,
                               mode="test")

    elif args.model_name == "moment":
        from baseline.MOMENT import MOMENT

        model = MOMENT(device=device)
        start_time = time.time()
        test_anomaly_score, flops, params = model.zero_shot(test_data)
        duration = time.time() - start_time
        eval_result = evaluate(ground_truth=test_label,
                               anomaly_scores=test_anomaly_score,
                               subsequence=subsequence_length,
                               use_ucr=use_ucr,
                               mode="test")

    elif args.model_name == "m2n2":
        model = M2N2(device=device, epochs=1)
        model.fit(train_data)
        start_time = time.time()
        test_anomaly_score, flops, params = model.decision_function(test_data)
        duration = time.time() - start_time
        eval_result = evaluate(ground_truth=test_label,
                               anomaly_scores=test_anomaly_score,
                               subsequence=subsequence_length,
                               use_ucr=use_ucr,
                               mode="test")

    elif args.model_name == "TFMAE":
        model = MTFA(win_size=100, seq_size=10, c_in=1, c_out=1, d_model=128,
                     e_layers=3, fr=0.4, tr=0.25, dev = args.device)
        train_loader, _ = get_dataloader(data=train_data,
                                         batch_size=256,
                                         window_length=100,
                                         train_stride=train_stride,
                                         mode="train")
        test_loader, test_window_converter = get_dataloader(data=test_data,
                                                            batch_size=256,
                                                            window_length=100,
                                                            test_stride=100,
                                                            mode="test")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # model.fit(train_loader, optimizer, epochs=100)
        # model = model.to(device)
        start_time = time.time()
        predict_windows_score, flops, params = model.predict(test_loader)
        duration = time.time() - start_time
        test_anomaly_score = test_window_converter.windows_to_sequence(predict_windows_score)
        test_anomaly_score = test_anomaly_score.squeeze()
        # eval_result = evaluate(ground_truth=test_label,
        #                        anomaly_scores=test_anomaly_score,
        #                        subsequence=subsequence_length,
        #                        use_ucr=use_ucr,
        #                        mode="test")

    else:
        raise ValueError("Invalid model name")

    args.group_name = args.group_name[0].zfill(3)

    # efficiency_result = EfficiencyResult(
    #     flops=flops,
    #     params=params,
    #     duration=duration)
    #
    # os.makedirs(f"{save_dir}/efficiency", exist_ok=True)
    # with open(os.path.join(save_dir, f"efficiency/group_{args.group_name}.json"), "w") as f:
    #     json.dump(asdict(efficiency_result), f, indent=4)

    np.save(os.path.join(save_dir, f"anomaly_score/group_{args.group_name}.npy"), test_anomaly_score)

    with open(os.path.join(save_dir, f"data/group_{args.group_name}.json"), "w") as f:
        json.dump(asdict(eval_result), f, indent=4)

    # save_path = os.path.join(save_dir, f"figure/group_{args.group_name}.png")
    # if args.plot:
    #     recon_plot(
    #         save_path=save_path,
    #         gap=subsequence_length,
    #         figure_width=10,
    #         figure_length=120,
    #         train_data=train_data,
    #         test_data=test_data,
    #         test_label=test_label,
    #         test_anomaly_score=test_anomaly_score
    #     )
