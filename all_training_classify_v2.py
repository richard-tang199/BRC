import argparse
import numpy as np
import tqdm
import torch
from evaluation.evaluator import evaluate
from toolkit.result_plot import recon_plot, get_segments
from model.one_to_all import PatchClassify, Patch_RWKV, PatchFrequency, PatchFrequencySeperate
from toolkit.utils import *
import json
import random
from dataclasses import dataclass, asdict
import torch.nn.init as init
from toolkit.get_anomaly_score import AnomalyScoreCalculator


seed = 42
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='freq', help='Name of the model')
parser.add_argument('--dataset_name', type=str, default='UCR', help='Name of the dataset')
parser.add_argument("--group_name", type=str, default=[1, 2, 4, 5, 11, 13, 15, 19, 24, 26, 27, 33, 37, 48, 55, 59, 65, 70, 72], help="group in the dataset")
parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs for training")
parser.add_argument("--lr", type=float, default=2e-3, help="learning rate for training")
parser.add_argument("--eval_gap", type=int, default=20, help="training epochs between evaluations")
parser.add_argument("--use_default_config", action='store_true', help="use default configuration for training")
parser.add_argument("--patch_size", type=int, default=8, help="length of the patch")

if __name__ == '__main__':

    args = parser.parse_args()
    # args.group_name = [args.group_name]
    print(f"Start training: {args.group_name}")
    train_data_list, test_data_list, test_label_list, subsequence_list = load_dataset(data_name=args.dataset_name,
                                                                                      group_name=args.group_name)
    use_train_data_list = train_data_list[:]
    train_subsequence_list = subsequence_list[:]

    if args.model_name == 'freq':
        model = PatchFrequency(patch_size=args.patch_size, expansion_ratio=2, num_layers=2)
    elif args.model_name == "raw":
        model = PatchClassify(patch_size=args.patch_size, expansion_ratio=2, num_layers=2)
    elif args.model_name == "sep":
        model = PatchFrequencySeperate(patch_size=args.patch_size, expansion_ratio=2, num_layers=2)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    multiple_list = [8 if len(train_data) // subsequence_length > 8 else len(train_data) // subsequence_length
                for train_data, subsequence_length in zip(train_data_list, subsequence_list)]
    use_train_data_list = [train_data[:, np.newaxis] for train_data in use_train_data_list]
    train_data_list = [train_data[:, np.newaxis] for train_data in train_data_list]

    train_window_length_list = [(multiple * subsequence_length) // args.patch_size * args.patch_size
                                for multiple, subsequence_length in zip(multiple_list, train_subsequence_list)]
    window_length_list = [(multiple * subsequence_length) // args.patch_size * args.patch_size
                          for multiple, subsequence_length in zip(multiple_list, subsequence_list)]


    train_loader = get_multiple_dataloader(
        train_data_list = use_train_data_list,
        batch_size=args.batch_size,
        window_length_list=train_window_length_list,
        train_stride=3
    )

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        for windows in train_loader:
            windows = [window.to(device) for window in windows]
            for i, window in enumerate(windows):
                loss = model(window)
                loss.backward()
                epoch_loss += loss.item()

            for param in model.parameters():
                if param.grad is not None:
                    param.grad /= len(windows)

            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

        scheduler.step()
        print(f"Epoch {epoch}: Loss {epoch_loss / (len(use_train_data_list) * len(train_loader))}")
        mean_loss = epoch_loss / (len(use_train_data_list) * len(train_loader))
        if mean_loss < 0.04:
            break

    val_anomaly_score_list = []
    val_window_converter_list = []
    for train_data, window_length in zip(train_data_list, window_length_list):
        val_loader, val_window_converter = get_dataloader(
            data=train_data,
            batch_size=args.batch_size,
            window_length=window_length,
            test_stride=window_length // 8
        )

        val_anomaly_score = None
        for batch_idx, (data, ) in enumerate(val_loader):
            data = data.to(device)
            with torch.no_grad():
                model.eval()
                predict_score = model.predict(data)
                if val_anomaly_score is None:
                    val_anomaly_score = predict_score
                else:
                    val_anomaly_score = torch.concat((val_anomaly_score, predict_score), dim=0)

        val_anomaly_score = val_anomaly_score.detach().cpu().numpy()
        val_anomaly_score = val_window_converter.windows_to_sequence(val_anomaly_score)
        val_anomaly_score[:val_window_converter.pad_length] = 0
        val_anomaly_score_list.append(val_anomaly_score)
        val_window_converter_list.append(val_window_converter)

    test_anomaly_score_list = []
    test_window_converter_list = []
    for i, (test_data, test_label, subsequence_length) in enumerate(zip(test_data_list, test_label_list, subsequence_list)):
        test_data = test_data[:, np.newaxis]
        window_length = subsequence_length * multiple_list[i]
        window_length = (window_length // args.patch_size) * args.patch_size
        test_loader, test_window_converter = get_dataloader(
            data=test_data,
            batch_size=args.batch_size,
            window_length=window_length,
            test_stride=subsequence_length
        )

        test_anomaly_score = None
        for batch_idx, (data, ) in enumerate(test_loader):
            data = data.to(device)
            with torch.no_grad():
                model.eval()
                predict_score = model.predict(data)
                if test_anomaly_score is None:
                    test_anomaly_score = predict_score
                else:
                    test_anomaly_score = torch.concat((test_anomaly_score, predict_score), dim=0)

        test_anomaly_score = test_anomaly_score.detach().cpu().numpy()
        test_anomaly_score = test_window_converter.windows_to_sequence(test_anomaly_score)
        test_anomaly_score_list.append(test_anomaly_score)
        test_window_converter_list.append(test_window_converter)

    anomaly_score_cal_list = [AnomalyScoreCalculator(
        mode = "error",
        average_window=subsequence_length
    ) for subsequence_length in subsequence_list]

    anomaly_score_list = [anomaly_score_cal.calculate_anomaly_score(
        raw_train_data=np.zeros_like(val_anomaly_score),
        raw_test_data=np.zeros_like(test_anomaly_score),
        recon_train_data=val_anomaly_score,
        recon_test_data=test_anomaly_score
    ) for anomaly_score_cal, val_anomaly_score, test_anomaly_score in zip(anomaly_score_cal_list,val_anomaly_score_list, test_anomaly_score_list)]


    val_score_list = []
    test_score_list = []
    for anomaly_score, val_window_converter, test_window_converter in zip(anomaly_score_list, val_window_converter_list, test_window_converter_list):
        val_score = anomaly_score.train_score_all
        val_score[:val_window_converter.pad_length] = 0
        val_score_list.append(val_score)
        test_score = anomaly_score.test_score_all
        test_score[:test_window_converter.pad_length] = 0
        test_score_list.append(test_score)

    save_dir = f"overall_results/{args.model_name}"

    for i, test_label in enumerate(args.group_name):
        eval_result = evaluate(
            ground_truth=test_label_list[i],
            anomaly_scores=test_score_list[i],
            subsequence=subsequence_list[i],
            use_ucr=True,
            mode="test"
        )

        result_path = os.path.join(save_dir, f"data/group_{args.group_name[i]}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(asdict(eval_result), f, indent=4)

        save_path = os.path.join(save_dir, f"figure/group_{args.group_name[i]}.png")
        recon_plot(
            save_path=save_path,
            gap= subsequence_list[i],
            figure_length=120,
            figure_width=10,
            train_data=train_data_list[i],
            test_data=test_data_list[i],
            test_label=test_label_list[i],
            test_anomaly_score=test_score_list[i],
            train_anomaly_score=val_score_list[i]
        )







