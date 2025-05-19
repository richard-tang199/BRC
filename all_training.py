# -*-coding:utf-8-*-
import argparse
from textwrap import indent

import torch
import numpy as np
import tqdm
from baselines.configs import *
from evaluation.evaluator import evaluate
from toolkit.result_plot import recon_plot
from toolkit.training import all_training
from toolkit.utils import *
from model.one_to_all import PatchDetectorGru, PatchDetectorAttention
import matplotlib.pyplot as plt
import json
import random
from dataclasses import dataclass, asdict

seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='all', help='Name of the model')
parser.add_argument('--dataset_name', type=str, default='UCR', help='Name of the dataset')
parser.add_argument("--group_name", type=list, default=list(range(1,251)), help="group in the dataset")
parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs for training")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for training")
parser.add_argument("--eval_gap", type=int, default=20, help="training epochs between evaluations")
parser.add_argument("--use_default_config", action='store_true', help="use default configuration for training")
parser.add_argument("--patch_size", type=int, default=16, help="length of the patch")

if __name__ == '__main__':
    args = parser.parse_args()
    train_data_list, test_data_list, test_label_list, subsequence_list = load_dataset(data_name=args.dataset_name,
                                                                                      group_name=args.group_name)
    use_train_data_list = train_data_list[:]
    train_subsequence_list = subsequence_list[:]

    model = PatchDetectorGru(patch_size=args.patch_size, expansion_ratio=6, num_layers=6)
    # model = PatchDetectorAttention(patch_size=16, expansion_ratio=6, num_layers=6)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    loss_dict = {key: [] for key in args.group_name}
    results = []

    for epoch in tqdm.tqdm(range(args.num_epochs)):
        loss_list = []
        model.train()
        for train_data, subsequence_length in zip(use_train_data_list, train_subsequence_list):
            train_data = train_data[:, np.newaxis]
            window_length = 8 * subsequence_length
            window_length = (window_length // args.patch_size) * args.patch_size
            try:
                train_loader, _ = get_dataloader(data=train_data, batch_size=args.batch_size,
                                                 window_length=window_length,
                                                 mode="train",
                                                 train_stride=subsequence_length // 16 + 1)
            except:
                print(subsequence_length)

            for batch_idx, (data, ) in enumerate(train_loader):
                data = data.to(device)
                loss = model(data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

        scheduler.step()
        print("Epoch: {}, Loss: {:.4f}".format(epoch, np.mean(loss_list)))

        if (epoch + 1) % args.eval_gap == 0 or epoch == args.num_epochs - 1:
            model.eval()
            for i, (test_data, test_label, subsequence_length) in enumerate(zip(test_data_list, test_label_list, subsequence_list)):
                current_group = args.group_name[i]
                os.makedirs(f"results/group_{args.group_name[i]}", exist_ok=True)
                test_data = test_data[:, np.newaxis]
                window_length = 4 * subsequence_length
                window_length = (window_length // args.patch_size) * args.patch_size
                test_loader, test_window_converter = get_dataloader(data=test_data,
                                                                    batch_size=args.batch_size,
                                                                    window_length=window_length,
                                                                    test_stride=subsequence_length
                                                                    )
                recon_out = None
                for batch_idx, (data, ) in enumerate(test_loader):
                    data = data.to(device)
                    recon = model.predict(data)
                    if recon_out is None:
                        recon_out = recon
                    else:
                        recon_out = torch.concat([recon_out, recon], dim=0)

                recon_out = recon_out.detach().cpu().numpy()
                recon_out = test_window_converter.windows_to_sequence(recon_out)

                group_epoch_loss = np.sqrt(np.mean((recon_out - test_data) ** 2))
                loss_dict[current_group].append(group_epoch_loss)

                save_path = f"results/group_{args.group_name[i]}/epoch_{epoch}.png"
                recon_plot(save_path=save_path,
                           gap=subsequence_length,
                           figure_width=10,
                           figure_length=120,
                           train_data=train_data_list[i],
                           test_data=test_data,
                           test_label=test_label,
                           recon_test_data=recon_out
                           )

                if epoch == args.num_epochs - 1:
                    try:
                        anomaly_scores = anomaly_score_func(raw_value=test_data,
                                                            predict_value=recon_out,
                                                            subsequence=subsequence_length)
                        eval_result = evaluate(ground_truth=test_label,
                                               anomaly_scores=anomaly_scores,
                                               subsequence=subsequence_length,
                                               use_ucr=True,
                                               mode="test")
                        results.append(eval_result)

                        with open(f"results/group_{args.group_name[i]}/result.json", "w", encoding="utf-8") as f:
                            json.dump(asdict(eval_result), f, indent=4)

                        save_path = f"results/group_{args.group_name[i]}/epoch_{epoch}.png"
                        recon_plot(save_path=save_path,
                                   gap=subsequence_length,
                                   figure_width=10,
                                   figure_length=120,
                                   train_data=train_data_list[i],
                                   test_data=test_data,
                                   test_label=test_label,
                                   recon_test_data=recon_out,
                                   test_anomaly_score = anomaly_scores
                                   )
                    except:
                        continue

    torch.save(model.state_dict(), f"models/{args.model_name}.pth")
    os.makedirs(f"results/all_groups", exist_ok=True)

    with open(f"results/all_groups/loss_dict.json", "w") as f:
        json.dump(loss_dict, f, indent=4)

    average_f1_result = [result.point_wise["f1_score"] for result in results]
    average_pr_score = [result.point_wise["auc_prc"] for result in results]
    average_vus_score = [result.vus["VUS_PR"] for result in results]
    average_result = {
        "f1_score": np.mean(average_f1_result),
        "auc_prc": np.mean(average_pr_score),
        "VUS_PR": np.mean(average_vus_score)
    }
    with open(f"results/all_groups/average_result.json", "w") as f:
        json.dump(average_result, f, indent=4)

    for key, value in loss_dict.items():
        plt.plot(value, label=key)
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title("RMSE vs Epoch")
        plt.savefig(f"results/all_groups/group_{key}.png")
        plt.close()









