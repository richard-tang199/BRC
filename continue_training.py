# -*-coding:utf-8-*-
import argparse
from textwrap import indent

import torch
import yaml
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
parser.add_argument("--group_name", type=list, default=(1, 4, 5), help="group in the dataset")
parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs for training")
parser.add_argument("--lr", type=float, default=5e-3, help="learning rate for training")
parser.add_argument("--eval_gap", type=int, default=20, help="training epochs between evaluations")
parser.add_argument("--use_default_config", action='store_true', help="use default configuration for training")
parser.add_argument("--patch_size", type=int, default=16, help="length of the patch")

if __name__ == '__main__':
    args = parser.parse_args()
    train_data_list, test_data_list, test_label_list, subsequence_list = load_dataset(data_name=args.dataset_name,
                                                                                      group_name=args.group_name)
    model = PatchDetectorGru(patch_size=args.patch_size, expansion_ratio=6, num_layers=6)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    for train_data, subsequence_length in zip(train_data_list, subsequence_list):
        train_data = train_data[:, np.newaxis]
        window_length = 4 * subsequence_length
        window_length = (window_length // args.patch_size) * args.patch_size
        train_loader, _ = get_dataloader(data=train_data,
                                         batch_size=args.batch_size,
                                         window_length=window_length,
                                         mode="train",
                                         train_stride=subsequence_length // 16 + 1)

        for epoch in tqdm.tqdm(range(args.num_epochs)):
            model.train()
            for batch_idx, (data, ) in enumerate(train_loader):
                data = data.to(device)
                loss = model(data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # scheduler.step()

    for i, (test_data, test_label, subsequence_length) in enumerate(zip(test_data_list, test_label_list, subsequence_list)):
        current_group = args.group_name[i]
        os.makedirs(f"continue/group_{args.group_name[i]}", exist_ok=True)
        test_data = test_data[:, np.newaxis]
        window_length = 4 * subsequence_length
        window_length = (window_length // args.patch_size) * args.patch_size
        test_loader, test_window_converter =  get_dataloader(data=test_data,
                                                             batch_size=args.batch_size,
                                                             window_length=window_length,
                                                             test_stride=subsequence_length)
        recon_out = None
        for batch_idx, (data,) in enumerate(test_loader):
            model.eval()
            data = data.to(device)
            recon = model.predict(data)
            if recon_out is None:
                recon_out = recon
            else:
                recon_out = torch.concat([recon_out, recon], dim=0)

        recon_out = recon_out.detach().cpu().numpy()
        recon_out = test_window_converter.windows_to_sequence(recon_out)

        save_path = f"continue/group_{args.group_name[i]}/result.png"
        recon_plot(save_path=save_path,
                   gap=subsequence_length,
                   figure_width=10,
                   figure_length=120,
                   train_data=train_data_list[i],
                   test_data=test_data,
                   test_label=test_label,
                   recon_test_data=recon_out
                   )



