# -*-coding:utf-8-*-
import argparse
import os.path

import numpy as np
import tqdm
import torch
from evaluation.evaluator import evaluate
from toolkit.result_plot import recon_plot
from model.one_to_all import PatchClassify, PatchUnified
from toolkit.utils import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='all', help='Name of the model')
parser.add_argument('--dataset_name', type=str, default='UCR', help='Name of the dataset')
parser.add_argument("--group_name", type=list, default=[1], help="group in the dataset")
parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs for training")
parser.add_argument("--lr", type=float, default=2e-3, help="learning rate for training")
parser.add_argument("--eval_gap", type=int, default=20, help="training epochs between evaluations")
parser.add_argument("--use_default_config", action='store_true', help="use default configuration for training")
parser.add_argument("--patch_size", type=int, default=8, help="length of the patch")

if __name__ == '__main__':
    args = parser.parse_args()
    train_data_list, test_data_list, test_label_list, subsequence_list = load_dataset(data_name=args.dataset_name,
                                                                                      group_name=args.group_name)

    model = PatchUnified(patch_size=args.patch_size, expansion_ratio=2, num_layers=2)
    model = model.to(device)
    optimizer_classifier = torch.optim.Adam(model.classifier_net.parameters(), lr=2 * args.lr)
    optimizer_recon = torch.optim.Adam(model.recon_net.parameters(), lr=args.lr)
    scheduler_classifier = torch.optim.lr_scheduler.StepLR(optimizer_classifier, step_size=1, gamma=0.99)
    scheduler_recon = torch.optim.lr_scheduler.StepLR(optimizer_recon, step_size=1, gamma=0.99)

    # initial tensorboard
    now = datetime.now().strftime("%m-%d-%H-%M")
    output_dir = os.path.join("new_results", "logs", now)
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(output_dir)

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss_classifier = 0
        epoch_loss_recon = 0
        for train_data, subsequence_length in zip(train_data_list, subsequence_list):
            train_data = train_data[:, np.newaxis]
            window_length = subsequence_length * 8
            window_length = (window_length // args.patch_size) * args.patch_size
            train_loader, _ = get_dataloader(
                data=train_data,
                batch_size=args.batch_size,
                window_length=window_length,
                mode="train",
                train_stride=subsequence_length // 16 + 1
            )

            for batch_idx, (data, ) in enumerate(train_loader):
                data = data.to(device)
                classify_loss, recon_loss = model(data)
                optimizer_classifier.zero_grad()
                optimizer_recon.zero_grad()
                classify_loss.backward()
                recon_loss.backward()
                optimizer_classifier.step()
                optimizer_recon.step()
                epoch_loss_classifier += classify_loss.item()
                epoch_loss_recon += recon_loss.item()

            scheduler_classifier.step()
            scheduler_recon.step()
            writer.add_scalar("train/classify_loss", classify_loss.item() / len(train_loader), epoch)
            writer.add_scalar("train/recon_loss", recon_loss.item() / len(train_loader), epoch)
            print(f"Epoch: {epoch}, Train Loss: {epoch_loss_classifier / len(train_loader)}, {epoch_loss_recon / len(train_loader)}")

    with torch.no_grad():
        model.eval()
        for i, (test_data, test_label, subsequence_length) in enumerate(zip(test_data_list, test_label_list, subsequence_list)):
            test_data = test_data[:, np.newaxis]
            window_length = subsequence_length * 8
            window_length = (window_length // args.patch_size) * args.patch_size
            test_loader, test_window_converter =  get_dataloader(
                data=test_data,
                batch_size=args.batch_size,
                window_length=window_length,
                test_stride=subsequence_length
            )

            recon_out = None
            for batch_idx, (data, ) in enumerate(test_loader):
                data = data.to(device)
                recon, classify_score = model.predict(data)
                if recon_out is None:
                    recon_out = recon
                else:
                    recon_out = torch.concat((recon_out, recon), dim=0)

            recon_out = recon_out.detach().cpu().numpy()
            recon_out = test_window_converter.windows_to_sequence(recon_out)
            anomaly_scores = anomaly_score_func(raw_value=test_data,
                                               predict_value=recon_out,
                                               subsequence=subsequence_length)



            eval_result = evaluate(test_label,
                                   anomaly_scores,
                                   subsequence_length,
                                   use_ucr=True,
                                    mode="test")

            save_path = os.path.join("new_results", f"group_{args.group_name[0]}", "result.png")

            recon_plot(
                save_path=save_path,
                gap=subsequence_length,
                figure_width=10,
                figure_length=120,
                train_data=train_data_list[i],
                test_data=test_data,
                test_label=test_label,
                recon_test_data=recon_out,
                test_anomaly_score=anomaly_scores
            )




