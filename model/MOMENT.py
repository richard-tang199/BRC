import os
import thop
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from momentfm import MOMENTPipeline
from momentfm.utils.masking import Masking
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import math


class ReconstructDataset_Moment(torch.utils.data.Dataset):
    def __init__(self, data, window_size, stride=1, normalize=True):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data

        self.univariate = self.data.shape[1] == 1
        self.sample_num = max((self.data.shape[0] - window_size) // stride + 1, 0)

        self.samples = self._generate_samples()
        self.input_mask = np.ones(self.window_size, dtype=np.float32)  # Fixed input mask

    def _normalize_data(self, data, epsilon=1e-8):
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        data = torch.tensor(self.data, dtype=torch.float32)
        indices = np.arange(0, self.sample_num * self.stride, self.stride)

        if self.univariate:
            X = torch.stack([data[i : i + self.window_size] for i in indices])
        else:
            X = torch.stack([data[i : i + self.window_size, :] for i in indices])

        return X

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index], self.input_mask

class MOMENT(torch.nn.Module):
    def __init__(self,
                 device,
                 win_size=256,
                 input_c=1,
                 batch_size=1024,
                 epochs=2,
                 validation_size=0,
                 lr=1e-4):
        super().__init__()

        self.model_name = 'MOMENT'
        self.win_size = win_size
        self.input_c = input_c
        self.batch_size = batch_size
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        self.epochs = epochs
        self.validation_size = validation_size
        self.lr = lr

        cuda = True
        self.cuda = cuda
        self.device = device

        self.model = MOMENTPipeline.from_pretrained(
            "baseline/momentfm",
            model_kwargs={
                "task_name": "reconstruction"
            },
        )
        self.model.init()
        self.model = self.model.to("cuda").float()
        # Optimize Mean Squarred Error using your favourite optimizer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.save_path = None

    def zero_shot(self, data):

        test_loader = DataLoader(
            dataset=ReconstructDataset_Moment(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False)

        trues, preds = [], []
        self.score_list = []
        self.step = 0
        with torch.no_grad():
            for batch_x, batch_masks in tqdm(test_loader, total=len(test_loader)):
                batch_x = batch_x.to("cuda").float()
                batch_masks = batch_masks.to("cuda")
                batch_x = batch_x.permute(0, 2, 1)

                # print('batch_x: ', batch_x.shape)             # [batch_size, n_channels, window_size]
                # print('batch_masks: ', batch_masks.shape)     # [batch_size, window_size]

                output = self.model(x_enc=batch_x, input_mask=batch_masks)  # [batch_size, n_channels, window_size]
                if self.step == 0:
                    flops, params = thop.profile(self.model, inputs=(batch_x, batch_masks))
                score = torch.mean(self.anomaly_criterion(batch_x, output.reconstruction),
                                   dim=-1).detach().cpu().numpy()[:, -1]
                self.score_list.append(score)

        self.__anomaly_score = np.concatenate(self.score_list, axis=0).reshape(-1)

        if self.__anomaly_score.shape[0] < len(data):
            self.__anomaly_score = np.array([self.__anomaly_score[0]] * math.ceil((self.win_size - 1) / 2) +
                                            list(self.__anomaly_score) + [self.__anomaly_score[-1]] * (
                                                        (self.win_size - 1) // 2))
        self.decision_scores_ = self.__anomaly_score

        return self.decision_scores_, flops, params

    def fit(self, data):
        tsTrain = data[:int((1 - self.validation_size) * len(data))]
        tsValid = data[int((1 - self.validation_size) * len(data)):]

        train_loader = DataLoader(
            dataset=ReconstructDataset_Moment(tsTrain, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=True
        )

        valid_loader = DataLoader(
            dataset=ReconstructDataset_Moment(tsValid, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False
        )

        mask_generator = Masking(mask_ratio=0.3)  # Mask 30% of patches randomly

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for batch_x, batch_masks in tqdm(train_loader, total=len(train_loader)):
                batch_x = batch_x.to(self.device).float()
                batch_x = batch_x.permute(0, 2, 1)
                # print('batch_x: ', batch_x.shape)

                original = batch_x
                n_channels = batch_x.shape[1]

                # Reshape to [batch_size * n_channels, 1, window_size]
                batch_x = batch_x.reshape((-1, 1, self.win_size))

                batch_masks = batch_masks.to(self.device).long()
                batch_masks = batch_masks.repeat_interleave(n_channels, axis=0)

                # Randomly mask some patches of data
                mask = mask_generator.generate_mask(
                    x=batch_x, input_mask=batch_masks).to(self.device).long()

                mask = torch.nn.functional.pad(mask, (0, batch_masks.size(1) - mask.size(1)), mode='constant', value=1)

                # Forward
                model_output = self.model(batch_x, input_mask=batch_masks, mask=mask).reconstruction
                model_output = torch.nn.functional.pad(model_output, (0, original.size(2) - model_output.size(2)),
                                                       mode='replicate')

                output = model_output.reshape(original.size(0), n_channels, self.win_size)

                # Compute loss
                loss = self.criterion(output, original)

                # print(f"loss: {loss.item()}")

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # self.model.eval()
            # avg_loss = 0
            # with torch.no_grad():
            #     for batch_x, batch_masks in tqdm(valid_loader, total=len(valid_loader)):
            #         batch_x = batch_x.to("cuda").float()
            #         batch_masks = batch_masks.to("cuda")
            #         batch_x = batch_x.permute(0,2,1)

            #         print('batch_x: ', batch_x.shape)
            #         print('batch_masks: ', batch_masks.shape)

            #         output = self.model(batch_x, input_mask=batch_masks)

            #         loss = self.criterion(output.reconstruction.reshape(-1, n_channels, self.win_size), batch_x)
            #         print(f"loss: {loss.item()}")
            #         avg_loss += loss.cpu().item()

            # valid_loss = avg_loss/max(len(valid_loader), 1)
            # self.scheduler.step()
            # self.early_stopping(valid_loss, self.model)
            # if self.early_stopping.early_stop:
            #     print("   Early stopping<<<")
            #     break

    def decision_function(self, data):
        """
        Not used, present for API consistency by convention.
        """

        test_loader = DataLoader(
            dataset=ReconstructDataset_Moment(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False)

        trues, preds = [], []
        self.score_list = []
        with torch.no_grad():
            for batch_x, batch_masks in tqdm(test_loader, total=len(test_loader)):
                batch_x = batch_x.to("cuda").float()
                batch_masks = batch_masks.to("cuda")
                batch_x = batch_x.permute(0, 2, 1)

                # print('batch_x: ', batch_x.shape)             # [batch_size, n_channels, window_size]
                # print('batch_masks: ', batch_masks.shape)     # [batch_size, window_size]

                output = self.model(batch_x, input_mask=batch_masks)
                score = torch.mean(self.anomaly_criterion(batch_x, output.reconstruction),
                                   dim=-1).detach().cpu().numpy()[:, -1]
                self.score_list.append(score)

        self.__anomaly_score = np.concatenate(self.score_list, axis=0).reshape(-1)

        if self.__anomaly_score.shape[0] < len(data):
            self.__anomaly_score = np.array([self.__anomaly_score[0]] * math.ceil((self.win_size - 1) / 2) +
                                            list(self.__anomaly_score) + [self.__anomaly_score[-1]] * (
                                                        (self.win_size - 1) // 2))

        return self.__anomaly_score

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = np.random.rand(1000, 1)
    model = MOMENT(device=device)
    test_scores = model.zero_shot(test_data)
    print(test_scores.shape)