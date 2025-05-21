from torch import nn
import torch
from torch.nn.modules.module import T
import numpy as np
from torch.utils.data import Dataset
import random
import tqdm

class Config(object):
    def __init__(self):
        # datasets
        self.dataset = 'UCR'
        # model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 64
        self.project = 2

        self.dropout = 0.45
        self.features_len = 10
        self.window_size = 64
        self.time_step = 16

        # training configs
        self.num_epoch = 300

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4
        self.weight = 5e-4

        # data parameters
        self.drop_last = False
        self.batch_size = 512
        # trend rate
        self.trend_rate = 0.01
        # negative sample rates
        self.rate = 1
        # minimum cut length
        self.cut_rate = 12

        # Anomaly quantile of fixed threshold
        self.detect_nu = 0.001
        # Methods for determining thresholds ("direct","fix","floating","one-anomaly")
        self.threshold_determine = 'one-anomaly'

def subsequences(sequence, window_size, time_step):
    # An array of non-contiguous memory is converted to an array of contiguous memory
    sq = np.ascontiguousarray(sequence)
    a = (sq.shape[0] - window_size + time_step) % time_step
    # label array
    if sq.ndim == 1:
        shape = (int((sq.shape[0] - window_size + time_step) / time_step), window_size)
        stride = sq.itemsize * np.array([time_step * 1, 1])
        if a != 0:
            sq = sq[:sq.shape[0] - a]
    # data array
    elif sq.ndim == 2:
        shape = (int((sq.shape[0] - window_size + time_step) / time_step), window_size, sq.shape[1])
        stride = sq.itemsize * np.array([time_step * sq.shape[1], sq.shape[1], 1])
        if a != 0:
            sq = sq[:sq.shape[0] - a, :]
    else:
        print('Array dimension error')
        os.exit()
    # print(sq.strides)
    sq = np.lib.stride_tricks.as_strided(sq, shape=shape, strides=stride)
    return sq

def cut_add_paste_outlier(train_x, configs):
    for i, x_i in enumerate(train_x):
        # radius = max(int(train_x.shape[1]/6), int(np.random.rand() * train_x.shape[1]))
        radius = max(int(configs.cut_rate), int(np.random.rand() * train_x.shape[1]))

        cut_random = int(random.uniform(0, train_x.shape[0]))
        cut_data = train_x[cut_random, :, :]
        from_position = int(np.random.rand() * (train_x.shape[1] - radius))
        cut_data = cut_data[from_position:from_position + radius, :]
        if cut_data.shape[1] > 1:
            elements = np.arange(0, cut_data.shape[1])
            if configs.dim > 1:
                dim_size = np.random.randint(1, configs.dim)
            else:
                dim_size = 1
            selected_elements = np.random.choice(elements, size=dim_size, replace=False)
            for item in selected_elements:
                factor = np.random.rand() * configs.trend_rate
                slope = np.random.choice([-1, 1]) * factor * np.arange(radius)
                cut_data[:, item] += slope
        else:
            factor = np.random.rand() * configs.trend_rate
            slope = np.random.choice([-1, 1]) * factor * np.arange(radius)
            cut_data[:, 0] += slope
        to_position = int(np.random.rand() * (train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, :] = cut_data[:, :]
    return train_x

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        # if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        #     X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if hasattr(config, 'augmentation'):
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if hasattr(self, 'aug1'):
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def data_generator(train_data, configs, seed=4):
    train_x = subsequences(train_data, configs.window_size, configs.time_step)
    train_labels = np.zeros(train_x.shape[0])
    train_y= np.zeros(train_x.shape[0])
    train_origin = train_x.copy()

    if ((configs.rate != 0) and (configs.rate <= 1)):
        train_aug_x = cut_add_paste_outlier(train_origin, configs)
        sample_num = int(configs.rate * len(train_origin))
        sample_list = [i for i in range(sample_num)]
        sample_list = random.sample(sample_list, sample_num)
        sample = train_aug_x[sample_list, :, :]
    elif configs.rate > 1:
        train_aug_x_1 = cut_add_paste_outlier(train_origin, configs)
        train_aug_x_2 = cut_add_paste_outlier(train_origin, configs)
        train_aug_x = np.concatenate((train_aug_x_1, train_aug_x_2), axis=0)
        sample_num = int(configs.rate * len(train_origin))
        sample_list = [i for i in range(sample_num)]
        sample_list = random.sample(sample_list, sample_num)
        sample = train_aug_x[sample_list, :, :]
    else:
        sample_num = 0
        sample = train_x[[], :, :]

    train_aug_x = sample
    train_aug_y = np.zeros(sample_num) + 1

    train_x = np.concatenate((train_x, train_aug_x), axis=0)
    train_y = np.concatenate((train_y, train_aug_y), axis=0)

    train_x = train_x.transpose((0, 2, 1))

    train_dat_dict = dict()
    train_dat_dict["samples"] = train_x
    train_dat_dict["labels"] = train_y

    train_dataset = Load_Dataset(train_dat_dict, configs)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)

    return train_loader

class CutAddPaste(nn.Module):
    def __init__(self, device):
        configs = Config()
        super(CutAddPaste, self).__init__()
        self.input_channels = configs.input_channels
        self.final_out_channels = configs.final_out_channels
        self.features_len = configs.features_len
        self.window_size = configs.window_size
        self.device = device
        self.kernel_size = configs.kernel_size
        self.stride = configs.stride
        self.dropout = configs.dropout
        self.project = configs.project

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(self.input_channels, 32, kernel_size=self.kernel_size,
                      stride=self.stride, bias=False, padding=(self.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(self.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, self.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(self.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.projection_head = nn.Sequential(
            nn.Linear(self.final_out_channels * self.features_len, self.final_out_channels * self.features_len // self.project),
            nn.BatchNorm1d(self.final_out_channels * self.features_len // self.project),
            nn.ReLU(inplace=True),
            nn.Linear(self.final_out_channels * self.features_len // self.project, 2),
        )
        self.logits = nn.Linear(self.final_out_channels * self.features_len, 2)

    def forward(self, x_in):
        if torch.isnan(x_in).any():
            print('tensor contain nan')
        # 1D CNN feature extraction
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # print(x.size())
        # Encoder
        hidden = x.permute(0, 2, 1)
        hidden = hidden.reshape(hidden.size(0), -1)
        logits = self.projection_head(hidden)
        # logits = self.logits(hidden)

        return logits

    def fit(self, train_data):
        """

        :param train_data: data_len, num_channels
        :return:
        """
        self.train()
        epochs = 300
        configs = Config()
        optimizer = torch.optim.Adam(self.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                      weight_decay=configs.weight)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        train_loader = data_generator(train_data, configs)
        loss_func = nn.CrossEntropyLoss()

        for epoch in tqdm.trange(epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.float().to(self.device), target.long().to(self.device)
                optimizer.zero_grad()
                logits = self(data)
                loss = loss_func(logits, target)
                score = torch.nn.Softmax(dim=1)(logits)[:, 1]
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            mean_loss = total_loss / len(train_loader)
            scheduler.step(mean_loss)

    def predict(self, test_loader):
        self.eval()
        all_scores = None
        for batch_idx, (data, ) in enumerate(test_loader):
            data = data.float().to(self.device)
            data = data.permute(0, 2, 1)
            with torch.no_grad():
                logits = self(data)
                score = torch.nn.Softmax(dim=1)(logits)[:, 1]
                if all_scores is None:
                    all_scores = score
                else:
                    all_scores = torch.cat((all_scores, score), dim=0)
        all_scores = all_scores.cpu().numpy()
        return all_scores

if __name__ == '__main__':
    train_data = np.random.rand(1024, 1)
    configs = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CutAddPaste(configs, device)
    model.to(device)
    model.fit(train_data)
    print('model training done')












