import torch
import numpy as np
import math
from torch import nn
from torch.nn import functional as F
import pandas as pd
from torch.utils.data import DataLoader
import tqdm

__all__ = ["complete_timestamp", "standardize_kpi"]


class UniDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        use_label,
        window,
        mode,
        sliding_window_size,
        train_data = None,
        test_data = None,
        test_label = None,
        data_pre_mode=0,
    ):
        self.window = window
        value_all = []
        label_all = []
        missing_all = []
        self.len = 0
        self.sample_num = 0

        if mode == "train":
            value = train_data
            label = np.zeros(len(value))
        elif mode == "valid":
            value = train_data[int(0.9*len(train_data)):]
            label = np.zeros(len(value))
        elif mode == "test":
            value = test_data
            label = test_label
        else:
            raise ValueError("Invalid mode")

        value = value.squeeze()
        df = pd.DataFrame(
            {"timestamp": np.arange(0, len(value)), "value": value, "label": label}
        )

        timestamp, missing, (value, label) = complete_timestamp(
            df["timestamp"], (df["value"], df["label"])
        )
        value = value.astype(float)
        missing2 = np.isnan(value)
        missing = np.logical_or(missing, missing2).astype(int)
        label = label.astype(float)
        label[np.where(missing == 1)[0]] = np.nan
        value[np.where(missing == 1)[0]] = np.nan
        df2 = pd.DataFrame()
        df2["timestamp"] = timestamp
        df2["value"] = value
        df2["label"] = label
        df2["missing"] = missing.astype(int)
        df2 = df2.fillna(method="bfill")
        df2 = df2.fillna(0)
        df2["label"] = df2["label"].astype(int)
        if data_pre_mode == 0:
            df2["value"], *_ = standardize_kpi(df2["value"])
        else:
            v = np.asarray(df2["value"])
            v = 2 * (v - train_min) / (train_max - train_min) - 1
            df2["value"] = v
        timestamp, values, labels = (
            np.asarray(df2["timestamp"]),
            np.clip(np.asarray(df2["value"]), -40, 40),
            np.asarray(df2["label"]),
        )
        values[np.where(missing == 1)[0]] = 0
        if (mode == "train" or mode == "valid") and use_label == 1:
            values[np.where(labels == 1)[0]] = 0
        elif (mode == "train" or mode == "valid") and use_label == 0:
            labels[:] = 0
        else:
            pass
        values = np.convolve(
            values,
            np.ones((sliding_window_size,)) / sliding_window_size,
            mode="valid",
        )
        timestamp = timestamp[sliding_window_size - 1 :]
        labels = labels[sliding_window_size - 1 :]
        missing = missing[sliding_window_size - 1 :]
        value_all.append(values)
        label_all.append(labels)
        missing_all.append(missing)
        self.sample_num += max(len(values) - window + 1, 0)
        self.samples, self.labels, self.miss_label = self.__getsamples(
            value_all, label_all, missing_all
        )

    def __getsamples(self, values, labels, missing):
        X = torch.zeros((self.sample_num, 1, self.window))
        Y = torch.zeros((self.sample_num, self.window))
        Z = torch.zeros((self.sample_num, self.window))
        i = 0
        for cnt in range(len(values)):
            v = values[cnt]
            l = labels[cnt]
            m = missing[cnt]
            for j in range(len(v) - self.window + 1):
                X[i, 0, :] = torch.from_numpy(v[j : j + self.window])
                Y[i, :] = torch.from_numpy(np.asarray(l[j : j + self.window]))
                Z[i, :] = torch.from_numpy(np.asarray(m[j : j + self.window]))
                i += 1
        return (X, Y, Z)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx, :], self.miss_label[idx, :]]
        return sample


def complete_timestamp(timestamp, arrays=None):
    """
    Complete `timestamp` such that the time interval is homogeneous.
    Zeros will be inserted into each array in `arrays`, at missing points.
    Also, an indicator array will be returned to indicate whether each
    point is missing or not.
    Args:
        timestamp (np.ndarray): 1-D int64 array, the timestamp values.
            It can be unsorted.
        arrays (Iterable[np.ndarray]): The 1-D arrays to be filled with zeros
            according to `timestamp`.
    Returns:
        np.ndarray: A 1-D int64 array, the completed timestamp.
        np.ndarray: A 1-D int32 array, indicating whether each point is missing.
        list[np.ndarray]: The arrays, missing points filled with zeros.
            (optional, return only if `arrays` is specified)
    """
    timestamp = np.asarray(timestamp, np.int64)
    if len(timestamp.shape) != 1:
        raise ValueError("`timestamp` must be a 1-D array")

    has_arrays = arrays is not None
    arrays = [np.asarray(array) for array in (arrays or ())]
    for i, array in enumerate(arrays):
        if array.shape != timestamp.shape:
            raise ValueError(
                "The shape of ``arrays[{}]`` does not agree with "
                "the shape of `timestamp` ({} vs {})".format(
                    i, array.shape, timestamp.shape
                )
            )
    src_index = np.argsort(timestamp)
    timestamp_sorted = timestamp[src_index]
    intervals = np.unique(np.diff(timestamp_sorted))
    interval = np.min(intervals)
    if interval == 0:
        raise ValueError("Duplicated values in `timestamp`")
    for itv in intervals:
        if itv % interval != 0:
            raise ValueError(
                "Not all intervals in `timestamp` are multiples "
                "of the minimum interval"
            )

    # prepare for the return arrays
    length = (timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1
    ret_timestamp = np.arange(
        timestamp_sorted[0], timestamp_sorted[-1] + interval, interval, dtype=np.int64
    )
    ret_missing = np.ones([length], dtype=np.int32)
    ret_arrays = [np.zeros([length], dtype=array.dtype) for array in arrays]

    # copy values to the return arrays
    dst_index = np.asarray(
        (timestamp_sorted - timestamp_sorted[0]) // interval, dtype=np.int32
    )
    ret_missing[dst_index] = 0
    for ret_array, array in zip(ret_arrays, arrays):
        ret_array[dst_index] = array[src_index]

    if has_arrays:
        return ret_timestamp, ret_missing, ret_arrays
    else:
        return ret_timestamp, ret_missing


def standardize_kpi(values, mean=None, std=None, excludes=None):
    """
    Standardize a
    Args:
        values (np.ndarray): 1-D `float32` array, the KPI observations.
        mean (float): If not :obj:`None`, will use this `mean` to standardize
            `values`. If :obj:`None`, `mean` will be computed from `values`.
            Note `mean` and `std` must be both :obj:`None` or not :obj:`None`.
            (default :obj:`None`)
        std (float): If not :obj:`None`, will use this `std` to standardize
            `values`. If :obj:`None`, `std` will be computed from `values`.
            Note `mean` and `std` must be both :obj:`None` or not :obj:`None`.
            (default :obj:`None`)
        excludes (np.ndarray): Optional, 1-D `int32` or `bool` array, the
            indicators of whether each point should be excluded for computing
            `mean` and `std`. Ignored if `mean` and `std` are not :obj:`None`.
            (default :obj:`None`)
    Returns:
        np.ndarray: The standardized `values`.
        float: The computed `mean` or the given `mean`.
        float: The computed `std` or the given `std`.
    """
    values = np.asarray(values, dtype=np.float32)
    if len(values.shape) != 1:
        raise ValueError("`values` must be a 1-D array")
    if (mean is None) != (std is None):
        raise ValueError("`mean` and `std` must be both None or not None")
    if excludes is not None:
        excludes = np.asarray(excludes, dtype=np.bool)
        if excludes.shape != values.shape:
            raise ValueError(
                "The shape of `excludes` does not agree with "
                "the shape of `values` ({} vs {})".format(excludes.shape, values.shape)
            )

    if mean is None:
        if excludes is not None:
            val = values[np.logical_not(excludes)]
        else:
            val = values
        mean = val.mean()
        std = val.std()

    return (values - mean) / std, mean, std

class Config(object):
    def __init__(self):
        self.window = 64
        self.latent_dim = 8
        self.only_test = 0
        self.max_epoch = 30
        self.batch_size = 512
        self.num_workers = 0
        self.learning_rate = 0.0005
        self.sliding_window_size = 1
        self.data_pre_mode = 0
        self.missing_data_rate = 0.01
        self.point_ano_rate = 0.05
        self.seg_ano_rate = 0.1
        self.eval_all = 0
        self.condition_emb_dim = 16
        self.d_model = 256
        self.d_inner = 512
        self.n_head = 8
        self.kernel_size = 16
        self.stride = 8
        self.mcmc_rate = 0.2
        self.mcmc_value = -5
        self.mcmc_mode = 2
        self.condition_mode = 2
        self.dropout_rate = 0.05
        self.gpu = 0
        self.use_label = 0

def missing_data_injection(x, y, z, rate):
    miss_size = int(rate * x.shape[0] * x.shape[1] * x.shape[2])
    row = torch.randint(low=0, high=x.shape[0], size=(miss_size,))
    col = torch.randint(low=0, high=x.shape[2], size=(miss_size,))
    # for i in range(miss_size):
    #     x[row[i], :, col[i]] = 0
    #     y[row[i], col[i]] = 1
    # z = torch.zeros_like(y)
    x[row, :, col] = 0
    z[row, col] = 1
    return x, y, z


def point_ano(x, y, z, rate):
    aug_size = int(rate * x.shape[0])
    id_x = torch.randint(low=0, high=x.shape[0], size=(aug_size,))
    x_aug = x[id_x].clone()
    y_aug = y[id_x].clone()
    z_aug = z[id_x].clone()
    if x_aug.shape[1] == 1:
        ano_noise1 = torch.randint(low=1, high=20, size=(int(aug_size / 2),))
        ano_noise2 = torch.randint(
            low=-20, high=-1, size=(aug_size - int(aug_size / 2),)
        )
        ano_noise = (torch.cat((ano_noise1, ano_noise2), dim=0) / 2).to(x_aug.device)
        x_aug[:, 0, -1] += ano_noise
        y_aug[:, -1] = torch.logical_or(y_aug[:, -1], torch.ones_like(y_aug[:, -1]))
    return x_aug, y_aug, z_aug


def seg_ano(x, y, z, rate, method):
    aug_size = int(rate * x.shape[0])
    idx_1 = torch.arange(aug_size)
    idx_2 = torch.arange(aug_size)
    while torch.any(idx_1 == idx_2):
        idx_1 = torch.randint(low=0, high=x.shape[0], size=(aug_size,))
        idx_2 = torch.randint(low=0, high=x.shape[0], size=(aug_size,))
    x_aug = x[idx_1].clone()
    y_aug = y[idx_1].clone()
    z_aug = z[idx_1].clone()
    time_start = torch.randint(low=7, high=x.shape[2], size=(aug_size,))  # seg start
    for i in range(len(idx_2)):
        if method == "swap":
            x_aug[i, :, time_start[i] :] = x[idx_2[i], :, time_start[i] :]
            y_aug[:, time_start[i] :] = torch.logical_or(
                y_aug[:, time_start[i] :], torch.ones_like(y_aug[:, time_start[i] :])
            )
    return x_aug, y_aug, z_aug


class EncoderLayer_selfattn(nn.Module):
    """Compose with two layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer_selfattn, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)
        output, attn = self.attention(q, k, v)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class CVAE(nn.Module):
    def __init__(
        self,
        hp,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = "C",
    ):
        super(CVAE, self).__init__()
        self.hp = hp
        self.num_iter = 0
        self.step_max = 0
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        modules = []
        in_channels = self.hp.window + 2 * self.hp.condition_emb_dim
        self.hidden_dims = [100, 100]
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh(),
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.hp.latent_dim)
        self.fc_var = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hp.latent_dim),
            nn.Softplus(),
        )
        modules = []
        self.decoder_input = nn.Linear(
            self.hp.latent_dim + 2 * self.hp.condition_emb_dim, self.hidden_dims[-1]
        )
        self.hidden_dims.reverse()
        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
                    nn.Tanh(),
                )
            )
        modules.append(
            nn.Sequential(
                nn.Linear(self.hidden_dims[-1], self.hp.window),
                nn.Tanh(),
            )
        )
        self.decoder = nn.Sequential(*modules)
        self.fc_mu_x = nn.Linear(self.hp.window, self.hp.window)
        self.fc_var_x = nn.Sequential(
            nn.Linear(self.hp.window, self.hp.window), nn.Softplus()
        )
        self.atten = nn.ModuleList(
            [
                EncoderLayer_selfattn(
                    self.hp.d_model,
                    self.hp.d_inner,
                    self.hp.n_head,
                    self.hp.d_inner // self.hp.n_head,
                    self.hp.d_inner // self.hp.n_head,
                    dropout=0.1,
                )
                for _ in range(1)
            ]
        )
        self.emb_local = nn.Sequential(
            nn.Linear(2 + self.hp.kernel_size, self.hp.d_model),
            nn.Tanh(),
        )
        self.out_linear = nn.Sequential(
            nn.Linear(self.hp.d_model, self.hp.condition_emb_dim),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(self.hp.dropout_rate)
        self.emb_global = nn.Sequential(
            nn.Linear(self.hp.window, self.hp.condition_emb_dim),
            nn.Tanh(),
        )

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        var = self.fc_var(result)
        return [mu, var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 1, self.hidden_dims[0])
        result = self.decoder(result)
        mu_x = self.fc_mu_x(result)
        var_x = self.fc_var_x(result)
        return mu_x, var_x

    def reparameterize(self, mu, var):
        std = torch.sqrt(1e-7 + var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, mode, y):
        if mode == "train" or mode == "valid":
            condition = self.get_conditon(input)
            condition = self.dropout(condition)
            mu, var = self.encode(torch.cat((input, condition), dim=2))
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(torch.cat((z, condition.squeeze(1)), dim=1))
            rec_x = self.reparameterize(mu_x, var_x)
            loss = self.loss_func(mu_x, var_x, input, mu, var, y, z)
            return [mu_x, var_x, rec_x, mu, var, loss]
        else:
            y = y.unsqueeze(1)
            return self.MCMC2(input)

    def get_conditon(self, x):
        x_g = x
        f_global = torch.fft.rfft(x_g[:, :, :-1], dim=-1)
        f_global = torch.cat((f_global.real, f_global.imag), dim=-1)
        f_global = self.emb_global(f_global)
        x_g = x_g.view(x.shape[0], 1, 1, -1)
        x_l = x_g.clone()
        x_l[:, :, :, -1] = 0
        unfold = nn.Unfold(
            kernel_size=(1, self.hp.kernel_size),
            dilation=1,
            padding=0,
            stride=(1, self.hp.stride),
        )
        unfold_x = unfold(x_l)
        unfold_x = unfold_x.transpose(1, 2)
        f_local = torch.fft.rfft(unfold_x, dim=-1)
        f_local = torch.cat((f_local.real, f_local.imag), dim=-1)
        f_local = self.emb_local(f_local)
        for enc_layer in self.atten:
            f_local, enc_slf_attn = enc_layer(f_local)
        f_local = self.out_linear(f_local)
        f_local = f_local[:, -1, :].unsqueeze(1)
        output = torch.cat((f_global, f_local), -1)
        return output

    def MCMC2(self, x):
        condition = self.get_conditon(x)
        origin_x = x.clone()
        for i in range(10):
            mu, var = self.encode(torch.cat((x, condition), dim=2))
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(torch.cat((z, condition.squeeze(1)), dim=1))
            recon = -0.5 * (torch.log(var_x) + (origin_x - mu_x) ** 2 / var_x)
            temp = (
                torch.from_numpy(np.percentile(recon.cpu(), self.hp.mcmc_rate, axis=-1))
                .unsqueeze(2)
                .repeat(1, 1, self.hp.window)
            ).to(recon.device)
            if self.hp.mcmc_mode == 0:
                l = (temp < recon).int()
                x = mu_x * (1 - l) + origin_x * l
            if self.hp.mcmc_mode == 1:
                l = (self.hp.mcmc_value < recon).int()
                x = origin_x * l + mu_x * (1 - l)
            if self.hp.mcmc_mode == 2:
                l = torch.ones_like(origin_x)
                l[:, :, -1] = 0
                x = origin_x * l + (1 - l) * mu_x
        prob_all = 0
        mu, var = self.encode(torch.cat((x, condition), dim=2))
        for i in range(128):
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(torch.cat((z, condition.squeeze(1)), dim=1))
            prob_all += -0.5 * (torch.log(var_x) + (origin_x - mu_x) ** 2 / var_x)
        return x, prob_all / 128

    def loss_func(self, mu_x, var_x, input, mu, var, y, z, mode="nottrain"):
        if mode == "train":
            self.num_iter += 1
            self.num_iter = self.num_iter % 100
        kld_weight = 0.005
        mu_x = mu_x.squeeze(1)
        var_x = var_x.squeeze(1)
        input = input.squeeze(1)
        recon_loss = torch.mean(
            0.5
            * torch.mean(y * (torch.log(var_x) + (input - mu_x) ** 2 / var_x), dim=1),
            dim=0,
        )
        m = (torch.sum(y, dim=1, keepdim=True) / self.hp.window).repeat(
            1, self.hp.latent_dim
        )
        kld_loss = torch.mean(
            0.5 * torch.mean(m * (z**2) - torch.log(var) - (z - mu) ** 2 / var, dim=1),
            dim=0,
        )
        if self.loss_type == "B":
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(
                self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0]
            )
            loss = recon_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        elif self.loss_type == "C":
            loss = recon_loss + kld_loss
        elif self.loss_type == "D":
            loss = recon_loss + self.num_iter / 100 * kld_loss
        else:
            raise ValueError("Undefined loss type.")
        return loss

class MyVAE(torch.nn.Module):
    """Frequency-enhenced CVAE"""

    def __init__(self):
        super(MyVAE, self).__init__()

        self.hp = Config()
        self.vae = CVAE(self.hp)
        self.atten = nn.ModuleList(
            [
                EncoderLayer_selfattn(
                    self.hp.d_model,
                    self.hp.d_inner,
                    self.hp.n_head,
                    self.hp.d_model // self.hp.n_head,
                    self.hp.d_model // self.hp.n_head,
                    dropout=0.1,
                )
                for _ in range(1)
            ]
        )

    def forward(self, x, mode, mask):
        x = x.view(-1, 1, self.hp.window)
        return self.vae.forward(x, mode, mask)

    def loss(self, x, y_all, z_all, mode="train"):
        y = (y_all[:, -1]).unsqueeze(1)
        mask = torch.logical_not(torch.logical_or(y_all, z_all))
        mu_x, var_x, rec_x, mu, var, loss = self.forward(
            x,
            "train",
            mask,
        )
        return loss

    def fit(self, train_data, device):
        dataset = UniDataset(
            use_label=self.hp.use_label,
            window=self.hp.window,
            mode = "train",
            sliding_window_size=self.hp.sliding_window_size,
            train_data=train_data,
            data_pre_mode=self.hp.data_pre_mode
        )

        batch_size = self.hp.batch_size
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=self.hp.batch_size,
            shuffle=False,
            num_workers=self.hp.num_workers
            )

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hp.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        for epoch in tqdm.trange(self.hp.max_epoch):
            self.train()
            for (x, y_all, z_all) in train_loader:
                x, y_all, z_all = x.to(device), y_all.to(device), z_all.to(device)
                x, y_all, z_all = self.batch_data_augmentation(x, y_all, z_all)
                optimizer.zero_grad()
                loss_val = self.loss(x, y_all, z_all)
                loss_val.backward()
                optimizer.step()
            scheduler.step()

    def predict(self, test_data, test_label, device):
        test_dataset = UniDataset(
            use_label=self.hp.use_label,
            window=self.hp.window,
            mode = "test",
            sliding_window_size=self.hp.sliding_window_size,
            train_data=None,
            test_data=test_data,
            test_label=test_label,
            data_pre_mode=self.hp.data_pre_mode
        )

        batch_size = self.hp.batch_size
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.hp.batch_size,
            shuffle=False,
            num_workers=self.hp.num_workers
            )

        all_y = None
        all_recon_prob = None

        for (x, y_all, z_all) in test_loader:
            x, y_all, z_all = x.to(device), y_all.to(device), z_all.to(device)
            y = (y_all[:, -1]).unsqueeze(1)
            with torch.no_grad():
                mu_x_test, recon_prob = self.forward(x, "test", z_all)
                mask = torch.logical_not(z_all)
                mu_x, var_x, rec_x, mu, var, loss = self.forward(x, "train", mask)
            recon_prob = recon_prob[:, :, -1]
            if all_y is None:
                all_y = y
                all_recon_prob = recon_prob
            else:
                all_y = torch.cat([all_y, y], dim=0)
                all_recon_prob = torch.cat([all_recon_prob, recon_prob], dim=0)

        score = -1 * all_recon_prob.squeeze(1).cpu().numpy()
        test_label = all_y.squeeze(1).long().cpu().numpy()

        return score, test_label

    def batch_data_augmentation(self, x, y, z):
        """missing data injection"""

        if self.hp.point_ano_rate > 0:
            x_a, y_a, z_a = point_ano(x, y, z, self.hp.point_ano_rate)
            x = torch.cat((x, x_a), dim=0)
            y = torch.cat((y, y_a), dim=0)
            z = torch.cat((z, z_a), dim=0)
        if self.hp.seg_ano_rate > 0:
            x_a, y_a, z_a = seg_ano(
                x, y, z, self.hp.seg_ano_rate, method="swap"
            )
            x = torch.cat((x, x_a), dim=0)
            y = torch.cat((y, y_a), dim=0)
            z = torch.cat((z, z_a), dim=0)
        x, y, z = missing_data_injection(
            x, y, z, self.hp.missing_data_rate
        )
        return x, y, z