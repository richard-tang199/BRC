import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.random import RandomState
from torch.utils.data import DataLoader
import os
import random
import numpy as np
import torch
import string
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import tqdm
import thop

mask = ''.join(random.sample(string.ascii_letters, 8))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, intermediate_dir,
                 patience=7, verbose=False, delta=5e-5, model_name="",
                 trace_func=print, structrue='torch'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.structure = structrue

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

        if structrue == 'torch':
            self.path = os.path.join(intermediate_dir, model_name, model_name + "." + mask + '_checkpoint.pt')
        elif structrue == 'keras':
            self.path = os.path.join(intermediate_dir, model_name, model_name + "." + mask + '_checkpoint.pt')

        self.trace_func = trace_func

        os.makedirs(os.path.split(self.path)[0], exist_ok=True)

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.structure == 'torch':
            torch.save(model.state_dict(), self.path)
        elif self.structure == 'keras':
            model.save(self.path)
        self.val_loss_min = val_loss

def get_sub_seqs(x_arr, seq_len, stride=1, start_discont=np.array([])):
    """
    :param start_discont: the start points of each sub-part in case the x_arr is just multiple parts joined together
    :param x_arr: dim 0 is time, dim 1 is channels
    :param seq_len: size of window used to create subsequences from the data
    :param stride: number of time points the window will move between two subsequences
    :return:
    """
    excluded_starts = []
    [excluded_starts.extend(range((start - seq_len + 1), start)) for start in start_discont if start > seq_len]
    seq_starts = np.delete(np.arange(0, x_arr.shape[0] - seq_len + 1, stride), excluded_starts)
    x_seqs = np.array([x_arr[i:i + seq_len] for i in seq_starts])
    return x_seqs

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 bias=True, dropout=0.2, residual=True):
        """
        Residual block

        :param n_inputs: int, input channels
        :param n_outputs: int, output channels
        :param kernel_size: int, convolutional kernel size
        :param stride: int,
        :param dilation: int,
        :param padding: int,
        :param dropout: float, dropout
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, bias=bias,
                                           dilation=dilation))

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, bias=bias,
                                           dilation=dilation))
        self.Chomp1d = Chomp1d(padding)
        self.dropout = torch.nn.Dropout(dropout)

        self.residual = residual

        self.net = nn.Sequential(self.conv1, Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
                                 self.conv2, Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout))
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()



    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)

        if self.residual:
            res = x if self.downsample is None else self.downsample(x)
            return out+res
        else:
            return out


class NetModule(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=32, rep_hidden=32, pretext_hidden=16,
                 emb_dim=10, kernel_size=2, dropout=0.2, out_dim=2,
                 tcn_bias=True, linear_bias=True,
                 dup=False, pretext=False):
        super(NetModule, self).__init__()

        self.layers = []

        if type(hidden_dims) == int: hidden_dims = [hidden_dims]
        num_layers = len(hidden_dims)
        for i in range(num_layers):
            dilation_size = 2 ** i
            padding_size = (kernel_size-1) * dilation_size
            in_channels = input_dim if i == 0 else hidden_dims[i-1]
            out_channels = hidden_dims[i]
            self.layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                          stride=1, dilation=dilation_size,
                                          padding=padding_size, dropout=dropout,
                                          bias=tcn_bias, residual=True)]
        self.network = nn.Sequential(*self.layers)
        self.l1 = nn.Linear(hidden_dims[-1], rep_hidden, bias=linear_bias)
        self.l2 = nn.Linear(rep_hidden, emb_dim, bias=linear_bias)
        self.act = torch.nn.LeakyReLU()

        self.dup = dup
        self.pretext = pretext

        if dup:
            self.l1_dup = nn.Linear(hidden_dims[-1], rep_hidden, bias=linear_bias)

        if pretext:
            self.pretext_l1 = nn.Linear(hidden_dims[-1], pretext_hidden, bias=linear_bias)
            self.pretext_l2 = nn.Linear(pretext_hidden, out_dim, bias=linear_bias)

    def forward(self, x):
        out = self.network(x.transpose(2, 1)).transpose(2, 1)
        out = out[:, -1]
        rep = self.l2(self.act(self.l1(out)))

        # pretext head
        if self.pretext:
            score = self.pretext_l2(self.act(self.pretext_l1(out)))

            if self.dup:
                rep_dup = self.l2(self.act(self.l1_dup(out)))
                return rep, rep_dup, score
            else:
                return rep, score

        else:
            if self.dup:
                rep_dup = self.l2(self.act(self.l1_dup(out)))
                return rep, rep_dup
            else:
                return rep



class COUTA:
    def __init__(self, sequence_length=100, stride=1,
                 num_epochs=40, batch_size=128, lr=1e-4, ss_type='FULL',
                 hidden_dims=16, emb_dim=16, rep_hidden=16, pretext_hidden=16,
                 kernel_size=2, dropout=0.0, bias=True,
                 alpha=0.1, neg_batch_ratio=0.2, es=False, train_val_pc=0.25,
                 seed=0, device='cuda:0',
                 logger=None, model_dir='couta_model/',
                 save_model_path=None, load_model_path=None,
                 nac=True, umc=True
                 ):
        """
        COUTA class for Calibrated One-class classifier for Unsupervised Time series Anomaly detection

        Parameters
        ----------
        sequence_length: integer, default=100
            sliding window length
        stride: integer, default=1
            sliding window stride
        num_epochs: integer, default=40
            the number of training epochs
        batch_size: integer, default=64
            the size of mini-batches
        lr: float, default=1e-4
            learning rate
        ss_type: string, default='FULL'
            types of perturbation operation type, which can be 'FULL' (using all
            three anomaly types), 'point', 'contextual', or 'collective'.
        hidden_dims: integer or list of integer, default=16,
            the number of neural units in the hidden layer
        emb_dim: integer, default=16
            the dimensionality of the feature space
        rep_hidden: integer, default=16
            the number of neural units of the hidden layer
        pretext_hidden: integer, default=16
        kernel_size: integer, default=2
            the size of the convolutional kernel in TCN
        dropout: float, default=0
            the dropout rate
        bias: bool, default=True
            the bias term of the linear layer
        alpha: float, default=0.1
            the weight of the classification head of NAC
        neg_batch_ratio: float, default=0.2
            the ratio of generated native anomaly examples
        es: bool, default=False
            early stopping
        seed: integer, default=42
            random state seed
        device: string, default='cuda'
        logger: logger or print, default=None
        model_dir: string, default='couta_model/'
            directory to store intermediate model files
        nac: bool, default=True
            used for ablation study
        umc: bool, default=True
            used for ablation study
        """

        self.seq_len = sequence_length
        self.stride = stride

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device

        self.ss_type = ss_type

        self.kernel_size = kernel_size
        self.dropout = dropout
        self.hidden_dims = hidden_dims
        self.rep_hidden = rep_hidden
        self.pretext_hidden = pretext_hidden
        self.emb_dim = emb_dim
        self.bias = bias

        self.alpha = alpha
        self.neg_batch_size = int(neg_batch_ratio * self.batch_size)
        self.max_cut_ratio = 0.5

        self.es = es
        self.train_val_pc = train_val_pc

        self.log_func = logger.info if logger is not None else print
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        param_lst = locals()
        del param_lst['self']
        del param_lst['device']
        del param_lst['logger']
        self.log_func(param_lst)

        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        self.save_model_path = save_model_path
        self.load_model_path = load_model_path

        self.net = None
        self.c = None
        self.test_df = None
        self.test_labels = None

        # for ablation study
        self.nac = nac
        self.umc = umc

        return

    def fit(self, X: pd.DataFrame):
        """
        Fit detector.

        Parameters
        ----------
        X: dataframe of pandas
            input training set
        """
        dim = X.shape[1]
        data = X.values

        sequences = get_sub_seqs(data, seq_len=self.seq_len, stride=self.stride)
        sequences = sequences[RandomState(42).permutation(len(sequences))]

        if self.train_val_pc > 0:
            train_seqs = sequences[: -int(self.train_val_pc * len(sequences))]
            val_seqs = sequences[-int(self.train_val_pc * len(sequences)):]
        else:
            train_seqs = sequences
            val_seqs = None

        self.net = self.network_init(dim)
        self.set_c(train_seqs)
        self.net = self.train(self.net, train_seqs, val_seqs)

        if self.save_model_path is not None:
            os.makedirs(os.path.split(self.save_model_path)[0], exist_ok=True)
            state = {'model_state': self.net.state_dict(), 'c': self.c}
            torch.save(state, self.save_model_path)


    def train(self, net, train_seqs, val_seqs=None):
        val_loader = DataLoader(dataset=SubseqData(val_seqs),
                                batch_size=self.batch_size,
                                drop_last=False, shuffle=False) if val_seqs is not None else None
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

        criterion_oc = DSVDDLoss(c=self.c)
        criterion_oc_umc = DSVDDUncLoss(c=self.c, reduction='mean')
        criterion_mse = torch.nn.MSELoss(reduction='mean')

        early_stp = EarlyStopping(intermediate_dir=self.model_dir,
                                  patience=7, delta=1e-6, model_name='couta', verbose=False)

        y0 = -1 * torch.ones(self.batch_size).float().to(self.device)

        net.train()
        for i in tqdm.trange(self.num_epochs):
            train_loader = DataLoader(dataset=SubseqData(train_seqs),
                                      batch_size=self.batch_size,
                                      drop_last=True, pin_memory=True, shuffle=True)

            rng = RandomState(seed=self.seed+i)
            epoch_seed = rng.randint(0, 1e+6, len(train_loader))
            loss_lst, loss_oc_lst, loss_ssl_lst, = [], [], []
            for ii, x0 in enumerate(train_loader):
                x0 = x0.float().to(self.device)

                x0_output = net(x0)

                if self.umc:
                    rep_x0 = x0_output[0]
                    rep_x0_dup = x0_output[1]
                    loss_oc = criterion_oc_umc(rep_x0, rep_x0_dup)
                else:
                    rep = x0_output[0]
                    loss_oc = criterion_oc(rep)

                if self.nac:
                    neg_cand_idx = RandomState(epoch_seed[ii]).randint(0, self.batch_size, self.neg_batch_size)
                    x1, y1 = create_batch_neg(batch_seqs=x0[neg_cand_idx],
                                              max_cut_ratio=self.max_cut_ratio,
                                              seed=epoch_seed[ii],
                                              return_mul_label=False,
                                              ss_type=self.ss_type)
                    x1, y1 = x1.to(self.device), y1.to(self.device)
                    y = torch.hstack([y0, y1])

                    x1_output = net(x1)
                    pred_x1 = x1_output[-1]
                    pred_x0 = x0_output[-1]

                    out = torch.cat([pred_x0, pred_x1]).view(-1)

                    loss_ssl = criterion_mse(out, y)
                else:
                    loss_ssl = 0.

                loss = loss_oc + self.alpha * loss_ssl

                net.zero_grad()
                loss.backward()
                optimizer.step()

                loss_lst.append(loss)
                loss_oc_lst.append(loss_oc)
                # loss_ssl_lst.append(loss_ssl)

            epoch_loss = torch.mean(torch.stack(loss_lst)).data.cpu().item()
            epoch_loss_oc = torch.mean(torch.stack(loss_oc_lst)).data.cpu().item()
            # epoch_loss_ssl = torch.mean(torch.stack(loss_ssl_lst)).data.cpu().item()

            # validation phase
            val_loss = np.nan
            if val_seqs is not None:
                val_loss = []
                with torch.no_grad():
                    for x in val_loader:
                        x = x.float().to(self.device)
                        x_out = net(x)
                        if self.umc:
                            loss = criterion_oc_umc(x_out[0], x_out[1])
                        else:
                            loss = criterion_oc(x_out[0])
                        loss = torch.mean(loss)
                        val_loss.append(loss)
                val_loss = torch.mean(torch.stack(val_loss)).data.cpu().item()

            if (i+1) % 10 == 0:
                self.log_func(
                    f'|>>> epoch: {i+1:02}  |   loss: {epoch_loss:.6f}, '
                    f'loss_oc: {epoch_loss_oc:.6f}, '
                    # f'loss_ssl: {epoch_loss_ssl:.6f}, <<<|'
                    f'val_loss: {val_loss:.6f}'
                )

            if self.es:
                # early_metric = val_loss+epoch_loss if val_loader is not None else epoch_loss
                early_metric = epoch_loss_oc
                early_stp(early_metric, model=net)

                if early_stp.early_stop:
                    net.load_state_dict(torch.load(early_stp.path))
                    self.log_func("early stop")
                    break
                if i == self.num_epochs - 1:
                    net.load_state_dict(torch.load(early_stp.path))
        return net


    def predict(self, X):
        """
        Predict raw anomaly score of X using the fitted detector.
        For consistency, outliers are assigned with larger anomaly scores.

        Parameters
        ----------
            X: pd.DataFrame
                testing dataframe

        Returns
        -------
            predictions_dic: dictionary of predicted results
            The anomaly score of the input samples.
        """
        data = X.values
        test_sub_seqs = get_sub_seqs(data, seq_len=self.seq_len, stride=1)
        test_dataset = SubseqData(test_sub_seqs)
        dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)

        if self.load_model_path is not None:
            state = torch.load(self.load_model_path)
            self.net = self.network_init(data.shape[1])
            self.net.load_state_dict(state['model_state'])
            self.c = state['c']

        representation_lst = []
        representation_lst2 = []
        self.net.eval()
        with torch.no_grad():
            for i, x in enumerate(dataloader):
                x = x.float().to(self.device)
                x_output = self.net(x)
                if i == 0:
                    flops, params = thop.profile(self.net, inputs=(x,))
                representation_lst.append(x_output[0])
                if self.umc:
                    representation_lst2.append(x_output[1])

        reps = torch.cat(representation_lst)
        dis = torch.sum((reps - self.c) ** 2, dim=1).data.cpu().numpy()

        if self.umc:
            reps_dup = torch.cat(representation_lst2)
            dis2 = torch.sum((reps_dup - self.c) ** 2, dim=1).data.cpu().numpy()
            dis = dis + dis2

        dis_pad = np.hstack([0 * np.ones(data.shape[0] - dis.shape[0]), dis])

        predictions_dic = {'score_t': dis_pad,
                           'score_tc': None,
                           'error_t': None,
                           'error_tc': None,
                           'recons_tc': None,
                           'flops': flops,
                           'params': params
                           }

        return predictions_dic

    def network_init(self, dim):
        net = NetModule(
            input_dim=dim,
            hidden_dims=self.hidden_dims,
            emb_dim=self.emb_dim,
            pretext_hidden=self.pretext_hidden,
            rep_hidden=self.rep_hidden,
            out_dim=1,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            linear_bias=self.bias,
            tcn_bias=self.bias,
            pretext=True if self.nac else False,
            dup=True if self.umc else False
        )
        net.to(self.device)
        return net

    def set_c(self, seqs, eps=0.1):
        """Initializing the center for the hypersphere"""
        dataloader = DataLoader(dataset=SubseqData(seqs), batch_size=self.batch_size,
                                  drop_last=True, pin_memory=True, shuffle=True)
        z_ = []
        self.net.eval()
        with torch.no_grad():
            for x in dataloader:
                x = x.float().to(self.device)
                x_output = self.net(x)
                rep = x_output[0]
                z_.append(rep.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        self.c = c


def create_batch_neg(batch_seqs, max_cut_ratio=0.5, seed=0, return_mul_label=False, ss_type='FULL'):
    """
    create a batch of negative samples based on the input sequences,
    the output batch size is the same as the input batch size
    :param batch_seqs: input sequences
    :param max_cut_ratio:
    :param seed:
    :param return_mul_label:
    :param type:
    :param ss_type:
    :return:

    """
    rng = np.random.RandomState(seed=seed)

    batch_size, l, dim = batch_seqs.shape
    cut_start = l - rng.randint(1, int(max_cut_ratio * l), size=batch_size)
    n_cut_dim = rng.randint(1, dim+1, size=batch_size)
    cut_dim = [rng.randint(dim, size=n_cut_dim[i]) for i in range(batch_size)]

    if type(batch_seqs) == np.ndarray:
        batch_neg = batch_seqs.copy()
        neg_labels = np.zeros(batch_size, dtype=int)
    else:
        batch_neg = batch_seqs.clone()
        neg_labels = torch.LongTensor(batch_size)


    if ss_type != 'FULL':
        pool = rng.randint(1e+6, size=int(1e+4))
        if ss_type == 'collective':
            pool = [a % 6 == 0 or a % 6 == 1 for a in pool]
        elif ss_type == 'contextual':
            pool = [a % 6 == 2 or a % 6 == 3 for a in pool]
        elif ss_type == 'point':
            pool = [a % 6 == 4 or a % 6 == 5 for a in pool]
        flags = rng.choice(pool, size=batch_size, replace=False)
    else:
        flags = rng.randint(1e+5, size=batch_size)

    n_types = 6
    for ii in range(batch_size):
        flag = flags[ii]

        # collective anomalies
        if flag % n_types == 0:
            batch_neg[ii, cut_start[ii]:, cut_dim[ii]] = 0
            neg_labels[ii] = 1

        elif flag % n_types == 1:
            batch_neg[ii, cut_start[ii]:, cut_dim[ii]] = 1
            neg_labels[ii] = 1

        # contextual anomalies
        elif flag % n_types == 2:
            mean = torch.mean(batch_neg[ii, -10:, cut_dim[ii]], dim=0)
            batch_neg[ii, -1, cut_dim[ii]] = mean + 0.5
            neg_labels[ii] = 2

        elif flag % n_types == 3:
            mean = torch.mean(batch_neg[ii, -10:, cut_dim[ii]], dim=0)
            batch_neg[ii, -1, cut_dim[ii]] = mean - 0.5
            neg_labels[ii] = 2

        # point anomalies
        elif flag % n_types == 4:
            batch_neg[ii, -1, cut_dim[ii]] = 2
            neg_labels[ii] = 3

        elif flag % n_types == 5:
            batch_neg[ii, -1, cut_dim[ii]] = -2
            neg_labels[ii] = 3

    if return_mul_label:
        return batch_neg, neg_labels
    else:
        neg_labels = torch.ones(batch_size).long()
        return batch_neg, neg_labels


class SubseqData(Dataset):
    def __init__(self, x, y=None, w1=None, w2=None):
        self.sub_seqs = x
        self.label = y
        self.sample_weight1 = w1
        self.sample_weight2 = w2

    def __len__(self):
        return len(self.sub_seqs)

    def __getitem__(self, idx):
        if self.label is not None and self.sample_weight1 is not None and self.sample_weight2 is not None:
            return self.sub_seqs[idx], self.label[idx], self.sample_weight1[idx], self.sample_weight2[idx]

        if self.label is not None:
            return self.sub_seqs[idx], self.label[idx]

        elif self.sample_weight1 is not None and self.sample_weight2 is None:
            return self.sub_seqs[idx], self.sample_weight[idx]

        elif self.sample_weight1 is not None and self.sample_weight2 is not None:
            return self.sub_seqs[idx], self.sample_weight1[idx], self.sample_weight2[idx]

        return self.sub_seqs[idx]


class DSVDDUncLoss(torch.nn.Module):
    def __init__(self, c, reduction='mean'):
        super(DSVDDUncLoss, self).__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, rep, rep2):
        dis1 = torch.sum((rep - self.c) ** 2, dim=1)
        dis2 = torch.sum((rep2 - self.c) ** 2, dim=1)
        var = (dis1 - dis2) ** 2

        loss = 0.5*torch.exp(torch.mul(-1, var)) * (dis1+dis2) + 0.5*var

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class DSVDDLoss(torch.nn.Module):
    def __init__(self, c, reduction='mean'):
        super(DSVDDLoss, self).__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, rep, sample_weight=None):
        loss = torch.sum((rep - self.c) ** 2, dim=1)
        if sample_weight is not None:
            loss = loss * sample_weight

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss