# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tqdm
import numpy as np
from toolkit.utils import *
import thop


class AttentionLayer(nn.Module):
    def __init__(self, d_model):
        super(AttentionLayer, self).__init__()

        self.norm = nn.LayerNorm(d_model)

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        # [B, T, D]
        B, T, D = x.shape

        queries = self.query_projection(x)
        keys = self.key_projection(x).transpose(1,2)
        values = self.value_projection(x)

        attn = torch.softmax(torch.matmul(queries, keys) / math.sqrt(D), -1)

        out = torch.matmul(attn, values) + x

        return self.out_projection(self.norm(out)) + out, attn

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe
        self.register_buffer('pe', pe)

    def forward(self, data=None, idx=None):
        if data != None:
            p = self.pe[:data].unsqueeze(0)
        else:
            p = self.pe.unsqueeze(0).repeat(idx.shape[0],1,1)[torch.arange(idx.shape[0])[:,None],idx,:]
        return p


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.05):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(data = x.shape[1])
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, T, D]A
        attlist = []
        for attn_layer in self.attn_layers:
            x, _ = attn_layer(x)
            attlist.append(_)

        if self.norm is not None:
            x = self.norm(x)

        return x, attlist


class FreEnc(nn.Module):
    def __init__(self, c_in, c_out, d_model, e_layers, win_size, fr):
        super(FreEnc, self).__init__()

        self.emb = DataEmbedding(c_in, d_model)

        self.enc = Encoder(
            [
                AttentionLayer(d_model) for l in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.mask_token = nn.Parameter(torch.zeros(1, d_model, 1, dtype=torch.cfloat))

        self.fr = fr

    def forward(self, x):
        # x: [B, T, C]
        ex = self.emb(x)  # [B, T, D]

        # converting to frequency domain and calculating the mag
        cx = torch.fft.rfft(ex.transpose(1, 2))
        mag = torch.sqrt(cx.real ** 2 + cx.imag ** 2)  # [B, D, Mag]

        # masking smaller mag
        quantile = torch.quantile(mag, self.fr, dim=2, keepdim=True)
        idx = torch.argwhere(mag < quantile)
        cx[mag < quantile] = self.mask_token.repeat(ex.shape[0], 1, mag.shape[-1])[idx[:, 0], idx[:, 1], idx[:, 2]]

        # converting to time domain
        ix = torch.fft.irfft(cx).transpose(1, 2)

        # encoding tokens
        dx, att = self.enc(ix)

        rec = self.pro(dx)
        att.append(rec)

        return att  # att(list): [B, T, T]


class TemEnc(nn.Module):
    def __init__(self, c_in, c_out, d_model, e_layers, win_size, seq_size, tr):
        super(TemEnc, self).__init__()

        self.emb = DataEmbedding(c_in, d_model)
        self.pos_emb = PositionalEmbedding(d_model)

        self.enc = Encoder(
            [
                AttentionLayer(d_model) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.dec = Encoder(
            [
                AttentionLayer(d_model) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.tr = int(tr * win_size)
        self.seq_size = seq_size

    def forward(self, x):
        # x: [B, T, C]
        ex = self.emb(x)  # [B, T, D]
        device = x.device
        filters = torch.ones(1, 1, self.seq_size).to(device)
        ex2 = ex ** 2

        # calculating summation of ex and ex2
        ltr = F.conv1d(ex.transpose(1, 2).reshape(-1, ex.shape[1]).unsqueeze(1), filters, padding=self.seq_size - 1)
        ltr[:, :, :self.seq_size - 1] /= torch.arange(1, self.seq_size).to(device)
        ltr[:, :, self.seq_size - 1:] /= self.seq_size
        ltr2 = F.conv1d(ex2.transpose(1, 2).reshape(-1, ex.shape[1]).unsqueeze(1), filters, padding=self.seq_size - 1)
        ltr2[:, :, :self.seq_size - 1] /= torch.arange(1, self.seq_size).to(device)
        ltr2[:, :, self.seq_size - 1:] /= self.seq_size

        # calculating mean and variance
        ltrd = (ltr2 - ltr ** 2)[:, :, :ltr.shape[-1] - self.seq_size + 1].squeeze(1).reshape(ex.shape[0], ex.shape[-1],
                                                                                              -1).transpose(1, 2)
        ltrm = ltr[:, :, :ltr.shape[-1] - self.seq_size + 1].squeeze(1).reshape(ex.shape[0], ex.shape[-1],
                                                                                -1).transpose(1, 2)
        score = ltrd.sum(-1) / ltrm.sum(-1)

        # mask time points
        masked_idx, unmasked_idx = score.topk(self.tr, dim=1, sorted=False)[1], \
        (-1 * score).topk(x.shape[1] - self.tr, dim=1, sorted=False)[1]
        unmasked_tokens = ex[torch.arange(ex.shape[0])[:, None], unmasked_idx, :]

        # encoding unmasked tokens and getting masked tokens
        ux, _ = self.enc(unmasked_tokens)
        masked_tokens = self.mask_token.repeat(ex.shape[0], masked_idx.shape[1], 1) + self.pos_emb(idx=masked_idx)

        tokens = torch.zeros(ex.shape, device=device)
        tokens[torch.arange(ex.shape[0])[:, None], unmasked_idx, :] = ux
        tokens[torch.arange(ex.shape[0])[:, None], masked_idx, :] = masked_tokens

        # decoding tokens
        dx, att = self.dec(tokens)

        rec = self.pro(dx)
        att.append(rec)

        return att  # att(list): [B, T, T]


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.sum(res, dim=-1)


class MTFA(nn.Module):
    def __init__(self, win_size, seq_size, c_in, c_out, d_model=512, e_layers=3, fr=0.4, tr=0.5, dev=None):
        super(MTFA, self).__init__()
        self.device = dev
        self.tem = TemEnc(c_in, c_out, d_model, e_layers, win_size, seq_size, tr)
        self.fre = FreEnc(c_in, c_out, d_model, e_layers, win_size, fr)
        self.tem = self.tem.to(dev)
        self.fre = self.fre.to(dev)


    def forward(self, x):
        # x: [B, T, C]
        tematt = self.tem(x)  # tematt: [B, T, T]
        freatt = self.fre(x)  # freatt: [B, T, T]
        return tematt, freatt

    def fit(self, train_loader, optimizer, epochs):
        self.train()

        for epoch in tqdm.trange(epochs):
            optimizer.zero_grad()
            loss_list = []

            for (input_data, ) in train_loader:
                input_data = input_data.to(self.device)
                tematt, freatt = self(input_data)

                adv_loss = 0.0
                con_loss = 0.0

                for u in range(len(freatt)):
                    adv_loss += (torch.mean(my_kl_loss(tematt[u], (
                            freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach())) + torch.mean(
                        my_kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach(),
                            tematt[u])))
                    con_loss += (torch.mean(
                        my_kl_loss((freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                                tematt[u].detach())) + torch.mean(
                        my_kl_loss(tematt[u].detach(),
                                (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)))))


                adv_loss /= len(freatt)
                con_loss /= len(freatt)

                loss = con_loss - adv_loss
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

            # print(f"Epoch {epoch+1}/{epochs}, Loss: {sum(loss_list)/len(loss_list)}")

    def predict(self, test_loader):
        self.eval()
        temperature = 50

        attens_energy = None
        with torch.no_grad():
            for i, (input_data, ) in enumerate(test_loader):
                input = input_data.float().to(self.device)
                tematt, freatt = self(input)
                if i == 0:
                    flops, params = thop.profile(self, inputs=(input,))

                adv_loss = 0.0
                con_loss = 0.0
                for u in range(len(freatt)):
                    if u == 0:
                        adv_loss = my_kl_loss(tematt[u], (
                                freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1),
                                                            dim=-1)).detach()) * temperature
                        con_loss = my_kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                            tematt[u].detach()) * temperature
                    else:
                        adv_loss += my_kl_loss(tematt[u], (
                                freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1),
                                                            dim=-1)).detach()) * temperature
                        con_loss += my_kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                            tematt[u].detach()) * temperature

                metric = torch.softmax((adv_loss + con_loss), dim=-1)
                cri = metric.detach().cpu().numpy()
                if attens_energy is None:
                    attens_energy = cri
                else:
                    attens_energy = np.concatenate((attens_energy, cri), axis=0)

            # attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_energy = attens_energy[:, :, np.newaxis]

        return test_energy, flops, params


if __name__ == '__main__':
    model = MTFA(win_size=100, seq_size=10, c_in=1, c_out=1, d_model=100, e_layers=3, fr=0.4, tr=0.25, dev='cuda:0')
    test_data = torch.randn(1000, 1)
    test_loader, _ = get_dataloader(
        data=test_data,
        window_length=100,
        batch_size=16,
        test_stride=100,
        train_stride=1,
        mode='test'
    )
    model.to('cuda:0')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.fit(train_loader=test_loader, optimizer=optimizer, epochs=10)
    pred = model.predict(test_loader)
    print(pred.shape)



