# -*-coding:utf-8-*-
import copy
import torch.nn as nn
import torch
from model.transformer import RoformerEncoder
from toolkit.distort import *
import random
from model.RWKV import RWKV_TimeMix, RWKVConfig
from model.frequency import time_to_timefreq, time_to_timefreq_v2
from model.PatchTSMixerLayer import PatchMixerEncoder, PatchDetectorConfig


class PatchDetectorGru(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = expansion_ratio * patch_size
        self.proj = nn.Linear(patch_size, self.hidden_size)
        self.head = nn.Linear(2 * self.hidden_size, patch_size)
        self.encoder = nn.GRU(input_size=self.hidden_size,
                              hidden_size=self.hidden_size,
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=True,
                              dropout=0.1)
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, x):
        """
        x: (batch_size, seq_len, num_channels) # seq_len is multiple of patch_size
        """
        batch_size, seq_len, num_channels = x.shape
        device = x.device
        x = x.permute(0, 2, 1) # (batch_size, num_channels, seq_len)
        x = x.reshape(batch_size * num_channels, seq_len)
        # transfer to patches and reshape to (batch_size * num_channels, seq_len//patch_size, patch_size)
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        _, patch_num, _ = patches.shape
        # generate mask for patches
        noise = torch.rand(batch_size * num_channels, patch_num, device=device)
        mask = torch.ones(batch_size * num_channels, patch_num, device=device)
        len_keep = int(patch_num * 0.5)
        mask[:, :len_keep] = 0
        ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [bs * num_channels x L]
        mask = torch.gather(mask, dim=-1, index=ids_restore)  # mask: [bs * num_channels x L]
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
        # apply mask to patches
        mask_patches = patches.masked_fill(mask.bool(), 0)

        # apply projection to patches
        proj_patches = self.proj(mask_patches)

        # apply GRU to patches
        encoder_output = self.encoder(proj_patches)[0]
        recons = self.head(encoder_output)

        recon_loss = self.loss_fn(recons, patches)
        loss = (recon_loss * mask).mean()

        return loss

    # def predict(self, x):
    #     """
    #     x: (batch_size, seq_len, num_channels) # seq_len is multiple of patch_size
    #     """
    #     batch_size, seq_len, num_channels = x.shape
    #     device = x.device
    #     x = x.permute(0, 2, 1) # (batch_size, num_channels, seq_len)
    #     x = x.reshape(batch_size * num_channels, seq_len)
    #
    #     # transfer to patches and reshape to (batch_size * num_channels, seq_len//patch_size, patch_size)
    #     patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
    #     _, patch_num, _ = patches.shape
    #
    #     # generate mask for patches
    #     noise = torch.rand(batch_size * num_channels, patch_num, device=device)
    #     mask = torch.ones(batch_size * num_channels, patch_num, device=device)
    #     len_keep = int(patch_num * 0.5)
    #     mask[:, :len_keep] = 0
    #     ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
    #     ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [bs * num_channels x L]
    #     mask = torch.gather(mask, dim=-1, index=ids_restore)  # mask: [bs * num_channels x L]
    #     mask_0 = mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
    #     mask_1 = (mask_0 == 0).long()
    #
    #     # apply mask to patches
    #     mask_patches_0 = patches.masked_fill(mask_0.bool(), 0)
    #     mask_patches_1 = patches.masked_fill(mask_1.bool(), 0)
    #
    #     # apply projection to patches
    #     proj_patches_0 = self.proj(mask_patches_0)
    #     proj_patches_1 = self.proj(mask_patches_1)
    #
    #     # apply GRU to patches
    #     encoder_output_0 = self.encoder(proj_patches_0)[0]
    #     encoder_output_1 = self.encoder(proj_patches_1)[0]
    #
    #     # restore patches
    #     recons_0 = self.head(encoder_output_0)
    #     recons_1 = self.head(encoder_output_1)
    #     recons = recons_0 * mask_0 + recons_1 * mask_1
    #     recons = recons.reshape(batch_size, num_channels, seq_len)
    #     recons = recons.permute(0, 2, 1)
    #
    #     return recons

    def predict(self, x, masking_matrix = None):
        """
        x: (batch_size, seq_len, num_channels) # seq_len is multiple of patch_size
        masking_matrix: (batch_size, num_channels, patch_num, patch_size)
        """
        batch_size, seq_len, num_channels = x.shape
        device = x.device
        x = x.permute(0, 2, 1) # (batch_size, num_channels, seq_len)
        x = x.reshape(batch_size * num_channels, seq_len)

        # transfer to patches and reshape to (batch_size * num_channels, seq_len//patch_size, patch_size)
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        _, patch_num, _ = patches.shape

        if masking_matrix is None:
            noise = torch.rand(batch_size * num_channels, patch_num, device=device)
            mask = torch.ones(batch_size * num_channels, patch_num, device=device)
            len_keep = int(patch_num * 0.5)
            mask[:, :len_keep] = 0
            ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [bs * num_channels x L]
            mask = torch.gather(mask, dim=-1, index=ids_restore)  # mask: [bs * num_channels x L]
            mask_0 = mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
            mask_1 = (mask_0 == 0).long()

            # apply mask to patches
            mask_patches_0 = patches.masked_fill(mask_0.bool(), 0)
            mask_patches_1 = patches.masked_fill(mask_1.bool(), 0)

            # apply projection to patches
            proj_patches_0 = self.proj(mask_patches_0)
            proj_patches_1 = self.proj(mask_patches_1)

            # apply GRU to patches
            encoder_output_0 = self.encoder(proj_patches_0)[0]
            encoder_output_1 = self.encoder(proj_patches_1)[0]

            # restore patches
            recons_0 = self.head(encoder_output_0)
            recons_1 = self.head(encoder_output_1)
            recons = recons_0 * mask_0 + recons_1 * mask_1
            recons = recons.reshape(batch_size, num_channels, seq_len)
            recons = recons.permute(0, 2, 1)

        else:
            # (batch_size, num_channels, patch_num, patch_size)
            masking_matrix = masking_matrix.reshape(batch_size * num_channels, patch_num, self.patch_size)
            # masked_patch: (batch_size * num_channels, patch_num, patch_size)
            masked_patch = patches * masking_matrix

            proj_patches_0 = self.proj(masked_patch)
            encoder_output_0 = self.encoder(proj_patches_0)[0]
            recons_0 = self.head(encoder_output_0)

            # get masked parts recon values, mask the unmasked parts
            recons_substitute = recons_0 * (~masking_matrix)

            # using recon values to predict unmasked parts
            proj_patches_1 = self.proj(recons_substitute)
            encoder_output_1 = self.encoder(proj_patches_1)[0]
            recons_1 = self.head(encoder_output_1)

            # unify two output
            recons = recons_0 * (~masking_matrix) + recons_1 * masking_matrix
            recons = recons.reshape(batch_size, num_channels, seq_len)
            recons = recons.permute(0, 2, 1)
            return recons

        return recons

    def generate_mask(self, x):
        # x (batch_size, seq_length, num_channels)
        batch_size, seq_length, num_channels = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size * num_channels, seq_length)
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        # (batch_size * num_channels, patch_num, 1)
        _, patch_num, _ = patches.shape
        patch_output, patch_hidden = self.encoder(patches)
        classify_result = self.classifier(patch_output)
        classify_result = classify_result.softmax(dim=-1)

        # (batch_size * num_channels, patch_num, 1)
        anomaly_score = classify_result[:, :, 1]

        # classify_score
        classify_score = copy.deepcopy(anomaly_score.detach())
        classify_score = classify_score.unsqueeze(-1).repeat(1, 1, self.patch_size)
        classify_score = classify_score.reshape(batch_size, num_channels, patch_num * self.patch_size)
        classify_score = classify_score.permute(0, 2, 1)

        # (batch_size, num_channels, patch_num)
        anomaly_score = anomaly_score.reshape(batch_size, num_channels, patch_num)
        # generate top%50 masking index
        mask_num = int(0.5 * patch_num)
        _, top_k_indices = torch.topk(anomaly_score, mask_num, dim=-1)
        # generate masking matrix
        masking_matrix = torch.ones_like(anomaly_score, dtype=torch.bool)
        masking_matrix.scatter_(-1, top_k_indices, 0)
        # (batch_size, num_channels, patch_num, patch_size)
        masking_matrix = masking_matrix.unsqueeze(-1).repeat(1, 1, 1, self.patch_size)

        return masking_matrix, classify_score

class PatchDetectorGruCenter(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers, mask_ratio=0.2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, patch_size)
        self.head = nn.Linear(2 * expansion_ratio * patch_size, patch_size)
        self.encoder = nn.GRU(input_size=patch_size,
                              hidden_size=patch_size*expansion_ratio,
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=True,
                              dropout=0.1)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.mask_ratio = mask_ratio

    def forward(self, x):
        """
        x: (batch_size, seq_len, num_channels) # seq_len is multiple of patch_size
        """
        batch_size, seq_len, num_channels = x.shape
        device = x.device
        x = x.permute(0, 2, 1) # (batch_size, num_channels, seq_len)
        x = x.reshape(batch_size * num_channels, seq_len)
        # transfer to patches and reshape to (batch_size * num_channels, seq_len//patch_size, patch_size)
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        _, patch_num, _ = patches.shape
        # generate mask for patches
        mask_centre = patch_num // 2
        mask_patch_num = int(patch_num * self.mask_ratio)
        mask = torch.zeros_like(patches, device=x.device)
        mask[:, mask_centre - mask_patch_num // 2 : mask_centre + mask_patch_num // 2, :] = 1
        # apply mask to patches
        mask_patches = patches.masked_fill(mask.bool(), 0)

        # apply projection to patches
        proj_patches = self.proj(mask_patches)

        # apply GRU to patches
        encoder_output = self.encoder(proj_patches)[0]
        recons = self.head(encoder_output)

        recon_loss = self.loss_fn(recons, patches)
        loss = (recon_loss * mask).mean()

        return loss

    def predict(self, x):
        """
        x: (batch_size, seq_len, num_channels) # seq_len is multiple of patch_size
        """
        batch_size, seq_len, num_channels = x.shape
        device = x.device
        x = x.permute(0, 2, 1) # (batch_size, num_channels, seq_len)
        x = x.reshape(batch_size * num_channels, seq_len)

        # transfer to patches and reshape to (batch_size * num_channels, seq_len//patch_size, patch_size)
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        _, patch_num, _ = patches.shape

        # generate mask for patches
        mask_centre = patch_num // 2
        mask_patch_num = int(patch_num * self.mask_ratio)
        mask = torch.zeros_like(patches, device=x.device)
        mask[:, mask_centre - mask_patch_num // 2: mask_centre + mask_patch_num // 2, :] = 1
        # apply mask to patches
        mask_patches = patches.masked_fill(mask.bool(), 0)

        # apply projection to patches
        proj_patches = self.proj(mask_patches)

        # apply GRU to patches
        encoder_output = self.encoder(proj_patches)[0]
        recons = self.head(encoder_output)
        recons = recons[:, mask_centre - mask_patch_num // 2: mask_centre + mask_patch_num // 2, :]
        recons = recons.reshape(batch_size, num_channels, -1)
        recons = recons.permute(0, 2, 1)

        return recons

class PatchDetectorAttention(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = patch_size * expansion_ratio
        self.proj = nn.Linear(self.patch_size, self.hidden_size)
        self.head = nn.Linear(self.hidden_size, self.patch_size)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.encoder = RoformerEncoder(emb_size=self.hidden_size,
                                       nhead=6,
                                       dim_feedforward=self.hidden_size,
                                       num_layers=num_layers,
                                       dropout=0.1)

    def forward(self, x):
        """

        Parameters
        ----------
        x: (batch_size, seq_len, num_channels) # seq_len is multiple of patch_size

        Returns
        -------

        """
        batch_size, seq_len, num_channels = x.shape
        device = x.device
        x = x.permute(0, 2, 1)  # (batch_size, num_channels, seq_len)
        x = x.reshape(batch_size * num_channels, seq_len)
        # transfer to patches and reshape to (batch_size * num_channels, seq_len//patch_size, patch_size)
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        _, patch_num, _ = patches.shape
        # generate mask for patches
        noise = torch.rand(batch_size * num_channels, patch_num, device=device)
        mask = torch.ones(batch_size * num_channels, patch_num, device=device)
        len_keep = int(patch_num * 0.5)
        mask[:, :len_keep] = 0
        ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [bs * num_channels x L]
        mask = torch.gather(mask, dim=-1, index=ids_restore)  # mask: [bs * num_channels x L]
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
        # apply mask to patches
        mask_patches = patches.masked_fill(mask.bool(), 0)

        # apply projection to patches
        proj_patches = self.proj(mask_patches) # [bs * num_channels x L x hidden_size]
        proj_patches = proj_patches.permute(1, 0, 2) # [L x bs * num_channels x hidden_size]
        encoder_out = self.encoder(proj_patches) # [L x bs * num_channels x hidden_size]
        encoder_out = encoder_out.permute(1, 0, 2) # [bs * num_channels x L x hidden_size]
        recons = self.head(encoder_out)

        recon_loss = self.loss_fn(recons, patches)
        loss = (recon_loss * mask).mean()

        return loss

    def predict(self, x):
        """
        x: (batch_size, seq_len, num_channels) # seq_len is multiple of patch_size
        """
        batch_size, seq_len, num_channels = x.shape
        device = x.device
        x = x.permute(0, 2, 1) # (batch_size, num_channels, seq_len)
        x = x.reshape(batch_size * num_channels, seq_len)

        # transfer to patches and reshape to (batch_size * num_channels, seq_len//patch_size, patch_size)
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        _, patch_num, _ = patches.shape

        # generate mask for patches
        noise = torch.rand(batch_size * num_channels, patch_num, device=device)
        mask = torch.ones(batch_size * num_channels, patch_num, device=device)
        len_keep = int(patch_num * 0.5)
        mask[:, :len_keep] = 0
        ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [bs * num_channels x L]
        mask = torch.gather(mask, dim=-1, index=ids_restore)  # mask: [bs * num_channels x L]
        mask_0 = mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
        mask_1 = (mask_0 == 0).long()

        # apply mask to patches
        mask_patches_0 = patches.masked_fill(mask_0.bool(), 0)
        mask_patches_1 = patches.masked_fill(mask_1.bool(), 0)

        # apply projection to patches
        proj_patches_0 = self.proj(mask_patches_0)
        proj_patches_1 = self.proj(mask_patches_1)
        proj_patches_0 = proj_patches_0.permute(1, 0, 2)
        proj_patches_1 = proj_patches_1.permute(1, 0, 2)

        # apply GRU to patches
        encoder_output_0 = self.encoder(proj_patches_0)
        encoder_output_1 = self.encoder(proj_patches_1)
        encoder_output_0 = encoder_output_0.permute(1, 0, 2)
        encoder_output_1 = encoder_output_1.permute(1, 0, 2)

        # restore patches
        recons_0 = self.head(encoder_output_0)
        recons_1 = self.head(encoder_output_1)
        recons = recons_0 * mask_0 + recons_1 * mask_1
        recons = recons.reshape(batch_size, num_channels, seq_len)
        recons = recons.permute(0, 2, 1)

        return recons


class PatchClassify(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = expansion_ratio * patch_size
        self.proj = nn.Linear(patch_size, self.hidden_size)
        self.encoder = nn.GRU(
            input_size=self.patch_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.classifier = nn.Linear(self.hidden_size * 2, 2)
        self.loss_fn = nn.CrossEntropyLoss()

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)


    def forward(self, x, subsequence_length = None, if_finetune = False):
        """
        Parameters
        ----------
        x: (batch_size, seq_length, num_channels)
        subsequence_length: int, subsequence length for each anomaly sequence
        if_finetune: bool, whether to retrain the model

        Returns
        -------

        """
        batch_size, seq_length, num_channels = x.shape
        distorted_functions = [uniform_distort, scale_distort, jitering_distort, original]
        # labels (batch_size, length)
        x_distorted, labels = random.choice(distorted_functions)(x, subsequence_length)
        x_distorted = x_distorted.permute(0, 2, 1) # (batch_size, num_channels, seq_len)
        # (batch_size * num_channels, seq_len)
        x_distorted = x_distorted.reshape(batch_size * num_channels, seq_length)
        # (batch_size * num_channels, patch_num, patch_size)
        patches_distorted = x_distorted.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        # (batch_size, patch_num, patch_size)
        labels_distorted = labels.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        _, patch_num, _ =  labels_distorted.shape
        labels_distorted = torch.sum(labels_distorted, dim=-1)
        patch_labels = (labels_distorted > 0).long()

        # gru embedding patch_output (batch_size * num_channels, patch_num, hidden_size * 2)
        patch_output, patch_hidden = self.encoder(patches_distorted)

        #  (batch_size * num_channels, patch_num, 2)
        classify_result = self.classifier(patch_output)
        classify_result = classify_result.reshape(batch_size * patch_num, -1)
        patch_labels = patch_labels.reshape(batch_size * patch_num)
        patch_labels = patch_labels.to(x.device)
        # classify_result = classify_result.softmax(dim=-1)

        # calculate loss
        loss = self.loss_fn(classify_result, patch_labels)

        return loss

    def predict(self, x):
        batch_size, seq_length, num_channels = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size * num_channels, seq_length)
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        # (batch_size * num_channels, patch_num, 1)
        _, patch_num, _ = patches.shape
        patch_output, patch_hidden = self.encoder(patches)
        classify_result = self.classifier(patch_output)
        classify_result = classify_result.softmax(dim=-1)
        # (batch_size * num_channels, patch_num, 1)
        anomaly_score = classify_result[:, :, 1]
        anomaly_score = anomaly_score.unsqueeze(-1)
        # (batch_size * num_channels, patch_num, patch_size)
        anomaly_score = anomaly_score.repeat(1, 1, self.patch_size)
        # (batch_size, num_channels, patch_num * patch_size)
        anomaly_score = anomaly_score.reshape(batch_size, num_channels, patch_num * self.patch_size)
        anomaly_score = anomaly_score.permute(0, 2, 1)
        return anomaly_score


class Patch_RWKV(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = expansion_ratio * patch_size
        self.proj = nn.Linear(patch_size, self.hidden_size)
        rwkv_config = RWKVConfig(n_layer=2, n_head=2, n_embd=self.hidden_size, dropout=0.1, bias=True)
        self.encoder = nn.ModuleList([RWKV_TimeMix(rwkv_config, i) for i in range(rwkv_config.n_layer)])
        self.classifier = nn.Linear(self.hidden_size, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, subsequence_length = None):
        """
        Parameters
        ----------
        x: (batch_size, seq_length, num_channels)

        Returns
        -------

        """
        batch_size, seq_length, num_channels = x.shape
        distorted_functions = [uniform_distort, scale_distort, jitering_distort, original]
        # labels (batch_size, length)
        x_distorted, labels = random.choice(distorted_functions)(x, subsequence_length)
        x_distorted = x_distorted.permute(0, 2, 1) # (batch_size, num_channels, seq_len)
        # (batch_size * num_channels, seq_len)
        x_distorted = x_distorted.reshape(batch_size * num_channels, seq_length)
        # (batch_size * num_channels, patch_num, patch_size)
        patches_distorted = x_distorted.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        # (batch_size, patch_num, patch_size)
        labels_distorted = labels.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        _, patch_num, _ =  labels_distorted.shape
        labels_distorted = torch.sum(labels_distorted, dim=-1)
        patch_labels = (labels_distorted > 0).long()

        patch_embd = self.proj(patches_distorted)
        # gru embedding patch_output (batch_size * num_channels, patch_num, hidden_size * 2)
        for i, layer in enumerate(self.encoder):
            patch_embd = layer(patch_embd)

        #  (batch_size * num_channels, patch_num, 2)
        classify_result = self.classifier(patch_embd)
        classify_result = classify_result.reshape(batch_size * patch_num, -1)
        patch_labels = patch_labels.reshape(batch_size * patch_num)
        patch_labels = patch_labels.to(x.device)
        # classify_result = classify_result.softmax(dim=-1)

        # calculate loss
        loss = self.loss_fn(classify_result, patch_labels)

        return loss

    def predict(self, x):
        batch_size, seq_length, num_channels = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size * num_channels, seq_length)
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        # (batch_size * num_channels, patch_num, 1)
        _, patch_num, _ = patches.shape

        patch_embd = self.proj(patches)
        for i, layer in enumerate(self.encoder):
            patch_embd = layer(patch_embd)

        classify_result = self.classifier(patch_embd)
        classify_result = classify_result.softmax(dim=-1)
        # (batch_size * num_channels, patch_num, 1)
        anomaly_score = classify_result[:, :, 1]
        anomaly_score = anomaly_score.unsqueeze(-1)
        # (batch_size * num_channels, patch_num, patch_size)
        anomaly_score = anomaly_score.repeat(1, 1, self.patch_size)
        # (batch_size, num_channels, patch_num * patch_size)
        anomaly_score = anomaly_score.reshape(batch_size, num_channels, patch_num * self.patch_size)
        anomaly_score = anomaly_score.permute(0, 2, 1)
        return anomaly_score

class PatchFlatten(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = expansion_ratio * patch_size
        self.proj = nn.Linear(patch_size, self.hidden_size)
        self.encoder = nn.GRU(
            input_size=self.patch_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.classifier = nn.Linear(self.hidden_size * 2, self.patch_size)
        self.loss_fn = nn.BCELoss()

    def forward(self, x, subsequence_length=None):
        """
        Parameters
        ----------
        x: (batch_size, seq_length, num_channels)

        Returns
        -------

        """
        batch_size, seq_length, num_channels = x.shape
        distorted_functions = [uniform_distort, scale_distort, jitering_distort, original]
        # labels (batch_size, length)
        x_distorted, labels = random.choice(distorted_functions)(x, subsequence_length)
        labels = labels.to(x.device)
        x_distorted = x_distorted.permute(0, 2, 1)  # (batch_size, num_channels, seq_len)
        # (batch_size * num_channels, seq_len)
        x_distorted = x_distorted.reshape(batch_size * num_channels, seq_length)
        # (batch_size * num_channels, patch_num, patch_size)
        patches_distorted = x_distorted.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        # (batch_size, patch_num, patch_size)
        # labels_distorted = labels.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        # _, patch_num, _ = labels_distorted.shape
        # labels_distorted = torch.sum(labels_distorted, dim=-1)
        # patch_labels = (labels_distorted > 0).long()

        # gru embedding patch_output (batch_size * num_channels, patch_num, hidden_size * 2)
        patch_output, patch_hidden = self.encoder(patches_distorted)

        #  (batch_size * num_channels, patch_num, patch_size)
        classify_result = self.classifier(patch_output)
        classify_result = classify_result.reshape(batch_size * num_channels, -1)
        classify_result = nn.Sigmoid()(classify_result)
        # classify_result = classify_result.softmax(dim=-1)

        # calculate loss
        loss = self.loss_fn(classify_result, labels)

        return loss

    def predict(self, x):
        batch_size, seq_length, num_channels = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size * num_channels, seq_length)
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        # (batch_size * num_channels, patch_num, 1)
        _, patch_num, _ = patches.shape
        patch_output, patch_hidden = self.encoder(patches)
        classify_result = self.classifier(patch_output)
        classify_result = nn.Sigmoid()(classify_result)
        # (batch_size * num_channels, patch_num, 1)
        anomaly_score = classify_result.reshape(batch_size, num_channels, -1)
        # (batch_size * num_channels, patch_num, patch_size)
        anomaly_score = anomaly_score.permute(0, 2, 1)
        return anomaly_score

class PatchUnified(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers):
        super().__init__()
        self.classifier_net = PatchClassify(patch_size=patch_size,
                                        expansion_ratio=expansion_ratio,
                                        num_layers=num_layers)

        self.recon_net = PatchDetectorGru(
            patch_size=patch_size,
            expansion_ratio=expansion_ratio,
            num_layers=num_layers
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, num_channels) # seq_len is multiple of patch_size
        """

        # classify_loss
        classify_loss = self.classifier_net(x)

        # recon_loss
        recon_loss = self.recon_net(x)

        return classify_loss, recon_loss

    def predict(self, x):
        """
        x: (batch_size, seq_len, num_channels) # seq_len is multiple of patch_size
        Parameters
        ----------
        x

        Returns
        -------
        """
        masking_matrix, classify_score = self.classifier_net.generate_mask(x)
        recons = self.recon_net.predict(x, masking_matrix)
        return recons, classify_score

class PatchFrequency(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers, n_fft=4):
        super().__init__()
        self.patch_size = patch_size
        self.n_fft = n_fft
        self.proj_size = patch_size * 2 * (self.n_fft // 2 + 1)
        self.hidden_size = expansion_ratio * self.proj_size
        self.proj = nn.Linear(self.proj_size, self.hidden_size)
        self.act = nn.SELU()
        self.encoder = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.classifier = nn.Linear(self.hidden_size * 2, 2)
        self.loss_fn = nn.CrossEntropyLoss()

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x,
                subsequence_length=None,
                if_finetune=False,
                x_label=None):
        """
        Parameters
        ----------
        x: (batch_size, seq_length, num_channels)

        Returns
        -------

        """
        batch_size, seq_length, num_channels = x.shape
        distorted_functions = [uniform_distort, scale_distort, jitering_distort, original, mirror_flip]
        # labels (batch_size, length)
        if not if_finetune:
            x_distorted, labels = random.choice(distorted_functions)(x, subsequence_length)
        else:
            x_distorted, labels = original(x, subsequence_length)
        x_distorted = x_distorted.permute(0, 2, 1)  # (batch_size, num_channels, seq_len)
        # (batch_size * num_channels, seq_len)
        x_distorted = x_distorted.reshape(batch_size * num_channels, seq_length)

        # x: (Batch_size * channel_num, seq_len, H * 2)
        labels_distorted = labels.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        _, patch_num, _ = labels_distorted.shape
        x_distorted_freq = time_to_timefreq(x_distorted, n_fft=self.n_fft, C=1)
        x_distorted_freq = x_distorted_freq[:, :seq_length, :]
        patch_distorted_freq = x_distorted_freq.view(batch_size, patch_num, -1)
        labels_distorted = torch.sum(labels_distorted, dim=-1)
        patch_labels = (labels_distorted > 0).long()

        # gru embedding patch_output (batch_size * num_channels, patch_num, hidden_size * 2)
        proj_freq = self.proj(patch_distorted_freq)
        proj_freq = self.act(proj_freq)
        patch_output, patch_hidden = self.encoder(proj_freq)

        #  (batch_size * num_channels, patch_num, 2)
        classify_result = self.classifier(patch_output)
        classify_result = classify_result.reshape(batch_size * patch_num, -1)
        patch_labels = patch_labels.reshape(batch_size * patch_num)
        patch_labels = patch_labels.to(x.device)
        # classify_result = classify_result.softmax(dim=-1)

        # calculate loss
        loss = self.loss_fn(classify_result, patch_labels)

        return loss

    def predict(self, x):
        batch_size, seq_length, num_channels = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size * num_channels, seq_length)

        x_freq = time_to_timefreq(x, n_fft=self.n_fft, C=1)
        x_freq = x_freq[:, :seq_length, :]
        patch_freq = x_freq.view(batch_size, seq_length // self.patch_size, -1)
        proj_freq = self.proj(patch_freq)
        proj_freq = self.act(proj_freq)
        patch_output, patch_hidden = self.encoder(proj_freq)

        #  (batch_size * num_channels, patch_num, 2)
        classify_result = self.classifier(patch_output)
        classify_result = classify_result.softmax(dim=-1)
        # (batch_size * num_channels, patch_num, 1)
        anomaly_score = classify_result[:, :, 1]
        anomaly_score = anomaly_score.unsqueeze(-1)
        # (batch_size * num_channels, patch_num, patch_size)
        anomaly_score = anomaly_score.repeat(1, 1, self.patch_size)
        # (batch_size, num_channels, patch_num * patch_size)
        anomaly_score = anomaly_score.reshape(batch_size, num_channels, seq_length // self.patch_size * self.patch_size)
        # (batch_size, seq_len, num_channels)
        anomaly_score = anomaly_score.permute(0, 2, 1)
        return anomaly_score

class PatchFrequencySeperate(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers, n_fft=4):
        super().__init__()
        self.patch_size = patch_size
        self.n_fft = n_fft
        self.proj_size = patch_size * 2 * (self.n_fft // 2 + 1)
        self.hidden_size = expansion_ratio * self.proj_size
        self.proj = nn.Conv2d(
            in_channels=2,
            out_channels=self.hidden_size,
            kernel_size=(1, self.patch_size),
            stride=(1, self.patch_size)
        )
        self.act = nn.SELU()
        self.encoder = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.classifier = nn.Linear(self.hidden_size * 2, 2)
        self.loss_fn = nn.CrossEntropyLoss()

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x,
                subsequence_length=None,
                if_finetune=False,
                x_label=None):
        """
        Parameters
        ----------
        x: (batch_size, seq_length, num_channels)

        Returns
        -------

        """
        batch_size, seq_length, num_channels = x.shape
        distorted_functions = [uniform_distort, scale_distort, jitering_distort, original]
        # labels (batch_size, length)
        if not if_finetune:
            x_distorted, labels = random.choice(distorted_functions)(x, subsequence_length)
        else:
            x_distorted, labels = original(x, subsequence_length)
        x_distorted = x_distorted.permute(0, 2, 1)  # (batch_size, num_channels, seq_len)
        # (batch_size * num_channels, seq_len)
        x_distorted = x_distorted.reshape(batch_size * num_channels, seq_length)

        # x: (Batch_size * channel_num, seq_len, H * 2)
        labels_distorted = labels.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        _, patch_num, _ = labels_distorted.shape
        # x_distorted_freq: (Batch_size * channel_num, 2, H, W)
        x_distorted_freq = time_to_timefreq_v2(x_distorted, n_fft=self.n_fft, C=1)
        x_distorted_freq = x_distorted_freq[:, :, :, :seq_length]
        # x_distorted_freq: (Batch_size * channel_num, hidden_size, H, patch_num)
        x_distorted_freq = self.proj(x_distorted_freq)
        # x_distorted_freq: (Batch_size * channel_num, H, patch_num, hidden_size)
        proj_freq = x_distorted_freq.permute(0, 2, 3, 1)
        # x_distorted_freq: (Batch_size * channel_num * H, patch_num, hidden_size)
        proj_freq = proj_freq.reshape(batch_size * num_channels * (self.n_fft // 2 + 1), patch_num, -1)

        labels_distorted = torch.sum(labels_distorted, dim=-1)
        patch_labels = (labels_distorted > 0).long()

        proj_freq = self.act(proj_freq)
        #  (batch_size * num_channels * H, patch_num, hidden_size)
        patch_output, patch_hidden = self.encoder(proj_freq)

        # (batch_size * num_channels * H, patch_num, 2)
        classify_result = self.classifier(patch_output)
        # (batch_size * num_channels, H, patch_num, 2)
        classify_result = classify_result.reshape(batch_size * num_channels, self.n_fft // 2 + 1, patch_num, -1)

        classify_result = torch.mean(classify_result, dim=1)
        classify_result = classify_result.reshape(batch_size * patch_num, -1)
        patch_labels = patch_labels.reshape(batch_size * patch_num)
        patch_labels = patch_labels.to(x.device)
        # classify_result = classify_result.softmax(dim=-1)

        # calculate loss
        loss = self.loss_fn(classify_result, patch_labels)

        return loss

    def predict(self, x):
        batch_size, seq_length, num_channels = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size * num_channels, seq_length)

        # proj_freq: (Batch_size * channel_num, 2, H, W)
        x_freq = time_to_timefreq_v2(x, n_fft=self.n_fft, C=1)
        x_freq = x_freq[:, :, :, :seq_length]
        # proj_freq (Batch_size * channel_num, hidden_size, H, patch_num)
        proj_freq = self.proj(x_freq)
        # proj_freq (Batch_size * channel_num, H, patch_num, hidden_size)
        proj_freq = proj_freq.permute(0, 2, 3, 1)
        # proj_freq (Batch_size * channel_num * H, patch_num, hidden_size)
        proj_freq = proj_freq.reshape(batch_size * num_channels * (self.n_fft // 2 + 1), seq_length // self.patch_size, -1)
        proj_freq = self.act(proj_freq)
        #  (batch_size * num_channels * H, patch_num, hidden_size)
        patch_output, patch_hidden = self.encoder(proj_freq)

        # (batch_size * num_channels * H, patch_num, 2)
        classify_result = self.classifier(patch_output)
        classify_result = classify_result.softmax(dim=-1)
        # (batch_size * num_channels * H, patch_num, 1)
        anomaly_score = classify_result[:, :, 1]
        # (batch_size * num_channels, H, patch_num, 1)
        anomaly_score = anomaly_score.reshape(batch_size * num_channels, self.n_fft // 2 + 1, seq_length // self.patch_size, -1)
        # (batch_size * num_channels, patch_num, 1)
        anomaly_score = torch.mean(anomaly_score, dim=1)
        # (batch_size * num_channels, patch_num, patch_size)
        anomaly_score = anomaly_score.repeat(1, 1, self.patch_size)
        # (batch_size, num_channels, patch_num * patch_size)
        anomaly_score = anomaly_score.reshape(batch_size, num_channels, seq_length // self.patch_size * self.patch_size)
        anomaly_score = anomaly_score.permute(0, 2, 1)
        return anomaly_score

class PatchFrequencyMultiple(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers, n_fft=4, levels = 3):
        super().__init__()
        self.patch_size = patch_size
        self.n_fft = n_fft
        self.proj_size = patch_size * 2 * (self.n_fft // 2 + 1)
        self.hidden_size = expansion_ratio * self.proj_size
        self.proj = nn.Linear(self.proj_size, self.hidden_size)
        self.act = nn.SELU()
        self.encoder = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.classifier = nn.Linear(self.hidden_size * 2, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.levels = levels

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, subsequence_length=None):
        """
        Parameters
        ----------
        x: (batch_size, seq_length, num_channels)

        Returns
        -------

        """
        batch_size, seq_length, num_channels = x.shape
        distorted_functions = [uniform_distort, scale_distort, jitering_distort, original]
        # labels (batch_size, length)
        x_distorted, labels = random.choice(distorted_functions)(x, subsequence_length)
        x_distorted = x_distorted.permute(0, 2, 1)  # (batch_size, num_channels, seq_len)
        # (batch_size * num_channels, seq_len)
        x_distorted = x_distorted.reshape(batch_size * num_channels, seq_length)

        # x: (Batch_size * channel_num, seq_len, H * 2)
        # labels_distorted = labels.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        # _, patch_num, _ = labels_distorted.shape


        x_distorted_freq = time_to_timefreq(x_distorted, n_fft=self.n_fft, C=1)
        x_distorted_freq = x_distorted_freq[:, :seq_length, :]
        patch_distorted_freq = x_distorted_freq.view(batch_size, seq_length // self.patch_size, -1)

        proj_freq = self.proj(patch_distorted_freq)
        proj_freq = self.act(proj_freq)
        patch_output_0, patch_hidden = self.encoder(proj_freq)

        loss_list = []
        patch_output_multiple = patch_output_0
        total_loss = 0

        for i in range(self.levels):
            patch_output_multiple = self.pool(patch_output_multiple.permute(0, 2, 1))
            patch_output_multiple = patch_output_multiple.permute(0, 2, 1)
            classify_result = self.classifier(patch_output_multiple)
            classify_result = classify_result.reshape(classify_result.shape[0] * classify_result.shape[1], -1)
            patch_labels = labels.unfold(dimension=-1, size=self.patch_size * 2 ** i, step = self.patch_size * 2 ** i)
            patch_labels = torch.sum(patch_labels, dim=-1)
            patch_labels = (patch_labels > 0).long()
            patch_labels = patch_labels.reshape(patch_labels.shape[0] * patch_labels.shape[1])
            patch_labels.to(x.device)
            loss = self.loss_fn(classify_result, patch_labels)
            total_loss += loss
            loss_list.append(loss.item())

        all_loss = torch.sum(loss_list)
        return all_loss, loss_list

    def predict(self, x):
        batch_size, seq_length, num_channels = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size * num_channels, seq_length)

        x_freq = time_to_timefreq(x, n_fft=self.n_fft, C=1)
        x_freq = x_freq[:, :seq_length, :]
        patch_freq = x_freq.view(batch_size, seq_length // self.patch_size, -1)
        proj_freq = self.proj(patch_freq)
        proj_freq = self.act(proj_freq)
        patch_output, patch_hidden = self.encoder(proj_freq)

        #  (batch_size * num_channels, patch_num, 2)
        classify_result = self.classifier(patch_output)
        classify_result = classify_result.softmax(dim=-1)
        # (batch_size * num_channels, patch_num, 1)
        anomaly_score = classify_result[:, :, 1]
        anomaly_score = anomaly_score.unsqueeze(-1)
        # (batch_size * num_channels, patch_num, patch_size)
        anomaly_score = anomaly_score.repeat(1, 1, self.patch_size)
        # (batch_size, num_channels, patch_num * patch_size)
        anomaly_score = anomaly_score.reshape(batch_size, num_channels, seq_length // self.patch_size * self.patch_size)
        anomaly_score = anomaly_score.permute(0, 2, 1)
        return anomaly_score


class PatchClassifyMixer(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers, window_length, n_fft=4):
        super().__init__()
        self.patch_size = patch_size
        self.n_fft = n_fft
        self.proj_size = patch_size * 2 * (self.n_fft // 2 + 1)
        self.hidden_size = expansion_ratio * self.proj_size
        self.act = nn.SELU()
        self.proj = nn.Linear(self.proj_size, self.hidden_size)
        mixer_config = PatchDetectorConfig(
            window_length= window_length,
            patch_length=self.patch_size,
            d_model=self.hidden_size,
            num_layers=num_layers
        )
        self.encoder = PatchMixerEncoder(mixer_config)
        self.classifier = nn.Linear(self.hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self,
                x: torch.Tensor,
                subsequence_length=None,
                if_finetune=False,
                x_label=None):
        """
        Parameters
        ----------
        x: (batch_size, seq_length, num_channels)

        Returns
        -------

        """
        batch_size, seq_length, num_channels = x.shape
        distorted_functions = [uniform_distort, scale_distort, jitering_distort, original, mirror_flip]
        # labels (batch_size, length)
        if not if_finetune:
            x_distorted, labels = random.choice(distorted_functions)(x, subsequence_length)
        else:
            x_distorted, labels = original(x, subsequence_length)
        x_distorted = x_distorted.permute(0, 2, 1)  # (batch_size, num_channels, seq_len)
        # (batch_size * num_channels, seq_len)
        x_distorted = x_distorted.reshape(batch_size * num_channels, seq_length)

        # x: (Batch_size * channel_num, seq_len, H * 2)
        labels_distorted = labels.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        _, patch_num, _ = labels_distorted.shape
        x_distorted_freq = time_to_timefreq(x_distorted, n_fft=self.n_fft, C=1)
        x_distorted_freq = x_distorted_freq[:, :seq_length, :]
        patch_distorted_freq = x_distorted_freq.view(batch_size, patch_num, -1)
        labels_distorted = torch.sum(labels_distorted, dim=-1)
        patch_labels = (labels_distorted > 0).float()

        # gru embedding patch_output (batch_size * num_channels, patch_num, hidden_size * 2)
        proj_freq = self.proj(patch_distorted_freq)
        proj_freq = self.act(proj_freq)
        patch_output = self.encoder(proj_freq)
        patch_output = patch_output.last_hidden_state

        #  (batch_size * num_channels, patch_num, 2)
        classify_result = self.classifier(patch_output)
        classify_result = classify_result.reshape(batch_size * patch_num, -1)
        patch_labels = patch_labels.reshape(batch_size * patch_num, -1)
        patch_labels = patch_labels.to(x.device)

        # calculate loss
        loss = self.loss_fn(classify_result, patch_labels)

        return loss

    def predict(self, x):
        batch_size, seq_length, num_channels = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size * num_channels, seq_length)

        x_freq = time_to_timefreq(x, n_fft=self.n_fft, C=1)
        x_freq = x_freq[:, :seq_length, :]
        patch_freq = x_freq.view(batch_size, seq_length // self.patch_size, -1)
        proj_freq = self.proj(patch_freq)
        proj_freq = self.act(proj_freq)
        patch_output = self.encoder(proj_freq)
        patch_output = patch_output.last_hidden_state

        #  (batch_size * num_channels, patch_num, 1)
        classify_result = self.classifier(patch_output)
        patch_anomaly_score = torch.nn.Sigmoid()(classify_result)
        # (batch_size * num_channels, patch_num, patch_size)
        anomaly_score = patch_anomaly_score.repeat(1, 1, self.patch_size)
        # (batch_size, num_channels, patch_num * patch_size)
        anomaly_score = anomaly_score.reshape(batch_size, num_channels, seq_length)
        # (batch_size, seq_len, num_channels)
        anomaly_score = anomaly_score.permute(0, 2, 1)
        return anomaly_score


if __name__ == '__main__':
    # model = PatchDetectorGruCenter(patch_size=16, expansion_ratio=2, num_layers=4, mask_ratio=0.2)
    model = PatchClassifyMixer(patch_size=16, expansion_ratio=2, num_layers=2, window_length=1024)
    x = torch.rand(32, 1024, 1)
    y = model(x)
    z = model.predict(x)
    print(y.shape)







