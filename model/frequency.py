# -*-coding:utf-8-*-
import torch

def time_to_timefreq(x: torch.Tensor, n_fft=4, C=1):
    """
    x: (Batch_size * channel_num, time_length)
    """
    x = torch.stft(x, n_fft, normalized=True,
                   return_complex=True, window=torch.hann_window(n_fft,
                                                                 device=x.device), hop_length=None)
    # x: (Batch_size * channel_num, H, W, 2)
    x = torch.view_as_real(x)
    # x: (Batch_size * channel_num, W, H, 2)
    x = x.permute(0, 2, 1, 3)
    # x: (Batch_size * channel_num, W, H * 2)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
    return x

def time_to_timefreq_v2(x: torch.Tensor, n_fft=4):
    """
    x: (Batch_size * channel_num, time_length)
    """
    x = torch.stft(x, n_fft, normalized=True,
                   return_complex=True, window=torch.hann_window(n_fft,
                                                                 device=x.device), hop_length=None)
    # x: (Batch_size * channel_num, H, W, 2)
    x = torch.view_as_real(x)
    # x: (Batch_size * channel_num, 2, W, H)
    x = x.permute(0, 3, 2, 1)
    return x

