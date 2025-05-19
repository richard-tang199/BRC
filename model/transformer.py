# -*-coding:utf-8-*-
import torch.nn as nn
import torch
from torch.nn import Transformer
from torch import Tensor
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)) # theta
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos): # pos -> p
        sinusoid_inp = torch.outer(pos.float().squeeze(0), self.inv_freq) # m * theta
        return torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1) # combine sin and cosine frequencies

def apply_rotary_pos_emb(x, sincos):
    sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=1) # split sin and cosine frequencies
    sin = sin.unsqueeze(1)
    cos = cos.unsqueeze(1)
    x1, x2 = x[..., :x.size(-1) // 2], x[..., x.size(-1) // 2:]
    return torch.cat([x1 * cos + x2 * sin, x2 * cos - x1 * sin], dim=-1) # rotate input vectors

# Wrapper around nn.MultiheadAttention
class RotaryAttention(nn.Module):
    def __init__(self, emb_size, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(emb_size, nhead, dropout=dropout)
        self.rotary_emb = RotaryEmbedding(emb_size)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        query_seq_len, key_seq_len, batch_size = query.size(0), key.size(0), query.size(1)
        query_pos = torch.arange(query_seq_len, device=query.device).unsqueeze(0) # Generates a range of position indices from 0 to query_seq_len-1
        query_pos_emb = self.rotary_emb(query_pos)

        key_pos = torch.arange(key_seq_len, device=key.device).unsqueeze(0) # Generates a range of position indices from 0 to key_seq_len-1
        key_pos_emb = self.rotary_emb(key_pos)

        # Apply rotary embeddings to queries and keys
        query, key = apply_rotary_pos_emb(query, query_pos_emb), apply_rotary_pos_emb(key, key_pos_emb)

        # Proceed with MHA
        return self.multihead_attn(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class RoformerEncoderLayer(nn.Module):
    def __init__(self, emb_size: int, nhead: int, dim_feedforward: int, dropout: float):
        super(RoformerEncoderLayer, self).__init__()
        self.self_attn = RotaryAttention(emb_size, nhead, dropout=dropout)
        self.linear1 = nn.Linear(emb_size, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, emb_size)

        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class RoformerEncoder(nn.Module):
    def __init__(self, emb_size, nhead, num_layers, dim_feedforward, dropout):
        super(RoformerEncoder, self).__init__()
        self.layers = nn.ModuleList([RoformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, src, mask = None, src_key_padding_mask = None):
        for layer in self.layers:
            src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        src = self.norm(src)
        return src

if __name__ == '__main__':
    # Test the RoformerEncoder
    encoder = RoformerEncoder(256, 8, 6, 256, 0.1)
    # length, batch_size, emb_size
    sample = torch.rand(10, 32, 256)
    # embedding
    embedding = encoder(sample, None, None)
    print(embedding.shape)


