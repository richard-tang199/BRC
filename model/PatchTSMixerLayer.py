import torch.nn as nn
import torch
from typing import Optional, Tuple
from torch import Tensor
from dataclasses import dataclass
from model.utility import PositionalEncoding


@dataclass
class PatchDetectorConfig:
    window_length: int = 1024
    patch_length: int = 16
    d_model: int = 64
    num_layers: int = 3
    instance_normalization: bool = False
    mode: str = "common_channel"  # "mix_channel" or "common_channel"
    use_position_encoder: bool = False
    positional_encoding_type: str = "sincos"  # "sincos" or "random"
    gated_attn: bool = True
    norm_mlp: str = "LayerNorm"
    self_attn: bool = False
    self_attn_heads: int = 1
    expansion_factor: int = 6
    dropout: float = 0.2
    norm_mlp: str = "LayerNorm"
    norm_eps: float = 1e-5

    def __post_init__(self):
        self.stride = self.patch_length  # patch_stride
        self.num_patches = self.window_length // self.patch_length


@dataclass
class PatchTSMixerEncoderOutput:
    """
    Base class for `PatchTSMixerEncoderOutput`, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, d_model)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class PatchTSMixerBatchNorm(nn.Module):
    """
    Compute batch normalization over the sequence length (time) dimension.
    """

    def __init__(self, config: PatchDetectorConfig):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(config.d_model, eps=1e-5)

    def forward(self, inputs: torch.Tensor):
        """
        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length, d_model)`):
                input for Batch norm calculation
        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, d_model)`
        """
        output = inputs.transpose(1, 2)  # output: (batch_size, d_model, sequence_length)
        output = self.batchnorm(output)
        return output.transpose(1, 2)


class PatchTSMixerNormLayer(nn.Module):
    """Normalization block

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the normalization layer.
        Returns:
            `torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`
        """

        inputs = self.norm(inputs)

        return inputs


class PatchTSMixerMLP(nn.Module):
    def __init__(self, in_features, expansion_factor=2):
        super().__init__()
        self.hidden_size = in_features * expansion_factor
        self.fc1 = nn.Linear(in_features, self.hidden_size)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(self.hidden_size, in_features)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the MLP layer.
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        inputs = self.dropout1(nn.functional.gelu(self.fc1(inputs)))
        inputs = self.fc2(inputs)
        inputs = self.dropout2(inputs)
        return inputs


class PatchTSMixerGatedAttention(nn.Module):
    """
    Module that applies gated attention to input data.

    Args:
        in_size (`int`): The input size.
        out_size (`int`): The output size.
    """

    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.attn_layer = nn.Linear(in_size, out_size)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        attn_weight = self.attn_softmax(self.attn_layer(inputs))
        inputs = inputs * attn_weight
        return inputs


class PatchMixerBlock(nn.Module):
    """This module mixes the patch dimension.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, hidden_size, num_patches, gated_attn=True, expansion_factor=2):
        super().__init__()

        self.norm = PatchTSMixerNormLayer(hidden_size=hidden_size, eps=1e-5)
        self.gated_attn = gated_attn

        self.mlp = PatchTSMixerMLP(
            in_features=num_patches,
            expansion_factor=expansion_factor
        )

        if self.gated_attn:
            self.gating_block = PatchTSMixerGatedAttention(in_size=num_patches, out_size=num_patches)

    def forward(self, hidden_state):
        """
        Args:
            hidden_state (`torch.Tensor`): Input tensor.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden_state

        hidden_state = self.norm(hidden_state)

        # Transpose so that num_patches is the last dimension
        # hidden: [bs * n_vars x num_patch x d_model]
        hidden_state = hidden_state.transpose(1, 2)  # hidden: [bs * n_vars x d_model x num_patch]
        hidden_state = self.mlp(hidden_state)

        if self.gated_attn:
            hidden_state = self.gating_block(hidden_state)

        # Transpose back
        hidden_state = hidden_state.transpose(1, 2)  # hidden: [bs * n_vars x num_patch x d_model]
        # out = hidden_state + residual
        out = hidden_state
        return out


class FeatureMixerBlock(nn.Module):
    """This module mixes the hidden feature dimension.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, hidden_size, gated_attn=True):
        super().__init__()

        self.norm = PatchTSMixerNormLayer(hidden_size=hidden_size)

        self.gated_attn = gated_attn

        self.mlp = PatchTSMixerMLP(
            in_features=hidden_size,
            expansion_factor=2
        )

        if self.gated_attn:
            self.gating_block = PatchTSMixerGatedAttention(in_size=hidden_size, out_size=hidden_size)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.mlp(hidden)

        if self.gated_attn:
            hidden = self.gating_block(hidden)

        # out = hidden + residual
        out = hidden
        return out


class PatchTSMixerLayer(nn.Module):
    """
    The `PatchTSMixer` layer that does all three kinds of mixing.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, hidden_size, num_patches, gated_attn=True, expansion_factor=2):
        super().__init__()

        self.patch_mixer = PatchMixerBlock(hidden_size=hidden_size,
                                           num_patches=num_patches,
                                           gated_attn=gated_attn,
                                           expansion_factor=expansion_factor)

        self.feature_mixer = FeatureMixerBlock(hidden_size=hidden_size,
                                               gated_attn=gated_attn)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """

        hidden = self.patch_mixer(hidden)  # hidden: [bs x n_vars x num_patch x d_model]
        hidden = self.feature_mixer(hidden)  # hidden: (batch_size x num_channels x num_patches x d_model)
        return hidden


class PatchTSMixerBlock(nn.Module):
    """The main computing framework of the `PatchTSMixer` model.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, num_layers, hidden_size, num_patches, gated_attn=True, expansion_factor=2):
        super().__init__()

        self.mixers = nn.ModuleList([PatchTSMixerLayer(
            hidden_size=hidden_size,
            num_patches=num_patches,
            gated_attn=gated_attn,
            expansion_factor=expansion_factor
        ) for _ in range(num_layers)])

    def forward(self, hidden_state, output_hidden_states: bool = False):
        """
        Args:
            hidden_state (`torch.Tensor`): The input tensor.
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well.

        Returns:
            `torch.Tensor`: The embedding. `list`: List of all hidden states if `output_hidden_states` is set to
            `True`.
        """
        all_hidden_states = []

        embedding = hidden_state  # [bs x n_vars x num_patch x d_model]

        for mod in self.mixers:
            embedding = mod(embedding)
            if output_hidden_states:
                all_hidden_states.append(embedding)

        if output_hidden_states:
            return embedding, all_hidden_states
        else:
            return embedding, None


class PatchMixerEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_patches, gated_attn=True,
                 expansion_factor=2):
        super().__init__()
        self.mlp_mixer_encoder = PatchTSMixerBlock(num_layers=num_layers,
                                                   hidden_size=hidden_size,
                                                   num_patches=num_patches,
                                                   gated_attn=gated_attn,
                                                   expansion_factor=expansion_factor)

    def forward(self,
                patch_inputs: Tensor,
                output_hidden_states: Optional[bool] = True):
        """
        Parameters:
           patch_inputs (`torch.Tensor` of shape `(batch_size *num_channels, num_patches, hidden_size)`)
                Masked patched input
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
        """

        last_hidden_state, hidden_states = self.mlp_mixer_encoder(
            hidden_state=patch_inputs,
            output_hidden_states=output_hidden_states)

        return last_hidden_state, hidden_states

if __name__ == '__main__':
    model = PatchMixerEncoder(num_layers=2, hidden_size=128, num_patches=16, gated_attn=True)
    sample = torch.rand(64, 16, 128)
    output = model(sample)
    print(output.last_hidden_state.shape)

