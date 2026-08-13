# EEGConformer
# Authors: Yonghao Song <eeyhsong@gmail.com>
#
from __future__ import annotations
import warnings

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn
import torch.nn.functional as F
import warnings
from collections import OrderedDict
from typing import Dict, Iterable, Optional

import numpy as np
from docstring_inheritance import NumpyDocstringInheritanceInitMeta
from torchinfo import ModelStatistics, summary


from .base import EEGModuleMixin

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

        self.rearrange_stack = Rearrange(
            "b n (h d) -> b h n d",
            h=num_heads,
        )
        self.rearrange_unstack = Rearrange(
            "b h n d -> b n (h d)",
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        queries = self.rearrange_stack(self.queries(x))
        keys = self.rearrange_stack(self.keys(x))
        values = self.rearrange_stack(self.values(x))
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = float("-inf")
            energy = energy.masked_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = self.rearrange_unstack(out)
        out = self.projection(out)
        return out
        
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p, activation: nn.Module = nn.GELU):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            activation(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class EEGConformer(EEGModuleMixin, nn.Module):

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        n_filters_time=40,
        filter_time_length=25,
        pool_time_length=75,
        pool_time_stride=15,
        drop_prob=0.5,
        att_depth=6,
        att_heads=10,
        att_drop_prob=0.5,
        final_fc_length="auto",
        return_features=False,
        activation: nn.Module = nn.ELU,
        activation_transfor: nn.Module = nn.GELU,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        self.mapping = {
            "classification_head.fc.6.weight": "final_layer.final_layer.0.weight",
            "classification_head.fc.6.bias": "final_layer.final_layer.0.bias",
        }

        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        if not (self.n_chans <= 64):
            warnings.warn(
                "This model has only been tested on no more "
                + "than 64 channels. no guarantee to work with "
                + "more channels.",
                UserWarning,
            )

        self.return_features = return_features

        self.patch_embedding = _PatchEmbedding(
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            n_channels=self.n_chans,
            pool_time_length=pool_time_length,
            stride_avg_pool=pool_time_stride,
            drop_prob=drop_prob,
            activation=activation,
        )

        if final_fc_length == "auto":
            assert self.n_times is not None
            self.final_fc_length = self.get_fc_size()
        else:
            self.final_fc_length = final_fc_length

        self.transformer = _TransformerEncoder(
            att_depth=att_depth,
            emb_size=n_filters_time,
            att_heads=att_heads,
            att_drop=att_drop_prob,
            activation=activation_transfor,
        )

        self.fc = _FullyConnected(
            final_fc_length=self.final_fc_length, activation=activation
        )

        self.final_layer = nn.Linear(self.fc.hidden_channels, self.n_outputs)

    def forward(self, x, return_feat=False):
        x = torch.unsqueeze(x, dim=1)  # add one extra dimension
        x = self.patch_embedding(x)
        feature = self.transformer(x)

        x = self.fc(feature)
        x = self.final_layer(x)
        if return_feat:
            return x, feature
        return x

    def get_fc_size(self):
        out = self.patch_embedding(torch.ones((1, 1, self.n_chans, self.n_times)))
        size_embedding_1 = out.detach().cpu().data.numpy().shape[1]
        size_embedding_2 = out.detach().cpu().data.numpy().shape[2]

        return size_embedding_1 * size_embedding_2


class _PatchEmbedding(nn.Module):
    def __init__(
        self,
        n_filters_time,
        filter_time_length,
        n_channels,
        pool_time_length,
        stride_avg_pool,
        drop_prob,
        activation: nn.Module = nn.ELU,
    ):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (1, filter_time_length), (1, 1)),
            nn.Conv2d(n_filters_time, n_filters_time, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(num_features=n_filters_time),
            activation(),
            nn.AvgPool2d(
                kernel_size=(1, pool_time_length), stride=(1, stride_avg_pool)
            ),
            # pooling acts as slicing to obtain 'patch' along the
            # time dimension as in ViT
            nn.Dropout(p=drop_prob),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(
                n_filters_time, n_filters_time, (1, 1), stride=(1, 1)
            ),  # transpose, conv could enhance fiting ability slightly
            Rearrange("b d_model 1 seq -> b seq d_model"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class _ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        res = x
        x = self.fn(x)
        x += res
        return x


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size,
        att_heads,
        att_drop,
        forward_expansion=4,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__(
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, att_heads, att_drop),
                    nn.Dropout(att_drop),
                )
            ),
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size,
                        expansion=forward_expansion,
                        drop_p=att_drop,
                        activation=activation,
                    ),
                    nn.Dropout(att_drop),
                )
            ),
        )


class _TransformerEncoder(nn.Sequential):
    """Transformer encoder module for the transformer encoder.

    Similar to the layers used in ViT.

    Parameters
    ----------
    att_depth : int
        Number of transformer encoder blocks.
    emb_size : int
        Embedding size of the transformer encoder.
    att_heads : int
        Number of attention heads.
    att_drop : float
        Dropout probability for the attention layers.

    """

    def __init__(
        self, att_depth, emb_size, att_heads, att_drop, activation: nn.Module = nn.GELU
    ):
        super().__init__(
            *[
                _TransformerEncoderBlock(
                    emb_size, att_heads, att_drop, activation=activation
                )
                for _ in range(att_depth)
            ]
        )


class _FullyConnected(nn.Module):
    def __init__(
        self,
        final_fc_length,
        drop_prob_1=0.5,
        drop_prob_2=0.3,
        out_channels=256,
        hidden_channels=32,
        activation: nn.Module = nn.ELU,
    ):
        """Fully-connected layer for the transformer encoder.

        Parameters
        ----------
        final_fc_length : int
            Length of the final fully connected layer.
        n_classes : int
            Number of classes for classification.
        drop_prob_1 : float
            Dropout probability for the first dropout layer.
        drop_prob_2 : float
            Dropout probability for the second dropout layer.
        out_channels : int
            Number of output channels for the first linear layer.
        hidden_channels : int
            Number of output channels for the second linear layer.
        return_features : bool
            Whether to return input features.
        """

        super().__init__()
        self.hidden_channels = hidden_channels
        self.fc = nn.Sequential(
            nn.Linear(final_fc_length, out_channels),
            activation(),
            nn.Dropout(drop_prob_1),
            nn.Linear(out_channels, hidden_channels),
            activation(),
            nn.Dropout(drop_prob_2),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out



if __name__ == '__main__':

    sample = torch.ones(32,30,500)
    model = EEGConformer(n_chans=30, n_times=500, n_outputs=5)
    out = model(sample)
    print(out.shape)  # torch.Size([32, 5])
    from torchinfo import summary
    summary(model, (32, 30, 500), device='cpu')

# ====================================================================================================
# Layer (type:depth-idx)                             Output Shape              Param #
# ====================================================================================================
# EEGConformer                                       [32, 5]                   --
# ├─_PatchEmbedding: 1-1                             [32, 27, 40]              --
# │    └─Sequential: 2-1                             [32, 40, 1, 27]           --
# │    │    └─Conv2d: 3-1                            [32, 40, 30, 476]         1,040
# │    │    └─Conv2d: 3-2                            [32, 40, 1, 476]          48,040
# │    │    └─BatchNorm2d: 3-3                       [32, 40, 1, 476]          80
# │    │    └─ELU: 3-4                               [32, 40, 1, 476]          --
# │    │    └─AvgPool2d: 3-5                         [32, 40, 1, 27]           --
# │    │    └─Dropout: 3-6                           [32, 40, 1, 27]           --
# │    └─Sequential: 2-2                             [32, 27, 40]              --
# │    │    └─Conv2d: 3-7                            [32, 40, 1, 27]           1,640
# │    │    └─Rearrange: 3-8                         [32, 27, 40]              --
# ├─_TransformerEncoder: 1-2                         [32, 27, 40]              --
# │    └─_TransformerEncoderBlock: 2-3               [32, 27, 40]              --
# │    │    └─_ResidualAdd: 3-9                      [32, 27, 40]              6,640
# │    │    └─_ResidualAdd: 3-10                     [32, 27, 40]              13,080
# │    └─_TransformerEncoderBlock: 2-4               [32, 27, 40]              --
# │    │    └─_ResidualAdd: 3-11                     [32, 27, 40]              6,640
# │    │    └─_ResidualAdd: 3-12                     [32, 27, 40]              13,080
# │    └─_TransformerEncoderBlock: 2-5               [32, 27, 40]              --
# │    │    └─_ResidualAdd: 3-13                     [32, 27, 40]              6,640
# │    │    └─_ResidualAdd: 3-14                     [32, 27, 40]              13,080
# │    └─_TransformerEncoderBlock: 2-6               [32, 27, 40]              --
# │    │    └─_ResidualAdd: 3-15                     [32, 27, 40]              6,640
# │    │    └─_ResidualAdd: 3-16                     [32, 27, 40]              13,080
# │    └─_TransformerEncoderBlock: 2-7               [32, 27, 40]              --
# │    │    └─_ResidualAdd: 3-17                     [32, 27, 40]              6,640
# │    │    └─_ResidualAdd: 3-18                     [32, 27, 40]              13,080
# │    └─_TransformerEncoderBlock: 2-8               [32, 27, 40]              --
# │    │    └─_ResidualAdd: 3-19                     [32, 27, 40]              6,640
# │    │    └─_ResidualAdd: 3-20                     [32, 27, 40]              13,080
# ├─_FullyConnected: 1-3                             [32, 32]                  --
# │    └─Sequential: 2-9                             [32, 32]                  --
# │    │    └─Linear: 3-21                           [32, 256]                 276,736
# │    │    └─ELU: 3-22                              [32, 256]                 --
# │    │    └─Dropout: 3-23                          [32, 256]                 --
# │    │    └─Linear: 3-24                           [32, 32]                  8,224
# │    │    └─ELU: 3-25                              [32, 32]                  --
# │    │    └─Dropout: 3-26                          [32, 32]                  --
# ├─Linear: 1-4                                      [32, 5]                   165
# ====================================================================================================
# Total params: 454,245
# Trainable params: 454,245
# Non-trainable params: 0
# Total mult-adds (Units.GIGABYTES): 1.22
# ====================================================================================================
# Input size (MB): 1.92
# Forward/backward pass size (MB): 174.57
# Params size (MB): 1.82
# Estimated Total Size (MB): 178.31
# ====================================================================================================