# Authors: Bruno Aristimunha <b.aristimunha>
#
# License: BSD (3-clause)

from __future__ import annotations

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from mne.utils import deprecated, warn

from .base import EEGModuleMixin


class TSception(EEGModuleMixin, nn.Module):
    """TSception model from Ding et al. (2020) from [ding2020]_.

    :bdg-success:`Convolution`

    TSception: A deep learning framework for emotion detection using EEG.

    .. figure:: https://user-images.githubusercontent.com/58539144/74716976-80415e00-526a-11ea-9433-02ab2b753f6b.PNG
        :align: center
        :alt: TSception Architecture

    The model consists of temporal and spatial convolutional layers
    (Tception and Sception) designed to learn temporal and spatial features
    from EEG data.

    Parameters
    ----------
    number_filter_temp : int
        Number of temporal convolutional filters.
    number_filter_spat : int
        Number of spatial convolutional filters.
    hidden_size : int
        Number of units in the hidden fully connected layer.
    drop_prob : float
        Dropout rate applied after the hidden layer.
    activation : nn.Module, optional
        Activation function class to apply. Should be a PyTorch activation
        module like ``nn.ReLU`` or ``nn.LeakyReLU``. Default is ``nn.LeakyReLU``.
    pool_size : int, optional
        Pooling size for the average pooling layers. Default is 8.
    inception_windows : list[float], optional
        List of window sizes (in seconds) for the inception modules.
        Default is [0.5, 0.25, 0.125].

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors. The modifications are minimal and the model is expected
    to work as intended. the original code from [code2020]_.

    References
    ----------
    .. [ding2020] Ding, Y., Robinson, N., Zeng, Q., Chen, D., Wai, A. A. P.,
        Lee, T. S., & Guan, C. (2020, July). Tsception: a deep learning framework
        for emotion detection using EEG. In 2020 international joint conference
        on neural networks (IJCNN) (pp. 1-7). IEEE.
    .. [code2020] Ding, Y., Robinson, N., Zeng, Q., Chen, D., Wai, A. A. P.,
        Lee, T. S., & Guan, C. (2020, July). Tsception: a deep learning framework
        for emotion detection using EEG.
        https://github.com/deepBrains/TSception/blob/master/Models.py
    """

    def __init__(
        self,
        # Braindecode parameters
        n_chans=None,
        n_outputs=None,
        input_window_seconds=None,
        chs_info=None,
        n_times=None,
        sfreq=None,
        # Model parameters
        number_filter_temp: int = 9,
        number_filter_spat: int = 6,
        hidden_size: int = 128,
        drop_prob: float = 0.5,
        activation: nn.Module = nn.LeakyReLU,
        pool_size: int = 8,
        inception_windows: tuple[float, float, float] = (0.5, 0.25, 0.125),
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.activation = activation
        self.pool_size = pool_size
        self.inception_windows = inception_windows
        self.number_filter_spat = number_filter_spat
        self.number_filter_temp = number_filter_temp
        self.drop_prob = drop_prob

        ### Layers
        self.ensuredim = Rearrange("batch nchans time -> batch 1 nchans time")
        if self.input_window_seconds < max(self.inception_windows):
            inception_windows = (
                self.input_window_seconds,
                self.input_window_seconds / 2,
                self.input_window_seconds / 4,
            )
            warning_msg = (
                "Input window size is smaller than the maximum inception window size. "
                "We are adjusting the input window size to match the maximum inception window size.\n"
                f"Original input window size: {self.inception_windows}, \n"
                f"Adjusted inception windows: {inception_windows}"
            )
            warn(warning_msg, UserWarning)
            self.inception_windows = inception_windows
        # Define temporal convolutional layers (Tception)
        self.temporal_blocks = nn.ModuleList()
        for window in self.inception_windows:
            # 1. Calculate the temporal kernel size for this block
            kernel_size_t = int(window * self.sfreq)

            # 2. Calculate the output length of the convolution
            conv_out_len = self.n_times - kernel_size_t + 1

            # 3. Ensure the pooling size is not larger than the conv output
            #    and is at least 1.
            dynamic_pool_size = max(1, min(self.pool_size, conv_out_len))

            # 4. Create the block with the dynamic pooling size
            block = self._conv_block(
                in_channels=1,
                out_channels=self.number_filter_temp,
                kernel_size=(1, kernel_size_t),
                stride=1,
                pool_size=dynamic_pool_size,  # Use the dynamic size
                activation=self.activation,
            )
            self.temporal_blocks.append(block)

        self.batch_temporal_lay = nn.BatchNorm2d(self.number_filter_temp)

        # Define spatial convolutional layers (Sception)

        pool_size_spat = self.pool_size // 4

        self.spatial_block_1 = self._conv_block(
            in_channels=self.number_filter_temp,
            out_channels=self.number_filter_spat,
            kernel_size=(self.n_chans, 1),
            stride=1,
            pool_size=pool_size_spat,
            activation=self.activation,
        )

        kernel_size_spat_2 = (max(1, self.n_chans // 2), 1)

        self.spatial_block_2 = self._conv_block(
            in_channels=self.number_filter_temp,
            out_channels=self.number_filter_spat,
            kernel_size=kernel_size_spat_2,
            stride=kernel_size_spat_2,
            pool_size=pool_size_spat,
            activation=self.activation,
        )
        self.batch_spatial_lay = nn.BatchNorm2d(self.number_filter_spat)

        # Calculate the size of the features after convolution and pooling layers
        self.feature_size = self._calculate_feature_size()
        # self.feature_size = self.number_filter_spat *
        # Define the final classification layers

        self.dense_layer = nn.Sequential(
            nn.Linear(self.feature_size, hidden_size),
            self.activation(),
            nn.Dropout(self.drop_prob),
        )

        self.final_layer = nn.Linear(hidden_size, self.n_outputs)

    def forward(self, x, return_feat=False):
        """
        Forward pass of the TSception model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_channels, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_classes).
        """
        # Temporal Convolution
        # shape: (batch_size, n_channels, n_times)
        x = self.ensuredim(x)
        # shape: (batch_size, 1, n_channels, n_times)

        t_features = [layer(x) for layer in self.temporal_blocks]
        # shape: (batch_size, number_filter_temp, n_channels,
        #
        t_out = torch.cat(t_features, dim=-1)

        t_out = self.batch_temporal_lay(t_out)

        # Spatial Convolution
        s_out1 = self.spatial_block_1(t_out)
        s_out2 = self.spatial_block_2(t_out)
        s_out = torch.cat((s_out1, s_out2), dim=2)
        s_out = self.batch_spatial_lay(s_out)
        feat = s_out
        # Flatten and apply final layers
        s_out = s_out.view(s_out.size(0), -1)
        output = self.dense_layer(s_out)
        output = self.final_layer(output)
        if return_feat:
            return output, feat
        return output

    def _calculate_feature_size(self) -> int:
        """
        Calculates the size of the features after convolution and pooling layers.

        Returns
        -------
        int
            Flattened size of the features after convolution and pooling layers.
        """
        with torch.no_grad():
            dummy_input = torch.ones(1, 1, self.n_chans, self.n_times)
            t_features = [layer(dummy_input) for layer in self.temporal_blocks]
            t_out = torch.cat(t_features, dim=-1)
            t_out = self.batch_temporal_lay(t_out)

            s_out1 = self.spatial_block_1(t_out)
            s_out2 = self.spatial_block_2(t_out)
            s_out = torch.cat((s_out1, s_out2), dim=2)
            s_out = self.batch_spatial_lay(s_out)

            feature_size = s_out.view(1, -1).size(1)
        return feature_size

    @staticmethod
    def _conv_block(
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple[int, int] | int,
        pool_size: int,
        activation: nn.Module,
    ) -> nn.Sequential:
        """
        Creates a convolutional block with Conv2d, activation, and AvgPool2d layers.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : tuple
            Size of the convolutional kernel.
        stride : int
            Stride of the convolution.
        pool_size : int
            Size of the pooling kernel.
        activation : nn.Module
            Activation function class.

        Returns
        -------
        nn.Sequential
            A sequential container of the convolutional block.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
            ),
            activation(),
            nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_size)),
        )


@deprecated(
    "`TSceptionV1` was renamed to `TSception` in v1.12; "
    "this alias will be removed in v1.14."
)
class TSceptionV1(TSception):
    """Deprecated alias for TSception."""

    pass

if __name__ == '__main__':
    # EEG 通道
    import matplotlib.pyplot as plt

    # EEG通道位置字典（简化为30通道）
    STANDARD_1005_CHANNEL_LOCATION_DICT = {
    'FP1': [-0.0294367, 0.0839171, -0.00699],
    'FP2': [0.0298723, 0.0848959, -0.00708],
    'F7': [-0.0702629, 0.0424743, -0.01142],
    'F3': [-0.0502438, 0.0531112, 0.042192],
    'FZ': [0.0003122, 0.058512, 0.066462],
    'F4': [0.0518362, 0.0543048, 0.040814],
    'F8': [0.0730431, 0.0444217, -0.012],
    'FC5': [-0.0772149, 0.0186433, 0.02446],
    'FC1': [-0.0340619, 0.0260111, 0.079987],
    'FC2': [0.0347841, 0.0264379, 0.078808],
    'FC6': [0.0795341, 0.0199357, 0.024438],
    'T7': [-0.0841611, -0.0160187, -0.009346],
    'C3': [-0.0653581, -0.0116317, 0.064358],
    'CZ': [0.0004009, -0.009167, 0.100244],
    'C4': [0.0671179, -0.0109003, 0.06358],
    'T8': [0.0850799, -0.0150203, -0.00949],
    'CP5': [-0.0795922, -0.0465507, 0.030949],
    'CP1': [-0.0355131, -0.0472919, 0.091315],
    'CP2': [0.0383838, -0.0470731, 0.090695],
    'CP6': [0.0833218, -0.0461013, 0.031206],
    'P7': [-0.0724343, -0.0734527, -0.002487],
    'P3': [-0.0530073, -0.0787878, 0.05594],
    'PZ': [0.0003247, -0.081115, 0.082615],
    'P4': [0.0556667, -0.0785602, 0.056561],
    'P8': [0.0730557, -0.0730683, -0.00254],
    'PO9': [-0.0549104, -0.0980448, -0.035465],
    'O1': [-0.0294134, -0.112449, 0.008839],
    'OZ': [0.0001076, -0.114892, 0.014657],
    'O2': [0.0298426, -0.112156, 0.0088],
    'PO10': [0.0549876, -0.0980911, -0.035541],
    }

    # 通道名称列表
    ch_names = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP5', 'CP1', 'CP2', 'CP6',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'PO9', 'O1', 'Oz', 'O2', 'PO10'
    ]

    chs_info = [
    {'name': name, 'loc': STANDARD_1005_CHANNEL_LOCATION_DICT[name.upper()]}
    for name in ch_names
    ]

    sample = torch.ones(32,30,500)
    model = TSception(n_times=500, n_outputs=5, chs_info=chs_info, n_chans=30, sfreq=100.0, input_window_seconds=5,)
    out = model(sample)
    print(out.shape)  # torch.Size([32, 5])
    from torchinfo import summary
    summary(model, (32, 30, 500), device='cpu')


# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# TSception                                [32, 5]                   --
# ├─Rearrange: 1-1                         [32, 1, 30, 500]          --
# ├─ModuleList: 1-2                        --                        --
# │    └─Sequential: 2-1                   [32, 9, 30, 56]           --
# │    │    └─Conv2d: 3-1                  [32, 9, 30, 451]          459
# │    │    └─LeakyReLU: 3-2               [32, 9, 30, 451]          --
# │    │    └─AvgPool2d: 3-3               [32, 9, 30, 56]           --
# │    └─Sequential: 2-2                   [32, 9, 30, 59]           --
# │    │    └─Conv2d: 3-4                  [32, 9, 30, 476]          234
# │    │    └─LeakyReLU: 3-5               [32, 9, 30, 476]          --
# │    │    └─AvgPool2d: 3-6               [32, 9, 30, 59]           --
# │    └─Sequential: 2-3                   [32, 9, 30, 61]           --
# │    │    └─Conv2d: 3-7                  [32, 9, 30, 489]          117
# │    │    └─LeakyReLU: 3-8               [32, 9, 30, 489]          --
# │    │    └─AvgPool2d: 3-9               [32, 9, 30, 61]           --
# ├─BatchNorm2d: 1-3                       [32, 9, 30, 176]          18
# ├─Sequential: 1-4                        [32, 6, 1, 88]            --
# │    └─Conv2d: 2-4                       [32, 6, 1, 176]           1,626
# │    └─LeakyReLU: 2-5                    [32, 6, 1, 176]           --
# │    └─AvgPool2d: 2-6                    [32, 6, 1, 88]            --
# ├─Sequential: 1-5                        [32, 6, 2, 88]            --
# │    └─Conv2d: 2-7                       [32, 6, 2, 176]           816
# │    └─LeakyReLU: 2-8                    [32, 6, 2, 176]           --
# │    └─AvgPool2d: 2-9                    [32, 6, 2, 88]            --
# ├─BatchNorm2d: 1-6                       [32, 6, 3, 88]            12
# ├─Sequential: 1-7                        [32, 128]                 --
# │    └─Linear: 2-10                      [32, 128]                 202,880
# │    └─LeakyReLU: 2-11                   [32, 128]                 --
# │    └─Dropout: 2-12                     [32, 128]                 --
# ├─Linear: 1-8                            [32, 5]                   645
# ==========================================================================================
# Total params: 206,807
# Trainable params: 206,807
# Non-trainable params: 0
# Total mult-adds (Units.MEGABYTES): 385.44
# ==========================================================================================
# Input size (MB): 1.92
# Forward/backward pass size (MB): 111.29
# Params size (MB): 0.83
# Estimated Total Size (MB): 114.04
# ==========================================================================================