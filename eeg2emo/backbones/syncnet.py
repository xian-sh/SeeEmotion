import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from numpy import arange, ceil

from .base import EEGModuleMixin


class SyncNet(EEGModuleMixin, nn.Module):
    """Synchronization Network (SyncNet) from Li, Y et al (2017) [Li2017]_.

    .. figure:: https://braindecode.org/dev/_static/model/SyncNet.png
        :align: center
        :alt: SyncNet Architecture

    SyncNet uses parameterized 1-dimensional convolutional filters inspired by
    the Morlet wavelet to extract features from EEG signals. The filters are
    dynamically generated based on learnable parameters that control the
    oscillation and decay characteristics.

    The filter for channel ``c`` and filter ``k`` is defined as:

    .. math::

        f_c^{(k)}(\\tau) = amplitude_c^{(k)} \\cos(\\omega^{(k)} \\tau + \\phi_c^{(k)}) \\exp(-\\beta^{(k)} \\tau^2)

    where:
    - :math:`amplitude_c^{(k)}` is the amplitude parameter (channel-specific).
    - :math:`\\omega^{(k)}` is the frequency parameter (shared across channels).
    - :math:`\\phi_c^{(k)}` is the phase shift (channel-specific).
    - :math:`\\beta^{(k)}` is the decay parameter (shared across channels).
    - :math:`\\tau` is the time index.

    Parameters
    ----------
    num_filters : int, optional
        Number of filters in the convolutional layer. Default is 1.
    filter_width : int, optional
        Width of the convolutional filters. Default is 40.
    pool_size : int, optional
        Size of the pooling window. Default is 40.
    activation : nn.Module, optional
        Activation function to apply after pooling. Default is ``nn.ReLU``.
    ampli_init_values : tuple of float, optional
        The initialization range for amplitude parameter using uniform
        distribution. Default is (-0.05, 0.05).
    omega_init_values : tuple of float, optional
        The initialization range for omega parameters using uniform
        distribution. Default is (0, 1).
    beta_init_values : tuple of float, optional
        The initialization range for beta parameters using uniform
        distribution. Default is (0, 1). Default is (0, 0.05).
    phase_init_values : tuple of float, optional
        The initialization range for phase parameters using `normal`
        distribution. Default is (0, 1). Default is (0, 0.05).


    Notes
    -----
    This implementation is not guaranteed to be correct! it has not been checked
    by original authors. The modifications are based on derivated code from
    [CodeICASSP2025]_.


    References
    ----------
    .. [Li2017] Li, Y., Dzirasa, K., Carin, L., & Carlson, D. E. (2017).
       Targeting EEG/LFP synchrony with neural nets. Advances in neural
       information processing systems, 30.
    .. [CodeICASSP2025] Code from Baselines for EEG-Music Emotion Recognition
       Grand Challenge at ICASSP 2025.
       https://github.com/SalvoCalcagno/eeg-music-challenge-icassp-2025-baselines

    """

    def __init__(
        self,
        # braindecode convention
        n_chans=None,
        n_times=None,
        n_outputs=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # model parameters
        num_filters=1,
        filter_width=40,
        pool_size=40,
        activation: nn.Module = nn.ReLU,
        ampli_init_values: tuple[float, float] = (-0.05, 0.05),
        omega_init_values: tuple[float, float] = (0.0, 1.0),
        beta_init_values: tuple[float, float] = (0.0, 0.05),
        phase_init_values: tuple[float, float] = (0.0, 0.05),
    ):
        super().__init__(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            chs_info=chs_info,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.num_filters = num_filters
        self.filter_width = filter_width
        self.pool_size = pool_size
        self.activation = activation()
        self.ampli_init_values = ampli_init_values
        self.omega_init_values = omega_init_values
        self.beta_init_values = beta_init_values
        self.phase_init_values = phase_init_values

        # Initialize parameters
        self.amplitude = nn.Parameter(
            torch.FloatTensor(1, 1, self.n_chans, self.num_filters).uniform_(
                self.ampli_init_values[0], self.ampli_init_values[1]
            )
        )
        self.omega = nn.Parameter(
            torch.FloatTensor(1, 1, 1, self.num_filters).uniform_(
                self.omega_init_values[0], self.omega_init_values[1]
            )
        )

        self.bias = nn.Parameter(torch.zeros(self.num_filters))

        # Calculate the output size after pooling
        self.classifier_input_size = int(
            ceil(float(self.n_times) / float(self.pool_size)) * self.num_filters
        )

        # Create time vector t
        if self.filter_width % 2 == 0:
            t_range = arange(-int(self.filter_width / 2), int(self.filter_width / 2))
        else:
            t_range = arange(
                -int((self.filter_width - 1) / 2), int((self.filter_width - 1) / 2) + 1
            )

        t_np = t_range.reshape(1, self.filter_width, 1, 1)
        self.t = nn.Parameter(torch.FloatTensor(t_np))
        # Phase Shift
        self.phi_ini = nn.Parameter(
            torch.FloatTensor(1, 1, self.n_chans, self.num_filters).normal_(
                self.beta_init_values[0], self.beta_init_values[1]
            )
        )
        self.beta = nn.Parameter(
            torch.FloatTensor(1, 1, 1, self.num_filters).uniform_(
                self.phase_init_values[0], self.phase_init_values[1]
            )
        )

        self.padding = self._compute_padding(filter_width=self.filter_width)
        self.pad_input = nn.ConstantPad1d(self.padding, 0.0)
        self.pad_res = nn.ConstantPad1d(self.padding, 0.0)

        # Define pooling and classifier layers
        self.pool = nn.MaxPool2d((1, self.pool_size), stride=(1, self.pool_size))

        self.ensuredim = Rearrange("batch ch time -> batch ch 1 time")

        self.final_layer = nn.Linear(self.classifier_input_size, self.n_outputs)

    def forward(self, x, return_feat=False):
        """Forward pass of the SyncNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_chans, n_times)

        Returns
        -------
        out : torch.Tensor
            Output tensor of shape (batch_size, n_outputs).

        """
        # Ensure input tensor has shape (batch_size, n_chans, 1, n_times)
        x = self.ensuredim(x)
        # Output: (batch_size, n_chans, 1, n_times)

        # Compute the oscillatory component
        W_osc = self.amplitude * torch.cos(self.t * self.omega + self.phi_ini)
        # W_osc is (1, filter_width, n_chans, 1)

        # Compute the decay component
        t_squared = torch.pow(self.t, 2)  # Shape: (filter_width,)
        t_squared_beta = t_squared * self.beta  # Shape: (filter_width, num_filters)
        W_decay = torch.exp(-t_squared_beta)
        # W_osc is (1, filter_width, 1, 1)

        # Combine oscillatory and decay components
        # W shape: (1, n_chans, num_filters, filter_width)
        W = W_osc * W_decay
        # W shape will be: (1, filter_width, n_chans, 1)

        W = W.view(self.num_filters, self.n_chans, 1, self.filter_width)

        # Apply convolution
        x_padded = self.pad_input(x.float())

        res = F.conv2d(x_padded, W.float(), bias=self.bias, stride=1)

        # Apply padding to the convolution result
        res_padded = self.pad_res(res)
        res_pooled = self.pool(res_padded)

        # Flatten the result
        res_flat = res_pooled.view(-1, self.classifier_input_size)

        # Ensure beta remains non-negative
        self.beta.data.clamp_(min=0)

        # Apply activation
        out = self.activation(res_flat)
        feat = out
        # Apply classifier
        out = self.final_layer(out)
        if return_feat:
            return out, feat
        return out

    @staticmethod
    def _compute_padding(filter_width):
        # Compute padding
        P = filter_width - 2
        if P % 2 == 0:
            padding = (P // 2, P // 2 + 1)
        else:
            padding = (P // 2, P // 2)
        return padding

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
    model = SyncNet(n_times=500, n_outputs=5, chs_info=chs_info, n_chans=30, sfreq=100.0, input_window_seconds=5,)
    out = model(sample)
    print(out.shape)  # torch.Size([32, 5])
    from torchinfo import summary
    summary(model, (32, 30, 500), device='cpu')

# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# SyncNet                                  [32, 5]                   103
# ├─Rearrange: 1-1                         [32, 30, 1, 500]          --
# ├─ConstantPad1d: 1-2                     [32, 30, 1, 539]          --
# ├─ConstantPad1d: 1-3                     [32, 1, 1, 539]           --
# ├─MaxPool2d: 1-4                         [32, 1, 1, 13]            --
# ├─ReLU: 1-5                              [32, 13]                  --
# ├─Linear: 1-6                            [32, 5]                   70
# ==========================================================================================
# Total params: 173
# Trainable params: 173
# Non-trainable params: 0
# Total mult-adds (Units.MEGABYTES): 0.00
# ==========================================================================================
# Input size (MB): 1.92
# Forward/backward pass size (MB): 0.00
# Params size (MB): 0.00
# Estimated Total Size (MB): 1.92
# ==========================================================================================