from math import ceil

import torch
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import init
from torch.nn.utils.parametrizations import weight_norm

from .base import EEGModuleMixin
from .base_util import Ensure4d


class TIDNet(EEGModuleMixin, nn.Module):
    """Thinker Invariance DenseNet model from Kostas et al. (2020) [TIDNet]_.

    :bdg-success:`Convolution`

    .. figure:: https://content.cld.iop.org/journals/1741-2552/17/5/056008/revision3/jneabb7a7f1_hr.jpg
        :align: center
        :alt: TIDNet Architecture

    See [TIDNet]_ for details.

    Parameters
    ----------
    s_growth : int
        DenseNet-style growth factor (added filters per DenseFilter)
    t_filters : int
        Number of temporal filters.
    drop_prob : float
        Dropout probability
    pooling : int
        Max temporal pooling (width and stride)
    temp_layers : int
        Number of temporal layers
    spat_layers : int
        Number of DenseFilters
    temp_span : float
        Percentage of n_times that defines the temporal filter length:
        temp_len = ceil(temp_span * n_times)
        e.g A value of 0.05 for temp_span with 1500 n_times will yield a temporal
        filter of length 75.
    bottleneck : int
        Bottleneck factor within Densefilter
    summary : int
        Output size of AdaptiveAvgPool1D layer. If set to -1, value will be calculated
        automatically (n_times // pooling).
    in_chans :
        Alias for n_chans.
    n_classes:
        Alias for n_outputs.
    input_window_samples :
        Alias for n_times.
    activation: nn.Module, default=nn.LeakyReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.LeakyReLU``.

    Notes
    -----
    Code adapted from: https://github.com/SPOClab-ca/ThinkerInvariance/

    References
    ----------
    .. [TIDNet] Kostas, D. & Rudzicz, F.
        Thinker invariance: enabling deep neural networks for BCI across more
        people.
        J. Neural Eng. 17, 056008 (2020).
        doi: 10.1088/1741-2552/abb7a7.
    """

    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        chs_info=None,
        s_growth: int = 24,
        t_filters: int = 32,
        drop_prob: float = 0.4,
        pooling: int = 15,
        temp_layers: int = 4,
        spat_layers: int = 4,
        temp_span: float = 0.05,
        bottleneck: int = 3,
        summary: int = -1,
        activation: nn.Module = nn.LeakyReLU,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            chs_info=chs_info,
        )
        del n_outputs, n_chans, n_times, input_window_seconds, sfreq, chs_info

        self.temp_len = ceil(temp_span * self.n_times)

        self.dscnn = _TIDNetFeatures(
            s_growth=s_growth,
            t_filters=t_filters,
            n_chans=self.n_chans,
            n_times=self.n_times,
            drop_prob=drop_prob,
            pooling=pooling,
            temp_layers=temp_layers,
            spat_layers=spat_layers,
            temp_span=temp_span,
            bottleneck=bottleneck,
            summary=summary,
            activation=activation,
        )

        self._num_features = self.dscnn.num_features

        self.flatten = nn.Flatten(start_dim=1)

        self.final_layer = self._create_classifier(self.num_features, self.n_outputs)

    def _create_classifier(self, incoming: int, n_outputs: int):
        classifier = nn.Linear(incoming, n_outputs)
        init.xavier_normal_(classifier.weight)
        classifier.bias.data.zero_()
        seq_clf = nn.Sequential(classifier, nn.Identity())

        return seq_clf

    def forward(self, x, return_feat=False):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """

        x = self.dscnn(x)
        feat = x
        x = self.flatten(x)
        x = self.final_layer(x)
        if return_feat:
            return x, feat
        return x

    @property
    def num_features(self):
        return self._num_features


class _BatchNormZG(nn.BatchNorm2d):
    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.zero_()
            self.bias.data.zero_()


class _ConvBlock2D(nn.Module):
    """Implements Convolution block with order:
    Convolution, dropout, activation, batch-norm
    """

    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        kernel: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        drop_prob: float = 0.5,
        batch_norm: bool = True,
        activation: type[nn.Module] = nn.LeakyReLU,
        residual: bool = False,
    ):
        super().__init__()
        self.kernel = kernel
        self.activation = activation()
        self.residual = residual

        self.conv = nn.Conv2d(
            in_filters,
            out_filters,
            kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=not batch_norm,
        )
        self.dropout = nn.Dropout2d(p=float(drop_prob))
        self.batch_norm = (
            _BatchNormZG(out_filters)
            if residual
            else nn.BatchNorm2d(out_filters)
            if batch_norm
            else nn.Identity()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        res = input
        input = self.conv(
            input,
        )
        input = self.dropout(input)
        input = self.activation(input)
        input = self.batch_norm(input)
        return input + res if self.residual else input


class _DenseFilter(nn.Module):
    def __init__(
        self,
        in_features: int,
        growth_rate: int,
        filter_len: int = 5,
        drop_prob: float = 0.5,
        bottleneck: int = 2,
        activation: type[nn.Module] = nn.LeakyReLU,
        dim: int = -2,
    ):
        super().__init__()
        dim = dim if dim > 0 else dim + 4
        if dim < 2 or dim > 3:
            raise ValueError("Only last two dimensions supported")
        kernel = (filter_len, 1) if dim == 2 else (1, filter_len)

        self.net = nn.Sequential(
            nn.BatchNorm2d(in_features),
            activation(),
            nn.Conv2d(in_features, bottleneck * growth_rate, 1),
            nn.BatchNorm2d(bottleneck * growth_rate),
            activation(),
            nn.Conv2d(
                bottleneck * growth_rate,
                growth_rate,
                kernel,
                padding=tuple((k // 2 for k in kernel)),
            ),
            nn.Dropout2d(p=float(drop_prob)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, self.net(x)), dim=1)


class _DenseSpatialFilter(nn.Module):
    def __init__(
        self,
        n_chans: int,
        growth: int,
        depth: int,
        in_ch: int = 1,
        bottleneck: int = 4,
        drop_prob: float = 0.0,
        activation: type[nn.Module] = nn.LeakyReLU,
        collapse: bool = True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            *[
                _DenseFilter(
                    in_ch + growth * d,
                    growth,
                    bottleneck=bottleneck,
                    drop_prob=drop_prob,
                    activation=activation,
                )
                for d in range(depth)
            ]
        )
        n_filters = in_ch + growth * depth
        self.collapse = collapse
        if collapse:
            self.channel_collapse = _ConvBlock2D(
                n_filters, n_filters, (n_chans, 1), drop_prob=0, activation=activation
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) < 4:
            x = x.unsqueeze(1).permute([0, 1, 3, 2])
        x = self.net(x)
        if self.collapse:
            return self.channel_collapse(x).squeeze(-2)
        return x


class _TemporalFilter(nn.Module):
    def __init__(
        self,
        n_chans: int,
        filters: int,
        depth: int,
        temp_len: int,
        drop_prob: float = 0.0,
        activation: type[nn.Module] = nn.LeakyReLU,
        residual: str = "netwise",
    ):
        super().__init__()
        temp_len = temp_len + 1 - temp_len % 2
        self.residual_style = str(residual)
        net = list()

        for i in range(depth):
            dil = depth - i
            conv = weight_norm(
                nn.Conv2d(
                    n_chans if i == 0 else filters,
                    filters,
                    kernel_size=(1, temp_len),
                    dilation=dil,
                    padding=(0, dil * (temp_len - 1) // 2),
                )
            )
            net.append(
                nn.Sequential(conv, activation(), nn.Dropout2d(p=float(drop_prob)))
            )
        if self.residual_style.lower() == "netwise":
            self.net = nn.Sequential(*net)
            self.residual = nn.Conv2d(n_chans, filters, (1, 1))
        elif residual.lower() == "dense":
            self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        style = self.residual_style.lower()
        if style == "netwise":
            return self.net(x) + self.residual(x)
        elif style == "dense":
            for layer in self.net:
                x = torch.cat((x, layer(x)), dim=1)
            return x
        # TorchScript now knows this path always returns or errors
        else:
            # Use an assertion so TorchScript can compile it
            assert False, f"Unsupported residual style: {self.residual_style}"


class _TIDNetFeatures(nn.Module):
    def __init__(
        self,
        s_growth: int,
        t_filters: int,
        n_chans: int,
        n_times: int,
        drop_prob: float,
        pooling: int,
        temp_layers: int,
        spat_layers: int,
        temp_span: float,
        bottleneck: int,
        summary: int,
        activation: type[nn.Module] = nn.LeakyReLU,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.temp_len = ceil(temp_span * n_times)

        self.temporal = nn.Sequential(
            Ensure4d(),
            Rearrange("batch C T 1 -> batch 1 C T"),
            _TemporalFilter(
                1,
                t_filters,
                depth=temp_layers,
                temp_len=self.temp_len,
                activation=activation,
            ),
            nn.MaxPool2d((1, pooling)),
            nn.Dropout2d(p=float(drop_prob)),
        )
        summary = n_times // pooling if summary == -1 else summary

        self.spatial = _DenseSpatialFilter(
            n_chans=n_chans,
            growth=s_growth,
            depth=spat_layers,
            in_ch=t_filters,
            drop_prob=drop_prob,
            bottleneck=bottleneck,
            activation=activation,
        )
        self.extract_features = nn.Sequential(
            nn.AdaptiveAvgPool1d(int(summary)), nn.Flatten(start_dim=1)
        )

        self._num_features = (t_filters + s_growth * spat_layers) * summary

    @property
    def num_features(self):
        return self._num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        return self.extract_features(x)

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
    model = TIDNet(n_times=500, n_outputs=5, chs_info=chs_info, n_chans=30, sfreq=100.0, input_window_seconds=5,)
    out = model(sample)
    print(out.shape)  # torch.Size([32, 5])
    from torchinfo import summary
    summary(model, (32, 30, 500), device='cpu')

# ===================================================================================================================
# Layer (type:depth-idx)                                            Output Shape              Param #
# ===================================================================================================================
# TIDNet                                                            [32, 5]                   --
# ├─_TIDNetFeatures: 1-1                                            [32, 2640]                --
# │    └─Sequential: 2-1                                            [32, 32, 30, 33]          --
# │    │    └─Ensure4d: 3-1                                         [32, 30, 500, 1]          --
# │    │    └─Rearrange: 3-2                                        [32, 1, 30, 500]          --
# │    │    └─_TemporalFilter: 3-3                                  [32, 32, 30, 500]         26,592
# │    │    └─MaxPool2d: 3-4                                        [32, 32, 30, 33]          --
# │    │    └─Dropout2d: 3-5                                        [32, 32, 30, 33]          --
# │    └─_DenseSpatialFilter: 2-2                                   [32, 80, 33]              --
# │    │    └─Sequential: 3-6                                       [32, 80, 30, 33]          24,272
# │    │    └─_ConvBlock2D: 3-7                                     [32, 80, 1, 33]           192,160
# │    └─Sequential: 2-3                                            [32, 2640]                --
# │    │    └─AdaptiveAvgPool1d: 3-8                                [32, 80, 33]              --
# │    │    └─Flatten: 3-9                                          [32, 2640]                --
# ├─Flatten: 1-2                                                    [32, 2640]                --
# ├─Sequential: 1-3                                                 [32, 5]                   --
# │    └─Linear: 2-4                                                [32, 5]                   13,205
# │    └─Identity: 2-5                                              [32, 5]                   --
# ===================================================================================================================
# Total params: 256,229
# Trainable params: 256,229
# Non-trainable params: 0
# Total mult-adds (Units.MEGABYTES): 988.15
# ===================================================================================================================
# Input size (MB): 1.92
# Forward/backward pass size (MB): 231.69
# Params size (MB): 0.92
# Estimated Total Size (MB): 234.53
# ===================================================================================================================