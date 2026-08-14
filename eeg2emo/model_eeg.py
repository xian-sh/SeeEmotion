import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== Fourier Embedding ==========
class FourierEmb(nn.Module):
    def __init__(self, dimension: int = 288, margin: float = 0.2):
        super().__init__()
        n_freqs = int((dimension // 2) ** 0.5)
        assert n_freqs ** 2 * 2 == dimension, \
            f"dimension={dimension} is invalid for FourierEmb, must be 2*n*n, e.g. 288, 512"
        self.dimension = dimension
        self.margin = margin
        self.n_freqs = n_freqs

    def forward(self, positions):
        *O, D = positions.shape
        assert D == 2
        freqs_y = torch.arange(self.n_freqs, device=positions.device)
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        positions = positions.unsqueeze(-2).unsqueeze(-2)  # [..., 1, 1, 2]
        loc = (positions[..., 0] * p_x + positions[..., 1] * p_y).reshape(*O, -1)
        emb = torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)
        return emb

# ========== DualPathRNN ==========
def pad_multiple(x: torch.Tensor, base: int):
    length = x.shape[-1]
    target = math.ceil(length / base) * base
    return F.pad(x, (0, target - length))

class DualPathRNN(nn.Module):
    def __init__(self, channels: int, depth: int, inner_length: int = 10):
        super().__init__()
        self.lstms = nn.ModuleList([nn.LSTM(channels, channels, 1) for _ in range(depth * 4)])
        self.inner_length = inner_length

    def forward(self, x: torch.Tensor):
        B, C, L = x.shape
        IL = self.inner_length
        x = pad_multiple(x, IL)
        x = x.permute(2, 0, 1).contiguous()  # [T, B, C]
        for idx, lstm in enumerate(self.lstms):
            y = x.reshape(-1, IL, B, C)
            if idx % 2 == 0:
                y = y.transpose(0, 1).reshape(IL, -1, C)
            else:
                y = y.reshape(-1, IL * B, C)
            y, _ = lstm(x)
            if idx % 2 == 0:
                y = y.reshape(IL, -1, B, C).transpose(0, 1).reshape(-1, B, C)
            else:
                y = y.reshape(-1, B, C)
            x = x + y
            if idx % 2 == 1:
                x = x.flip(dims=(0,))
        x = x[:L].permute(1, 2, 0).contiguous()  # [B, C, T]
        return x

# ========== LayerScale ==========
class LayerScale(nn.Module):
    def __init__(self, channels: int, init: float = 0.1, boost: float = 5.):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels))
        self.scale.data[:] = init / boost
        self.boost = boost
    def forward(self, x):
        return (self.boost * self.scale[:, None]) * x

# ========== ConvSequence ==========
class ConvSequence(nn.Module):
    def __init__(self, channels, kernel=4, dilation_growth=1, dilation_period=None, stride=2,
                 dropout=0.0, leakiness=0.0, groups=1, skip=False, scale=None,
                 activation=None, batch_norm=False, activation_on_last=True,
                 glu=0, glu_context=0, glu_glu=True):
        super().__init__()
        dilation = 1
        channels = tuple(channels)
        self.skip = skip
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()
        if activation is None:
            activation = nn.ReLU
        Conv = nn.Conv1d
        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers = []
            is_last = k == len(channels) - 2
            pad = kernel // 2 * dilation
            layers.append(Conv(chin, chout, kernel, stride, pad,
                               dilation=dilation, groups=groups if k > 0 else 1))
            dilation *= dilation_growth
            if activation_on_last or not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(num_features=chout))
                layers.append(activation())
                if dropout:
                    layers.append(nn.Dropout(dropout))
            if chin == chout and skip:
                if scale is not None:
                    layers.append(LayerScale(chout, scale))
            self.sequence.append(nn.Sequential(*layers))
            if glu and (k + 1) % glu == 0:
                ch = 2 * chout if glu_glu else chout
                act = nn.GLU(dim=1) if glu_glu else activation()
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(chout, ch, 1 + 2 * glu_context, padding=glu_context), act))
            else:
                self.glus.append(None)
    def forward(self, x):
        for module_idx, module in enumerate(self.sequence):
            old_x = x
            x = module(x)
            if self.skip and x.shape == old_x.shape:
                x = x + old_x
            glu = self.glus[module_idx]
            if glu is not None:
                x = glu(x)
        return x

# ========== ChannelMerger ==========
class ChannelMerger(nn.Module):
    def __init__(self, chout: int, pos_dim: int = 288,
                 dropout: float = 0, usage_penalty: float = 0.):
        super().__init__()
        assert pos_dim % 4 == 0
        self.heads = nn.Parameter(torch.randn(chout, pos_dim))
        self.heads.data /= pos_dim ** 0.5
        self.dropout = dropout
        self.embedding = FourierEmb(pos_dim)
        self.usage_penalty = usage_penalty
        self._penalty = torch.tensor(0.)

    @property
    def training_penalty(self):
        return self._penalty.to(next(self.parameters()).device)

    def get_position_features(self, device, batch_size):
        """Return compact position features for classifier fusion."""
        return torch.zeros(batch_size, 16, device=device)  

    def forward(self, data, positions):
        B, C, T = data.shape
        device = data.device
        if positions.ndim == 2:
            positions = positions.unsqueeze(0).expand(B, -1, -1)  # [B, C, 2]
        embedding = self.embedding(positions)
        score_offset = torch.zeros(B, C, device=device)
        if self.training and self.dropout:
            center_to_ban = torch.rand(2, device=device)
            radius_to_ban = self.dropout
            banned = (positions - center_to_ban).norm(dim=-1) <= radius_to_ban
            score_offset[banned] = float('-inf')
        heads = self.heads[None].expand(B, -1, -1)
        scores = torch.einsum("bcd,bod->boc", embedding, heads)
        scores += score_offset[:, None]
        weights = torch.softmax(scores, dim=2)
        out = torch.einsum("bct,boc->bot", data, weights)
        if self.training and self.usage_penalty > 0.:
            usage = weights.mean(dim=(0, 1)).sum()
            self._penalty = self.usage_penalty * usage
        return out

# ========== SimpleConv with ChannelMerger ==========
class SimpleConv(nn.Module):
    def __init__(self,
                 in_channels: int = 30,
                 out_channels: int = 32,
                 hidden: int = 320,
                 depth: int = 5,
                 kernel_size: int = 3,
                 dilation_growth: int = 2,
                 dilation_period: int = 5,
                 skip: bool = True,
                 glu: int = 2,
                 glu_context: int = 1,
                 gelu: bool = True,
                 batch_norm: bool = True,
                 conv_dropout: float = 0.0,
                 dropout_input: float = 0.0,
                 linear_out: bool = False,
                 complex_out: bool = True,
                 initial_linear: int = 320,
                 initial_depth: int = 1,
                 merger: bool = False,
                 merger_pos_dim: int = 288,
                 merger_channels: int = 30,
                 merger_dropout: float = 0.0,
                 merger_penalty: float = 0.0,
                 **kwargs):
        super().__init__()

        self.merger = None
        self._use_merger = merger
        if merger:
            self.merger = ChannelMerger(
                chout=merger_channels,
                pos_dim=merger_pos_dim,
                dropout=merger_dropout,
                usage_penalty=merger_penalty
            )
            in_channels = merger_channels

        if gelu:
            activation = nn.GELU
        else:
            activation = nn.ReLU
        self.initial_linear = None
        if initial_linear:
            init = [nn.Conv1d(in_channels, initial_linear, 1)]
            for _ in range(initial_depth - 1):
                init += [activation(), nn.Conv1d(initial_linear, initial_linear, 1)]
            self.initial_linear = nn.Sequential(*init)
            in_channels = initial_linear

        sizes = [in_channels] + [hidden] * depth
        params = dict(kernel=kernel_size, stride=1,
                      dropout=conv_dropout,
                      batch_norm=batch_norm, dilation_growth=dilation_growth,
                      dilation_period=dilation_period, skip=skip,
                      glu=glu, glu_context=glu_context, activation=activation)
        self.encoder = ConvSequence(sizes, **params)
        

    def forward(self, x, positions=None):
        # x: (B, C, T); positions: (C, 2) or (B, C, 2)
        if self.merger is not None:
            if positions is not None:
                positions = positions.to(x.device)
            x = self.merger(x, positions)
        if self.initial_linear is not None:
            x = self.initial_linear(x)
        x = self.encoder(x)
        
        return x

# ========== EEG Classifier ==========
class EEGClassifier(nn.Module):
    """EEG-only classifier for emotion recognition."""

    def __init__(self, n_classes=10, eeg_channels=30, hidden_dim=320, depth=5,
                 use_dual_path=1,
                 merger=False, merger_pos_dim=288, merger_channels=30,
                 merger_dropout=0.0, merger_penalty=0.0,
                 positions=None,
                 dilation_period=5):
        super().__init__()

        self.positions = positions
        self.out_channels = hidden_dim

        self.encoder = SimpleConv(
            in_channels=eeg_channels,
            out_channels=hidden_dim,
            hidden=hidden_dim,
            depth=depth,
            kernel_size=3,
            dilation_growth=2,
            dilation_period=dilation_period,
            skip=True,
            glu=2,
            glu_context=1,
            gelu=True,
            batch_norm=True,
            conv_dropout=0.0,
            dropout_input=0.1,
            linear_out=False,
            complex_out=True,
            initial_linear=hidden_dim,
            initial_depth=1,
            dual_path=use_dual_path,
            merger=merger,
            merger_pos_dim=merger_pos_dim,
            merger_channels=merger_channels,
            merger_dropout=merger_dropout,
            merger_penalty=merger_penalty,
        )

        self.dual_path = DualPathRNN(hidden_dim, use_dual_path) if use_dual_path else None

        self.final = nn.Sequential(
            nn.Conv1d(hidden_dim, 2 * hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(2 * hidden_dim, hidden_dim, 1),
        )
        self.classification_pool = nn.AdaptiveAvgPool1d(1)

        classifier_input_dim = hidden_dim
        if merger and hasattr(self.encoder, 'merger') and self.encoder.merger is not None:
            classifier_input_dim += 16
        self.classifier = nn.Linear(classifier_input_dim, n_classes)

    def forward(self, x):
        """Return emotion-classification logits from EEG input."""
        if hasattr(self.encoder, 'merger') and self.encoder.merger is not None and self.positions is not None:
            x = self.encoder(x, self.positions)
        else:
            x = self.encoder(x)

        if self.dual_path is not None:
            x = self.dual_path(x)

        features = self.final(x)
        pooled_features = self.classification_pool(features).squeeze(-1)

        if hasattr(self.encoder, 'merger') and self.encoder.merger is not None:
            pos_features = self.encoder.merger.get_position_features(x.device, x.shape[0])
            pooled_features = torch.cat([pooled_features, pos_features], dim=-1)

        return self.classifier(pooled_features)

if __name__ == '__main__':
    from util import EAV_EEG_CHANNEL_POSITIONS

    batch_size, channels, n_times = 32, 30, 500
    sample = torch.randn(batch_size, channels, n_times)
    model = EEGClassifier(
        n_classes=5,
        eeg_channels=channels,
        hidden_dim=32,
        depth=5,
        merger=True,
        merger_channels=6,
        use_dual_path=0,
        positions=EAV_EEG_CHANNEL_POSITIONS,
    )
    logits = model(sample)
    print('Classification output shape:', logits.shape)
    print('Model parameters:', sum(p.numel() for p in model.parameters()))
