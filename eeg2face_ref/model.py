from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from config import DataConfig, ModelConfig


def fallback_positions(num_channels: int) -> torch.Tensor:
    angles = torch.linspace(0.0, 2.0 * math.pi, steps=num_channels + 1)[:-1]
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1).float()


class FourierEmbedding(nn.Module):
    def __init__(self, dimension: int = 288, margin: float = 0.2) -> None:
        super().__init__()
        n_freqs = int((dimension // 2) ** 0.5)
        if n_freqs * n_freqs * 2 != dimension:
            raise ValueError("Fourier dimension must be 2*n*n, for example 288.")
        self.n_freqs = n_freqs
        self.margin = margin

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        *prefix, dims = positions.shape
        if dims != 2:
            raise ValueError(f"Expected positions with last dim 2, got {positions.shape}.")
        freqs_y = torch.arange(self.n_freqs, device=positions.device)
        freqs_x = freqs_y[:, None]
        width = 1.0 + 2.0 * self.margin
        positions = positions + self.margin
        phase_x = 2.0 * math.pi * freqs_x / width
        phase_y = 2.0 * math.pi * freqs_y / width
        positions = positions.unsqueeze(-2).unsqueeze(-2)
        phase = (positions[..., 0] * phase_x + positions[..., 1] * phase_y).reshape(*prefix, -1)
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


class ChannelMerger(nn.Module):
    def __init__(self, out_channels: int, pos_dim: int = 288) -> None:
        super().__init__()
        self.embedding = FourierEmbedding(pos_dim)
        self.heads = nn.Parameter(torch.randn(out_channels, pos_dim) / pos_dim**0.5)

    def forward(self, eeg: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch_size = eeg.shape[0]
        positions = positions.to(eeg.device).unsqueeze(0).expand(batch_size, -1, -1)
        embeddings = self.embedding(positions)
        weights = torch.softmax(torch.einsum("bcd,od->boc", embeddings, self.heads), dim=-1)
        return torch.einsum("bct,boc->bot", eeg, weights)


class LayerScale(nn.Module):
    def __init__(self, channels: int, init: float = 0.1, boost: float = 5.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), init / boost))
        self.boost = boost

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.boost * self.scale[:, None] * x


class ConvSequence(nn.Module):
    def __init__(self, channels: list[int], kernel_size: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        layers = []
        dilation = 1
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            padding = (kernel_size // 2) * dilation
            block = [
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            ]
            if dropout > 0:
                block.append(nn.Dropout(dropout))
            if in_channels == out_channels:
                block.append(LayerScale(out_channels))
            layers.append(nn.Sequential(*block))
            dilation *= 2
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            residual = x
            x = block(x)
            if x.shape == residual.shape:
                x = x + residual
        return x


class FMENetBackbone(nn.Module):
    def __init__(self, data_config: DataConfig, model_config: ModelConfig) -> None:
        super().__init__()
        self.register_buffer("positions", fallback_positions(data_config.eeg_channels), persistent=False)
        self.channel_merger = (
            ChannelMerger(model_config.merger_channels, model_config.merger_pos_dim)
            if model_config.use_channel_merger
            else None
        )
        in_channels = model_config.merger_channels if self.channel_merger else data_config.eeg_channels
        hidden_dim = model_config.backbone_hidden_dim
        self.input_projection = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.temporal_encoder = ConvSequence([hidden_dim] * (model_config.backbone_depth + 1), dropout=model_config.dropout)
        self.output_projection = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if self.channel_merger is not None:
            eeg = self.channel_merger(eeg, self.positions)
        x = self.input_projection(eeg)
        x = self.temporal_encoder(x)
        return self.output_projection(x)


class EEGToFaceLatent(nn.Module):
    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        latent_shape: tuple[int, int, int],
        num_sample_identities: int,
    ) -> None:
        super().__init__()
        self.latent_shape = latent_shape
        self.backbone = FMENetBackbone(data_config, model_config)
        self.pool = nn.AdaptiveAvgPool1d(1)
        input_dim = model_config.backbone_hidden_dim

        self.use_subject_embedding = model_config.use_subject_embedding
        if self.use_subject_embedding:
            self.subject_embedding = nn.Embedding(data_config.max_subject_id + 1, model_config.subject_embedding_dim)
            input_dim += model_config.subject_embedding_dim
        else:
            self.subject_embedding = None

        self.use_trial_frame_embedding = model_config.use_trial_frame_embedding
        if self.use_trial_frame_embedding:
            self.trial_embedding = nn.Embedding((data_config.max_subject_id + 1) * data_config.max_trials_per_subject, model_config.trial_embedding_dim)
            self.frame_embedding = nn.Embedding(512, model_config.frame_embedding_dim)
            input_dim += model_config.trial_embedding_dim + model_config.frame_embedding_dim
        else:
            self.trial_embedding = None
            self.frame_embedding = None

        self.use_sample_embedding = model_config.use_sample_embedding
        if self.use_sample_embedding:
            self.sample_embedding = nn.Embedding(num_sample_identities, model_config.sample_embedding_dim)
            input_dim += model_config.sample_embedding_dim
        else:
            self.sample_embedding = None

        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in model_config.hidden_dims:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
            if model_config.dropout > 0:
                layers.append(nn.Dropout(model_config.dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, int(np.prod(latent_shape))))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        eeg: torch.Tensor,
        subject_id: Optional[torch.Tensor] = None,
        trial_key: Optional[torch.Tensor] = None,
        frame_id: Optional[torch.Tensor] = None,
        identity_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.pool(self.backbone(eeg)).squeeze(-1)
        if self.use_subject_embedding:
            if subject_id is None:
                raise ValueError("subject_id is required.")
            features = torch.cat([features, self.subject_embedding(subject_id.long())], dim=1)
        if self.use_trial_frame_embedding:
            if trial_key is None or frame_id is None:
                raise ValueError("trial_key and frame_id are required.")
            features = torch.cat([features, self.trial_embedding(trial_key.long()), self.frame_embedding(frame_id.long())], dim=1)
        if self.use_sample_embedding:
            if identity_id is None:
                raise ValueError("identity_id is required.")
            features = torch.cat([features, self.sample_embedding(identity_id.long())], dim=1)
        latent = self.head(features)
        return latent.view(eeg.shape[0], *self.latent_shape)
