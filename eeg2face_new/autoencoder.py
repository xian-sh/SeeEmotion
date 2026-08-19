from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def create_autoencoder_kl(model_type: str, emoji_size: int, latent_dim: int, device: str) -> nn.Module:
    try:
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
    except ImportError as error:
        raise ImportError("Install diffusers to use the pretrained Face2Face AutoencoderKL.") from error

    configs = {
        "light": {"block_out_channels": (64, 128, 256), "layers_per_block": 1},
        "medium": {"block_out_channels": (64, 128, 256, 512), "layers_per_block": 1},
        "heavy": {"block_out_channels": (128, 256, 512, 512), "layers_per_block": 2},
    }
    cfg = configs[model_type]
    n_blocks = len(cfg["block_out_channels"])
    model = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        down_block_types=("DownEncoderBlock2D",) * n_blocks,
        up_block_types=("UpDecoderBlock2D",) * n_blocks,
        block_out_channels=cfg["block_out_channels"],
        layers_per_block=cfg["layers_per_block"],
        act_fn="silu",
        latent_channels=latent_dim // 2,
        sample_size=emoji_size,
    )
    return model.to(device)


def load_checkpoint(path: Path, device: str) -> Mapping:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_autoencoder_state(checkpoint: Mapping) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint.get("face_autoencoder"), Mapping):
        state = dict(checkpoint["face_autoencoder"])
    elif isinstance(checkpoint.get("state_dict"), Mapping):
        state = dict(checkpoint["state_dict"])
    elif checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        state = dict(checkpoint)
    else:
        raise KeyError("Expected checkpoint key 'face_autoencoder', 'state_dict', or a raw tensor state dict.")

    for prefix in ("module.face_autoencoder.", "face_autoencoder.", "module."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix) :]: value for key, value in state.items()}
            break
    return state


class FrozenAutoencoderKL(nn.Module):
    def __init__(
        self,
        checkpoint: Path,
        model_type: str,
        emoji_size: int,
        latent_dim: int,
        device: str,
        sample_latent: bool = False,
    ) -> None:
        super().__init__()
        self.autoencoder = create_autoencoder_kl(model_type, emoji_size, latent_dim, device)
        state = extract_autoencoder_state(load_checkpoint(checkpoint, device))
        self.autoencoder.load_state_dict(state, strict=True)
        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.sample_latent = sample_latent
        self.latent_shape = self._infer_latent_shape(emoji_size, device)

    @torch.no_grad()
    def _infer_latent_shape(self, emoji_size: int, device: str) -> tuple[int, int, int]:
        dummy = torch.zeros(1, 1, emoji_size, emoji_size, device=device)
        latent = self.encode(dummy)
        return tuple(latent.shape[1:])

    @torch.no_grad()
    def encode(self, faces: torch.Tensor) -> torch.Tensor:
        posterior = self.autoencoder.encode(faces).latent_dist
        return posterior.sample() if self.sample_latent else posterior.mode()

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(latents).sample.clamp(0.0, 1.0)


@dataclass
class LatentStats:
    mean: torch.Tensor
    std: torch.Tensor


class LatentNormalizer(nn.Module):
    def __init__(self, latent_shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, *latent_shape))
        self.register_buffer("std", torch.ones(1, *latent_shape))

    @torch.no_grad()
    def fit(self, autoencoder: FrozenAutoencoderKL, loader, device: str, max_batches: int = 0) -> LatentStats:
        total = 0
        mean = torch.zeros_like(self.mean, device=device)
        m2 = torch.zeros_like(self.mean, device=device)

        for batch_idx, batch in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            faces = batch["face"].to(device, non_blocking=True).float()
            latents = autoencoder.encode(faces)
            batch_count = latents.shape[0]
            batch_mean = latents.mean(dim=0, keepdim=True)
            batch_var = latents.var(dim=0, keepdim=True, unbiased=False)
            delta = batch_mean - mean
            new_total = total + batch_count
            mean = mean + delta * (batch_count / max(1, new_total))
            m2 = m2 + batch_var * batch_count + delta.pow(2) * total * batch_count / max(1, new_total)
            total = new_total

        if total > 1:
            std = torch.sqrt(m2 / total).clamp_min(1e-5)
            self.mean.copy_(mean.detach())
            self.std.copy_(std.detach())
        return LatentStats(self.mean.detach().clone(), self.std.detach().clone())

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.mean) / self.std

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.std + self.mean
