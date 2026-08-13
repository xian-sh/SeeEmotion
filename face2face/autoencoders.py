import torch.nn as nn

from config import Face2FaceConfig
from models import MultitaskModel


def count_parameters_mb(model):
    """Count model parameters and estimate storage in MB using float32."""
    total_params = sum(p.numel() for p in model.parameters())
    total_params_mb = total_params * 4 / (1024 * 1024)
    return total_params, total_params_mb


def get_autoencoder(
    model_type="light",
    model_class="AutoencoderKL",
    input_size=56,
    latent_dim=512,
    device="cuda",
):
    """Create a diffusers autoencoder with the selected architecture and size."""
    model_configs = {
        "light": {
            "block_out_channels": (64, 128, 256),
            "down_block_types": ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            "up_block_types": ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
            "layers_per_block": 1,
        },
        "medium": {
            "block_out_channels": (64, 128, 256, 512),
            "down_block_types": (
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
            "up_block_types": (
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ),
            "layers_per_block": 1,
        },
        "heavy": {
            "block_out_channels": (128, 256, 512, 512),
            "down_block_types": (
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
            "up_block_types": (
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ),
            "layers_per_block": 2,
        },
    }

    if model_type not in model_configs:
        print(f"Invalid model_type: {model_type}. Use light instead.")
        model_type = "light"

    config = model_configs[model_type]
    print(f"Using {model_class} autoencoder with {model_type} configuration")

    if model_class == "AutoencoderKL":
        try:
            from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

            model = AutoencoderKL(
                in_channels=1,
                out_channels=1,
                down_block_types=config["down_block_types"],
                up_block_types=config["up_block_types"],
                block_out_channels=config["block_out_channels"],
                layers_per_block=config["layers_per_block"],
                act_fn="silu",
                latent_channels=latent_dim // 2,
                sample_size=input_size,
            )
            is_kl_model = True
        except ImportError:
            print("Failed to import diffusers. Please install it first.")
            return None, False

    elif model_class == "AutoencoderRAE":
        try:
            from diffusers.models.autoencoders.autoencoder_rae import AutoencoderRAE

            model = AutoencoderRAE(
                in_channels=1,
                out_channels=1,
                down_block_types=config["down_block_types"],
                up_block_types=config["up_block_types"],
                block_out_channels=config["block_out_channels"],
                layers_per_block=config["layers_per_block"],
                act_fn="silu",
                latent_channels=latent_dim,
                sample_size=input_size,
            )
            is_kl_model = False
        except ImportError:
            print("Failed to import diffusers. Please install it first.")
            return None, False

    elif model_class == "VQModel":
        try:
            from diffusers.models.autoencoders.vq_model import VQModel

            model = VQModel(
                in_channels=1,
                out_channels=1,
                down_block_types=config["down_block_types"],
                up_block_types=config["up_block_types"],
                block_out_channels=config["block_out_channels"],
                layers_per_block=config["layers_per_block"],
                act_fn="silu",
                latent_channels=latent_dim,
                sample_size=input_size,
            )
            is_kl_model = False
        except ImportError:
            print("Failed to import diffusers. Please install it first.")
            return None, False

    elif model_class == "AutoencoderTiny":
        try:
            from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

            encoder_channels = config["block_out_channels"]
            decoder_channels = list(reversed(encoder_channels))

            model = AutoencoderTiny(
                in_channels=1,
                out_channels=1,
                encoder_block_out_channels=encoder_channels,
                decoder_block_out_channels=decoder_channels,
                act_fn="silu",
                latent_channels=latent_dim,
            )
            is_kl_model = False
        except ImportError:
            print("Failed to import diffusers. Please install it first.")
            return None, False

    else:
        print(f"Model class {model_class} is not recognized.")
        return None, False

    total_params, total_params_mb = count_parameters_mb(model)
    print(f"Model created: {model_class}")
    print(f"Total parameters: {total_params:,} ({total_params_mb:.2f} MB)")
    print(f"Input size: {input_size}x{input_size}")
    print(f"Latent space: {latent_dim // 2}*2 (KL model)" if is_kl_model else f"Latent space: {latent_dim}")

    model.to(device)
    return model, is_kl_model


class FaceAutoencoderWrapper(nn.Module):
    """Wrap a diffusers autoencoder with a simple face2face interface."""

    def __init__(self, autoencoder, is_kl_model=False):
        super().__init__()
        self.face_autoencoder = autoencoder
        self.is_kl_model = is_kl_model

    def face2face(self, face):
        if self.is_kl_model:
            encoder_output = self.face_autoencoder.encode(face)
            latent = encoder_output.latent_dist.sample()
        else:
            latent = self.face_autoencoder.encode(face).latents

        reconstructed = self.face_autoencoder.decode(latent).sample
        return reconstructed, latent

    def forward(self, face):
        return self.face2face(face)


def create_face_autoencoder(config: Face2FaceConfig):
    """Create the selected face autoencoder with a fallback standard model."""
    autoencoder, is_kl_model = get_autoencoder(
        model_type=config.model_type,
        model_class=config.model_class,
        input_size=config.emoji_size,
        latent_dim=config.latent_dim,
        device=config.device,
    )

    if autoencoder is None:
        print("Failed to create diffusers autoencoder. Falling back to MultitaskModel.")
        model = MultitaskModel(
            eeg_dim=1000,
            face_emb_dim=config.latent_dim,
            eeg_emb_dim=config.eeg_emb_dim,
            emoji_size=config.emoji_size,
            num_classes=5,
        )
        return model.to(config.device), False

    return FaceAutoencoderWrapper(autoencoder, is_kl_model=is_kl_model).to(config.device), is_kl_model

