import torch
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class EgoExoSequenceClassifier(nn.Module):
    def __init__(
        self,
        num_floor_height_classes: int = 13,
        num_layers: int = 8,
        num_channels: int = 256,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_channels = num_channels

        # Define the 1D convolutional layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(
                in_channels=6 + 3 * 6 * 2,
                out_channels=num_channels,
                kernel_size=3,
                padding=1,
            )
        )
        for _ in range(1, num_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=3,
                    padding=1,
                )
            )

        # Define a fully connected layer
        self.fc1 = nn.Linear(
            num_channels, 128
        )  # Adjust input size according to the flattened conv output

        # Classification heads
        self.take_type = nn.Linear(128, 2)
        self.floor_height_logits = nn.Linear(128, num_floor_height_classes)
        self.floor_height_regress = nn.Linear(128, 1)

    def forward(self, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # positions should have shape (batch_size, 3, sequence_length)
        B, dim, len = positions.shape
        assert dim == 3

        # x = positions * torch.tensor([0, 0, 1], device=positions.device)[None, :, None]
        x = positions
        x = torch.cat([x[:, :, 1:], x[:, :, 1:] - x[:, :, :-1]], dim=-2)

        x = fourier_encode(x, 3)
        for conv in self.conv_layers:
            x = conv(x)
            x = F.relu(x)

        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        # Pass through the fully connected layer
        x = F.relu(self.fc1(x))

        # Output heads
        take_type_output = self.take_type(x)
        floor_height_output = self.floor_height_logits(x)
        floor_height_output_2 = self.floor_height_regress(x)

        return take_type_output, floor_height_output, floor_height_output_2


def fourier_encode(
    x: Float[Tensor, "*#batch channels"], freqs: int
) -> Float[Tensor, "*#batch channels+2*freqs*channels"]:
    """Apply Fourier encoding to a tensor."""
    *batch_axes, x_dim, t = x.shape
    coeffs = 2.0 ** torch.arange(freqs, device=x.device)
    scaled = (x[..., None, :] * coeffs[:, None]).reshape(
        (*batch_axes, x_dim * freqs, t)
    )
    return torch.cat(
        [
            x,
            torch.sin(torch.cat([scaled, scaled + torch.pi / 2.0], dim=-2)),
        ],
        dim=-2,
    )
