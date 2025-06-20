import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable
from torch import Tensor, nn
from layers.RevIN import RevIN

def time_to_feature(x: torch.Tensor) -> torch.Tensor:
    """Converts a time series tensor to a feature tensor."""
    return x.permute(0, 2, 1)

feature_to_time = time_to_feature


class TimeBatchNorm2d(nn.BatchNorm1d):
    """A batch normalization layer that normalizes over the last two dimensions of a
    sequence in PyTorch, mimicking Keras behavior.

    This class extends nn.BatchNorm1d to apply batch normalization across time and
    feature dimensions.

    Attributes:
        num_time_steps (int): Number of time steps in the input.
        num_channels (int): Number of channels in the input.
    """

    def __init__(self, normalized_shape):
        """Initializes the TimeBatchNorm2d module.

        Args:
            normalized_shape (tuple[int, int]): A tuple (num_time_steps, num_channels)
                representing the shape of the time and feature dimensions to normalize.
        """
        num_time_steps, num_channels = normalized_shape
        super().__init__(num_channels * num_time_steps)
        self.num_time_steps = num_time_steps
        self.num_channels = num_channels

    def forward(self, x: Tensor) -> Tensor:
        """Applies the batch normalization over the last two dimensions of the input tensor.

        Args:
            x (Tensor): A 3D tensor with shape (N, S, C), where N is the batch size,
                S is the number of time steps, and C is the number of channels.

        Returns:
            Tensor: A 3D tensor with batch normalization applied over the last two dims.

        Raises:
            ValueError: If the input tensor is not 3D.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input tensor, but got {x.ndim}D tensor instead.")

        # Reshaping input to combine time and feature dimensions for normalization
        x = x.reshape(x.shape[0], -1, 1)

        # Applying batch normalization
        x = super().forward(x)

        # Reshaping back to original dimensions (N, S, C)
        x = x.reshape(x.shape[0], self.num_time_steps, self.num_channels)

        return x
    

class FeatureMixing(nn.Module):
    """A module for feature mixing with flexibility in normalization and activation.

    This module provides options for batch normalization before or after mixing features,
    uses dropout for regularization, and allows for different activation functions.

    Args:
        sequence_length: The length of the sequences to be transformed.
        input_channels: The number of input channels to the module.
        output_channels: The number of output channels from the module.
        ff_dim: The dimension of the feed-forward network internal to the module.
        activation_fn: The activation function used within the feed-forward network.
        dropout_rate: The dropout probability used for regularization.
        normalize_before: A boolean indicating whether to apply normalization before
            the rest of the operations.
    """

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        output_channels: int,
        ff_dim: int,
        activation_fn: F.relu,
        dropout_rate: 0.1,
        normalize_before: True,
        norm_type: TimeBatchNorm2d,
    ):
        """Initializes the FeatureMixing module with the provided parameters."""
        super().__init__()

        self.norm_before = (
            norm_type((sequence_length, input_channels))
            if normalize_before
            else nn.Identity()
        )
        self.norm_after = (
            norm_type((sequence_length, output_channels))
            if not normalize_before
            else nn.Identity()
        )

        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_channels, ff_dim)
        self.fc2 = nn.Linear(ff_dim, output_channels)

        self.projection = (
            nn.Linear(input_channels, output_channels)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the FeatureMixing module.

        Args:
            x: A 3D tensor with shape (N, C, L) where C is the channel dimension.

        Returns:
            The output tensor after feature mixing.
        """
        x_proj = self.projection(x)

        x = self.norm_before(x)

        x = self.fc1(x)  # Apply the first linear transformation.
        x = self.activation_fn(x)  # Apply the activation function.
        x = self.dropout(x)  # Apply dropout for regularization.
        x = self.fc2(x)  # Apply the second linear transformation.
        x = self.dropout(x)  # Apply dropout again if needed.

        x = x_proj + x  # Add the projection shortcut to the transformed features.

        return self.norm_after(x)


class ConditionalFeatureMixing(nn.Module):
    """Conditional feature mixing module that incorporates static features.

    This module extends the feature mixing process by including static features. It uses
    a linear transformation to integrate static features into the dynamic feature space,
    then applies the feature mixing on the concatenated features.

    Args:
        input_channels: The number of input channels of the dynamic features.
        output_channels: The number of output channels after feature mixing.
        static_channels: The number of channels in the static feature input.
        ff_dim: The inner dimension of the feedforward network used in feature mixing.
        activation_fn: The activation function used in feature mixing.
        dropout_rate: The dropout probability used in the feature mixing operation.
    """

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        output_channels: int,
        static_channels: int,
        ff_dim: int,
        activation_fn:F.relu,
        dropout_rate:  0.1,
        normalize_before: False,
        norm_type: nn.LayerNorm,
    ):
        super().__init__()

        self.fr_static = nn.Linear(static_channels, output_channels)
        self.fm = FeatureMixing(
            sequence_length,
            input_channels + output_channels,
            output_channels,
            ff_dim,
            activation_fn,
            dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

    def forward(
        self, x: torch.Tensor, x_static: torch.Tensor
    ):
        """Applies conditional feature mixing using both dynamic and static inputs.

        Args:
            x: A tensor representing dynamic features, typically with shape
               [batch_size, time_steps, input_channels].
            x_static: A tensor representing static features, typically with shape
               [batch_size, static_channels].

        Returns:
            A tuple containing:
            - The output tensor after applying conditional feature mixing.
            - The transformed static features tensor for monitoring or further processing.
        """
        v = self.fr_static(x_static)  # Transform static features to match output channels.
        v = v.unsqueeze(1).repeat(
            1, x.shape[1], 1
        )  # Repeat static features across time steps.

        return (
            self.fm(
                torch.cat([x, v], dim=-1)
            ),  # Apply feature mixing on concatenated features.
            v.detach(),  # Return detached static feature for monitoring or further use.
        )


class TimeMixing(nn.Module):
    """Applies a transformation over the time dimension of a sequence.

    This module applies a linear transformation followed by an activation function
    and dropout over the sequence length of the input feature tensor after converting
    feature maps to the time dimension and then back.

    Args:
        input_channels: The number of input channels to the module.
        sequence_length: The length of the sequences to be transformed.
        activation_fn: The activation function to be used after the linear transformation.
        dropout_rate: The dropout probability to be used after the activation function.
    """

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        activation_fn: F.relu,
        dropout_rate: 0.1,
        norm_type: TimeBatchNorm2d,
    ):
        """Initializes the TimeMixing module with the specified parameters."""
        super().__init__()
        self.norm = norm_type((sequence_length, input_channels))
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(sequence_length, sequence_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the time mixing operations on the input tensor.

        Args:
            x: A 3D tensor with shape (N, C, L), where C = channel dimension and
                L = sequence length.

        Returns:
            The normalized output tensor after time mixing transformations.
        """
        x_temp = feature_to_time(
            x
        )  # Convert feature maps to time dimension. Assumes definition elsewhere.
        x_temp = self.activation_fn(self.fc1(x_temp))
        x_temp = self.dropout(x_temp)
        x_res = time_to_feature(x_temp)  # Convert back from time to feature maps.

        return self.norm(x + x_res)  # Apply normalization and combine with original input.
    
class MixerLayer(nn.Module):
    """A residual block that combines time and feature mixing for sequence data.

    This module sequentially applies time mixing and feature mixing, which are forms
    of data augmentation and feature transformation that can help in learning temporal
    dependencies and feature interactions respectively.

    Args:
        sequence_length: The length of the input sequences.
        input_channels: The number of input channels to the module.
        output_channels: The number of output channels from the module.
        ff_dim: The inner dimension of the feedforward network used in feature mixing.
        activation_fn: The activation function used in both time and feature mixing.
        dropout_rate: The dropout probability used in both mixing operations.
    """

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        output_channels: int,
        ff_dim: int,
        activation_fn:F.relu,
        dropout_rate:  0.1,
        normalize_before:  False,
        norm_type:nn.LayerNorm,
    ):
        """Initializes the MixLayer with time and feature mixing modules."""
        super().__init__()

        self.time_mixing = TimeMixing(
            sequence_length,
            input_channels,
            activation_fn,
            dropout_rate,
            norm_type=norm_type,
        )
        self.feature_mixing = FeatureMixing(
            sequence_length,
            input_channels,
            output_channels,
            ff_dim,
            activation_fn,
            dropout_rate,
            norm_type=norm_type,
            normalize_before=normalize_before,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MixLayer module.

        Args:
            x: A 3D tensor with shape (N, C, L) to be processed by the mixing layers.

        Returns:
            The output tensor after applying time and feature mixing operations.
        """
        x = self.time_mixing(x)  # Apply time mixing first.
        x = self.feature_mixing(x)  # Then apply feature mixing.

        return x




class Model(nn.Module):
    """TSMixer model for time series forecasting.

    This model uses a series of mixer layers to process time series data,
    followed by a linear transformation to project the output to the desired
    prediction length.

    Attributes:
        mixer_layers: Sequential container of mixer layers.
        temporal_projection: Linear layer for temporal projection.

    Args:
        sequence_length: Length of the input time series sequence.
        prediction_length: Desired length of the output prediction sequence.
        input_channels: Number of input channels.
        output_channels: Number of output channels. Defaults to None.
        activation_fn: Activation function to use. Defaults to "relu".
        num_blocks: Number of mixer blocks. Defaults to 2.
        dropout_rate: Dropout rate for regularization. Defaults to 0.1.
        ff_dim: Dimension of feedforward network inside mixer layer. Defaults to 64.
        normalize_before: Whether to apply layer normalization before or after mixer layer.
        norm_type: Type of normalization to use. "batch" or "layer". Defaults to "batch".
    """

    def __init__(
        self, configs,
        activation_fn: str = "relu",
        normalize_before: bool = True,
        norm_type: str = "batch",
    ):
        super().__init__()
        self.configs = configs
        sequence_length = configs.seq_len
        prediction_length = configs.pred_len
        input_channels = configs.enc_in
        output_channels = configs.dec_in
        num_blocks = configs.e_layers
        dropout_rate = configs.dropout
        ff_dim = configs.d_ff
        if self.configs.revin: self.revin_layer = RevIN(self.configs.enc_in, affine=True, subtract_last=False)


        # Transform activation_fn to callable
        activation_fn = getattr(F, activation_fn)

        # Transform norm_type to callable
        assert norm_type in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {norm_type}, must be one of batch, layer."
        norm_type = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm

        # Build mixer layers
        self.mixer_layers = self._build_mixer(
            num_blocks,
            input_channels,
            output_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            sequence_length=sequence_length,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

        # Temporal projection layer
        self.temporal_projection = nn.Linear(sequence_length, prediction_length)

    def _build_mixer(
        self, num_blocks: int, input_channels: int, output_channels: int, **kwargs
    ):
        """Build the mixer blocks for the model.

        Args:
            num_blocks (int): Number of mixer blocks to be built.
            input_channels (int): Number of input channels for the first block.
            output_channels (int): Number of output channels for the last block.
            **kwargs: Additional keyword arguments for mixer layer configuration.

        Returns:
            nn.Sequential: Sequential container of mixer layers.
        """
        output_channels = output_channels if output_channels is not None else input_channels
        channels = [input_channels] * (num_blocks - 1) + [output_channels]

        return nn.Sequential(
            *[
                MixerLayer(input_channels=in_ch, output_channels=out_ch, **kwargs)
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TSMixer model.

        Args:
            x_hist (torch.Tensor): Input time series tensor.

        Returns:
            torch.Tensor: The output tensor after processing by the model.
        """
        if self.configs.revin:
            #x = x.permute(0,2,1)
            x = self.revin_layer(x, 'norm')
            #x = x.permute(0,2,1)

        x = self.mixer_layers(x)
        x_temp = feature_to_time(x)
        x_temp = self.temporal_projection(x_temp)
        x = time_to_feature(x_temp)
        if self.configs.revin:
            #x = x.permute(0,2,1)
            x = self.revin_layer(x, 'denorm')
            #x = x.permute(0,2,1)

        return x


