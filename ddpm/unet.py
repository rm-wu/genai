import equinox as eqx
from equinox import nn
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray
from typing import Callable, Optional


from icecream import ic

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class ResidualBlock(eqx.Module):
    """
    Residual block with a 3x3 convolution, group norm, and SiLU activation.
    (It should be similar to the ResidualBlock employed in PixelCNN++ and BigGAN)
    """

    in_channels: int
    out_channels: int
    kernel_size: int
    groups: int
    conv: nn.Conv2d
    norm: nn.GroupNorm
    act: Callable

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 8,
        *,
        key: PRNGKeyArray
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            key=key,
        )
        self.norm = nn.GroupNorm(
            groups=groups,
            channels=out_channels,
        )

        self.act = jax.nn.silu

    def __call__(
        self, x: Array, scale: Optional[Array] = None, shift: Optional[Array] = None
    ) -> Array:
        x = self.conv(x)
        x = self.norm(x)
        if scale and shift:
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResNetBlock(eqx.Module):
    """
    The ResNetBlock is composed of two ResidualBlocks, the skip connection (that
    may require an additional 1x1 convolution if the input and output channels
    are different), and a time embedding MLP (if the time embedding dimension is
    set).
    """

    conv1: ResidualBlock
    conv2: ResidualBlock
    res_conv: nn.Conv2d
    time_mlp: Optional[nn.MLP] = None
    # takes the time embedding and outputs the scale and shift used to scale the
    # output of the Res
    
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 8,
        time_embedding_dim: Optional[int] = None,
        *,
        key: PRNGKeyArray
    ):
        super().__init__()

        # Split the key into 4 keys for initialize the blocks
        keys = jr.split(key, 4)

        # Initialize the two ResidualBlocks
        self.conv1 = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            groutps=groups,
            key=keys[0],
        )
        self.conv2 = ResidualBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            groutps=groups,
            key=keys[1],
        )

        # Use the 1x1 convolution if the input and output channels are different
        # otherwise use an identity function
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                key=keys[2],
            )
        else:
            self.res_conv = nn.Identity()

        if time_embedding_dim:
            self.time_mlp = nn.Sequential(
                jax.nn.silu,
                nn.Linear(
                    in_features=time_embedding_dim,
                    out_features=out_channels * 2,
                    key=keys[3],
                ),
            )

    def __call__(self, x: Array, time_embedding: Optional[Array] = None) -> Array:
        if time_embedding and self.time_mlp:
            time_embedding = self.time_mlp(time_embedding)


class UNet:
    pass


if __name__ == "__main__":
    import jax.numpy as jnp
    import numpy as np

    x = np.random.randn(3, 32, 32)
    x = jnp.array(x)

    residualblock = ResidualBlock(in_channels=3, out_channels=16, key=jax.random.PRNGKey(0))

    ic(residualblock)
    ic(residualblock(x).shape)

    resenetblock = ResNetBlock(
        in_channels=3, 
        out_channels=16, 
        tmp_embedding_dim=16,
        key=jax.random.PRNGKey(0))
    