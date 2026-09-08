"""Learning utilities adapted from research/MjDex."""

from typing import Any, Optional, Sequence
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding module."""

    dim: int

    @nn.compact
    def __call__(self, x):
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class Identity(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow("intermediates", "feature", x)
        return x


class ResMLP(nn.Module):
    """Residual MLP.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: If True, it works as an intermediate layer; if False, it works as a standalone neural network.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x):
        assert self.layer_norm

        x = nn.Dense(self.hidden_dims[0], kernel_init=self.kernel_init)(x)
        x = nn.LayerNorm()(x)
        x = self.activations(x)
        num_res_blocks = (
            len(self.hidden_dims) if self.activate_final else len(self.hidden_dims) - 1
        )

        for i in range(num_res_blocks):
            size = self.hidden_dims[i]
            residual = x
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            x = nn.LayerNorm()(x)
            x = self.activations(x)
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            x = nn.LayerNorm()(x)
            x = x + residual
        x = nn.LayerNorm()(x)

        if not self.activate_final:
            x = nn.Dense(self.hidden_dims[-1], kernel_init=self.kernel_init)(x)

        return x


class ResMLPDiffusion(nn.Module):
    """Residual MLP diffusion policy network.

    The module can be used in two modes:
    - Encoding mode: encode observations only (for encoder caching).
    - Diffusion mode: predict action velocities given (encoded) observations,
      noisy actions, and diffusion timesteps.

    Attributes:
        hidden_dims: Hidden layer dimensions of the residual MLP.
        time_step_embed_dim: Dimensionality of the timestep embedding.
        horizon_steps: Number of action steps predicted in parallel.
        action_dim: Dimensionality of a single action.
        layer_norm: Whether to apply layer normalization in the MLP.
        activations: Activation function used in the network.
        kernel_init: Weight initialization for linear layers.
        encoder: Optional encoder module to encode observations before diffusion.
    """

    hidden_dims: Sequence[int]
    time_step_embed_dim: int
    horizon_steps: int
    action_dim: int
    layer_norm: bool
    activations: Any = nn.gelu
    kernel_init: Any = default_init()
    encoder: nn.Module = None

    def setup(self):
        self.time_mlp = nn.Sequential(
            [
                SinusoidalPosEmb(self.time_step_embed_dim),
                nn.Dense(self.time_step_embed_dim * 4, kernel_init=self.kernel_init),
                self.activations,
                nn.Dense(self.time_step_embed_dim, kernel_init=self.kernel_init),
            ]
        )

        hidden_dims = self.hidden_dims + (self.action_dim * self.horizon_steps,)
        self.res_mlp = ResMLP(
            hidden_dims=hidden_dims,
            activations=self.activations,
            layer_norm=self.layer_norm,
            activate_final=False,
        )

    def __call__(self, ob, actions=None, times=None, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            ob = self.encoder(ob)

        batch_size, horizon_steps, action_dim = actions.shape
        actions = actions.reshape(batch_size, -1)

        time_embed = self.time_mlp(times)
        x = jnp.concatenate([actions, time_embed, ob], axis=-1)
        x = self.res_mlp(x)
        return x.reshape(batch_size, horizon_steps, action_dim)


class Upsample1d(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(
            features=self.dim,
            kernel_size=(4,),
            strides=(2,),
            padding="SAME",
        )(x.transpose(0, 2, 1))
        return x.transpose(0, 2, 1)


class Downsample1d(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.dim,
            kernel_size=(3,),
            strides=(2,),
            padding=((1, 1),),
        )(x.transpose(0, 2, 1))
        return x.transpose(0, 2, 1)


class TransposeModule(nn.Module):

    @nn.compact
    def __call__(self, x):
        return x.transpose(0, 2, 1)


class Conv1dBlock(nn.Module):
    input_channels: int
    output_channels: int
    kernel_size: int
    n_groups: int
    activation: Any = nn.gelu
    eps: float = 1e-5

    def setup(self):
        self.conv = nn.Conv(
            features=self.output_channels,
            kernel_size=(self.kernel_size,),
            padding="SAME",
        )
        self.groupnorm = nn.GroupNorm(
            num_groups=self.n_groups,
            epsilon=self.eps,
        )

    def __call__(self, x):
        x = x.transpose(0, 2, 1)
        x = self.conv(x)
        if self.n_groups is not None:
            x = self.groupnorm(x)
        x = self.activation(x)
        return x.transpose(0, 2, 1)


class ResidualBlock1d(nn.Module):
    in_channels: int
    out_channels: int
    cond_dim: int
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    eps: float = 1e-5
    activation: Any = nn.gelu
    kernel_init: Any = default_init()

    def setup(self):
        self.block1 = Conv1dBlock(
            input_channels=self.in_channels,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
            eps=self.eps,
        )
        self.block2 = Conv1dBlock(
            input_channels=self.in_channels,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
            eps=self.eps,
        )

        cond_out_features = (
            self.out_channels * 2 if self.cond_predict_scale else self.out_channels
        )
        self.cond_mlp = nn.Sequential(
            [
                self.activation,
                nn.Dense(features=cond_out_features, kernel_init=self.kernel_init),
            ]
        )

        if self.in_channels != self.out_channels:
            self.residual_conv = nn.Conv(
                features=self.out_channels,
                kernel_size=(1,),
                padding="SAME",
            )
        else:
            self.residual_conv = None

    def __call__(self, x, cond):
        out = self.block1(x)

        cond = self.cond_mlp(cond)
        cond = jnp.expand_dims(cond, axis=-1)

        if self.cond_predict_scale:
            cond = cond.reshape(cond.shape[0], 2, self.out_channels, 1)
            scale = cond[:, 0, ...]
            bias = cond[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + cond

        out = self.block2(out)
        if self.residual_conv is not None:
            residual = self.residual_conv(x.transpose(0, 2, 1))
            residual = residual.transpose(0, 2, 1)
        else:
            residual = x
        return out + residual


class UNet(nn.Module):
    action_dim: int
    cond_dim: int
    time_step_embed_dim: int
    dim: int
    dim_mults: tuple
    kernel_size: int = 5
    n_groups: int = 8
    activation: Any = nn.gelu
    cond_predict_scale: bool = True
    eps: float = 1e-5
    kernel_init: Any = default_init()

    def setup(self):

        dims = [self.action_dim, *map(lambda m: self.dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.time_mlp = nn.Sequential(
            [
                SinusoidalPosEmb(self.time_step_embed_dim),
                nn.Dense(
                    features=self.time_step_embed_dim * 4, kernel_init=self.kernel_init
                ),
                self.activation,
                nn.Dense(
                    features=self.time_step_embed_dim, kernel_init=self.kernel_init
                ),
            ]
        )

        cond_block_dim = self.time_step_embed_dim + self.cond_dim

        mid_dims = dims[-1]
        self.mid_modules = [
            ResidualBlock1d(
                in_channels=mid_dims,
                out_channels=mid_dims,
                cond_dim=cond_block_dim,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                cond_predict_scale=self.cond_predict_scale,
                eps=self.eps,
            ),
            ResidualBlock1d(
                in_channels=mid_dims,
                out_channels=mid_dims,
                cond_dim=cond_block_dim,
                kernel_size=self.kernel_size,
                n_groups=self.n_groups,
                cond_predict_scale=self.cond_predict_scale,
                eps=self.eps,
            ),
        ]
        down_modules = []
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= (len(in_out) - 1)
            block = [
                ResidualBlock1d(
                    in_channels=dim_in,
                    out_channels=dim_out,
                    cond_dim=cond_block_dim,
                    kernel_size=self.kernel_size,
                    n_groups=self.n_groups,
                    cond_predict_scale=self.cond_predict_scale,
                    eps=self.eps,
                ),
                ResidualBlock1d(
                    in_channels=dim_in,
                    out_channels=dim_out,
                    cond_dim=cond_block_dim,
                    kernel_size=self.kernel_size,
                    n_groups=self.n_groups,
                    cond_predict_scale=self.cond_predict_scale,
                    eps=self.eps,
                ),
                Downsample1d(dim=dim_out) if not is_last else Identity(),
            ]
            down_modules.append(block)
        self.down_modules = down_modules

        up_modules = []
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = idx >= len(in_out) - 1
            block = [
                ResidualBlock1d(
                    in_channels=dim_out * 2,
                    out_channels=dim_in,
                    cond_dim=cond_block_dim,
                    kernel_size=self.kernel_size,
                    n_groups=self.n_groups,
                    cond_predict_scale=self.cond_predict_scale,
                    eps=self.eps,
                ),
                ResidualBlock1d(
                    in_channels=dim_in,
                    out_channels=dim_in,
                    cond_dim=cond_block_dim,
                    kernel_size=self.kernel_size,
                    n_groups=self.n_groups,
                    cond_predict_scale=self.cond_predict_scale,
                    eps=self.eps,
                ),
                Upsample1d(dim=dim_in) if not is_last else Identity(),
            ]
            up_modules.append(block)
        self.up_modules = up_modules

        self.final_conv = nn.Sequential(
            [
                Conv1dBlock(
                    input_channels=self.dim,
                    output_channels=self.dim,
                    kernel_size=self.kernel_size,
                    n_groups=self.n_groups,
                    eps=self.eps,
                ),
                TransposeModule(),
                nn.Conv(
                    features=self.action_dim,
                    kernel_size=(1,),
                    padding="SAME",
                ),
            ]
        )

    def __call__(self, x, time, cond):
        B = x.shape[0]
        x = x.transpose(0, 2, 1)

        time = jnp.broadcast_to(time, (B,))
        time = self.time_mlp(time)

        global_features = jnp.concatenate([time, cond], axis=-1)
        h = []
        h_local = list()
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_features)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_features)
            h.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_features)
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = jnp.concatenate((x, h.pop()), axis=1)
            x = resnet(x, global_features)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_features)
            x = upsample(x)
        x = self.final_conv(x)
        return x


class UNetActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    action_dim: int
    cond_dim: int
    time_step_embed_dim: int
    dim: int
    dim_mults: tuple
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    eps: float = 1e-5
    encoder: nn.Module = None

    def setup(self) -> None:
        self.unet = UNet(
            action_dim=self.action_dim,
            cond_dim=self.cond_dim,
            time_step_embed_dim=self.time_step_embed_dim,
            dim=self.dim,
            dim_mults=self.dim_mults,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
            cond_predict_scale=self.cond_predict_scale,
            eps=self.eps,
        )

    @nn.compact
    def __call__(
        self,
        observations,
        actions=None,
        times=None,
        is_encoded=False,
    ):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        v = self.unet(x=actions, time=times, cond=observations)

        return v
