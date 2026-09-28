import jax.numpy as jnp
import flax.linen as nn
from jax.nn.initializers import variance_scaling, zeros, normal


def dense(features):
    return nn.Dense(features, kernel_init=variance_scaling(1 / 3, "fan_in", "uniform"), bias_init=zeros)


def dense_norm_act(x, width):
    return nn.swish(nn.LayerNorm()(dense(width)(x)))


class ResidualBlock(nn.Module):
    width: int

    @nn.compact
    def __call__(self, x):
        y = x
        for _ in range(4):
            y = dense_norm_act(y, self.width)
        return x + y


class MLP(nn.Module):
    width: int
    depth: int

    @nn.compact
    def __call__(self, x):
        if self.depth < 4:
            for _ in range(self.depth):
                x = dense_norm_act(x, self.width)
            return x
        x = dense_norm_act(x, self.width)
        for _ in range(self.depth // 4):
            x = ResidualBlock(self.width)(x)
        return x


class Encoder(nn.Module):
    width: int
    depth: int
    embedding_dim: int
    layer_norm: bool
    constant_input: bool = False

    @nn.compact
    def __call__(self, x):
        if self.constant_input:
            x = jnp.concatenate([x, jnp.ones(x.shape[:-1] + (1,), x.dtype)], axis=-1)
        z = dense(self.embedding_dim)(MLP(self.width, self.depth)(x))
        return nn.LayerNorm()(z) if self.layer_norm else z


class HazardHead(nn.Module):
    nr_bins: int
    width: int
    depth: int
    embedding_dim: int
    nr_occupancy_bins: int = 0
    rank: int = 64

    @nn.compact
    def __call__(self, z_sa, z_g):
        film = dense(2 * self.embedding_dim)(z_g)
        z = (1.0 + 0.1 * film[..., :self.embedding_dim]) * z_sa + 0.1 * film[..., self.embedding_dim:]
        h = dense_norm_act(jnp.concatenate([z, z_g, z - z_g, z * z_g], axis=-1), self.width)
        for _ in range(self.depth // 4):
            h = ResidualBlock(self.width)(h)
        logit0 = dense(1)(h)[..., 0]
        basis = self.param("time_basis", normal(0.02), (self.nr_bins, self.rank))
        bias = self.param("time_bias", zeros, (self.nr_bins,))
        logits = dense(self.rank)(h) @ basis.T + bias
        if self.nr_occupancy_bins == 0:
            return logit0, logits
        return logit0, logits, dense(self.nr_occupancy_bins)(h)


class SurvivalCritic(nn.Module):
    nr_bins: int
    width: int
    depth: int
    encoder_depth: int
    embedding_dim: int
    layer_norm: bool
    goal_constant_input: bool
    nr_occupancy_bins: int = 0

    @nn.compact
    def __call__(self, state, action, goal):
        z_sa = Encoder(self.width, self.encoder_depth, self.embedding_dim, self.layer_norm)(jnp.concatenate([state, action], axis=-1))
        z_g = Encoder(self.width, self.encoder_depth, self.embedding_dim, self.layer_norm, self.goal_constant_input)(goal)
        return HazardHead(self.nr_bins, self.width, self.depth, self.embedding_dim, self.nr_occupancy_bins)(z_sa, z_g) + (z_sa, z_g)


class ContrastiveCritic(nn.Module):
    width: int
    depth: int
    embedding_dim: int
    layer_norm: bool
    goal_constant_input: bool

    @nn.compact
    def __call__(self, state, action, goal):
        z_sa = Encoder(self.width, self.depth, self.embedding_dim, self.layer_norm)(jnp.concatenate([state, action], axis=-1))
        z_g = Encoder(self.width, self.depth, self.embedding_dim, self.layer_norm, self.goal_constant_input)(goal)
        return z_sa, z_g


class Policy(nn.Module):
    action_size: int
    width: int
    depth: int

    @nn.compact
    def __call__(self, x):
        x = MLP(self.width, self.depth)(x)
        mean = dense(self.action_size)(x)
        log_std = -5.0 + 3.5 * (nn.tanh(dense(self.action_size)(x)) + 1.0)
        return mean, log_std


def pairwise_distances(z_sa, z_g):
    return jnp.sqrt(jnp.sum((z_sa[:, None, :] - z_g[None, :, :]) ** 2, axis=-1) + 1e-12)
