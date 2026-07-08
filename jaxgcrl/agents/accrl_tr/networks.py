import logging

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling


class Encoder(nn.Module):
    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = (
        # 0 for no skip connections, >= 0 means the frequency of skip connections (every X layers)
        0
    )
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(self, data: jnp.ndarray):
        logging.info("encoder input shape: %s", data.shape)
        lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.use_ln:
            def normalize(x): return nn.LayerNorm()(x)
        else:
            def normalize(x): return x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        x = data
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                         bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)

            if self.skip_connections:
                if i == 0:
                    skip = x
                if i > 0 and i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        x = nn.Dense(self.repr_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class Actor(nn.Module):
    action_size: int
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = (
        # 0 for no skip connections, >= 0 means the frequency of skip connections (every X layers)
        0
    )
    use_relu: bool = False
    use_ln: bool = False
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        if self.use_ln:
            def normalize(x): return nn.LayerNorm()(x)
        else:
            def normalize(x): return x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        logging.info("actor input shape: %s", x.shape)
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_unfirom,
                         bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)

            if self.skip_connections:
                if i == 0:
                    skip = x
                if i > 0 and i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom,
                        bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom,
                           bias_init=bias_init)(x)

        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std


def rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(x, sin, cos):
    # Broadcast over batch and heads
    sin = sin[None, None, :, :]
    cos = cos[None, None, :, :]

    return x * cos + rotate_half(x) * sin


def rotary_embedding(seq_len, head_dim, theta=10000.0):
    half_dim = head_dim // 2

    freq = 1.0 / (
        theta ** (jnp.arange(half_dim) / half_dim)
    )

    positions = jnp.arange(seq_len)

    angles = positions[:, None] * freq[None, :]

    sin = jnp.sin(angles)
    cos = jnp.cos(angles)

    # Duplicate for pair dimensions
    sin = jnp.repeat(sin, 2, axis=-1)
    cos = jnp.repeat(cos, 2, axis=-1)

    return sin, cos


class MultiHeadAttentionRoPE(nn.Module):
    d_model: int
    num_heads: int

    def setup(self):
        assert self.d_model % self.num_heads == 0

        self.head_dim = self.d_model // self.num_heads

        self.q_proj = nn.Dense(
            self.d_model, kernel_init=nn.xavier_uniform(), bias_init=nn.initializers.zeros)
        self.k_proj = nn.Dense(
            self.d_model, kernel_init=nn.xavier_uniform(), bias_init=nn.initializers.zeros)
        self.v_proj = nn.Dense(
            self.d_model, kernel_init=nn.xavier_uniform(), bias_init=nn.initializers.zeros)
        self.out_proj = nn.Dense(
            self.d_model, kernel_init=nn.xavier_uniform(), bias_init=nn.initializers.zeros)

    def __call__(self, x):
        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B,T,D] -> [B,H,T,Dh]
        def split_heads(y):
            y = y.reshape(
                B, T, self.num_heads, self.head_dim
            )
            return y.transpose(0, 2, 1, 3)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        sin, cos = rotary_embedding(
            T,
            self.head_dim
        )

        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

        attn = jnp.matmul(
            q,
            k.transpose(0, 1, 3, 2)
        )

        attn = attn / jnp.sqrt(self.head_dim)

        # causal mask: allow attending only to current and previous tokens
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))

        # broadcast to [B, H, T, T]
        attn = jnp.where(
            mask[None, None, :, :],
            attn,
            -1e10
        )

        weights = nn.softmax(attn, axis=-1)

        out = jnp.matmul(weights, v)

        # [B,H,T,Dh] -> [B,T,D]
        out = out.transpose(0, 2, 1, 3)
        out = out.reshape(B, T, self.d_model)

        return self.out_proj(out)


class TransformerBlockRoPE(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int

    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

        self.attn = MultiHeadAttentionRoPE(
            d_model=self.d_model,
            num_heads=self.num_heads,
        )

        self.mlp = nn.Sequential([
            nn.Dense(self.mlp_dim, kernel_init=nn.xavier_uniform(),
                     bias_init=nn.initializers.zeros),
            nn.swish,
            nn.Dense(self.d_model, kernel_init=nn.xavier_uniform(),
                     bias_init=nn.initializers.zeros),
        ])

    def __call__(self, x):
        #  x shape: [B, T, D]
        h = self.norm1(x)
        h = self.attn(h)

        x = x + h

        h = self.norm2(x)
        h = self.mlp(h)

        x = x + h

        return x


class TransformerCritic(nn.Module):
    state_encoder: nn.Module
    action_encoder: nn.Module
    d_model: int
    num_heads: int
    mlp_dim: int
    num_layers: int

    def setup(self):
        self.layers = [
            TransformerBlockRoPE(
                d_model=self.d_model,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, state, actions):
        # state shape: [B, D_state]
        # actions shape: [B, T, D_action]
        state_repr = self.state_encoder(state)
        action_repr = self.action_encoder(actions)
        x = jnp.concatenate([state_repr[:, None, :], action_repr], axis=1)
        for layer in self.layers:
            x = layer(x)

        return x


def get_default_transformer_critic(state_dim, action_dim, repr_dim):
    state_encoder = Encoder(repr_dim=repr_dim, network_width=256,
                            network_depth=4, skip_connections=4, use_relu=False, use_ln=True)
    action_encoder = Encoder(repr_dim=repr_dim, network_width=64,
                             network_depth=1, skip_connections=0, use_relu=False, use_ln=True)
    return TransformerCritic(
        state_encoder=state_encoder,
        action_encoder=action_encoder,
        d_model=repr_dim,
        num_heads=4,
        mlp_dim=256,
        num_layers=2,
    )
