import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


class ReplayBuffer:
    def __init__(self, dummy_transition, nr_envs, max_size, window, nr_windows):
        flat, unflatten = ravel_pytree(dummy_transition)
        self.flatten = jax.vmap(jax.vmap(lambda x: ravel_pytree(x)[0]))
        self.unflatten = jax.vmap(jax.vmap(unflatten))
        self.shape = (max_size, nr_envs, flat.shape[0])
        self.nr_envs = nr_envs
        self.max_size = max_size
        self.window = window
        self.nr_windows = nr_windows

    def init(self):
        return {"data": jnp.zeros(self.shape, dtype=jnp.float32), "position": jnp.zeros((), dtype=jnp.int32)}

    def insert(self, buffer_state, transitions):
        update = self.flatten(transitions)
        roll = jnp.minimum(0, self.max_size - buffer_state["position"] - update.shape[0])
        data = jax.lax.cond(roll < 0, lambda: jnp.roll(buffer_state["data"], roll, axis=0), lambda: buffer_state["data"])
        position = buffer_state["position"] + roll
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        return {"data": data, "position": position + update.shape[0]}

    def sample(self, buffer_state, key):
        key_envs, key_start = jax.random.split(key)
        envs = jax.random.choice(key_envs, self.nr_envs, shape=(self.nr_windows,), replace=False)
        starts = jax.random.randint(key_start, (self.nr_windows,), 0, jnp.maximum(0, buffer_state["position"] - self.window) + 1)
        indices = starts[:, None] + jnp.arange(self.window)[None, :]
        windows = jax.vmap(lambda data, index: data[index], in_axes=(1, 0))(buffer_state["data"][:, envs], indices)
        return self.unflatten(windows)
