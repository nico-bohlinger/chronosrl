import numpy as np
import jax
import jax.numpy as jnp
from ml_collections import config_dict

from chronosrl.networks import SurvivalCritic
from chronosrl.relabeling import bin_discounts


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name
    config.total_timesteps = 100_000_000
    config.nr_epochs = 100
    config.unroll_length = 62
    config.min_replay_size = 1000
    config.max_replay_size = 10000
    config.nr_windows = -1
    config.batch_size = 512
    config.nr_sgd_batches = 800
    config.learning_rate = 3e-4
    config.width = 256
    config.depth = 8
    config.embedding_dim = 128
    config.gamma = 0.999
    config.window = 1000
    config.goal_probabilities = (0.85, 0.05, 0.1)
    config.future_goal_gamma = 0.999
    config.goal_sequence_length = 1
    config.cost_weight = 0.0
    config.action_chunk_length = 1
    config.goal_constant_input = False
    config.entropy_per_action_dim = -0.5

    return config


def hazard_nll(logit0, logits, is_event, tau, censor):
    log1mh_prefix = jnp.cumsum(jax.nn.log_sigmoid(-logits), axis=-1)
    log1mh_before = jnp.concatenate([jnp.zeros_like(log1mh_prefix[:, :1]), log1mh_prefix[:, :-1]], axis=-1)
    rows = jnp.arange(logits.shape[0])
    k = jnp.clip(tau, 0, logits.shape[1] - 1)
    ll_event = jnp.where(tau == 0, jax.nn.log_sigmoid(logit0), jax.nn.log_sigmoid(-logit0) + log1mh_before[rows, k] + jax.nn.log_sigmoid(logits)[rows, k])
    ll_censor = jax.nn.log_sigmoid(-logit0) + jnp.where(censor == 0, 0.0, log1mh_prefix[rows, jnp.clip(censor, 1, logits.shape[1]) - 1])
    return -jnp.where(is_event > 0.5, ll_event, ll_censor)


def hazard_value(logit0, logits, discounts):
    log1mh_prefix = jnp.cumsum(jax.nn.log_sigmoid(-logits), axis=-1)
    log1mh_before = jnp.concatenate([jnp.zeros_like(log1mh_prefix[:, :1]), log1mh_prefix[:, :-1]], axis=-1)
    return -jnp.sum(jnp.exp(jax.nn.log_sigmoid(-logit0)[:, None] + log1mh_before) * discounts[None, :], axis=-1)


class SRL:
    def __init__(self, config, env):
        self.state_dim = env.state_dim
        self.critic = SurvivalCritic(config.window, config.width, config.depth, config.depth // 2, config.embedding_dim, True, config.goal_constant_input)
        self.discounts = bin_discounts(np.arange(config.window + 1), config.gamma)
        self.occupancy_edges = None


    def critic_loss(self, params, batch):
        logit0, logits, _, _ = self.critic.apply(params, batch["observation"][:, :self.state_dim], batch["action"], batch["goal"])
        nll = hazard_nll(logit0, logits, batch["is_event"], batch["tau"], batch["censor"])
        loss = jnp.sum(batch["valid"] * nll) / jnp.maximum(jnp.sum(batch["valid"]), 1.0)
        return loss, {"loss/hazard_loss": loss}


    def actor_value(self, params, state, action, goal, batch):
        logit0, logits, _, _ = self.critic.apply(params, state, action, goal)
        return hazard_value(logit0, logits, self.discounts)
