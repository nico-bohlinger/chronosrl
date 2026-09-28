import numpy as np
import jax
import jax.numpy as jnp
from ml_collections import config_dict

from chronosrl.networks import SurvivalCritic, pairwise_distances
from chronosrl.relabeling import geometric_bins, bin_discounts
from chronosrl.algorithms.srl import hazard_nll, hazard_value


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
    config.embedding_dim = 64
    config.gamma = 0.999
    config.window = 1000
    config.goal_probabilities = (0.85, 0.05, 0.1)
    config.future_goal_gamma = 0.99
    config.goal_sequence_length = 1
    config.cost_weight = 0.0
    config.action_chunk_length = 2
    config.goal_constant_input = False
    config.entropy_per_action_dim = -0.5
    config.nr_occupancy_bins = 32
    config.occupancy_radii = 3

    return config


class ChronoSRL:
    def __init__(self, config, env):
        self.state_dim = env.state_dim
        self.occupancy_edges = geometric_bins(config.window, config.nr_occupancy_bins)
        self.critic = SurvivalCritic(config.window, config.width, config.depth, config.depth, config.embedding_dim, False, config.goal_constant_input, len(self.occupancy_edges) - 1)
        self.discounts = bin_discounts(np.arange(config.window + 1), config.gamma)
        self.occupancy_discounts = bin_discounts(self.occupancy_edges, config.gamma).at[0].set(0.0)
        self.kappa = -jnp.log(config.gamma)
        self.margin = self.kappa * config.window
        self.goal_radius = env.goal_radius


    def critic_loss(self, params, batch):
        logit0, logits, occupancy_logits, z_sa, z_g = self.critic.apply(params, batch["observation"][:, :self.state_dim], batch["action"], batch["goal"])
        valid, is_event, tau = batch["valid"], batch["is_event"], batch["tau"]

        nll = hazard_nll(logit0, logits, is_event, tau, batch["censor"])
        hazard_loss = jnp.sum(valid * nll) / jnp.maximum(jnp.sum(valid), 1.0)

        occupancy_mask = batch["occupancy_mask"] * valid[:, None]
        bce = jax.nn.softplus(occupancy_logits) - occupancy_logits * batch["occupancy"]
        occupancy_loss = jnp.sum(occupancy_mask * bce) / jnp.maximum(jnp.sum(occupancy_mask), 1.0)

        distances = pairwise_distances(z_sa, z_g)
        residual = jnp.diagonal(distances) - self.kappa * tau
        huber = jnp.where(jnp.abs(residual) <= 1.0, 0.5 * residual ** 2, jnp.abs(residual) - 0.5)
        reached = is_event * valid * (tau >= 1)
        time_loss = jnp.sum(reached * huber) / jnp.maximum(jnp.sum(reached), 1.0)

        hinge = jnp.maximum(0.0, self.margin - distances)
        goal_distances = jnp.sqrt(jnp.sum((batch["goal"][:, None, :] - batch["goal"][None, :, :]) ** 2, axis=-1) + 1e-12)
        pairs = (1.0 - jnp.eye(distances.shape[0])) * (goal_distances > self.goal_radius)
        censored = (1.0 - is_event) * valid
        separation_loss = (jnp.sum(pairs * hinge) + jnp.sum(censored * jnp.diagonal(hinge))) / jnp.maximum(jnp.sum(pairs) + jnp.sum(censored), 1.0)

        loss = hazard_loss + time_loss + separation_loss + occupancy_loss
        return loss, {
            "loss/hazard_loss": hazard_loss,
            "loss/occupancy_loss": occupancy_loss,
            "loss/time_loss": time_loss,
            "loss/separation_loss": separation_loss,
        }


    def actor_value(self, params, state, action, goal, batch):
        logit0, logits, occupancy_logits, _, _ = self.critic.apply(params, state, action, goal)
        beta = jnp.sum(batch["terminates"] * batch["valid"]) / jnp.maximum(jnp.sum(batch["valid"]), 1.0)
        return hazard_value(logit0, logits, self.discounts) + beta * jnp.sum(jax.nn.sigmoid(occupancy_logits) * self.occupancy_discounts, axis=-1)
