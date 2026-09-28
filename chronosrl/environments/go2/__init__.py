import jax.numpy as jnp
from ml_collections import config_dict

from chronosrl.environments.go2.goal_environment import Go2GoalEnv


class EvalWrapper:
    def __init__(self, env):
        self.env = env


    def __getattr__(self, name):
        return getattr(self.env, name)


    def reset(self, keys):
        state = self.env.reset(keys)
        return state.replace(info={**state.info, "episode_metrics": {name: jnp.zeros_like(value) for name, value in state.metrics.items()},
                                   "active_episodes": jnp.ones_like(state.done), "episode_steps": jnp.zeros_like(state.done)})


    def step(self, state, action):
        next_state = self.env.step(state, action)
        active = state.info["active_episodes"]
        return next_state.replace(info={
            **next_state.info,
            "episode_metrics": {name: state.info["episode_metrics"][name] + value * active for name, value in next_state.metrics.items()},
            "active_episodes": active * (1.0 - next_state.done),
            "episode_steps": jnp.where(active, next_state.info["steps"], state.info["episode_steps"]),
        })


    def evaluation_metrics(self, state):
        episode_metrics = {name: jnp.mean(value) for name, value in state.info["episode_metrics"].items()}
        metrics = {"time_at_goal": episode_metrics["success"], "episode_length": jnp.mean(state.info["episode_steps"])}
        if self.env.task == "velocity":
            metrics["forward_time_at_goal"] = episode_metrics["forward_success"]
            metrics["forward_tracking_error"] = episode_metrics["forward_error"] / jnp.maximum(episode_metrics["forward_steps"], 1.0)
            metrics["forward_episode_length"] = episode_metrics["forward_steps"]
        return metrics


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name
    config.seed = 1000
    config.nr_envs = 4096
    config.nr_eval_envs = 4096
    config.episode_length = 1000
    config.gait_amplitude = 0.3
    config.gait_frequency = 3.0
    config.travel_cost_weight = 0.3 if environment_name == "go2_position" else 0.0

    return config


def algorithm_overrides(algorithm_name, environment_name):
    overrides = {"total_timesteps": 500_000_000, "unroll_length": 4, "batch_size": 8192, "nr_windows": 192, "goal_constant_input": True}
    if algorithm_name in ("chronosrl", "srl"):
        overrides.update({"goal_sequence_length": 8, "cost_weight": 10.0})
    else:
        overrides["embedding_layer_norm"] = True
    return overrides


def create_env(config):
    env_config = config.environment
    task = env_config.name.split("_")[1]
    make = lambda nr_envs, seed, is_eval: Go2GoalEnv(task, nr_envs, seed, is_eval, env_config.gait_amplitude, env_config.gait_frequency, env_config.travel_cost_weight)
    return make(env_config.nr_envs, env_config.seed, False), EvalWrapper(make(env_config.nr_eval_envs, env_config.seed + 1, True))
