import jax.numpy as jnp
from ml_collections import config_dict
from mujoco import mjx
from brax import envs

from chronosrl.environments.jaxgcrl.ant_maze import AntMaze
from chronosrl.environments.jaxgcrl.humanoid import Humanoid, HumanoidMaze


# brax 0.10.1 still calls mjx.ncon, which MuJoCo 3.2 removed
if not hasattr(mjx, "ncon"):
    mjx.ncon = lambda sys: mjx.make_data(sys).ncon

TASKS = {
    "ant_u4_maze": lambda: AntMaze("u4_maze"),
    "ant_big_maze": lambda: AntMaze("big_maze"),
    "ant_hardest_maze": lambda: AntMaze("hardest_maze"),
    "ant_u5_maze": lambda: AntMaze("u5_maze"),
    "humanoid": lambda: Humanoid(),
    "humanoid_u_maze": lambda: HumanoidMaze("u_maze"),
    "humanoid_big_maze": lambda: HumanoidMaze("big_maze"),
}


class EvalWrapper(envs.training.EvalWrapper):
    def evaluation_metrics(self, state):
        metrics = state.info["eval_metrics"]
        success = metrics.episode_metrics["success"]
        return {"time_at_goal": jnp.mean(success), "goal_reached": jnp.mean(success > 0), "episode_length": jnp.mean(metrics.episode_steps)}


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name
    config.seed = 1000
    config.nr_envs = 512
    config.nr_eval_envs = 100
    config.episode_length = 1000

    return config


def algorithm_overrides(algorithm_name, environment_name):
    if algorithm_name in ("chronosrl", "srl") and environment_name.startswith("humanoid"):
        return {"goal_sequence_length": 32, "batch_size": 1024}
    return {}


def create_env(config):
    train_env = envs.training.wrap(TASKS[config.environment.name](), episode_length=config.environment.episode_length)
    eval_env = EvalWrapper(envs.training.wrap(TASKS[config.environment.name](), episode_length=config.environment.episode_length))
    return train_env, eval_env
