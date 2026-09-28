import os
import sys
import time
from absl import app
from ml_collections import config_dict, config_flags
import wandb

from chronosrl.algorithms import ALGORITHMS
from chronosrl.environments import get_environment
from chronosrl.trainer import Trainer


def get_runner_config():
    config = config_dict.ConfigDict()

    config.project_name = "chronosrl"
    config.exp_name = "default"
    config.run_name = f"{int(time.time())}"
    config.track_wandb = False
    config.wandb_entity = ""
    config.save_model = False

    return config


class Runner:
    def __init__(self):
        algorithm_name = self.pop_argument("algorithm.name", "chronosrl")
        environment_name = self.pop_argument("environment.name", "ant_u4_maze")
        get_algorithm_config, self.algorithm_class = ALGORITHMS[algorithm_name]
        self.environment = get_environment(environment_name)
        algorithm_config = get_algorithm_config(algorithm_name)
        algorithm_config.update(self.environment.algorithm_overrides(algorithm_name, environment_name))
        self.runner_config_flag = config_flags.DEFINE_config_dict("runner", get_runner_config())
        self.algorithm_config_flag = config_flags.DEFINE_config_dict("algorithm", algorithm_config)
        self.environment_config_flag = config_flags.DEFINE_config_dict("environment", self.environment.get_config(environment_name))


    @staticmethod
    def pop_argument(name, default):
        for argument in sys.argv:
            if argument.startswith(f"--{name}="):
                sys.argv.remove(argument)
                return argument.split("=", 1)[1]
        return default


    def run(self):
        app.run(self.train)


    def train(self, _):
        config = config_dict.ConfigDict()
        config.runner = self.runner_config_flag.value
        config.algorithm = self.algorithm_config_flag.value
        config.environment = self.environment_config_flag.value
        print(config, flush=True)

        run_path = os.path.abspath(f"runs/{config.runner.project_name}/{config.runner.exp_name}/{config.runner.run_name}")
        if config.runner.track_wandb:
            wandb.init(entity=config.runner.wandb_entity or None, project=config.runner.project_name, group=config.runner.exp_name, name=config.runner.run_name, config=config.to_dict())
            wandb.define_metric("*", step_metric="global_step")

        train_env, eval_env = self.environment.create_env(config)
        Trainer(config, self.algorithm_class, train_env, eval_env, run_path).train()

        if config.runner.track_wandb:
            wandb.finish()
