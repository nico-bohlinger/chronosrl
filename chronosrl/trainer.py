import os
import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import wandb

from chronosrl.networks import Policy
from chronosrl.relabeling import make_relabel_fn
from chronosrl.replay_buffer import ReplayBuffer


class Trainer:
    def __init__(self, config, algorithm_class, train_env, eval_env, run_path):
        self.train_env = train_env
        self.eval_env = eval_env
        self.run_path = run_path
        self.save_model = config.runner.save_model
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.nr_envs = config.environment.nr_envs
        self.nr_eval_envs = config.environment.nr_eval_envs
        self.episode_length = config.environment.episode_length
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_epochs = config.algorithm.nr_epochs
        self.unroll_length = config.algorithm.unroll_length
        self.min_replay_size = config.algorithm.min_replay_size
        self.nr_windows = self.nr_envs if config.algorithm.nr_windows == -1 else config.algorithm.nr_windows
        self.batch_size = config.algorithm.batch_size
        self.nr_sgd_batches = config.algorithm.nr_sgd_batches
        self.learning_rate = config.algorithm.learning_rate
        self.action_chunk_length = config.algorithm.action_chunk_length
        self.goal_sequence_length = config.algorithm.goal_sequence_length
        self.state_dim = train_env.state_dim
        self.action_size = train_env.action_size
        self.goal_size = train_env.goal_indices[1] - train_env.goal_indices[0]
        self.target_entropy = config.algorithm.entropy_per_action_dim * self.action_size * self.action_chunk_length
        self.env_steps_per_training_step = self.unroll_length * self.nr_envs
        self.nr_prefill_steps = self.min_replay_size // self.unroll_length
        self.nr_training_steps_per_epoch = (self.total_timesteps - self.min_replay_size * self.nr_envs) // (self.nr_epochs * self.env_steps_per_training_step)

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key = jax.random.split(self.key, 3)

        self.algorithm = algorithm_class(config.algorithm, train_env)
        self.policy = Policy(self.action_size * self.action_chunk_length, config.algorithm.width, config.algorithm.depth)

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, train_env.policy_observation(jnp.ones((1, train_env.observation_size)))),
            tx=optax.adam(self.learning_rate),
        )
        self.critic_state = TrainState.create(
            apply_fn=self.algorithm.critic.apply,
            params=self.algorithm.critic.init(critic_key, jnp.ones((1, self.state_dim)), jnp.ones((1, self.action_size * self.action_chunk_length)), jnp.ones((1, self.goal_size * self.goal_sequence_length))),
            tx=optax.adam(self.learning_rate),
        )
        self.alpha_state = TrainState.create(apply_fn=None, params={"log_alpha": jnp.zeros(())}, tx=optax.adam(self.learning_rate))

        dummy_transition = {"observation": jnp.zeros(train_env.observation_size), "action": jnp.zeros(self.action_size), "discount": 0.0, "truncation": 0.0, "seed": 0.0, "cost": 0.0}
        self.replay_buffer = ReplayBuffer(dummy_transition, self.nr_envs, config.algorithm.max_replay_size, config.algorithm.window, self.nr_windows)
        self.relabel = jax.vmap(make_relabel_fn(config.algorithm, train_env, self.algorithm.occupancy_edges))

        if self.save_model:
            os.makedirs(self.run_path, exist_ok=True)


    def act(self, policy_params, env_state, chunk, chunk_index, key):
        mean, log_std = self.policy.apply(policy_params, self.train_env.policy_observation(env_state.obs))
        new_chunk = jnp.tanh(mean if key is None else mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape))
        chunk_index = jnp.where((env_state.info["steps"] == 0) | (env_state.done > 0), 0, chunk_index)
        chunk = jnp.where((chunk_index == 0)[:, None, None], new_chunk.reshape(chunk.shape), chunk)
        return chunk[jnp.arange(chunk.shape[0]), chunk_index], chunk, (chunk_index + 1) % self.action_chunk_length


    def evaluate(self, policy_params, key):
        def step(carry, _):
            env_state, chunk, chunk_index = carry
            action, chunk, chunk_index = self.act(policy_params, env_state, chunk, chunk_index, None)
            return (self.eval_env.step(env_state, action), chunk, chunk_index), None

        env_state = self.eval_env.reset(jax.random.split(key, self.nr_eval_envs))
        chunk = jnp.zeros((self.nr_eval_envs, self.action_chunk_length, self.action_size))
        chunk_index = jnp.zeros((self.nr_eval_envs,), dtype=jnp.int32)
        (env_state, _, _), _ = jax.lax.scan(step, (env_state, chunk, chunk_index), None, length=self.episode_length)
        return self.eval_env.evaluation_metrics(env_state)


    def train(self):
        def collect(policy_params, env_state, key):
            def step(carry, _):
                env_state, chunk, chunk_index, key = carry
                key, subkey = jax.random.split(key)
                action, chunk, chunk_index = self.act(policy_params, env_state, chunk, chunk_index, subkey)
                next_env_state = self.train_env.step(env_state, action)
                transition = {
                    "observation": env_state.obs,
                    "action": action,
                    "discount": 1.0 - next_env_state.done,
                    "truncation": env_state.info["truncation"],
                    "seed": env_state.info["seed"],
                    "cost": next_env_state.reward,
                }
                return (next_env_state, chunk, chunk_index, key), transition

            chunk = jnp.zeros((self.nr_envs, self.action_chunk_length, self.action_size))
            chunk_index = jnp.zeros((self.nr_envs,), dtype=jnp.int32)
            (env_state, _, _, _), transitions = jax.lax.scan(step, (env_state, chunk, chunk_index, key), None, length=self.unroll_length)
            return env_state, transitions


        def sample_batches(buffer_state, key):
            sample_key, relabel_key, permutation_key, selection_key = jax.random.split(key, 4)
            windows = self.replay_buffer.sample(buffer_state, sample_key)
            batches = self.relabel(windows, jax.random.split(relabel_key, self.nr_windows))
            batches = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), batches)
            nr_samples = batches["valid"].shape[0]
            nr_batches = nr_samples // self.batch_size
            permutation = jax.random.permutation(permutation_key, nr_samples)[:nr_batches * self.batch_size]
            batches = jax.tree_util.tree_map(lambda x: x[permutation].reshape((nr_batches, self.batch_size) + x.shape[1:]), batches)
            if self.nr_sgd_batches != -1:
                selection = jax.random.permutation(selection_key, nr_batches)[:self.nr_sgd_batches]
                batches = jax.tree_util.tree_map(lambda x: x[selection], batches)
            return batches


        def update(carry, batch):
            policy_state, critic_state, alpha_state, key = carry
            key, actor_key = jax.random.split(key)

            # Critic
            (_, critic_metrics), critic_gradients = jax.value_and_grad(self.algorithm.critic_loss, has_aux=True)(critic_state.params, batch)
            critic_state = critic_state.apply_gradients(grads=critic_gradients)

            # Policy
            observation = batch["observation"]
            state = observation[:, :self.state_dim]
            goal = jnp.tile(observation[:, self.state_dim:], (1, self.goal_sequence_length))
            alpha = jnp.exp(alpha_state.params["log_alpha"])

            def policy_loss_fn(policy_params):
                mean, log_std = self.policy.apply(policy_params, self.train_env.policy_observation(observation))
                std = jnp.exp(log_std)
                x = mean + std * jax.random.normal(actor_key, mean.shape)
                action = jnp.tanh(x)
                log_prob = jnp.sum(jax.scipy.stats.norm.logpdf(x, loc=mean, scale=std) - jnp.log(1.0 - action ** 2 + 1e-6), axis=-1)
                q = self.algorithm.actor_value(critic_state.params, state, action, goal, batch)
                return jnp.mean(alpha * log_prob - q), (log_prob, q)

            (policy_loss, (log_prob, q)), policy_gradients = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_state.params)
            policy_state = policy_state.apply_gradients(grads=policy_gradients)

            # Entropy coefficient
            alpha_loss_fn = lambda params: jnp.mean(jnp.exp(params["log_alpha"]) * jax.lax.stop_gradient(-log_prob - self.target_entropy))
            alpha_gradients = jax.grad(alpha_loss_fn)(alpha_state.params)
            alpha_state = alpha_state.apply_gradients(grads=alpha_gradients)

            metrics = {
                **critic_metrics,
                "loss/policy_loss": policy_loss,
                "q_value/q_value": jnp.mean(q),
                "entropy/entropy": -jnp.mean(log_prob),
                "entropy/alpha": alpha,
                "labels/reached": jnp.mean(batch["is_event"]),
                "labels/valid": jnp.mean(batch["valid"]),
            }
            return (policy_state, critic_state, alpha_state, key), metrics


        def training_step(carry, _):
            policy_state, critic_state, alpha_state, env_state, buffer_state, key = carry
            key, collect_key, sample_key, update_key = jax.random.split(key, 4)
            env_state, transitions = collect(policy_state.params, env_state, collect_key)
            buffer_state = self.replay_buffer.insert(buffer_state, transitions)
            batches = sample_batches(buffer_state, sample_key)
            (policy_state, critic_state, alpha_state, _), metrics = jax.lax.scan(update, (policy_state, critic_state, alpha_state, update_key), batches)
            return (policy_state, critic_state, alpha_state, env_state, buffer_state, key), metrics


        @jax.jit
        def training_epoch(carry):
            carry, metrics = jax.lax.scan(training_step, carry, None, length=self.nr_training_steps_per_epoch)
            return carry, jax.tree_util.tree_map(jnp.mean, metrics)


        @jax.jit
        def prefill(policy_params, env_state, buffer_state, key):
            def step(carry, _):
                env_state, buffer_state, key = carry
                key, subkey = jax.random.split(key)
                env_state, transitions = collect(policy_params, env_state, subkey)
                return (env_state, self.replay_buffer.insert(buffer_state, transitions), key), None
            (env_state, buffer_state, _), _ = jax.lax.scan(step, (env_state, buffer_state, key), None, length=self.nr_prefill_steps)
            return env_state, buffer_state


        evaluate = jax.jit(self.evaluate)
        self.key, env_key, eval_key, prefill_key, train_key = jax.random.split(self.key, 5)
        env_state = jax.jit(self.train_env.reset)(jax.random.split(env_key, self.nr_envs))
        env_state, buffer_state = prefill(self.policy_state.params, env_state, self.replay_buffer.init(), prefill_key)
        carry = (self.policy_state, self.critic_state, self.alpha_state, env_state, buffer_state, train_key)
        del env_state, buffer_state
        env_steps = self.nr_prefill_steps * self.env_steps_per_training_step

        for epoch in range(self.nr_epochs + 1):
            eval_key, subkey = jax.random.split(eval_key)
            self.log({"eval/" + name: value for name, value in evaluate(carry[0].params, subkey).items()}, env_steps)
            if epoch == self.nr_epochs:
                break
            start_time = time.time()
            carry, metrics = training_epoch(carry)
            metrics = jax.tree_util.tree_map(lambda x: np.asarray(x).item(), {**metrics, **self.train_env.training_metrics(carry[3])})
            env_steps += self.nr_training_steps_per_epoch * self.env_steps_per_training_step
            metrics["time/sps"] = self.nr_training_steps_per_epoch * self.env_steps_per_training_step / (time.time() - start_time)
            self.log(metrics, env_steps)
            if self.save_model:
                with open(os.path.join(self.run_path, "model.pkl"), "wb") as f:
                    pickle.dump(jax.device_get({"policy": carry[0].params, "critic": carry[1].params, "alpha": carry[2].params}), f)


    def log(self, metrics, step):
        print(f"step {step:>11} | " + " | ".join(f"{name} {value:.4g}" for name, value in sorted(metrics.items())), flush=True)
        if self.track_wandb:
            wandb.log({"global_step": step, **metrics})
