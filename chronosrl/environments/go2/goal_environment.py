from pathlib import Path
from typing import Any, Dict
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

from chronosrl.environments.go2.locomotion.default_config import get_config
from chronosrl.environments.go2.locomotion.environment import LocomotionEnv
from chronosrl.environments.go2.unitree_go2.robot_config import robot_config


PENALTIES = (
    "z_velocity", "imu_acceleration", "angular_velocity", "angular_position", "actuator_joint_nominal_diff", "joint_position_limit",
    "joint_velocity_limit", "joint_velocity", "joint_acceleration", "joint_torque", "power_draw_penalty", "action_rate", "action_smoothness",
    "collision", "ground_penetration", "base_height", "foot_air_time", "feet_lateral_min_distance", "symmetry_air", "foot_slip",
    "foot_z_velocity", "foot_flat_contact", "feet_orientation",
)


@struct.dataclass
class State:
    locomotion: Any
    goal: jax.Array
    start: jax.Array
    start_yaw: jax.Array
    key: jax.Array
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array]
    info: Dict[str, Any]


class Go2GoalEnv:
    def __init__(self, task, nr_envs, seed, is_eval, gait_amplitude, gait_frequency, travel_cost_weight):
        self.task = task
        self.nr_envs = nr_envs
        self.is_eval = is_eval
        self.gait_amplitude = gait_amplitude
        self.gait_frequency = gait_frequency
        self.travel_cost_weight = travel_cost_weight
        self.velocity_task = task == "velocity"

        config = get_config()
        config.seed = seed
        if task == "box":
            config.terrain.type = "plane_box"
            config.njmax = 256
            config.naconmax_per_env = 24
            config.env_curriculum_success_error = 0.5
            config.env_curriculum_success_episode_length = 0
        if not self.velocity_task:
            config.reward.curriculum_coeff_floor = 0.3
        if is_eval:
            config.termination.curriculum_coeff = 1.0
        self.env = LocomotionEnv({**robot_config, "directory_path": Path(__file__).parent / "unitree_go2"}, config, nr_envs)
        self.dt = float(self.env.dt)

        self.goal_size = 3 if self.velocity_task else 4
        self.goal_indices = (0, self.goal_size)
        self.goal_radius = 0.25 if self.velocity_task else 0.5
        self.heading_scale = 0.3
        self.nr_forward_envs = int(round(0.25 * nr_envs)) if (is_eval and self.velocity_task) else 0

        # Observation: [achieved goal | heading | locomotion observation | gait clock | goal relative to the robot | goal]
        locomotion_observation_size = int(self.env.single_observation_space.shape[0])
        self.state_dim = 2 + 2 * self.goal_size + locomotion_observation_size + 2
        self.observation_size = self.state_dim + self.goal_size
        self.action_size = int(self.env.nr_actuator_joints)
        command_indices = {int(i) for i in np.asarray(self.env.goal_velocities_obs_idx)}
        offset = self.goal_size + 2
        self.policy_observation_indices = jnp.array([offset + int(i) for i in np.asarray(self.env.policy_observation_indices) if int(i) not in command_indices] +
                                                    [offset + locomotion_observation_size, offset + locomotion_observation_size + 1])

        # Trot-shaped gait prior on the joint targets: thighs +2, calves -2, diagonal legs in phase
        phase = {"FL": 0.0, "RR": 0.0, "FR": jnp.pi, "RL": jnp.pi}
        self.gait_phase = jnp.array([phase[name.upper()[:2]] for name in self.env.actuator_joint_names], jnp.float32)
        self.gait_gain = jnp.array([2.0 if "thigh" in name else (-2.0 if "calf" in name else 0.0) for name in self.env.actuator_joint_names], jnp.float32)


    def policy_observation(self, obs):
        x = jnp.take(obs, self.policy_observation_indices, axis=-1)
        goal = obs[..., self.state_dim:self.state_dim + self.goal_size]
        if self.velocity_task:
            return jnp.concatenate([x, goal], axis=-1)
        achieved, cos_heading, sin_heading = obs[..., :self.goal_size], obs[..., self.goal_size], obs[..., self.goal_size + 1]
        rotate = lambda d: jnp.stack([cos_heading * d[..., 0] + sin_heading * d[..., 1], -sin_heading * d[..., 0] + cos_heading * d[..., 1]], axis=-1)
        return jnp.concatenate([x, rotate(goal[..., :2] - achieved[..., :2]), rotate(goal[..., 2:] - achieved[..., 2:])], axis=-1)


    def training_metrics(self, state):
        curriculum_coeff = jnp.mean(state.locomotion.internal_state["env_curriculum_coeff"])
        metrics = {"train/curriculum_coefficient": curriculum_coeff}
        if self.task == "box":
            metrics["train/box_height"] = self.env.env_config["terrain"]["box_height_delta"] * curriculum_coeff
        return metrics


    def pose(self, locomotion_state):
        qpos = locomotion_state.data.qpos
        w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
        return qpos[:, :3], jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


    def achieved_goal(self, locomotion_state, base, yaw, start, start_yaw):
        if self.velocity_task:
            qvel = locomotion_state.data.qvel
            cos_yaw, sin_yaw = jnp.cos(yaw), jnp.sin(yaw)
            return jnp.stack([cos_yaw * qvel[:, 0] + sin_yaw * qvel[:, 1], -sin_yaw * qvel[:, 0] + cos_yaw * qvel[:, 1], qvel[:, 5]], axis=-1)
        difference = base[:, :3] - start
        cos_start, sin_start = jnp.cos(start_yaw), jnp.sin(start_yaw)
        position = jnp.stack([cos_start * difference[:, 0] + sin_start * difference[:, 1], -sin_start * difference[:, 0] + cos_start * difference[:, 1]], axis=-1)
        relative_yaw = yaw - start_yaw
        return jnp.concatenate([position, self.heading_scale * jnp.cos(relative_yaw)[:, None], self.heading_scale * jnp.sin(relative_yaw)[:, None]], axis=-1)


    def sample_goal(self, key):
        if self.velocity_task:
            key1, key2 = jax.random.split(key)
            velocity_key, all_zero_key, single_zero_key = jax.random.split(key1, 3)
            goal = jax.random.uniform(velocity_key, (self.nr_envs, 3), minval=-1.0, maxval=1.0)
            goal = jnp.where(jnp.abs(goal) < 0.1, 0.0, goal)
            goal = jnp.where(jax.random.bernoulli(all_zero_key, 0.04, (self.nr_envs, 1)), 0.0, goal)
            goal = jnp.where(jax.random.uniform(single_zero_key, (self.nr_envs, 3)) < 0.005, 0.0, goal)
            goal = goal * (jax.random.uniform(key2, (self.nr_envs, 1)) > 0.04)
            forward = jnp.zeros((self.nr_envs, 3), goal.dtype).at[:, 0].set(1.0)
            return jnp.where((jnp.arange(self.nr_envs) < self.nr_forward_envs)[:, None], forward, goal)
        distance_key, angle_key = jax.random.split(key)
        distance = jax.random.uniform(distance_key, (self.nr_envs,), minval=1.0, maxval=5.0)
        angle = jax.random.uniform(angle_key, (self.nr_envs,), minval=0.0, maxval=2.0 * jnp.pi)
        return jnp.stack([distance * jnp.cos(angle), distance * jnp.sin(angle), self.heading_scale * jnp.cos(angle), self.heading_scale * jnp.sin(angle)], axis=-1)


    def place_box(self, locomotion_state, goal, start, start_yaw, mask):
        if self.task != "box":
            return locomotion_state
        cos_start, sin_start = jnp.cos(start_yaw), jnp.sin(start_yaw)
        internal_state = locomotion_state.internal_state
        internal_state["box_centre_x"] = jnp.where(mask, start[:, 0] + cos_start * goal[:, 0] - sin_start * goal[:, 1], internal_state["box_centre_x"]).astype(internal_state["box_centre_x"].dtype)
        internal_state["box_centre_y"] = jnp.where(mask, start[:, 1] + sin_start * goal[:, 0] + cos_start * goal[:, 1], internal_state["box_centre_y"]).astype(internal_state["box_centre_y"].dtype)
        internal_state["box_yaw"] = jnp.where(mask, start_yaw + jnp.arctan2(goal[:, 1], goal[:, 0]) - 0.5 * jnp.pi, internal_state["box_yaw"]).astype(internal_state["box_yaw"].dtype)
        return locomotion_state


    def observation(self, locomotion_state, yaw, start_yaw, goal, achieved):
        if self.velocity_task:
            relative_goal = goal
            cos_heading, sin_heading = jnp.cos(yaw), jnp.sin(yaw)
        else:
            cos_heading, sin_heading = jnp.cos(yaw - start_yaw), jnp.sin(yaw - start_yaw)
            difference = goal[:, :2] - achieved[:, :2]
            relative_goal = jnp.stack([cos_heading * difference[:, 0] + sin_heading * difference[:, 1], -sin_heading * difference[:, 0] + cos_heading * difference[:, 1]], axis=-1)
            relative_goal = jnp.concatenate([relative_goal, goal[:, 2:4] - achieved[:, 2:4]], axis=-1)
        phase = locomotion_state.info_episode_store["episode_step"] * (2.0 * jnp.pi * self.gait_frequency * self.dt)
        obs = jnp.concatenate([achieved[:, :self.goal_size], cos_heading[:, None], sin_heading[:, None], locomotion_state.next_observation,
                               jnp.cos(phase)[:, None], jnp.sin(phase)[:, None], relative_goal, goal], axis=-1).astype(jnp.float32)
        return jnp.where(jnp.isfinite(obs), obs, 0.0)


    def evaluation_step_metrics(self, goal, achieved):
        success = (jnp.linalg.norm(achieved - goal, axis=-1) < self.goal_radius).astype(jnp.float32)
        forward = (jnp.arange(self.nr_envs) < self.nr_forward_envs).astype(jnp.float32) * (float(self.nr_envs) / max(self.nr_forward_envs, 1))
        return {
            "success": success,
            "forward_success": success * forward,
            "forward_error": jnp.linalg.norm(achieved[:, :3] - goal[:, :3], axis=-1) * forward,
            "forward_steps": forward,
        }


    def match_step_shapes(self, locomotion_state):
        target = jax.eval_shape(self.env.step, locomotion_state, jnp.zeros((self.nr_envs, self.action_size), jnp.float32))
        return jax.tree_util.tree_map(lambda x, t: jnp.broadcast_to(jnp.asarray(x), t.shape).astype(t.dtype), locomotion_state, target)


    def reset(self, keys):
        key, goal_key = jax.random.split(keys[0])
        locomotion_state = self.match_step_shapes(self.env.reset(keys))
        if self.is_eval:
            locomotion_state.internal_state["env_curriculum_coeff"] = jnp.zeros_like(locomotion_state.internal_state["env_curriculum_coeff"])
        base, yaw = self.pose(locomotion_state)
        achieved = self.achieved_goal(locomotion_state, base, yaw, base, yaw)
        goal = self.sample_goal(goal_key)
        locomotion_state = self.place_box(locomotion_state, goal, base, yaw, jnp.ones((self.nr_envs,), bool))
        zeros = jnp.zeros((self.nr_envs,), jnp.float32)
        if not self.velocity_task:
            locomotion_state.internal_state["task_error_sum"] = zeros
            locomotion_state.internal_state["task_error_count"] = zeros
            if self.task == "box":
                locomotion_state.internal_state["on_box_steps"] = zeros
        info = {"truncation": zeros, "seed": jnp.zeros((self.nr_envs,), jnp.int32), "steps": zeros}
        return State(locomotion=locomotion_state, goal=goal, start=base, start_yaw=yaw, key=key, obs=self.observation(locomotion_state, yaw, yaw, goal, achieved),
                     reward=zeros, done=zeros, metrics=self.evaluation_step_metrics(goal, achieved), info=info)


    def step(self, state, action):
        phase = state.locomotion.info_episode_store["episode_step"] * (2.0 * jnp.pi * self.gait_frequency * self.dt)
        action = jnp.clip(action, -1.0, 1.0) + self.gait_amplitude * self.gait_gain[None, :] * jnp.sin(phase[:, None] + self.gait_phase[None, :])
        if self.velocity_task:
            command = state.goal[:, :3]
            state.locomotion.internal_state["goal_velocities"] = command.astype(state.locomotion.internal_state["goal_velocities"].dtype)
            state.locomotion.internal_state["actuator_joint_keep_nominal"] = jnp.where(jnp.all(command == 0.0, axis=-1, keepdims=True), True, self.env.command_function.default_actuator_joint_keep_nominal[None, :])
        locomotion_state = self.env.step(state.locomotion, action)
        done = locomotion_state.terminated | locomotion_state.truncated

        key, goal_key = jax.random.split(state.key)
        base, yaw = self.pose(locomotion_state)
        start = jnp.where(done[:, None], base, state.start)
        start_yaw = jnp.where(done, yaw, state.start_yaw)
        achieved = self.achieved_goal(locomotion_state, base, yaw, start, start_yaw)
        goal = jnp.where(done[:, None], self.sample_goal(goal_key), state.goal)
        locomotion_state = self.place_box(locomotion_state, goal, start, start_yaw, done)

        # Per-step cost for the shaped clock: the penalty terms of the locomotion reward plus the cost for travelling backward or sideways
        velocity = jnp.nan_to_num(locomotion_state.data.qvel[:, :2].astype(jnp.float32))
        forward_velocity = jnp.cos(yaw) * velocity[:, 0] + jnp.sin(yaw) * velocity[:, 1]
        speed = jnp.hypot(forward_velocity, -jnp.sin(yaw) * velocity[:, 0] + jnp.cos(yaw) * velocity[:, 1])
        travel_away = 0.5 * (1.0 - forward_velocity / jnp.maximum(speed, 1e-6)) * jnp.clip(speed / 0.2, 0.0, 1.0)
        penalty = sum(locomotion_state.info[f"reward/{name}"] for name in PENALTIES)
        cost = jnp.clip(jnp.nan_to_num(-penalty) + self.travel_cost_weight * travel_away, 0.0, 50.0).astype(jnp.float32)

        # Task error for the curriculum: mean normalized distance to the goal position, or whether the robot stood on the box
        if not self.velocity_task:
            internal_state = locomotion_state.internal_state
            if self.task == "box":
                qpos = locomotion_state.data.qpos
                dx, dy = qpos[:, 0] - internal_state["box_centre_x"], qpos[:, 1] - internal_state["box_centre_y"]
                local_x = jnp.cos(internal_state["box_yaw"]) * dx + jnp.sin(internal_state["box_yaw"]) * dy
                local_y = -jnp.sin(internal_state["box_yaw"]) * dx + jnp.cos(internal_state["box_yaw"]) * dy
                on_box = (jnp.abs(local_x) < internal_state["box_half_x"]) & (jnp.abs(local_y) < internal_state["box_half_y"]) & (qpos[:, 2] - 2.0 * internal_state["box_half_height"] > 0.2)
                internal_state["on_box_steps"] = jnp.where(done | ~on_box, 0.0, internal_state["on_box_steps"] + 1.0).astype(jnp.float32)
                error = (internal_state["on_box_steps"] < 50.0).astype(jnp.float32)
                internal_state["task_error_sum"] = jnp.where(done, error, jnp.minimum(internal_state["task_error_sum"], error)).astype(jnp.float32)
                internal_state["task_error_count"] = jnp.ones_like(internal_state["task_error_count"])
            else:
                error = jnp.linalg.norm(achieved[:, :2] - goal[:, :2], axis=-1) / jnp.maximum(jnp.linalg.norm(goal[:, :2], axis=-1), self.goal_radius)
                internal_state["task_error_sum"] = jnp.where(done, error, internal_state["task_error_sum"] + error).astype(jnp.float32)
                internal_state["task_error_count"] = jnp.where(done, 1.0, internal_state["task_error_count"] + 1.0).astype(jnp.float32)

        steps = jnp.where(state.done > 0.5, 0.0, state.info["steps"])
        info = {
            **state.info,
            "seed": state.info["seed"] + (steps == 0).astype(jnp.int32),
            "truncation": (locomotion_state.truncated & ~locomotion_state.terminated).astype(jnp.float32),
            "steps": steps + 1.0,
        }
        return state.replace(locomotion=locomotion_state, goal=goal, start=start, start_yaw=start_yaw, key=key, obs=self.observation(locomotion_state, yaw, start_yaw, goal, achieved),
                             reward=jnp.where(jnp.isfinite(cost), cost, 0.0), done=done.astype(jnp.float32), metrics=self.evaluation_step_metrics(goal, achieved), info=info)
