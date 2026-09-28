# Adapted from JaxGCRL (https://github.com/MichalBortkiewicz/JaxGCRL, Apache-2.0) in the version used by
# https://github.com/wang-kevin3290/scaling-crl. Changes: only the layouts of the paper, no reward terms.
import os
import xml.etree.ElementTree as ET
import jax
from jax import numpy as jp
from brax import actuator, base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

from chronosrl.environments.jaxgcrl.ant_maze import make_maze


GEAR = [350.0] * 11 + [100.0] * 6
TARGET_Z_COORD = 1.25

LAYOUTS = {
    "u_maze": [
        [1, 1, 1, 1, 1],
        [1, "r", "g", "g", 1],
        [1, 1, 1, "g", 1],
        [1, "g", "g", "g", 1],
        [1, 1, 1, 1, 1],
    ],
    "big_maze": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, "r", "g", 1, 1, "g", "g", 1],
        [1, "g", "g", 1, "g", "g", "g", 1],
        [1, 1, "g", "g", "g", 1, 1, 1],
        [1, "g", "g", 1, "g", "g", "g", 1],
        [1, "g", 1, "g", "g", 1, "g", 1],
        [1, "g", "g", "g", 1, "g", "g", 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
}


class Humanoid(PipelineEnv):
    def __init__(self):
        sys = mjcf.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "humanoid.xml"))
        sys = sys.tree_replace({"opt.timestep": 0.0015})
        self._init(sys)

    def _init(self, sys):
        sys = sys.replace(actuator=sys.actuator.replace(gear=jp.array(GEAR)))
        super().__init__(sys=sys, backend="spring", n_frames=10)
        self.healthy_z_range = (1.0, 2.0)
        self.state_dim = 268
        self.goal_indices = (0, 3)
        self.goal_radius = 0.5

    def policy_observation(self, obs):
        return obs


    def training_metrics(self, state):
        return {}


    def reset(self, rng):
        _, rng1, rng2 = jax.random.split(jax.random.split(rng, 3)[0], 3)
        dist = jax.random.uniform(rng1, minval=1.0, maxval=5.0)
        angle = jp.pi * 2.0 * jax.random.uniform(rng2)
        qpos = self.sys.init_q.at[-2:].set(jp.array([dist * jp.cos(angle), dist * jp.sin(angle)]))
        return self._reset(qpos, jp.zeros(self.sys.qd_size()))

    def _reset(self, qpos, qvel):
        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)
        state = State(pipeline_state, obs, reward, done, {"dist": zero, "success": zero})
        state.info.update({"seed": 0})
        return state

    def step(self, state, action):
        seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1) if "steps" in state.info else state.info["seed"]
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        min_z, max_z = self.healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        obs = self._get_obs(pipeline_state, action)
        dist = jp.linalg.norm(obs[:3] - obs[-3:])
        state.metrics.update(dist=dist, success=jp.array(dist < 0.5, dtype=float))
        state.info.update({"seed": seed})
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=jp.zeros(()), done=1.0 - is_healthy)

    def _get_obs(self, pipeline_state, action):
        com, inertia, mass_sum, x_i = self._com(pipeline_state)
        cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        com_inertia = jp.hstack([cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]])
        xd_i = base.Transform.create(pos=x_i.pos - pipeline_state.x.pos).vmap().do(pipeline_state.xd)
        com_velocity = jp.hstack([inertia.mass[:, None] * xd_i.vel / mass_sum, xd_i.ang])
        qfrc_actuator = actuator.to_tau(self.sys, action, pipeline_state.q, pipeline_state.qd)
        return jp.concatenate([
            pipeline_state.q, pipeline_state.qd, com_inertia.ravel(), com_velocity.ravel(), qfrc_actuator,
            pipeline_state.x.pos[-1][:2], jp.array([TARGET_Z_COORD])
        ])

    def _com(self, pipeline_state):
        inertia = self.sys.link.inertia
        inertia = inertia.replace(
            i=jax.vmap(jp.diag)(jax.vmap(jp.diagonal)(inertia.i) ** (1 - self.sys.spring_inertia_scale)),
            mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
        )
        mass_sum = jp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        return com, inertia, mass_sum, x_i


class HumanoidMaze(Humanoid):
    def __init__(self, layout_name):
        tree, starts, self.possible_goals = make_maze(LAYOUTS[layout_name], 2.0, "humanoid_maze.xml")
        self.possible_starts = jp.array(starts)
        sys = mjcf.loads(ET.tostring(tree.getroot()))
        sys = sys.replace(dt=0.0015)
        self._init(sys)

    def reset(self, rng):
        rng, _, _, rng3 = jax.random.split(rng, 4)
        start = self.possible_starts[jax.random.randint(rng3, (1,), 0, len(self.possible_starts))][0]
        target = self.possible_goals[jax.random.randint(rng, (1,), 0, len(self.possible_goals))][0]
        qpos = self.sys.init_q.at[:2].set(start).at[-2:].set(target)
        return self._reset(qpos, jp.zeros(self.sys.qd_size()))
