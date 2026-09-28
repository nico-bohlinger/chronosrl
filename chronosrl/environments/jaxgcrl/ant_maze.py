# Adapted from JaxGCRL (https://github.com/MichalBortkiewicz/JaxGCRL, Apache-2.0) in the version used by
# https://github.com/wang-kevin3290/scaling-crl. Changes: only the layouts of the paper, no reward terms.
import os
import xml.etree.ElementTree as ET
import jax
from jax import numpy as jp
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf


R = "r"
G = "g"

LAYOUTS = {
    "u4_maze": [
        [1, 1, 1, 1, 1],
        [1, G, G, G, 1],
        [1, R, 1, G, 1],
        [1, 1, 1, G, 1],
        [1, G, 1, G, 1],
        [1, G, G, G, 1],
        [1, 1, 1, 1, 1],
    ],
    "u5_maze": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, G, G, G, G, G, G, 1],
        [1, R, 1, 1, 1, 1, G, 1],
        [1, 1, 1, 1, 1, 1, G, 1],
        [1, G, 1, 1, 1, 1, G, 1],
        [1, G, G, G, G, G, G, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "big_maze": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, R, G, 1, 1, G, G, 1],
        [1, G, G, 1, G, G, G, 1],
        [1, 1, G, G, G, 1, 1, 1],
        [1, G, G, 1, G, G, G, 1],
        [1, G, 1, G, G, 1, G, 1],
        [1, G, G, G, 1, G, G, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "hardest_maze": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, R, G, G, G, 1, G, G, G, G, G, 1],
        [1, G, 1, 1, G, 1, G, 1, G, 1, G, 1],
        [1, G, G, G, G, G, G, 1, G, G, G, 1],
        [1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1],
        [1, G, G, 1, G, 1, G, G, G, G, G, 1],
        [1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1],
        [1, G, G, 1, G, G, G, 1, G, G, G, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
}
MAZE_HEIGHT = 0.5


def make_maze(layout, scaling, xml_file):
    tree = ET.parse(os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", xml_file))
    worldbody = tree.find(".//worldbody")
    starts, goals = [], []
    for i in range(len(layout)):
        for j in range(len(layout[0])):
            if layout[i][j] == R:
                starts.append([i * scaling, j * scaling])
            elif layout[i][j] == G:
                goals.append([i * scaling, j * scaling])
            elif layout[i][j] == 1:
                ET.SubElement(
                    worldbody, "geom", name="block_%d_%d" % (i, j),
                    pos="%f %f %f" % (i * scaling, j * scaling, MAZE_HEIGHT / 2 * scaling),
                    size="%f %f %f" % (0.5 * scaling, 0.5 * scaling, MAZE_HEIGHT / 2 * scaling),
                    type="box", material="", contype="1", conaffinity="1", rgba="0.7 0.5 0.3 1.0",
                )
    return tree, starts, jp.array(goals)


class AntMaze(PipelineEnv):
    def __init__(self, layout_name):
        scaling = 4.0
        tree, starts, self.possible_goals = make_maze(LAYOUTS[layout_name], scaling, "ant_maze.xml")
        init_qpos = tree.find(".//numeric[@name='init_qpos']")
        init_qpos.set("data", f"{starts[0][0]} {starts[0][1]} " + init_qpos.get("data"))
        sys = mjcf.loads(ET.tostring(tree.getroot()))
        sys = sys.replace(dt=0.005)
        super().__init__(sys=sys, backend="spring", n_frames=10)
        self.reset_noise_scale = 0.1
        self.healthy_z_range = (0.2, 1.0)
        self.state_dim = 29
        self.goal_indices = (0, 2)
        self.goal_radius = 0.5

    def policy_observation(self, obs):
        return obs


    def training_metrics(self, state):
        return {}


    def reset(self, rng):
        rng, rng1, rng2 = jax.random.split(rng, 3)
        low, hi = -self.reset_noise_scale, self.reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))
        target = self.possible_goals[jax.random.randint(rng, (1,), 0, len(self.possible_goals))][0]
        q = q.at[-2:].set(target)
        qd = qd.at[-2:].set(0)
        pipeline_state = self.pipeline_init(q, qd)
        reward, done, zero = jp.zeros(3)
        state = State(pipeline_state, self._get_obs(pipeline_state), reward, done, {"dist": zero, "success": zero})
        state.info.update({"seed": 0})
        return state

    def step(self, state, action):
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1) if "steps" in state.info else state.info["seed"]
        min_z, max_z = self.healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        obs = self._get_obs(pipeline_state)
        dist = jp.linalg.norm(obs[:2] - obs[-2:])
        state.metrics.update(dist=dist, success=jp.array(dist < 0.5, dtype=float))
        state.info.update({"seed": seed})
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=jp.zeros(()), done=1.0 - is_healthy)

    def _get_obs(self, pipeline_state):
        return jp.concatenate([pipeline_state.q[:-2], pipeline_state.qd[:-2], pipeline_state.x.pos[-1][:2]])
