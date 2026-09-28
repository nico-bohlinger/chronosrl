from copy import deepcopy
import warp as wp
import numpy as np
import mujoco
from mujoco import mjx
from mujoco.mjx._src import io as _mjx_io
from mujoco.mjx._src import types as _mjx_types
import mujoco.mjx.third_party.mujoco_warp._src.io as _mjwp_io
import mujoco.mjx.warp.types as _mjxw_types
from dm_control import mjcf
from jax.scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp

try:
    from warp._src.jax_experimental.ffi import GraphMode as _WarpGraphMode
    _WARP_GRAPH_MODE_MAP = {
        "jax": _WarpGraphMode.JAX,
        "warp": _WarpGraphMode.WARP,
        "warp_staged": _WarpGraphMode.WARP_STAGED,
        "warp_staged_ex": _WarpGraphMode.WARP_STAGED_EX,
    }
except ImportError:
    _WarpGraphMode = None
    _WARP_GRAPH_MODE_MAP = {}

from chronosrl.environments.go2.locomotion.state import State
from chronosrl.environments.go2.locomotion.box_space import BoxSpace
from chronosrl.environments.go2.locomotion.buffer_sizing import measure_per_world_buffer_sizes
from chronosrl.environments.go2.locomotion.control import PDControl
from chronosrl.environments.go2.locomotion.commands import RandomCommands
from chronosrl.environments.go2.locomotion.sampling import StepProbabilitySampling, StepProbabilityAndResetSampling
from chronosrl.environments.go2.locomotion.reward import DefaultReward
from chronosrl.environments.go2.locomotion.termination import BelowHeightTermination
from chronosrl.environments.go2.locomotion.exteroception import NoneExteroceptiveObservation, HeightOverGroundExteroceptiveObservation
from chronosrl.environments.go2.locomotion.terrain.plane import PlaneTerrainGeneration
from chronosrl.environments.go2.locomotion.terrain.plane_box import PlaneBoxTerrainGeneration
from chronosrl.environments.go2.locomotion.domain_randomization.initial_state import RandomDRInitialState
from chronosrl.environments.go2.locomotion.domain_randomization.action_delay import DefaultActionDelay
from chronosrl.environments.go2.locomotion.domain_randomization.mujoco_model import DefaultDRMuJoCoModel
from chronosrl.environments.go2.locomotion.domain_randomization.seen_robot import DefaultDRSeenRobotFunction
from chronosrl.environments.go2.locomotion.domain_randomization.unseen_robot import DefaultDRUnseenRobotFunction
from chronosrl.environments.go2.locomotion.domain_randomization.perturbation import DefaultDRPerturbation
from chronosrl.environments.go2.locomotion.domain_randomization.observation_noise import DefaultDRObservationNoise
from chronosrl.environments.go2.locomotion.domain_randomization.joint_dropout import DefaultDRJointDropout


def make_batched_warp_data(mj_model, nworld, naconmax, njmax):
    def _wp_to_np(wp_field):
        if isinstance(wp_field, wp.array):
            return wp_field.numpy()
        wp_dtype = type(wp_field)
        if wp_dtype in wp.types.warp_type_to_np_dtype:
            return wp.types.warp_type_to_np_dtype[wp_dtype](wp_field)
        return wp_field

    with wp.ScopedDevice("cpu"):
        dw = _mjwp_io.make_data(mj_model, nworld=nworld, naconmax=naconmax, njmax=njmax)

    fields = _mjx_io._make_data_public_fields(mj_model)
    for k in list(fields.keys()):
        if k in {"userdata", "plugin_state", "history"}:
            continue
        if not hasattr(dw, k):
            continue
        fields[k] = _wp_to_np(getattr(dw, k))

    impl_fields = {}
    for k in _mjxw_types.DataWarp.__annotations__.keys():
        raw = _mjx_io._get_nested_attr(dw, k, split="__")
        impl_fields[k] = _wp_to_np(raw)

    eq_active_batched = np.tile(mj_model.eq_active0.reshape(1, -1), (nworld, 1)).astype(bool)

    data = _mjx_types.Data(
        qpos=np.tile(mj_model.qpos0, (nworld, 1)).astype(np.float32),
        eq_active=eq_active_batched,
        **{k: v for k, v in fields.items() if k not in ("qpos", "eq_active")},
        _impl=_mjxw_types.DataWarp(**impl_fields),
    )
    return jax.device_put(data)


class LocomotionEnv:
    def __init__(self, robot_config, env_config, nr_envs):

        self.robot_config = robot_config
        self.env_config = env_config
        self.nr_envs = nr_envs
        self.graph_mode = env_config["graph_mode"]
        # naconmax (total, all worlds) and njmax (per world) are finalized once the model is built; if
        # either is None in the config they are measured from the robot and terrain (see below)
        self.naconmax = None
        self.njmax = None

        xml_path = (self.robot_config["directory_path"] / "plane.xml").as_posix()
        xml_handle = mjcf.from_path(xml_path)

        # Remove all unnecessary assets, materials, meshes and geoms during training
        # This removes all geoms besides feet and floor, if the contacts for other geoms should be enabled this needs to be changed
        for texture in xml_handle.asset.find_all("texture"):
            texture.remove()
        for material in xml_handle.asset.find_all("material"):
            material.remove()
        for mesh in xml_handle.asset.find_all("mesh"):
            mesh.remove()
        box_terrain = env_config["terrain"]["type"] == "plane_box"
        self.box_terrain = box_terrain
        nr_named_calf_geoms = 0
        for geom in xml_handle.find_all("geom"):
            is_foot_geom = geom.name and "foot" in geom.name
            is_floor_geom = geom.name == "floor"
            is_reward_collision_sphere_geom = geom.dclass and geom.dclass.dclass == "reward_collision_sphere"
            is_box_calf_geom = box_terrain and geom.type == "cylinder" and "calf" in (getattr(geom.parent, "name", None) or "")
            if is_box_calf_geom:
                if not geom.name:
                    geom.name = f"{geom.parent.name}_shin{nr_named_calf_geoms}"
                    nr_named_calf_geoms += 1
                geom.type = "capsule"
            if not is_foot_geom and not is_floor_geom and not is_reward_collision_sphere_geom and not is_box_calf_geom:
                geom.remove()
            if is_floor_geom:
                geom.material = ""

        # The box is a mocap body, so MuJoCo-Warp can place it per world, and it collides with the feet and the calves
        if box_terrain:
            floor = xml_handle.find("geom", "floor")
            terrain_config = env_config["terrain"]
            box_body = xml_handle.worldbody.add("body", name="climb_box_body", mocap="true", pos=f"{float(terrain_config['box_offset_in_meters'])} 0 0.0001")
            box_body.add("geom", name="climb_box", type="box", size=f"{0.5 * float(terrain_config['box_length_in_meters'])} {0.5 * float(terrain_config['box_width_in_meters'])} 0.0001", pos="0 0 0", group="0",
                         contype=floor.contype, conaffinity=floor.conaffinity, condim=floor.condim, friction=floor.friction, rgba="0.55 0.45 0.35 1")
            timeconst = float(terrain_config["box_contact_timeconst"])
            for name in [geom.name for geom in xml_handle.find_all("geom") if geom.name and "foot" in geom.name]:
                xml_handle.contact.add("pair", name=f"climb_box_{name}", geom1="climb_box", geom2=name, condim=6, friction=[0.8, 0.8, 0.02, 0.01, 0.01], solref=[timeconst, 1], solimp=[0.8, 0.95, 0.001])
            for name in [geom.name for geom in xml_handle.find_all("geom") if geom.name and "calf" in geom.name]:
                xml_handle.contact.add("pair", name=f"climb_box_{name}", geom1="climb_box", geom2=name, condim=3, friction=[0.8, 0.8, 0.005, 0.0001, 0.0001], solref=[timeconst, 1])

        self.initial_mj_model = mujoco.MjModel.from_xml_string(xml=xml_handle.to_xml_string(), assets=xml_handle.get_assets())
        self.initial_mj_model.opt.timestep = env_config["timestep"]
        self.data = mujoco.MjData(self.initial_mj_model)

        naconmax_per_env = env_config["naconmax_per_env"]
        njmax = env_config["njmax"]
        if naconmax_per_env is None or njmax is None:
            measured_naconmax_per_env, measured_njmax = measure_per_world_buffer_sizes(self.initial_mj_model, env_config["seed"])
            naconmax_per_env = measured_naconmax_per_env if naconmax_per_env is None else naconmax_per_env
            njmax = measured_njmax if njmax is None else njmax
        self.naconmax = naconmax_per_env * nr_envs
        self.njmax = njmax

        gm = _WARP_GRAPH_MODE_MAP.get(self.graph_mode.lower(), None)
        put_model_kwargs = {"impl": "warp"}
        if gm is not None:
            put_model_kwargs["graph_mode"] = gm
        self.initial_mjx_model = mjx.put_model(self.initial_mj_model, **put_model_kwargs)
        self.mjx_data = make_batched_warp_data(self.initial_mj_model, nworld=self.nr_envs, naconmax=self.naconmax, njmax=self.njmax)
        self.mjx_data = mjx.forward(self.initial_mjx_model, self.mjx_data)  # Necessary because of error with toddlerbot
        self.c_model = deepcopy(self.initial_mj_model)
        self.c_data = mujoco.MjData(self.c_model)
        self.c_data.qpos = self.initial_mj_model.keyframe("home").qpos
        mujoco.mj_forward(self.c_model, self.c_data)

        self.imu_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.trunk_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.actuator_joint_max_velocities = jnp.array(robot_config["actuator_joint_max_velocities"])
        self.initial_qpos = jnp.array(self.initial_mj_model.keyframe("home").qpos)
        self.initial_imu_orientation_rotation_inverse = Rotation.from_matrix(self.c_data.site_xmat[self.imu_site_id].reshape(3, 3)).inv()
        self.initial_imu_height = self.c_data.site_xpos[self.imu_site_id, 2]
        self.actuator_joint_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, actuator_trnid[0]) for actuator_trnid in self.initial_mj_model.actuator_trnid]
        self.actuator_joint_mask_joints = jnp.array([self.initial_mj_model.joint(joint_name).id for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qpos = jnp.array([self.initial_mj_model.joint(joint_name).qposadr[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qvel = jnp.array([self.initial_mj_model.joint(joint_name).dofadr[0] for joint_name in self.actuator_joint_names])
        self.nr_actuator_joints = len(self.actuator_joint_names)
        self.nr_joints = self.initial_mj_model.njnt

        imu_angular_velocity_sensor_id = self.initial_mj_model.sensor("imu_angular_velocity").id
        self.imu_angular_velocity_sensor_adr = self.initial_mj_model.sensor_adr[imu_angular_velocity_sensor_id]
        self.imu_angular_velocity_sensor_dim = self.initial_mj_model.sensor_dim[imu_angular_velocity_sensor_id]
        imu_linear_velocity_sensor_id = self.initial_mj_model.sensor("imu_linear_velocity").id
        self.imu_linear_velocity_sensor_adr = self.initial_mj_model.sensor_adr[imu_linear_velocity_sensor_id]
        self.imu_linear_velocity_sensor_dim = self.initial_mj_model.sensor_dim[imu_linear_velocity_sensor_id]

        geom_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) for geom_id in range(self.initial_mj_model.ngeom)]
        self.feet_names = [geom_name for geom_name in geom_names if geom_name and "foot" in geom_name]
        self.foot_geom_indices = jnp.array([mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, foot_name) for foot_name in self.feet_names])
        self.nr_feet = len(self.feet_names)

        feet_xpos = self.c_data.geom_xpos[self.foot_geom_indices]
        x_pos, y_pos, z_pos = feet_xpos[:, 0], feet_xpos[:, 1], feet_xpos[:, 2]
        abs_y_feet_xpos = np.array([x_pos, jnp.abs(y_pos), z_pos]).T
        distances_between_abs_y_feet = np.linalg.norm(abs_y_feet_xpos[:, None] - abs_y_feet_xpos[None], axis=-1)
        min_dist_indices = np.argmin(distances_between_abs_y_feet + np.eye(len(abs_y_feet_xpos)) * 1000, axis=1)
        feet_symmetry_set = set([(min(i, min_dist_indices[i]), max(i, min_dist_indices[i])) for i in range(len(min_dist_indices)) if min_dist_indices[min_dist_indices[i]] == i])
        self.feet_symmetry_pairs = jnp.array([list(pair) for pair in feet_symmetry_set]).reshape(-1, 2)
        feet_deltas = feet_xpos[np.asarray(self.feet_symmetry_pairs)[:, 0], :2] - feet_xpos[np.asarray(self.feet_symmetry_pairs)[:, 1], :2]
        nominal_imu_rotation = self.c_data.site_xmat[self.imu_site_id].reshape(3, 3)
        nominal_imu_yaw = np.arctan2(nominal_imu_rotation[1, 0], nominal_imu_rotation[0, 0])
        self.nominal_feet_lateral_distances = jnp.array(np.abs(-np.sin(nominal_imu_yaw) * feet_deltas[:, 0] + np.cos(nominal_imu_yaw) * feet_deltas[:, 1]))
        self.body_ids_of_feet = jnp.array([self.initial_mj_model.geom(geom_id).bodyid[0] for geom_id in self.foot_geom_indices])
        nominal_feet_rotations = self.c_data.xmat[np.asarray(self.body_ids_of_feet)].reshape(-1, 3, 3)
        self.nominal_feet_tilt = jnp.array(np.sqrt(nominal_feet_rotations[:, 2, 0] ** 2 + nominal_feet_rotations[:, 2, 1] ** 2))
        all_feet_are_sphere = jnp.all(self.initial_mjx_model.geom_type[self.foot_geom_indices] == 2)
        all_feet_are_box = jnp.all(self.initial_mjx_model.geom_type[self.foot_geom_indices] == 6)
        if not all_feet_are_sphere | all_feet_are_box:
            raise ValueError("Foot geoms are not all of type sphere or box.")
        self.foot_type = "sphere" if all_feet_are_sphere else "box"
        self.foot_type_int = 0 if self.foot_type == "sphere" else 1

        feet_global_linear_velocity_sensor_ids = [self.initial_mj_model.sensor(f"{foot_name}_global_linear_velocity").id for foot_name in self.feet_names]
        self.feet_global_linear_velocity_sensor_adrs_start = jnp.array([self.initial_mj_model.sensor_adr[sensor_id] for sensor_id in feet_global_linear_velocity_sensor_ids])

        body_to_parentid = jnp.array([self.initial_mj_model.body(body_id).parentid[0] for body_id in range(self.initial_mj_model.nbody)])
        body_to_children_count = jnp.array([jnp.sum(body_to_parentid == body_id) for body_id in range(self.initial_mj_model.nbody)])
        self.body_ids_of_actuator_joints = jnp.array([self.initial_mj_model.joint(joint_name).bodyid[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_nr_direct_child_actuator_joints = body_to_children_count[self.body_ids_of_actuator_joints]

        self.floor_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.box_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "climb_box")

        self.reward_collision_sphere_geom_ids = jnp.array([geom.id for geom in [self.initial_mj_model.geom(geom_id) for geom_id in range(self.initial_mj_model.ngeom)] if geom.group[0] == 5])

        self.has_equality_constraints = len(self.initial_mj_model.eq_data) > 0

        self.robot_dimensions_mean = 0.5  # This can be calculated smartly...

        self.env_curriculum_nr_levels = env_config["env_curriculum_nr_levels"]
        self.env_curriculum_success_error = env_config["env_curriculum_success_error"]
        self.env_curriculum_success_episode_length = env_config["env_curriculum_success_episode_length"]

        self.control_function = PDControl(self)
        self.control_frequency_hz = self.control_function.control_frequency_hz
        self.nr_substeps = int(round(1 / self.control_frequency_hz / env_config["timestep"]))
        self.dt = env_config["timestep"] * self.nr_substeps
        self.horizon = int(round(env_config["episode_length_in_seconds"] * self.control_frequency_hz))
        self.command_function = RandomCommands(self)
        self.command_sampling_function = StepProbabilityAndResetSampling(self)
        self.initial_state_function = RandomDRInitialState(self)
        self.reward_function = DefaultReward(self)
        self.termination_function = BelowHeightTermination(self)
        self.policy_exteroceptive_observation_function = NoneExteroceptiveObservation(self)
        self.critic_exteroceptive_observation_function = HeightOverGroundExteroceptiveObservation(self)
        self.terrain_function = PlaneBoxTerrainGeneration(self) if box_terrain else PlaneTerrainGeneration(self)
        self.domain_randomization_sampling_function = StepProbabilityAndResetSampling(self)
        self.domain_randomization_action_delay_function = DefaultActionDelay(self)
        self.domain_randomization_mujoco_model_function = DefaultDRMuJoCoModel(self)
        self.domain_randomization_seen_robot_function = DefaultDRSeenRobotFunction(self)
        self.domain_randomization_unseen_robot_function = DefaultDRUnseenRobotFunction(self)
        self.domain_randomization_perturbation_function = DefaultDRPerturbation(self)
        self.domain_randomization_perturbation_sampling_function = StepProbabilitySampling(self)
        self.observation_noise_function = DefaultDRObservationNoise(self)
        self.joint_dropout_function = DefaultDRJointDropout(self)

        action_space_size = self.nr_actuator_joints
        lower_joint_limit, upper_joint_limit = self.initial_mj_model.jnt_range[self.actuator_joint_mask_joints].T
        nominal_joint_positions = self.initial_qpos[self.actuator_joint_mask_qpos]
        action_scale_factor = robot_config["scaling_factor"]
        # The action space attributes are fixed and do not change with domain randomization, if they are randomized heavily the algorithm using them might need to be adapted
        self.single_action_space = BoxSpace(low=lower_joint_limit, high=upper_joint_limit, shape=(action_space_size,), dtype=jnp.float32, center=nominal_joint_positions, scale=action_scale_factor)

        self.single_observation_space = self.get_observation_space()

        self.observation_noise_function.init_attributes()

        del self.c_model, self.c_data


    def feet_bottom_extent(self, data, mjx_model):
        geom_size = mjx_model.geom_size
        if geom_size.ndim == 2:
            geom_size = jnp.broadcast_to(geom_size, (self.nr_envs,) + geom_size.shape)
        feet_sizes = geom_size[:, self.foot_geom_indices]
        if self.foot_type == "sphere":
            return feet_sizes[:, :, 0]
        feet_rotations = data.geom_xmat[:, self.foot_geom_indices].reshape(self.nr_envs, self.nr_feet, 3, 3)
        return jnp.sum(feet_sizes * jnp.abs(feet_rotations[:, :, 2, :]), axis=-1)


    def reset(self, keys):
        nr_envs = self.nr_envs
        key = keys[0]
        mjx_model = self.initial_mjx_model
        data = self.mjx_data

        next_observation = jnp.zeros((nr_envs,) + self.single_observation_space.shape, dtype=jnp.float32)
        reward = jnp.zeros(nr_envs, dtype=jnp.float32)
        terminated = jnp.zeros(nr_envs, dtype=bool)
        truncated = jnp.zeros(nr_envs, dtype=bool)

        internal_state = {
            "env_curriculum_coeff": jnp.zeros(nr_envs),
            "env_curriculum_levels_in_a_row": jnp.zeros(nr_envs),
            "actuator_joint_nominal_positions": jnp.tile(self.initial_qpos[self.actuator_joint_mask_qpos][None], (nr_envs, 1)),
            "actuator_joint_max_velocities": jnp.tile(self.actuator_joint_max_velocities[None], (nr_envs, 1)),
            "goal_velocities": jnp.zeros((nr_envs, 3)),
            "imu_orientation_rotation": Rotation.from_quat(jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0]), (nr_envs, 1))),
            "imu_orientation_rotation_inverse": Rotation.from_quat(jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0]), (nr_envs, 1))).inv(),
            "imu_orientation_euler": jnp.zeros((nr_envs, 3)),
            "last_action": jnp.zeros((nr_envs, self.nr_actuator_joints)),
            "second_last_action": jnp.zeros((nr_envs, self.nr_actuator_joints)),
            "joint_dropout_mask": jnp.ones((nr_envs, self.nr_actuator_joints), dtype=bool),
            "robot_dimensions_mean": jnp.full(nr_envs, self.robot_dimensions_mean),
            "max_command_velocity": jnp.full(nr_envs, jnp.minimum(self.robot_dimensions_mean * self.command_function.max_velocity_per_m_factor, self.command_function.clip_max_velocity)),
            "nr_collisions_in_nominal": jnp.zeros(nr_envs),
            "nr_ground_penetrations_in_nominal": jnp.zeros((nr_envs, self.reward_collision_sphere_geom_ids.shape[0])),
            "nominal_feet_tilt": jnp.tile(self.nominal_feet_tilt[None], (nr_envs, 1)),
            "nominal_feet_lateral_distances": jnp.tile(self.nominal_feet_lateral_distances[None], (nr_envs, 1)),
        }
        self.command_function.init(internal_state)
        self.reward_function.init(internal_state, mjx_model)
        self.terrain_function.init(internal_state)
        self.joint_dropout_function.init(internal_state)
        self.domain_randomization_action_delay_function.init(internal_state)
        self.domain_randomization_seen_robot_function.init(internal_state)
        self.domain_randomization_unseen_robot_function.init(internal_state)

        info = {}
        self.reward_function.reward_and_info(data, mjx_model, internal_state, jnp.zeros((nr_envs, self.nr_actuator_joints)), info)
        info["rollout/episode_return"] = reward
        info["rollout/episode_length"] = jnp.zeros(nr_envs)
        info["env_curriculum/coefficient"] = internal_state["env_curriculum_coeff"]
        info_episode_store = {
            "episode_return": jnp.zeros(nr_envs),
            "episode_step": jnp.zeros(nr_envs),
            "episode_total_xy_velocity_diff_abs": jnp.zeros(nr_envs),
        }

        state = State(mjx_model, data, next_observation, next_observation, reward, terminated, truncated, info, info_episode_store, internal_state, key)

        return self._reset(state)


    def _reset(self, state):
        nr_envs = self.nr_envs
        key, initial_state_key, terrain_key, domain_randomization_key, command_sampling_key, command_key, observation_key = jax.random.split(state.key, 7)
        state = state.replace(key=key)

        new_state = state

        # The curriculum judges the task error of the goal environment when it provides one, else the velocity tracking error
        mean_xy_velocity_diff_abs = new_state.info_episode_store["episode_total_xy_velocity_diff_abs"] / jnp.maximum(new_state.info_episode_store["episode_step"], 1)
        episode_error = mean_xy_velocity_diff_abs / jnp.maximum(new_state.internal_state["max_command_velocity"], 1e-6)
        if "task_error_sum" in new_state.internal_state:
            episode_error = new_state.internal_state["task_error_sum"] / jnp.maximum(new_state.internal_state["task_error_count"], 1.0)
        episode_success = (episode_error <= self.env_curriculum_success_error) & (new_state.info_episode_store["episode_step"] >= self.env_curriculum_success_episode_length)
        new_state.internal_state["env_curriculum_levels_in_a_row"] = jnp.where(episode_success,
            jnp.where(new_state.internal_state["env_curriculum_levels_in_a_row"] >= 0,
                new_state.internal_state["env_curriculum_levels_in_a_row"] + 1,
                1
            ),
            jnp.where(new_state.internal_state["env_curriculum_levels_in_a_row"] < 0,
                new_state.internal_state["env_curriculum_levels_in_a_row"] - 1,
                -1
            )
        )
        new_state.internal_state["env_curriculum_coeff"] = jnp.clip(new_state.internal_state["env_curriculum_coeff"] + new_state.internal_state["env_curriculum_levels_in_a_row"] / self.env_curriculum_nr_levels, 0.0, 1.0)

        mjx_model = self.terrain_function.sample(state.mjx_model, state.internal_state, terrain_key)

        data = self.mjx_data
        qpos, qvel = self.initial_state_function.setup(mjx_model, state.internal_state, initial_state_key)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros((nr_envs, self.nr_actuator_joints)))

        new_state.internal_state["last_action"] = jnp.zeros((nr_envs, self.nr_actuator_joints))
        new_state.internal_state["second_last_action"] = jnp.zeros((nr_envs, self.nr_actuator_joints))
        self.reward_function.setup(new_state.internal_state)
        self.domain_randomization_action_delay_function.setup(new_state.internal_state)
        data, mjx_model = self.handle_domain_randomization(new_state.internal_state, mjx_model, data, domain_randomization_key, is_episode_start=True)
        if self.box_terrain:
            mjx_model = self.terrain_function.box_geometry(mjx_model, new_state.internal_state)
            data = self.terrain_function.box_pose(data, new_state.internal_state)
        data = mjx.forward(mjx_model, data)
        new_state.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(data.site_xmat[:, self.imu_site_id].reshape(nr_envs, 3, 3))
        new_state.internal_state["imu_orientation_rotation_inverse"] = new_state.internal_state["imu_orientation_rotation"].inv()
        new_state.internal_state["imu_orientation_euler"] = new_state.internal_state["imu_orientation_rotation"].as_euler("xyz")
        self.terrain_function.pre_step(data, new_state.internal_state)
        should_sample_commands = self.command_sampling_function.setup(command_sampling_key)
        self.command_function.get_next_command(new_state.internal_state, should_sample_commands, command_key)

        next_observation = self.get_observation(data, mjx_model, new_state.internal_state, observation_key, jnp.zeros((nr_envs, self.nr_actuator_joints)))
        reward = jnp.zeros(nr_envs, dtype=jnp.float32)
        terminated = jnp.zeros(nr_envs, dtype=bool)
        truncated = jnp.zeros(nr_envs, dtype=bool)
        info_episode_store = {
            "episode_return": jnp.zeros(nr_envs),
            "episode_step": jnp.zeros(nr_envs),
            "episode_total_xy_velocity_diff_abs": jnp.zeros(nr_envs),
        }

        # Reset everything besides parts of the internal_state, info and the key
        new_state = new_state.replace(
            mjx_model=mjx_model,
            data=data,
            next_observation=next_observation, actual_next_observation=next_observation,
            reward=reward,
            terminated=terminated, truncated=truncated,
            info_episode_store=info_episode_store
        )

        return new_state


    def step(self, state, action):
        nr_envs = self.nr_envs
        key, domain_randomization_key, command_sampling_key, command_key, observation_key, terrain_key = jax.random.split(state.key, 6)
        state = state.replace(key=key)

        data, mjx_model = self.handle_domain_randomization(state.internal_state, state.mjx_model, state.data, domain_randomization_key)
        if self.box_terrain:
            mjx_model = self.terrain_function.box_geometry(mjx_model, state.internal_state)
        data = self.terrain_function.before_physics_step(data, mjx_model, state.internal_state, terrain_key)
        state = state.replace(data=data, mjx_model=mjx_model)

        chosen_action = action[:, :self.nr_actuator_joints]
        delayed_actions = self.domain_randomization_action_delay_function.delay_action(chosen_action, state.internal_state)

        data = state.data
        data = data.replace(qacc_warmstart=jnp.nan_to_num(data.qacc_warmstart, nan=0.0, posinf=0.0, neginf=0.0))
        for delayed_action in delayed_actions:
            data = data.replace(ctrl=self.control_function.process_action(delayed_action, state.internal_state))
            data = mjx.step(state.mjx_model, data)
        data = data.replace(qvel=jnp.clip(data.qvel, -100, 100))

        state.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(data.site_xmat[:, self.imu_site_id].reshape(nr_envs, 3, 3))
        state.internal_state["imu_orientation_rotation_inverse"] = state.internal_state["imu_orientation_rotation"].inv()
        state.internal_state["imu_orientation_euler"] = state.internal_state["imu_orientation_rotation"].as_euler("xyz")

        self.terrain_function.pre_step(data, state.internal_state)

        reward = self.reward_function.reward_and_info(data, mjx_model, state.internal_state, chosen_action, state.info)
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        self.reward_function.step(data, state.internal_state)

        should_sample_commands = self.command_sampling_function.step(command_sampling_key)
        self.command_function.get_next_command(state.internal_state, should_sample_commands, command_key)

        next_observation = self.get_observation(data, mjx_model, state.internal_state, observation_key, chosen_action)
        terminated = self.termination_function.should_terminate(state.internal_state) | jnp.any(jnp.abs(data.qvel) >= 100.0, axis=-1) | \
                     ~(jnp.all(jnp.isfinite(data.qpos), axis=-1) & jnp.all(jnp.isfinite(data.qvel), axis=-1))
        truncated = state.info_episode_store["episode_step"] >= (self.horizon - 1)
        done = terminated | truncated

        state.internal_state["second_last_action"] = state.internal_state["last_action"]
        state.internal_state["last_action"] = chosen_action
        state.info_episode_store["episode_step"] = state.info_episode_store["episode_step"] + 1
        state.info_episode_store["episode_return"] = state.info_episode_store["episode_return"] + reward
        state.info_episode_store["episode_total_xy_velocity_diff_abs"] = state.info_episode_store["episode_total_xy_velocity_diff_abs"] + state.info["env_info/xy_vel_diff_abs"]
        state.info["rollout/episode_return"] = jnp.where(done, state.info_episode_store["episode_return"], state.info["rollout/episode_return"])
        state.info["rollout/episode_length"] = jnp.where(done, state.info_episode_store["episode_step"], state.info["rollout/episode_length"])
        state.info["env_curriculum/coefficient"] = state.internal_state["env_curriculum_coeff"]

        state = state.replace(data=data)
        # Snapshot the post-step internal_state so the not-done branch is unaffected by the in-place
        # mutations that _reset performs on the (shared) internal_state dict
        not_done_state = state.replace(
            internal_state=dict(state.internal_state),
            next_observation=next_observation, actual_next_observation=next_observation,
            reward=reward, terminated=terminated, truncated=truncated
        )
        done_state = self._reset(state)
        done_state = done_state.replace(actual_next_observation=next_observation, reward=reward, terminated=terminated, truncated=truncated)

        return self.merge_done(done, done_state, not_done_state)


    def merge_done(self, done, done_state, not_done_state):
        def where_leaf(reset_leaf, keep_leaf):
            if jnp.ndim(reset_leaf) == 0:
                return keep_leaf
            d = jnp.reshape(done, (self.nr_envs,) + (1,) * (jnp.ndim(reset_leaf) - 1))
            return jnp.where(d, reset_leaf, keep_leaf)

        mjx_model = jax.tree_util.tree_map(
            lambda reset_leaf, keep_leaf: where_leaf(reset_leaf, keep_leaf)
            if jnp.ndim(reset_leaf) > 0 and jnp.shape(reset_leaf)[0] == self.nr_envs else keep_leaf,
            done_state.mjx_model, not_done_state.mjx_model
        )
        # The Warp-backed contact / constraint buffers do not carry a leading env axis, so only the
        # per-env qpos/qvel/ctrl/time/warm start are merged; all derived quantities are recomputed by the next mjx.step
        data = not_done_state.data.replace(
            qpos=where_leaf(done_state.data.qpos, not_done_state.data.qpos),
            qvel=where_leaf(done_state.data.qvel, not_done_state.data.qvel),
            ctrl=where_leaf(done_state.data.ctrl, not_done_state.data.ctrl),
            time=where_leaf(done_state.data.time, not_done_state.data.time),
            qacc_warmstart=where_leaf(done_state.data.qacc_warmstart, not_done_state.data.qacc_warmstart),
        )
        internal_state = jax.tree_util.tree_map(where_leaf, done_state.internal_state, not_done_state.internal_state)
        info_episode_store = jax.tree_util.tree_map(where_leaf, done_state.info_episode_store, not_done_state.info_episode_store)
        next_observation = where_leaf(done_state.next_observation, not_done_state.next_observation)

        return not_done_state.replace(
            mjx_model=mjx_model,
            data=data,
            next_observation=next_observation,
            info_episode_store=info_episode_store,
            internal_state=internal_state,
            key=done_state.key,
        )


    def get_observation(self, data, mjx_model, internal_state, key, action):
        nr_envs = self.nr_envs
        gravity_vector = internal_state["imu_orientation_rotation_inverse"].apply(jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), (nr_envs, 3)))
        observation = jnp.concatenate([
            data.qpos[:, self.actuator_joint_mask_qpos],
            data.qvel[:, self.actuator_joint_mask_qvel],
            action,
            self.terrain_function.check_feet_floor_contact(data),
            internal_state["feet_time_on_ground"],
            internal_state["feet_time_in_air"],
            data.sensordata[:, self.imu_linear_velocity_sensor_adr:self.imu_linear_velocity_sensor_adr + self.imu_linear_velocity_sensor_dim],
            data.sensordata[:, self.imu_angular_velocity_sensor_adr:self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim],
            internal_state["goal_velocities"],
            gravity_vector,
            self.policy_exteroceptive_observation_function.get_exteroceptive_observation(data, mjx_model, internal_state).reshape(nr_envs, -1),
            self.critic_exteroceptive_observation_function.get_exteroceptive_observation(data, mjx_model, internal_state).reshape(nr_envs, -1),
        ], axis=-1)

        # Add noise
        observation = self.observation_noise_function.modify_observation(internal_state, observation, key)

        # Normalize and clip
        observation = observation.at[:, self.joint_positions_obs_idx].set((observation[:, self.joint_positions_obs_idx] - internal_state["actuator_joint_nominal_positions"]) / 3.14)
        observation = observation.at[:, self.joint_velocities_obs_idx].set(observation[:, self.joint_velocities_obs_idx] / 100.0)
        observation = observation.at[:, self.joint_previous_actions_obs_idx].set(observation[:, self.joint_previous_actions_obs_idx] / 10.0)
        observation = observation.at[:, self.feet_ground_contact_obs_idx].set((observation[:, self.feet_ground_contact_obs_idx] / 0.5) - 1.0)
        observation = observation.at[:, self.feet_time_on_ground_obs_idx].set(jnp.clip((observation[:, self.feet_time_on_ground_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0))
        observation = observation.at[:, self.feet_time_in_air_obs_idx].set(jnp.clip((observation[:, self.feet_time_in_air_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0))
        observation = observation.at[:, self.imu_linear_vel_obs_idx].set(jnp.clip(observation[:, self.imu_linear_vel_obs_idx] / 10.0, -1.0, 1.0))
        observation = observation.at[:, self.imu_angular_vel_obs_idx].set(jnp.clip(observation[:, self.imu_angular_vel_obs_idx] / 50.0, -1.0, 1.0))
        if len(self.policy_exteroception_obs_idx) > 0:
            observation = observation.at[:, self.policy_exteroception_obs_idx].set(jnp.clip((observation[:, self.policy_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0))
        if len(self.critic_exteroception_obs_idx) > 0:
            observation = observation.at[:, self.critic_exteroception_obs_idx].set(jnp.clip((observation[:, self.critic_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0))

        observation = jnp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = jnp.clip(observation, -10.0, 10.0)

        return observation


    def handle_domain_randomization(self, internal_state, mjx_model, data, key, is_episode_start=False):
        domain_sampling_key, domain_perturbation_sampling_key, seen_robot_key, unseen_robot_key, mujoco_model_key, action_delay_key, joint_dropout_key, perturbation_key = jax.random.split(key, 8)

        should_randomize_domain_episode_start = self.domain_randomization_sampling_function.setup(domain_sampling_key)
        should_randomize_domain_perturbation_episode_start = self.domain_randomization_perturbation_sampling_function.setup(domain_perturbation_sampling_key, internal_state["env_curriculum_coeff"])
        should_randomize_domain_step = self.domain_randomization_sampling_function.step(domain_sampling_key)
        should_randomize_domain_perturbation_step = self.domain_randomization_perturbation_sampling_function.step(domain_perturbation_sampling_key, internal_state["env_curriculum_coeff"])
        should_randomize_domain = jnp.where(is_episode_start, should_randomize_domain_episode_start, should_randomize_domain_step)
        should_randomize_domain_perturbation = jnp.where(is_episode_start, should_randomize_domain_perturbation_episode_start, should_randomize_domain_perturbation_step)

        self.domain_randomization_unseen_robot_function.sample(internal_state, should_randomize_domain, unseen_robot_key)
        mjx_model, data = self.domain_randomization_seen_robot_function.sample(internal_state, mjx_model, data, should_randomize_domain, seen_robot_key)
        mjx_model = self.domain_randomization_mujoco_model_function.sample(internal_state, mjx_model, should_randomize_domain, mujoco_model_key)
        self.domain_randomization_action_delay_function.sample(internal_state, should_randomize_domain, action_delay_key)
        mjx_model = self.joint_dropout_function.sample(internal_state, mjx_model, should_randomize_domain, joint_dropout_key)
        self.reward_function.handle_model_change(internal_state, mjx_model, should_randomize_domain)

        data = self.domain_randomization_perturbation_function.sample(internal_state, mjx_model, data, should_randomize_domain_perturbation, perturbation_key)

        return data, mjx_model


    def get_observation_space(self):
        current_observation_idx = 0

        self.joint_positions_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_actuator_joints)])
        current_observation_idx += self.nr_actuator_joints
        self.joint_velocities_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_actuator_joints)])
        current_observation_idx += self.nr_actuator_joints
        self.joint_previous_actions_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_actuator_joints)])
        current_observation_idx += self.nr_actuator_joints
        self.feet_ground_contact_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_feet)])
        current_observation_idx += self.nr_feet
        self.feet_time_on_ground_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_feet)])
        current_observation_idx += self.nr_feet
        self.feet_time_in_air_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_feet)])
        current_observation_idx += self.nr_feet
        self.imu_linear_vel_obs_idx = jnp.array([current_observation_idx + i for i in range(self.imu_linear_velocity_sensor_dim)])
        current_observation_idx += self.imu_linear_velocity_sensor_dim
        self.imu_angular_vel_obs_idx = jnp.array([current_observation_idx + i for i in range(self.imu_angular_velocity_sensor_dim)])
        current_observation_idx += self.imu_angular_velocity_sensor_dim
        self.goal_velocities_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3
        self.gravity_vector_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3
        self.policy_exteroception_obs_idx = jnp.array([current_observation_idx + i for i in range(self.policy_exteroceptive_observation_function.nr_exteroceptive_observations)])
        current_observation_idx += self.policy_exteroceptive_observation_function.nr_exteroceptive_observations
        self.critic_exteroception_obs_idx = jnp.array([current_observation_idx + i for i in range(self.critic_exteroceptive_observation_function.nr_exteroceptive_observations)])
        current_observation_idx += self.critic_exteroceptive_observation_function.nr_exteroceptive_observations

        self.policy_observation_indices = jnp.concatenate([
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.imu_angular_vel_obs_idx,
            self.goal_velocities_obs_idx,
            self.gravity_vector_obs_idx,
            self.policy_exteroception_obs_idx,
        ], dtype=int)

        self.critic_observation_indices = jnp.concatenate([
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.feet_ground_contact_obs_idx,
            self.feet_time_on_ground_obs_idx,
            self.feet_time_in_air_obs_idx,
            self.imu_linear_vel_obs_idx,
            self.imu_angular_vel_obs_idx,
            self.goal_velocities_obs_idx,
            self.gravity_vector_obs_idx,
            self.critic_exteroception_obs_idx,
        ], dtype=int)

        return BoxSpace(low=-jnp.inf, high=jnp.inf, shape=(current_observation_idx,), dtype=jnp.float32)
