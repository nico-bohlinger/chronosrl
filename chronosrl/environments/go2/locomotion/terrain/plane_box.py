import mujoco
import jax
import jax.numpy as jnp

from chronosrl.environments.go2.locomotion.terrain.plane import PlaneTerrainGeneration


class PlaneBoxTerrainGeneration(PlaneTerrainGeneration):
    def __init__(self, env):
        super().__init__(env)
        terrain_config = env.env_config["terrain"]
        self.box_length_in_meters = terrain_config["box_length_in_meters"]
        self.box_width_in_meters = terrain_config["box_width_in_meters"]
        self.box_offset_in_meters = terrain_config["box_offset_in_meters"]
        self.box_height_delta = terrain_config["box_height_delta"]
        self.box_random_x_range = terrain_config["box_random_x_range"]
        self.box_random_y_range = terrain_config["box_random_y_range"]
        self.box_size_random_x_range = terrain_config["box_size_random_x_range"]
        self.box_size_random_y_range = terrain_config["box_size_random_y_range"]
        self.box_geom_id = int(env.box_geom_id)
        box_body_id = mujoco.mj_name2id(env.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "climb_box_body")
        self.box_mocap_id = int(env.initial_mj_model.body_mocapid[box_body_id])


    def init(self, internal_state):
        super().init(internal_state)
        nr_envs = self.env.nr_envs
        internal_state["box_half_height"] = jnp.zeros(nr_envs)
        internal_state["box_centre_x"] = jnp.full(nr_envs, self.box_offset_in_meters)
        internal_state["box_centre_y"] = jnp.zeros(nr_envs)
        internal_state["box_half_x"] = jnp.full(nr_envs, 0.5 * self.box_length_in_meters)
        internal_state["box_half_y"] = jnp.full(nr_envs, 0.5 * self.box_width_in_meters)
        internal_state["box_yaw"] = jnp.zeros(nr_envs)


    def box_geometry(self, mjx_model, internal_state):
        nr_envs = self.env.nr_envs
        half_height = jnp.maximum(internal_state["box_half_height"], 1e-4)
        size = jnp.stack([internal_state["box_half_x"], internal_state["box_half_y"], half_height], axis=-1)
        geom_size = mjx_model.geom_size
        if geom_size.ndim == 2:
            geom_size = jnp.broadcast_to(geom_size[None], (nr_envs,) + geom_size.shape)
        geom_size = geom_size.at[:, self.box_geom_id].set(size)
        rbound = mjx_model.geom_rbound
        if rbound.ndim == 1:
            rbound = jnp.broadcast_to(rbound[None], (nr_envs,) + rbound.shape)
        rbound = rbound.at[:, self.box_geom_id].set(jnp.linalg.norm(size, axis=-1))
        aabb = mjx_model.geom_aabb
        if aabb.ndim == 3:
            aabb = jnp.broadcast_to(aabb[None], (nr_envs,) + aabb.shape)
        aabb = aabb.at[:, self.box_geom_id, 0].set(jnp.zeros_like(size))
        aabb = aabb.at[:, self.box_geom_id, 1].set(size)
        return mjx_model.replace(geom_size=geom_size, geom_rbound=rbound, geom_aabb=aabb)


    def box_pose(self, data, internal_state):
        half_height = jnp.maximum(internal_state["box_half_height"], 1e-4)
        pos = jnp.stack([internal_state["box_centre_x"], internal_state["box_centre_y"], half_height + self.ground_height], axis=-1)
        half_yaw = 0.5 * internal_state["box_yaw"]
        quat = jnp.stack([jnp.cos(half_yaw), jnp.zeros_like(half_yaw), jnp.zeros_like(half_yaw), jnp.sin(half_yaw)], axis=-1)
        mocap_pos = data.mocap_pos.at[:, self.box_mocap_id].set(pos.astype(data.mocap_pos.dtype))
        mocap_quat = data.mocap_quat.at[:, self.box_mocap_id].set(quat.astype(data.mocap_quat.dtype))
        return data.replace(mocap_pos=mocap_pos, mocap_quat=mocap_quat)


    def before_physics_step(self, data, mjx_model, internal_state, key):
        return self.box_pose(data, internal_state)


    def sample(self, mjx_model, internal_state, key):
        nr_envs = self.env.nr_envs
        _, _, x_key, y_key, size_x_key, size_y_key = jax.random.split(key, 6)
        coeff = internal_state["env_curriculum_coeff"]
        internal_state["box_half_height"] = 0.5 * coeff * self.box_height_delta
        internal_state["box_centre_x"] = self.box_offset_in_meters + coeff * jax.random.uniform(x_key, (nr_envs,), minval=-self.box_random_x_range, maxval=self.box_random_x_range)
        internal_state["box_centre_y"] = coeff * jax.random.uniform(y_key, (nr_envs,), minval=-self.box_random_y_range, maxval=self.box_random_y_range)
        internal_state["box_half_x"] = jnp.maximum(0.5 * self.box_length_in_meters + coeff * jax.random.uniform(size_x_key, (nr_envs,), minval=-self.box_size_random_x_range, maxval=self.box_size_random_x_range), 0.1)
        internal_state["box_half_y"] = jnp.maximum(0.5 * self.box_width_in_meters + coeff * jax.random.uniform(size_y_key, (nr_envs,), minval=-self.box_size_random_y_range, maxval=self.box_size_random_y_range), 0.1)
        return self.box_geometry(mjx_model, internal_state)


    def ground_height_at(self, internal_state, x_in_m, y_in_m):
        base = super().ground_height_at(internal_state, x_in_m, y_in_m)
        shape = (self.env.nr_envs,) + (1,) * (x_in_m.ndim - 1)
        centre_x, centre_y = internal_state["box_centre_x"].reshape(shape), internal_state["box_centre_y"].reshape(shape)
        cos_yaw, sin_yaw = jnp.cos(internal_state["box_yaw"]).reshape(shape), jnp.sin(internal_state["box_yaw"]).reshape(shape)
        local_x = cos_yaw * (x_in_m - centre_x) + sin_yaw * (y_in_m - centre_y)
        local_y = -sin_yaw * (x_in_m - centre_x) + cos_yaw * (y_in_m - centre_y)
        inside = (jnp.abs(local_x) <= internal_state["box_half_x"].reshape(shape)) & (jnp.abs(local_y) <= internal_state["box_half_y"].reshape(shape))
        return base + jnp.where(inside, 2.0 * internal_state["box_half_height"].reshape(shape), 0.0)


    def pre_step(self, data, internal_state):
        imu = data.site_xpos[:, self.env.imu_site_id]
        internal_state["robot_imu_height_over_ground"] = imu[:, 2] - self.ground_height_at(internal_state, imu[:, 0], imu[:, 1])


    def check_feet_floor_contact(self, data):
        nr_envs = self.env.nr_envs
        contact_geom = data._impl.contact__geom
        contact_dist = data._impl.contact__dist
        contact_worldid = data._impl.contact__worldid
        valid_contact = jnp.arange(contact_geom.shape[0]) < data._impl.nacon[0]
        feet = self.env.foot_geom_indices
        hit = jnp.zeros((contact_geom.shape[0], feet.shape[0]), dtype=bool)
        for ground_id in (self.env.floor_geom_id, self.box_geom_id):
            pairs = jnp.stack([jnp.full_like(feet, ground_id), feet], axis=1)
            pairs_reversed = jnp.stack([feet, jnp.full_like(feet, ground_id)], axis=1)
            hit = hit | (contact_geom[:, None, :] == pairs[None, :, :]).all(axis=2)
            hit = hit | (contact_geom[:, None, :] == pairs_reversed[None, :, :]).all(axis=2)
        penetrating = (contact_dist < 0.0) & valid_contact
        per_contact = (hit & penetrating[:, None]).astype(jnp.float32)
        worldid = jnp.clip(contact_worldid, 0, nr_envs - 1)
        in_contact = jnp.zeros((nr_envs, feet.shape[0]), dtype=jnp.float32).at[worldid].add(per_contact)
        return in_contact > 0.0
