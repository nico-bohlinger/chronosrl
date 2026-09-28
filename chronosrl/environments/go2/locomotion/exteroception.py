import jax.numpy as jnp


class NoneExteroceptiveObservation:
    def __init__(self, env):
        self.env = env

        self.nr_exteroceptive_observations = 0


    def get_exteroceptive_observation(self, data, mjx_model, internal_state):
        return jnp.zeros((self.env.nr_envs, 0))


class HeightOverGroundExteroceptiveObservation:
    def __init__(self, env):
        self.env = env

        self.nr_exteroceptive_observations = 1


    def get_exteroceptive_observation(self, data, mjx_model, internal_state):
        return internal_state["robot_imu_height_over_ground"]
