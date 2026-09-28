class BelowHeightTermination:
    def __init__(self, env):
        self.env = env

        self.height_percentage_threshold = self.env.env_config["termination"]["height_percentage_threshold"]
        self.curriculum_coeff = self.env.env_config["termination"]["curriculum_coeff"]


    def should_terminate(self, internal_state):
        curriculum_coeff = internal_state["env_curriculum_coeff"] if self.curriculum_coeff is None else self.curriculum_coeff
        below_height = internal_state["robot_imu_height_over_ground"] < ((1 - curriculum_coeff) * self.height_percentage_threshold * internal_state["robot_nominal_imu_height_over_ground"])

        return below_height
