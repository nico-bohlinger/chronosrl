import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def measure_per_world_buffer_sizes(mj_model, seed, nr_samples=256, contact_safety_factor=1.6, constraint_safety_factor=1.6, min_contacts=8, min_constraints=16):
    rng = np.random.default_rng(int(seed))
    data = mujoco.MjData(mj_model)
    home_qpos = np.array(mj_model.keyframe("home").qpos, dtype=np.float64)

    max_nr_contacts = 0
    max_nr_constraints = 0
    for _ in range(nr_samples):
        qpos = home_qpos.copy()
        for joint_id in range(mj_model.njnt):
            joint_type = mj_model.jnt_type[joint_id]
            qpos_adr = mj_model.jnt_qposadr[joint_id]
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                qpos[qpos_adr:qpos_adr + 2] = rng.uniform(-1.0, 1.0, 2)
                qpos[qpos_adr + 2] = home_qpos[2] + rng.uniform(-0.2, 0.0)
                quat_xyzw = Rotation.from_euler("xyz", rng.uniform(-0.5, 0.5, 3)).as_quat()
                qpos[qpos_adr + 3:qpos_adr + 7] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            elif joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                lower_limit, upper_limit = mj_model.jnt_range[joint_id]
                if mj_model.jnt_limited[joint_id] and upper_limit > lower_limit:
                    qpos[qpos_adr] = rng.uniform(lower_limit, upper_limit)
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, data)
        max_nr_contacts = max(max_nr_contacts, int(data.ncon))
        max_nr_constraints = max(max_nr_constraints, int(data.nefc))

    naconmax_per_env = max(min_contacts, int(np.ceil(max_nr_contacts * contact_safety_factor)))
    njmax = max(min_constraints, int(np.ceil(max_nr_constraints * constraint_safety_factor)))
    return naconmax_per_env, njmax
