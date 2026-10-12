"""Reproduce Preview 4 GPU aggregate corruption with the original robot only.

Use --max-shapes 179 for the failing case and 193 for the corrected capacity.
Run in the Gym environment with an idle CUDA_VISIBLE_DEVICES and a process timeout.
This scene fixture performs no policy evaluation or training.
"""

import argparse
import faulthandler
from pathlib import Path

from isaacgym import gymapi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shapes", type=int, required=True)
    args = parser.parse_args()
    faulthandler.enable()
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1 / 60
    params.substeps = 2
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.use_gpu_pipeline = True
    params.physx.use_gpu = True
    params.physx.solver_type = 1
    params.physx.num_threads = 4
    params.physx.num_position_iterations = 8
    params.physx.num_velocity_iterations = 0
    params.physx.contact_offset = 0.002
    params.physx.rest_offset = 0.0
    params.physx.max_depenetration_velocity = 1000.0
    params.physx.default_buffer_size_multiplier = 5.0
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    options = gymapi.AssetOptions()
    options.flip_visual_attachments = False
    options.fix_base_link = True
    options.collapse_fixed_joints = False
    options.disable_gravity = True
    options.thickness = 0.001
    options.angular_damping = 0.01
    options.linear_damping = 0.01
    options.use_physx_armature = True
    options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    root = Path(__file__).resolve().parents[1] / "dex4d_policy/assets"
    robot = gym.load_asset(sim, str(root), "urdf/xarm6_leap_description/xarm6_leap_left_2023.urdf", options)
    print("ROBOT_LOADED", gym.get_asset_rigid_body_count(robot), gym.get_asset_rigid_shape_count(robot), flush=True)
    for name in ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"]:
        index = gym.find_asset_rigid_body_index(robot, name)
        assert index >= 0
        gym.create_asset_force_sensor(robot, index, gymapi.Transform())
    for index in range(4):
        print("CREATE_ENV", index, flush=True)
        env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 2)
        gym.begin_aggregate(env, 82, args.max_shapes, True)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.5, 0, 0.6)
        gym.create_actor(env, robot, pose, "robot", index, -1, 0)
        gym.end_aggregate(env)
    gym.prepare_sim(sim)
    print("MINIMAL_ROBOT_SCENE_OK", flush=True)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
