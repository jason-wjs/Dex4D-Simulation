"""Calibrate post-prepare native shape filters before the hand/table diagnostic."""
import argparse
import json
from pathlib import Path

from isaacgym import gymapi, gymtorch
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt, params.substeps = 1/60, 2
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.use_gpu_pipeline = params.physx.use_gpu = True
    params.physx.num_position_iterations = 8
    params.physx.num_velocity_iterations = 0
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    assert sim is not None
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)
    fixed = gymapi.AssetOptions()
    fixed.fix_base_link = True
    table = gym.create_box(sim, 1.2, 1.2, .6, fixed)
    box = gym.create_box(sim, .1, .1, .1, gymapi.AssetOptions())
    scenes, indices = [], []
    for i in range(2):
        env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 2)
        actors = []
        for asset, name, x, z in [(table, 'table', 0, .3), (box, 'lower', -.3, .75),
                                   (box, 'upper', -.3, .95), (box, 'object', .3, .9)]:
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(x, 0, z)
            actor = gym.create_actor(env, asset, pose, name, i, 0)
            actors.append(actor)
            indices.append(gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM))
        scenes.append((env, actors))
    gym.prepare_sim(sim)
    # Same timing as the replay: after prepare_sim, before first simulate.
    env, actors = scenes[1]
    for actor, mask in zip(actors, [3, 1, 2, 0]):
        props = gym.get_actor_rigid_shape_properties(env, actor)
        assert len(props) == 1 and props[0].filter == 0
        props[0].filter = mask
        assert gym.set_actor_rigid_shape_properties(env, actor, props)
        assert gym.get_actor_rigid_shape_properties(env, actor)[0].filter == mask
    state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    trace = []
    for _ in range(180):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        trace.append(state[indices].detach().cpu().numpy().copy())
    data = np.stack(trace)
    z = data[-30:, :, 2].reshape(30, 2, 4)
    expected = np.array([[.3, .65, .75, .65], [.3, .05, .15, .65]])
    errors = np.max(np.abs(z - expected), axis=0)
    report = {'passed': bool(np.isfinite(data).all() and (errors < .005).all()),
              'actor_order': ['table', 'lower', 'upper', 'object'],
              'baseline_final_z': z[-1, 0].tolist(), 'filtered_final_z': z[-1, 1].tolist(),
              'expected_z': expected.tolist(), 'max_settled_z_error_m': errors.tolist(),
              'filtered_masks': [3, 1, 2, 0], 'native_filter_readback_exact': True,
              'steps': 180, 'scope': 'GPU PhysX post-prepare filter setter; table contacts removed for two distinct-mask bodies, their mutual stacking retained, mask-zero object table support retained. Primitive calibration, not hand task evidence.'}
    np.savez_compressed(args.output / 'trace.npz', root_state=data)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    gym.destroy_sim(sim)
    assert report['passed'], report


if __name__ == '__main__':
    main()
