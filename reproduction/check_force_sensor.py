"""Bounded calibration of contact and constraint-sensor force on free boxes."""
import argparse
import hashlib
import json
from pathlib import Path

from isaacgym import gymapi, gymtorch
import numpy as np
import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    config = root / 'dex4d_policy/dex4d/cfg/xarm6_leap_hand_ap2ap_stage_3.yaml'
    cfg = yaml.safe_load(config.read_text())['sim']
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt, params.substeps = 1 / 60, cfg['substeps']
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.use_gpu_pipeline = params.physx.use_gpu = True
    for key, value in cfg['physx'].items():
        setattr(params.physx, key, value)
    params.physx.contact_collection = gymapi.ContactCollection.CC_ALL_SUBSTEPS
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    assert sim is not None
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)
    asset = gym.create_box(sim, .1, .1, .1, gymapi.AssetOptions())
    sensor_props = gymapi.ForceSensorProperties()
    sensor_props.enable_forward_dynamics_forces = False
    sensor_props.enable_constraint_solver_forces = True
    sensor_props.use_world_frame = True
    assert gym.create_asset_force_sensor(asset, 0, gymapi.Transform(), sensor_props) == 0
    sensor_ids, masses = [], []
    for i, z in enumerate([.05, 20.]):
        env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 21), 2)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, z)
        actor = gym.create_actor(env, asset, pose, 'box', i, 0)
        assert gym.get_actor_force_sensor_count(env, actor) == 1
        sensor_ids.append(gym.get_actor_force_sensor(env, actor, 0).get_global_index())
        masses.append(gym.get_actor_rigid_body_properties(env, actor)[0].mass)
    gym.prepare_sim(sim)
    force = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
    sensor = gymtorch.wrap_tensor(gym.acquire_force_sensor_tensor(sim))
    state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    rows = []
    for _ in range(60):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_net_contact_force_tensor(sim)
        gym.refresh_force_sensor_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)
        rows.append({name: value.detach().cpu().numpy().copy() for name, value in
                     [('contact', force), ('sensor', sensor[sensor_ids]), ('state', state)]})
    arrays = {name: np.stack([r[name] for r in rows]) for name in rows[0]}
    np.savez_compressed(args.output / 'trace.npz', **arrays)
    expected = masses[0] * 9.81
    settled = slice(30, 60)
    checks = {
        'finite': all(np.isfinite(a).all() for a in arrays.values()),
        'contact_support_error_N': float(np.abs(arrays['contact'][settled, 0, 2] - expected).max()),
        'sensor_support_error_N': float(np.abs(arrays['sensor'][settled, 0, 2] - expected).max()),
        'airborne_contact_max_N': float(np.abs(arrays['contact'][:, 1]).max()),
        'airborne_sensor_max_N': float(np.abs(arrays['sensor'][:, 1, :3]).max()),
        'contact_sensor_difference_max_N': float(np.abs(arrays['contact'] - arrays['sensor'][:, :, :3]).max()),
    }
    report = {'checks': checks, 'mass_kg': masses, 'sensor_global_indices': sensor_ids,
              'dt': params.dt, 'substeps': params.substeps, 'physx': cfg['physx'],
              'contact_collection': int(params.physx.contact_collection),
              'steps': 60, 'settled_steps': [31, 60],
              'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
              'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Two independent free boxes, static support and airborne control; world-frame constraint-only sensor. Original Dex4D dt/substeps/solver settings, no hand, mesh or trained policy.'}
    report['passed'] = bool(checks['finite'] and checks['contact_support_error_N'] < .1
                            and checks['sensor_support_error_N'] < .1
                            and checks['airborne_contact_max_N'] < 1e-5
                            and checks['airborne_sensor_max_N'] < 1e-5)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report), flush=True)
    gym.destroy_sim(sim)
    assert report['passed'], 'Force calibration failed; inspect retained trace and report'


if __name__ == '__main__':
    main()
