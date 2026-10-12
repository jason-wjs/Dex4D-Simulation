"""Run the original teacher setup, then record reset and three policy steps.

Accepts train.py arguments. Run with the same cwd/runtime as train.py; outputs
go beside --model_dir. This replaces only PPO.run in this diagnostic process.
"""
import hashlib
import json
from pathlib import Path
import random
import runpy
import subprocess
import sys

from isaacgym import gymapi  # Native runtime must precede torch.
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dex4d_policy/dex4d'))
from algorithms.rl.ppo import PPO


def probe(self, num_learning_iterations, log_interval=1):
    assert self.is_testing and not self.apply_reset
    task = self.vec_env.task
    output = Path(self.model_dir).parent
    assert not (output / 'initial_state_summary.json').exists()
    stages = []
    object_properties = []
    for env in task.envs:
        actor = task.gym.find_actor_handle(env, 'object')
        bodies = task.gym.get_actor_rigid_body_properties(env, actor)
        assert len(bodies) == 1
        body = bodies[0]
        object_properties.append({'mass': body.mass,
                                  'com': [getattr(body.com, axis) for axis in 'xyz'],
                                  'inertia': [[getattr(getattr(body.inertia, row), col) for col in 'xyz'] for row in 'xyz'],
                                  'shapes': task.gym.get_actor_rigid_shape_count(env, actor)})

    def snapshot(name, obs=None, action=None):
        values = {key: getattr(task, key) for key in [
            'robot_dof_pos', 'robot_dof_vel', 'root_state_tensor', 'rigid_body_states',
            'object_keypoint_buf', 'masked_keypoint_buf', 'visual_feat_buf', 'goal_state',
            'cur_targets', 'obs_buf', 'states_buf', 'reset_buf', 'progress_buf',
            'object_indices', 'table_indices']}
        values['torch_cpu_rng'] = torch.get_rng_state()
        values['torch_cuda_rng'] = torch.cuda.get_rng_state(task.device)
        if obs is not None:
            values['policy_observation'] = obs
        if action is not None:
            values['policy_action'] = action
        arrays = {key: value.detach().cpu().numpy().copy() for key, value in values.items()}
        rng = np.random.get_state()
        arrays['numpy_rng'] = rng[1]
        np.savez_compressed(output / (name + '.npz'), **arrays)
        stages.append({'name': name, 'finite': {k: bool(np.isfinite(v).all()) for k, v in arrays.items()},
                       'numpy_rng_position': int(rng[2]), 'python_rng': repr(random.getstate())})

    task.random_time = False
    snapshot('constructed')
    original_pre_step = task.pre_physics_step

    def record_reset_input(actions):
        original_pre_step(actions)
        snapshot('reset_pre_physics')

    task.pre_physics_step = record_reset_input
    try:
        current_obs = self.vec_env.reset()
    finally:
        task.pre_physics_step = original_pre_step
    self.vec_env.get_state()  # Preserve the original call/order.
    snapshot('after_reset', current_obs)
    task.iteration = self.current_learning_iteration
    task.update_curriculum()
    snapshot('before_first_action', current_obs)
    for step in range(3):
        with torch.no_grad():
            action = self.actor_critic.act_inference(current_obs)
            snapshot('before_step_' + str(step), current_obs, action)
            next_obs, _, _, _ = self.vec_env.step(action, step)
            current_obs.copy_(next_obs)
        snapshot('after_step_' + str(step), current_obs)
    summary = {'diagnostic': 'Original setup and reset, three original inference/step calls, read-only snapshots',
               'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'checkpoint_sha256': hashlib.sha256(Path(self.model_dir).read_bytes()).hexdigest(),
               'stages': stages, 'num_envs': task.num_envs, 'seed': task.cfg['seed'],
               'object_properties': object_properties,
               'table_dims': [float(getattr(task.table_dims, axis)) for axis in 'xyz']}
    (output / 'initial_state_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    assert all(all(s['finite'].values()) for s in stages)
    print('INITIAL_STATE_PROBE_COMPLETE', flush=True)


if __name__ == '__main__':
    PPO.run = probe
    runpy.run_path(str(ROOT / 'dex4d_policy/dex4d/train.py'), run_name='__main__')
