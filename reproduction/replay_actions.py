"""Finite open-loop dynamics diagnostic using recorded states and actions."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

from isaacgym import gymtorch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dex4d_policy/dex4d'))
from algorithms.rl.ppo import PPO
from utils.trajectory_audit import TrajectoryAudit


def replay(self, num_learning_iterations, log_interval=1):
    task = self.vec_env.task
    assert self.is_testing and not task.use_osc_control
    output = Path(self.model_dir).parent
    spec_path = output / 'action_replay.json'
    spec = json.loads(spec_path.read_text())
    initial_path, action_path = Path(spec['initial_snapshot']), Path(spec['action_archive'])
    case_path, meta_path = initial_path.parent / 'evaluation_cases.json', action_path.parent / 'trajectory_audit.json'
    assert json.loads(case_path.read_text()) == task.object_code_and_scale_str_for_envs
    meta = json.loads(meta_path.read_text())
    ids = meta['env_ids']
    assert [task.object_code_and_scale_str_for_envs[i] for i in ids] == meta['cases']
    initial = np.load(initial_path, allow_pickle=False)
    actions = np.load(action_path, allow_pickle=False)['policy_action']
    steps = spec['steps']
    assert 0 < steps < task.max_episode_length - 2 and steps <= len(actions)
    assert np.isfinite(actions).all()
    task.random_time = False
    task.iteration = self.current_learning_iteration
    task.update_curriculum()

    # Install a common measured post-reset state into a fresh scene. Simulator
    # contact caches are not serialized; neither condition performs warm-up.
    for name in ['robot_dof_pos', 'robot_dof_vel', 'root_state_tensor', 'goal_state', 'cur_targets', 'progress_buf']:
        target = getattr(task, name)
        value = torch.as_tensor(initial[name], device=task.device, dtype=target.dtype)
        assert target.shape == value.shape and torch.isfinite(value).all()
        target.copy_(value)
    task.prev_targets.copy_(task.cur_targets)
    task.last_robot_dof_vel.copy_(task.robot_dof_vel)
    task.reset_buf.zero_()
    task.reset_goal_buf.zero_()
    assert task.gym.set_dof_state_tensor(task.sim, gymtorch.unwrap_tensor(task.dof_state))
    assert task.gym.set_actor_root_state_tensor(task.sim, gymtorch.unwrap_tensor(task.root_state_tensor))
    assert task.gym.set_dof_position_target_tensor(task.sim, gymtorch.unwrap_tensor(task.cur_targets))
    np.savez_compressed(output / 'replay_initial.npz', **{
        name: getattr(task, name).detach().cpu().numpy().copy() for name in
        ['robot_dof_pos', 'robot_dof_vel', 'root_state_tensor', 'goal_state', 'cur_targets', 'progress_buf']})

    audit = TrajectoryAudit(task, output, ids)
    source_actions = torch.as_tensor(actions[:steps], device=task.device)
    all_actions = torch.zeros((task.num_envs, task.num_actions), device=task.device)
    for step in range(steps):
        assert not task.reset_buf.any(), 'Unexpected environment reset in open-loop diagnostic'
        task.reset_goal_buf.zero_()  # Fixed goals; no goal resampling in this diagnostic.
        all_actions[ids] = source_actions[step]
        with torch.no_grad():
            self.vec_env.step(all_actions, step)
        audit.capture(step + 1, all_actions)
        if (step + 1) % 60 == 0:
            print(f'REPLAY_STEP {step + 1}/{steps}', flush=True)
    audit.save()
    native = task.gym.get_sim_params(task.sim)
    summary = {'steps': steps, 'env_ids': ids, 'num_envs': task.num_envs,
               'position_iterations': native.physx.num_position_iterations,
               'velocity_iterations': native.physx.num_velocity_iterations,
               'dt': native.dt, 'substeps': native.substeps,
               'inputs_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [initial_path, action_path, case_path, meta_path]},
               'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'scope': 'Fresh simulator, common recorded q/qd/root/goal/targets/progress; original action processing. Recorded five-env actions; all other environments receive zero actions. Goals fixed, no episode resets. No policy inference, contact-cache restoration or task-success claim.'}
    (output / 'action_replay_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print('ACTION_REPLAY_COMPLETE', flush=True)


if __name__ == '__main__':
    PPO.run = replay
    runpy.run_path(str(ROOT / 'dex4d_policy/dex4d/train.py'), run_name='__main__')
