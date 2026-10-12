"""Run the real CLI with readback checks around training-state restoration."""
import json
from pathlib import Path
import runpy
import sys

from isaacgym import gymapi
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dex4d_policy/dex4d'))
from algorithms.rl.ppo import PPO
from tasks.ap2ap_checkpoint import capture


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


original_load, original_run = PPO.load_training_checkpoint, PPO.run


def physical_check(saved, restored):
    # Native unchanged get/set/get measured 3.153e-7 relative Frobenius error.
    # Only inertia gets a four-float32-epsilon bound; all other fields stay exact.
    exact = saved.keys() == restored.keys() and same(saved['gravity'], restored['gravity'])
    exact = exact and len(saved['actors']) == len(restored['actors'])
    errors = []
    for a, b in zip(saved['actors'], restored['actors']):
        exact = exact and a.keys() == b.keys()
        for name in a:
            x, y = a[name], b[name]
            exact = exact and x.keys() == y.keys()
            exact = exact and all(same(v, y[k]) for k, v in x.items() if k != 'bodies')
            exact = exact and len(x['bodies']) == len(y['bodies'])
            for u, v in zip(x['bodies'], y['bodies']):
                exact = exact and u.keys() == v.keys()
                exact = exact and all(same(value, v[k]) for k, value in u.items() if k != 'inertia')
                i, j = np.asarray(u['inertia']), np.asarray(v['inertia'])
                norm, error = np.linalg.norm(i), np.linalg.norm(i-j)
                errors.append(float(error/norm) if norm else (0. if error == 0 else float('inf')))
    bound = float(4 * np.finfo(np.float32).eps)
    return {'non_inertia_exact': bool(exact), 'max_relative_inertia_error': max(errors),
            'relative_inertia_bound': bound,
            'passed': bool(exact and all(np.isfinite(e) and e <= bound for e in errors))}


def checked_load(self, path):
    saved = torch.load(path, map_location='cpu')
    original_load(self, path)
    task = self.vec_env.task
    actual = capture(task)
    report = {'task_groups_exact': {k: same(v, actual[k]) for k, v in saved['task'].items()},
              'learner_exact': same(saved['learner'], self.learner_state_dict()),
              'loop_exact': same(saved['loop'], self._pending_training_loop),
              'native_readback': {},
              'physical_check': physical_check(saved['task']['physical'], actual['physical'])}
    task.gym.refresh_dof_state_tensor(task.sim)
    task.gym.refresh_actor_root_state_tensor(task.sim)
    for name in ['dof_state', 'root_state_tensor']:
        report['native_readback'][name] = float((getattr(task, name).cpu() - saved['task']['buffers'][name]).abs().max())
    out = Path(self.log_dir)
    torch.save({'saved': saved['task'], 'restored': actual}, out / 'restored_task_state.pt')
    (out / 'restore_check.json').write_text(json.dumps(report, indent=2) + '\n')
    assert all(v for k, v in report['task_groups_exact'].items() if k != 'physical'), report
    assert report['physical_check']['passed'], report
    assert report['learner_exact'] and report['loop_exact']
    assert all(v <= 1e-6 for v in report['native_readback'].values())
    original_act = self.actor_critic.act

    def first_act(obs, states):
        result = {'obs_exact': same(obs, saved['loop']['current_obs']),
                  'states_exact': same(states, saved['loop']['current_states']),
                  'task_buffers_exact': all(same(getattr(task, k), v) for k, v in saved['task']['buffers'].items()),
                  'logical_frame_unchanged': capture(task)['frame_count'] == saved['task']['frame_count']}
        (out / 'first_rollout_check.json').write_text(json.dumps(result, indent=2) + '\n')
        assert all(result.values()), result
        self.actor_critic.act = original_act
        return original_act(obs, states)

    self.actor_critic.act = first_act


def checked_run(self, *args, **kwargs):
    if self._pending_training_loop is None:
        # Exercise nonzero original randomization schedules in a short fixture.
        self.vec_env.task.checkpoint_frame_offset = 30000
    return original_run(self, *args, **kwargs)


if __name__ == '__main__':
    PPO.load_training_checkpoint = checked_load
    PPO.run = checked_run
    runpy.run_path(str(ROOT / 'dex4d_policy/dex4d/train.py'), run_name='__main__')
