"""Native unchanged-property setter/getter control; exits before policy rollout."""
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
from tasks.ap2ap_checkpoint import physical_state


def check(self, *args, **kwargs):
    task = self.vec_env.task
    before = physical_state(task)
    for env in task.envs:
        for name in ['robot', 'object']:
            actor = task.gym.find_actor_handle(env, name)
            props = task.gym.get_actor_rigid_body_properties(env, actor)
            task.gym.set_actor_rigid_body_properties(env, actor, props, False)
    after = physical_state(task)
    errors = []
    assert before['gravity'] == after['gravity']
    for a, b in zip(before['actors'], after['actors']):
        for name in a:
            assert a[name]['friction'] == b[name]['friction']
            if name == 'robot':
                assert np.array_equal(a[name]['dofs'], b[name]['dofs'])
            for x, y in zip(a[name]['bodies'], b[name]['bodies']):
                for key in ['mass', 'com', 'flags']:
                    assert x[key] == y[key]
                u, v = np.asarray(x['inertia']), np.asarray(y['inertia'])
                norm = np.linalg.norm(u)
                error = np.linalg.norm(u - v)
                errors.append({'max_abs': float(np.abs(u-v).max()),
                               'relative_norm': float(error/norm) if norm else float(error)})
    report = {'bodies': len(errors), 'non_inertia_fields_exact': True,
              'max_abs_inertia_error': max(v['max_abs'] for v in errors),
              'max_relative_inertia_error': max(v['relative_norm'] for v in errors),
              'float32_epsilon': float(np.finfo(np.float32).eps),
              'scope': 'Native get/set/get with unchanged RigidBodyProperties and recomputeInertia=False; no policy rollout or task update.'}
    out = Path(self.log_dir)
    torch.save({'before': before, 'after': after}, out / 'inertia_roundtrip.pt')
    (out / 'inertia_roundtrip.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    PPO.run = check
    runpy.run_path(str(ROOT / 'dex4d_policy/dex4d/train.py'), run_name='__main__')
