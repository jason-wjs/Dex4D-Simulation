"""Offline acceptance of the bounded 2→3 native teacher resume fixture."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fresh', type=Path)
    parser.add_argument('resumed', type=Path)
    args = parser.parse_args()
    paths = [p / 'training_state_latest.pt' for p in [args.fresh, args.resumed]]
    before, after = [torch.load(p, map_location='cpu') for p in paths]
    restore = json.loads((args.resumed / 'restore_check.json').read_text())
    rollout = json.loads((args.resumed / 'first_rollout_check.json').read_text())
    assert all(v for k, v in restore['task_groups_exact'].items() if k != 'physical')
    assert restore['physical_check']['passed']
    assert restore['learner_exact'] and restore['loop_exact']
    assert all(v <= 1e-6 for v in restore['native_readback'].values())
    assert all(rollout.values())
    a, b = before['learner'], after['learner']
    num_envs = before['task']['cfg']['env']['numEnvs']
    transitions = [2 * num_envs * 8, 3 * num_envs * 8]
    assert a['settings'] == b['settings']
    assert [a['next_iteration'], b['next_iteration']] == [2, 3]
    assert [a['tot_timesteps'], b['tot_timesteps']] == transitions
    assert a['rng']['cuda'] is not None and b['rng']['cuda'] is not None
    assert a['optimizer']['state'].keys() == b['optimizer']['state'].keys()
    steps = [[float(v['step']) for v in state['optimizer']['state'].values()] for state in [a, b]]
    assert set(steps[0]) == {40.} and set(steps[1]) == {60.}
    delta = max(float((v - a['model'][k]).abs().max()) for k, v in b['model'].items())
    assert delta > 0 and all(torch.isfinite(v).all() for v in b['model'].values())
    for state in [a, b]:
        assert all(torch.isfinite(v).all() for row in state['optimizer']['state'].values()
                   for v in row.values() if isinstance(v, torch.Tensor))
    frames = [p['task']['frame_count'] for p in [before, after]]
    assert frames[1] - frames[0] == 8 * before['task']['cfg']['env']['controlFrequencyInv']
    counters = [p['task']['scalars']['common_step_counter'] for p in [before, after]]
    assert counters[1] - counters[0] == 8
    assert before['task']['cases'] == after['task']['cases']
    noise = {k: float(v['var']) for k, v in before['task']['noise'].items()}
    assert set(noise) == {'observations', 'actions'} and all(v > 0 for v in noise.values())
    report = {'passed': True, 'iterations': [2, 3], 'transitions': transitions, 'num_envs': num_envs,
              'adam_steps': [[min(v), max(v)] for v in steps], 'max_parameter_delta': delta,
              'model_and_adam_finite': True, 'cuda_rng_present_and_restored_exactly': True,
              'logical_frames': frames, 'control_step_counters': counters,
              'saved_noise_standard_deviations': noise,
              'adaptive_step_size': [a['step_size'], b['step_size']],
              'checkpoint_sha256': [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
              'restore': restore, 'first_rollout': rollout,
              'scope': 'Native teacher network/PPO with environment scale recorded above; diagnostic DR frame offset 30000. Separate-process continuation, not convergence, accepted physical behavior or bitwise future physics equality.'}
    (args.resumed / 'continuation_result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
