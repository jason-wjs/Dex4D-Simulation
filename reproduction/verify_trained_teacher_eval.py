"""Verify a completed split evaluation against its actual resumed training run."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml


def verify(training, evaluation):
    state_path = training / 'training_state_latest.pt'
    state = torch.load(state_path, map_location='cpu')
    assert state['format'] == 'dex4d_teacher_training_v1'
    manifest = json.loads((evaluation / 'run_manifest.json').read_text())
    assert manifest['status'] == 'completed' and manifest['exit_code'] == 0
    checkpoint = Path(manifest['checkpoint_source'])
    assert checkpoint.parent.resolve() == training.resolve()
    weights = torch.load(evaluation / checkpoint.name, map_location='cpu')
    assert weights.keys() == state['learner']['model'].keys()
    assert all(torch.equal(v, state['learner']['model'][k]) and torch.isfinite(v).all()
               for k, v in weights.items()), 'Evaluated weights differ from the saved learner'
    for path in [checkpoint, evaluation / checkpoint.name, evaluation / 'train.yaml', evaluation / 'env.yaml']:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['hashes'][str(path.resolve())]
    assert (training / 'train.yaml').read_bytes() == (evaluation / 'train.yaml').read_bytes()
    source = yaml.safe_load((training / 'env.yaml').read_text())
    prepared = yaml.safe_load((evaluation / 'env.yaml').read_text())
    expected = json.loads((evaluation / 'requested_cases.json').read_text())
    assert expected['base_config_sha256'] == hashlib.sha256((training / 'env.yaml').read_bytes()).hexdigest()
    assert prepared['env']['evaluation_object_pairs'] == expected['pairs']
    protocol = json.loads((evaluation / 'evaluation_protocol.json').read_text())
    assert protocol['checkpoint_iteration'] == state['learner']['next_iteration']
    assert protocol['num_envs'] == expected['entries'] == prepared['env']['numEnvs']
    assert protocol['episode_length'] == source['env']['episodeLength']
    assert protocol['control_frequency_inv'] == source['env']['controlFrequencyInv']
    assert not protocol['randomize'] and not protocol['random_time']
    for name in ['numEnvs', 'object_code_dict_file', 'evaluation_object_pairs', 'trajectory_audit_env_ids']:
        source['env'].pop(name, None)
        prepared['env'].pop(name, None)
    assert source == prepared, 'Evaluation changed the checkpoint environment recipe'
    actual = json.loads((evaluation / 'evaluation_cases.json').read_text())
    assert actual == [{'object_code': code, 'scale_str': f'{round(scale*100):03d}'} for code, scale in expected['pairs']]
    rollouts = json.loads((evaluation / 'evaluation_rollouts.json').read_text())
    assert rollouts == manifest['rollouts'] and len(rollouts) == 2
    report = {'passed': True, 'evaluated_weights_equal_training_state': True,
              'config_equal_except_test_cases_count_and_optional_audit': True,
              'training_transitions': state['learner']['tot_timesteps'],
              'training_state_sha256': hashlib.sha256(state_path.read_bytes()).hexdigest(),
              'model_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'protocol': protocol, 'verified_entries': len(actual),
              'scope': 'Actual saved learner weights evaluated with their source recipe and native test behavior; no convergence or physical-quality claim.'}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.training_run.resolve(), args.evaluation.resolve())
    (args.evaluation / 'trained_checkpoint_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
