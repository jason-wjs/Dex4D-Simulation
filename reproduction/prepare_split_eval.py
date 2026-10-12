"""Write an explicit test-pair config from a shipped split, without changing task math."""
import argparse
import hashlib
import json
from pathlib import Path

import yaml

from audit_dataset import split_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['test_set_seen_cat', 'test_set_unseen_cat'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-env-ids', type=int, nargs='+', help='Record these original split-entry indices at every control step')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source, pairs = split_pairs(root, args.split)
    base = root / 'dex4d_policy/dex4d/cfg/xarm6_leap_hand_ap2ap_stage_3.yaml'
    config = yaml.safe_load(base.read_text())
    config['env']['evaluation_object_pairs'] = [[code, scale] for code, scale in pairs]
    config['env']['numEnvs'] = len(pairs)
    if args.audit_env_ids is not None:
        assert len(set(args.audit_env_ids)) == len(args.audit_env_ids)
        assert all(0 <= i < len(pairs) for i in args.audit_env_ids)
        config['env']['trajectory_audit_env_ids'] = args.audit_env_ids
    # The explicit test pairs replace the training-list source for this config.
    del config['env']['object_code_dict_file']
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'env.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {'split': args.split, 'entries': len(pairs),
                'unique_meshes': len({code for code, _ in pairs}),
                'unique_pairs': len(set(pairs)), 'pairs': pairs,
                'split_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'base_config_sha256': hashlib.sha256(base.read_bytes()).hexdigest(),
                'protocol': 'One environment per original split entry, including every scale and repeated entry; original stage3 task and two-rollout test metrics.'}
    (args.output / 'requested_cases.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'pairs'}, indent=2))


if __name__ == '__main__':
    main()
