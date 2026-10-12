"""Run one prepared teacher split on an idle GPU, with a finite wall-time cap."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True, help='Directory from prepare_split_eval.py')
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--max-seconds', type=int, default=1200)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--initial-state-probe', action='store_true', help='Record reset and three policy steps instead of full evaluation')
    modes.add_argument('--action-replay', action='store_true', help='Use the finite open-loop diagnostic specified by action_replay.json')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    requested = json.loads((output / 'requested_cases.json').read_text())
    expected = [{'object_code': code, 'scale_str': f'{round(scale*100):03d}'} for code, scale in requested['pairs']]

    def gpu_pids():
        text = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid',
                                        '--format=csv,noheader,nounits'], text=True)
        return {int(line.strip()) for line in text.splitlines() if line.strip()}

    assert not gpu_pids(), 'Selected GPU is occupied'
    checkpoint = root / 'dex4d_policy/dex4d/example_models/teacher_policy_stage_3/model_34000_best.pt'
    shutil.copy2(checkpoint, output / checkpoint.name)
    command = [sys.executable, 'train.py', '--task', 'XArm6LeapHandAP2AP', '--algo', 'ppo',
               '--seed', '0', '--rl_device', 'cuda:0', '--sim_device', 'cuda:0', '--test', '--headless',
               '--num_envs', str(requested['entries']), '--vis_env_id', '0',
               '--model_dir', str(output / checkpoint.name), '--cfg_env', str(output / 'env.yaml'),
               '--cfg_train', 'cfg/ppo/config_stage_3.yaml']
    if args.initial_state_probe:
        command[1] = str(root / 'reproduction/check_initial_state.py')
    elif args.action_replay:
        command[1] = str(root / 'reproduction/replay_actions.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4', MAX_JOBS='2', PYTHONUNBUFFERED='1')
    env['PATH'] = str(root / '.venv-gym/bin') + ':' + env['PATH']
    env['LD_LIBRARY_PATH'] = str(root / '.python/cpython-3.8.20-linux-x86_64-gnu/lib')
    manifest = {'command': command, 'cwd': str(root / 'dex4d_policy/dex4d'), 'gpu': args.gpu,
                'max_seconds': args.max_seconds, 'split': requested['split'], 'status': 'starting',
                'initial_state_probe': args.initial_state_probe,
                'action_replay': args.action_replay,
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                'source_patch': subprocess.check_output(['git', 'diff'], cwd=root, text=True),
                'gpu_identity': subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=name,uuid', '--format=csv,noheader'], text=True).strip(),
                'hashes': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [checkpoint, output / 'env.yaml', Path(__file__).resolve(), root / 'dex4d_policy/dex4d/cfg/ppo/config_stage_3.yaml']}}
    manifest_path = output / 'run_manifest.json'
    with manifest_path.open('x') as f:
        json.dump(manifest, f, indent=2)
    start = time.monotonic()
    with (output / 'run.log').open('w') as log:
        process = subprocess.Popen(command, cwd=manifest['cwd'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        manifest['pid'] = process.pid
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'Running {requested["split"]}, {requested["entries"]} entries, PID {process.pid}', flush=True)
        try:
            while process.poll() is None:
                if time.monotonic() - start >= args.max_seconds:
                    manifest['status'] = 'stopped_wall_limit'
                    break
                foreign = gpu_pids() - {process.pid}
                if foreign:
                    manifest.update(status='stopped_gpu_occupied', foreign_pids=sorted(foreign))
                    break
                time.sleep(10)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            manifest.update(exit_code=process.returncode, wall_seconds=time.monotonic()-start)
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    if process.returncode != 0:
        raise SystemExit(f'Incomplete evaluation: {manifest["status"]}, exit {process.returncode}')
    actual = json.loads((output / 'evaluation_cases.json').read_text())
    assert actual == expected, 'Actual case coverage mismatch'
    if args.action_replay:
        summary = json.loads((output / 'action_replay_summary.json').read_text())
        audit = json.loads((output / 'trajectory_audit.json').read_text())
        assert summary['num_envs'] == len(expected) and summary['steps'] == audit['samples']
        assert all(audit['finite'].values())
        manifest.update(status='completed', verified_entries=len(actual), diagnostic_summary=summary)
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'status': 'completed', 'diagnostic': 'action_replay', 'wall_seconds': manifest['wall_seconds']}), flush=True)
        return
    if args.initial_state_probe:
        summary = json.loads((output / 'initial_state_summary.json').read_text())
        assert summary['num_envs'] == len(expected)
        assert [s['name'] for s in summary['stages']] == ['constructed', 'reset_pre_physics', 'after_reset', 'before_first_action'] + [
            name for i in range(3) for name in [f'before_step_{i}', f'after_step_{i}']]
        assert all(all(s['finite'].values()) for s in summary['stages'])
        manifest.update(status='completed', verified_entries=len(actual), diagnostic_summary=summary)
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'status': 'completed', 'diagnostic': 'initial_state_probe', 'wall_seconds': manifest['wall_seconds']}), flush=True)
        return
    rollouts = json.loads((output / 'evaluation_rollouts.json').read_text())
    assert len(rollouts) == 2 and [r['policy_step'] for r in rollouts] == [399, 799]
    for row in rollouts:
        for name, values in row['per_environment'].items():
            assert len(values) == len(expected)
            assert abs(sum(values)/len(values)-row[name]) < 1e-5
    manifest.update(status='completed', verified_entries=len(actual), rollouts=rollouts)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': manifest['status'], 'entries': len(actual), 'wall_seconds': manifest['wall_seconds'],
                      'rollouts': [{k: v for k, v in r.items() if k != 'per_environment'} for r in rollouts]}), flush=True)


if __name__ == '__main__':
    main()
