"""Bounded native teacher save/exit/restore fixture; no formal training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--source-run', type=Path, help='Reuse a completed two-iteration fixture when debugging restore')
    parser.add_argument('--full-scale', action='store_true',
                        help='Keep original 4096 environments and full training object list')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    task_root = root / 'dex4d_policy/dex4d'
    cfg = yaml.safe_load((task_root / 'cfg/xarm6_leap_hand_ap2ap_stage_1_2.yaml').read_text())
    if not args.full_scale:
        cfg['env']['numEnvs'], cfg['env']['object_num'] = 16, 4
    wall_limit = 900 if args.full_scale else 420
    train = yaml.safe_load((task_root / 'cfg/ppo/config_stage_1_2.yaml').read_text())
    train['learn']['save_interval'] = 1
    (output / 'env.yaml').write_text(yaml.safe_dump(cfg))
    (output / 'train.yaml').write_text(yaml.safe_dump(train))
    sources = ['reproduction/check_native_teacher_resume.py', 'reproduction/run_teacher_resume_probe.py',
               'dex4d_policy/dex4d/tasks/ap2ap_checkpoint.py', 'dex4d_policy/dex4d/tasks/hand_base/base_task.py',
               'dex4d_policy/dex4d/algorithms/rl/ppo/ppo.py', 'dex4d_policy/dex4d/utils/process_sarl.py',
               'dex4d_policy/dex4d/utils/config.py']
    sources.append('dex4d_policy/dex4d/utils/util.py')
    manifest = {'runs': [], 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                'source_diff': subprocess.check_output(['git', 'diff'], cwd=root, text=True),
                'source_hashes': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
                'scope': '16 environments, first four filtered training objects, full teacher/PPO and original randomization. Diagnostic logical-frame offset 30000 exercises nonzero DR. Native iterations 2 then separate-process 3. Not convergence, full-scale throughput or bitwise future physics validation.'}
    manifest.update(num_envs=cfg['env']['numEnvs'], object_limit=cfg['env']['object_num'],
                    max_seconds_per_process=wall_limit)
    if args.full_scale:
        manifest['scope'] = ('Original 4096 environments and complete training list; full teacher/PPO and original randomization. '
                             'Diagnostic logical-frame offset 30000 exercises nonzero DR. Native iterations 2 then separate-process 3. '
                             'Full-scale recovery check, not convergence, accepted physical behavior or bitwise future physics validation.')
    def pids():
        return {int(v) for v in subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).split()}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               MAX_JOBS='2', WANDB_MODE='disabled', PYTHONUNBUFFERED='1', VK_LOADER_DRIVERS_DISABLE='*')
    env['PATH'] = str(root / '.venv-gym/bin') + ':' + env['PATH']
    env['LD_LIBRARY_PATH'] = str((root / '.venv-gym/bin/python').resolve().parent.parent / 'lib')
    phases = [('fresh', 2), ('resumed', 3)]
    checkpoint = output / 'fresh/training_state_latest.pt'
    if args.source_run is not None:
        source = args.source_run.resolve()
        checkpoint = source / 'fresh/training_state_latest.pt'
        assert checkpoint.is_file()
        assert (output / 'env.yaml').read_bytes() == (source / 'env.yaml').read_bytes()
        assert (output / 'train.yaml').read_bytes() == (source / 'train.yaml').read_bytes()
        manifest['source_checkpoint'] = {'path': str(checkpoint), 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        phases = [('resumed', 3)]
    for phase, limit in phases:
        assert not pids(), 'Selected GPU occupied'
        command = [str(root / '.venv-gym/bin/python'), str(root / 'reproduction/check_native_teacher_resume.py'),
                   '--task', 'XArm6LeapHandAP2AP', '--algo', 'ppo', '--seed', '42', '--headless',
                   '--rl_device', 'cuda:0', '--sim_device', 'cuda:0', '--cfg_env', str(output / 'env.yaml'),
                   '--cfg_train', str(output / 'train.yaml'), '--logdir', str(output / phase),
                   '--max_iterations', str(limit), '--save_training_state']
        if phase == 'resumed':
            command += ['--training_state', str(checkpoint)]
        record = {'phase': phase, 'command': command, 'status': 'running', 'max_seconds': wall_limit,
                  'gpu': subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True).strip()}
        manifest['runs'].append(record)
        started = time.monotonic()
        with (output / (phase + '.log')).open('w') as log:
            process = subprocess.Popen(command, cwd=task_root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = process.pid
            (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            print(phase, 'PID', process.pid, flush=True)
            try:
                while process.poll() is None:
                    if time.monotonic() - started > wall_limit or pids() - {process.pid}:
                        record['status'] = 'stopped_limit_or_gpu_occupied'
                        break
                    time.sleep(5)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            record.update(exit_code=process.returncode, wall_seconds=time.monotonic() - started)
            if record['status'] == 'running':
                record['status'] = 'completed' if process.returncode == 0 else 'failed'
            (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        assert record['status'] == 'completed' and (output / phase / 'training_state_latest.pt').is_file(), record
    for name in ['restore_check.json', 'first_rollout_check.json']:
        assert (output / 'resumed' / name).is_file()
    print('NATIVE_TEACHER_RESUME_PROCESSES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
