"""Cross-process full-teacher PPO learner test with a synthetic CPU VecEnv."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

from isaacgym import gymapi  # Gym must precede torch.
from gym.spaces import Box
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dex4d_policy/dex4d'))
from algorithms.rl.ppo import PPO, ActorCriticPointNet


class SyntheticEnv:
    def __init__(self):
        self.num_envs = 4
        self.task = SimpleNamespace(num_keypoints=128, max_episode_length=400,
                                    update_curriculum=lambda: None)
        self.observation_space = Box(-np.inf, np.inf, (1041,), dtype=np.float32)
        self.state_space = self.observation_space
        self.action_space = Box(-1, 1, (22,), dtype=np.float32)

    def reset(self):
        self.obs = torch.randn(4, 1041) * .1 + float(np.random.uniform(-.01, .01)) + random.uniform(-.01, .01)
        return self.obs.clone()

    def get_state(self):
        return self.obs.clone()

    def step(self, actions, id):
        rewards = 1 - actions.square().mean(dim=-1)
        observations = self.reset()
        return observations, rewards, torch.zeros(4, dtype=torch.uint8), {'current_successes': torch.zeros(4)}


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'resume'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    seed = 42 if args.phase == 'prepare' else 777
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = yaml.safe_load((ROOT / 'dex4d_policy/dex4d/cfg/ppo/config_stage_1_2.yaml').read_text())
    c = cfg['learn']
    run = args.output / args.phase
    run.mkdir(parents=True, exist_ok=False)
    learner = PPO(SyntheticEnv(), ActorCriticPointNet, c['nsteps'], c['noptepochs'], c['nminibatches'],
                  clip_param=c['cliprange'], gamma=c['gamma'], lam=c['lam'],
                  init_noise_std=c['init_noise_std'], entropy_coef=c['ent_coef'],
                  learning_rate=c['optim_stepsize'], max_grad_norm=c['max_grad_norm'],
                  schedule=c['schedule'], desired_kl=c['desired_kl'], model_cfg=copy.deepcopy(cfg['policy']),
                  use_clipped_value_loss=False, value_loss_coef=2., device='cpu',
                  is_testing=True, print_log=False, log_dir=str(run))
    # Suppress constructor W&B only; exercise the real training loop and update.
    learner.is_testing = False
    if args.phase == 'prepare':
        learner.run(2, log_interval=1)
        state = learner.learner_state_dict()
        assert state['next_iteration'] == 2 and state['tot_timesteps'] == 64
        assert state['optimizer']['state']
        torch.save(state, args.output / 'learner_2.pt')
        learner.run(3, log_interval=1)
        torch.save(learner.learner_state_dict(), args.output / 'reference_3.pt')
        print('PREPARED_LEARNER_AND_REFERENCE', flush=True)
        return
    state = torch.load(args.output / 'learner_2.pt', map_location='cpu')
    learner.restore_learner_state(state)
    assert equal(state, learner.learner_state_dict()), 'Restore mismatch'
    learner.run(3, log_interval=1)
    final = learner.learner_state_dict()
    reference = torch.load(args.output / 'reference_3.pt', map_location='cpu')
    checks = {k: equal(final[k], reference[k]) for k in final if k != 'tot_time'}
    assert all(checks.values()), checks
    assert final['next_iteration'] == 3 and final['tot_timesteps'] == 96
    before_steps = [float(v['step']) for v in state['optimizer']['state'].values()]
    after_steps = [float(v['step']) for v in final['optimizer']['state'].values()]
    assert all(b > a for a, b in zip(before_steps, after_steps))
    assert all(torch.isfinite(t).all() for t in final['model'].values())
    assert any(not torch.equal(t, state['model'][k]) for k, t in final['model'].items())
    # Concrete rejection checks at the rollout/recipe boundary.
    learner.storage.step = 1
    try:
        learner.learner_state_dict()
    except AssertionError:
        pass
    else:
        raise AssertionError('Partial rollout incorrectly accepted')
    learner.storage.clear()
    changed = copy.deepcopy(state)
    changed['settings']['gamma'] = .5
    try:
        learner.restore_learner_state(changed)
    except AssertionError as error:
        assert 'Training recipe mismatch' in str(error)
    else:
        raise AssertionError('Different recipe incorrectly accepted')
    report = {'checks': checks, 'iterations': [2, 3], 'timesteps': [64, 96],
              'adam_steps': [[min(before_steps), max(before_steps)], [min(after_steps), max(after_steps)]],
              'adaptive_step_size': [state['step_size'], final['step_size']],
              'partial_rollout_rejected': True, 'recipe_mismatch_rejected': True,
              'scope': 'Two separate CPU processes, full original teacher and actual PPO.run/update with four synthetic environments, original 8-step/5-epoch/4-minibatch recipe. Both reference and resumed call run with fresh synthetic resets; no physical task state, native rollout, CUDA RNG or full simulator continuation claim. Wall-time accumulator checked at restore, excluded only from post-update equality.'}
    (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    torch.save(final, args.output / 'resumed_3.pt')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
