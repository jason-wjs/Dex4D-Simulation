"""Audit original teacher save/load using a synthetic CPU optimizer fixture."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from isaacgym import gymapi  # noqa: F401: Gym must precede torch.
import numpy as np
import torch
import yaml
from gym.spaces import Box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dex4d_policy/dex4d'))
from algorithms.rl.ppo import PPO, ActorCriticPointNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg_root = ROOT / 'dex4d_policy/dex4d/cfg'
    cfg = yaml.safe_load((cfg_root / 'ppo/config_stage_1_2.yaml').read_text())
    task = yaml.safe_load((cfg_root / 'xarm6_leap_hand_ap2ap_stage_1_2.yaml').read_text())['env']
    assert not task['asymmetric_observations']
    keypoints = task['numKeypoints']
    width = 273 + 6 * keypoints
    env = SimpleNamespace(num_envs=4, task=SimpleNamespace(num_keypoints=keypoints),
                          observation_space=Box(-np.inf, np.inf, (width,), dtype=np.float32),
                          state_space=Box(-np.inf, np.inf, (width,), dtype=np.float32),
                          action_space=Box(-1, 1, (22,), dtype=np.float32))

    def agent():
        return PPO(env, ActorCriticPointNet, cfg['learn']['nsteps'], cfg['learn']['noptepochs'],
                   cfg['learn']['nminibatches'], model_cfg=copy.deepcopy(cfg['policy']),
                   init_noise_std=cfg['learn']['init_noise_std'],
                   learning_rate=cfg['learn']['optim_stepsize'], device='cpu', is_testing=True)

    source = agent()
    observations = torch.randn(4, width) * 0.1
    actions = torch.zeros(4, 22)

    def update(learner):
        log_prob, entropy, values, _, _ = learner.actor_critic.evaluate(observations, observations, actions)
        loss = log_prob.square().mean() + values.square().mean() - 0.01 * entropy.mean()
        learner.optimizer.zero_grad()
        loss.backward()
        learner.optimizer.step()
        assert torch.isfinite(loss)
        return float(loss.detach())

    update(source)
    update(source)
    path = args.output / 'model_2.pt'
    source.save(str(path))
    resumed = agent()
    resumed.load(str(path))
    saved = torch.load(path, map_location='cpu')
    assert all(isinstance(v, torch.Tensor) for v in saved.values())
    model_equal = all(torch.equal(v, resumed.actor_critic.state_dict()[k]) for k, v in source.actor_critic.state_dict().items())
    source_states = len(source.optimizer.state)
    resumed_states = len(resumed.optimizer.state)
    source_loss, resumed_loss = update(source), update(resumed)
    delta = max(float((v - resumed.actor_critic.state_dict()[k]).abs().max()) for k, v in source.actor_critic.state_dict().items())
    result = {'saved_fields': list(saved), 'model_equal_after_load': model_equal,
              'source_adam_state_entries': source_states, 'loaded_adam_state_entries': resumed_states,
              'loaded_iteration_from_filename': resumed.current_learning_iteration,
              'identical_batch_next_loss': [source_loss, resumed_loss],
              'next_update_model_max_abs_difference': delta,
              'scope': 'Original full stage-1/2 PointNet teacher, PPO constructor/save/load and Adam. Synthetic differentiable loss, four CPU inputs; no simulator, native rollout or PPO learning-performance claim. is_testing only suppresses constructor W&B initialization.'}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    assert model_equal and source_states > 0 and resumed_states == 0 and delta > 0
    print(json.dumps({k: v for k, v in result.items() if k != 'saved_fields'}, indent=2))


if __name__ == '__main__':
    main()
