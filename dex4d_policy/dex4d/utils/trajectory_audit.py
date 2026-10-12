"""Read-only control-step trajectories; GPU net force is not contact depth."""
import json
from pathlib import Path

from isaacgym import gymapi, gymtorch
import numpy as np
import torch


class TrajectoryAudit:
    def __init__(self, task, output_dir, env_ids):
        assert task.cfg['test']
        assert env_ids and len(set(env_ids)) == len(env_ids)
        assert all(0 <= i < task.num_envs for i in env_ids)
        self.task = task
        self.output = Path(output_dir)
        self.env_ids = env_ids
        self.indices = torch.tensor(env_ids, device=task.device, dtype=torch.long)
        self.forces = gymtorch.wrap_tensor(task.gym.acquire_net_contact_force_tensor(task.sim)).view(task.num_envs, -1, 3)
        self.frames = []
        bodies = []
        dof_properties = []
        for index in env_ids:
            handle = task.envs[index]
            actors = {}
            for label in ['robot', 'object', 'goal_object', 'table']:
                actor = task.gym.find_actor_handle(handle, label)
                names = task.gym.get_actor_rigid_body_names(handle, actor)
                actors[label] = [{'name': name, 'index': task.gym.get_actor_rigid_body_index(handle, actor, i, gymapi.DOMAIN_ENV)}
                                 for i, name in enumerate(names)]
            bodies.append(actors)
            properties = task.gym.get_actor_dof_properties(handle, task.robots[index])
            dof_properties.append({name: properties[name].tolist() for name in properties.dtype.names})
        self.metadata = {
            'env_ids': env_ids, 'cases': [task.object_code_and_scale_str_for_envs[i] for i in env_ids],
            'bodies': bodies, 'joint_names': task.gym.get_actor_dof_names(task.envs[0], task.robots[0]),
            'native_dof_properties': dof_properties,
            'joint_lower': task.robot_dof_lower_limits.detach().cpu().tolist(),
            'joint_upper': task.robot_dof_upper_limits.detach().cpu().tolist(),
            'table_dims': [float(getattr(task.table_dims, axis)) for axis in 'xyz'],
            'simulation_dt': task.sim_params.dt, 'substeps': task.sim_params.substeps,
            'control_frequency_inv': task.cfg['env'].get('controlFrequencyInv', 1),
            'contact_collection': int(task.sim_params.physx.contact_collection),
            'sampling': 'After each completed policy control step and before the next pre-step reset. No physics-substep samples.',
            'quaternion_order': 'xyzw',
            'force_scope': 'Native per-body net contact force; no contact pairs, signed separation or penetration depth.',
        }

    def capture(self, policy_step, policy_action):
        task, ids = self.task, self.indices
        task.gym.refresh_net_contact_force_tensor(task.sim)
        values = {
            'joint_pos': task.robot_dof_pos[ids], 'joint_vel': task.robot_dof_vel[ids],
            'targets': task.cur_targets[ids], 'policy_action': policy_action[ids],
            'applied_action': task.actions[ids], 'rigid_body_state': task.rigid_body_states[ids],
            'object_state': task.root_state_tensor[task.object_indices[ids]],
            'goal_state': task.goal_state[ids], 'net_contact_force': self.forces[ids],
            'reset': task.reset_buf[ids], 'goal_reset': task.reset_goal_buf[ids],
            'progress': task.progress_buf[ids], 'goal_keypoint_distance': task.goal_obj_dist[ids],
        }
        row = {name: value.detach().cpu().numpy().copy() for name, value in values.items()}
        row['policy_step'] = np.asarray(policy_step)
        self.frames.append(row)

    def save(self):
        arrays = {name: np.stack([row[name] for row in self.frames]) for name in self.frames[0]}
        np.savez_compressed(self.output / 'trajectory_audit.npz', **arrays)
        finite = {name: bool(np.isfinite(value).all()) for name, value in arrays.items()}
        q = arrays['joint_pos']
        violation = np.maximum(np.asarray(self.metadata['joint_lower']) - q,
                               q - np.asarray(self.metadata['joint_upper'])).clip(min=0)
        report = {**self.metadata, 'samples': len(self.frames), 'finite': finite,
                  'max_joint_limit_violation_rad': float(violation.max()),
                  'max_body_net_contact_force_N': float(np.linalg.norm(arrays['net_contact_force'], axis=-1).max())}
        (self.output / 'trajectory_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        assert all(finite.values()), 'Non-finite trajectory data; inspect the retained raw arrays'
