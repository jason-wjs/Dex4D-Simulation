"""Record free-object state and net force after each simulate(dt) call."""
import json

from isaacgym import gymapi, gymtorch
import numpy as np
import torch


class SimulationStepAudit:
    def __init__(self, task, output, env_ids):
        self.task, self.output = task, output
        self.ids = torch.tensor(env_ids, dtype=torch.long, device=task.device)
        self.force = gymtorch.wrap_tensor(task.gym.acquire_net_contact_force_tensor(task.sim)).view(task.num_envs, -1, 3)
        props, body_indices, global_body_indices, sensor_indices = [], [], [], []
        for env_id in env_ids:
            env = task.envs[env_id]
            actor = task.gym.find_actor_handle(env, 'object')
            bodies = task.gym.get_actor_rigid_body_properties(env, actor)
            assert len(bodies) == 1 and task.gym.get_actor_dof_count(env, actor) == 0
            body = bodies[0]
            props.append({'mass': body.mass, 'com': [getattr(body.com, a) for a in 'xyz'], 'flags': int(body.flags)})
            body_indices.append(task.gym.get_actor_rigid_body_index(env, actor, 0, gymapi.DOMAIN_ENV))
            global_index = task.gym.get_actor_rigid_body_index(env, actor, 0, gymapi.DOMAIN_SIM)
            assert global_index == env_id * self.force.shape[1] + body_indices[-1]
            global_body_indices.append(global_index)
            if task.object_force_sensor_audit:
                assert task.gym.get_actor_force_sensor_count(env, actor) == 1
                sensor_indices.append(task.gym.get_actor_force_sensor(env, actor, 0).get_global_index())
        if task.object_force_sensor_audit:
            self.sensors = gymtorch.wrap_tensor(task.gym.acquire_force_sensor_tensor(task.sim)).view(-1, 6)
            self.sensor_ids = torch.tensor(sensor_indices, dtype=torch.long, device=task.device)
        self.body_ids = torch.tensor(body_indices, dtype=torch.long, device=task.device)
        native = task.gym.get_sim_params(task.sim)
        self.meta = {'env_ids': env_ids, 'object_properties': props,
                     'object_body_indices_env': body_indices,
                     'object_body_indices_sim': global_body_indices,
                     'gravity_disabled_flag': int(gymapi.RIGID_BODY_DISABLE_GRAVITY),
                     'gravity': [getattr(native.gravity, a) for a in 'xyz'],
                     'dt': native.dt, 'substeps': native.substeps,
                     'contact_collection': int(native.physx.contact_collection),
                     'control_frequency_inv': task.control_freq_inv,
                     'object_constraint_sensor_indices': sensor_indices,
                     'object_constraint_sensor_frame': 'world; forward dynamics disabled, constraint solver enabled' if sensor_indices else None,
                     'initial_object_state': task.root_state_tensor[task.object_indices[self.ids]].detach().cpu().tolist(),
                     'scope': 'After every native simulate(dt), including all configured internal substeps. Tensor refreshes and CPU copies add synchronization. Not an internal PhysX substep/contact-pair trace.'}
        self.frames = []

    def capture(self, policy_step, simulation_step):
        t = self.task
        t.gym.refresh_actor_root_state_tensor(t.sim)
        t.gym.refresh_rigid_body_state_tensor(t.sim)
        t.gym.refresh_dof_state_tensor(t.sim)
        t.gym.refresh_net_contact_force_tensor(t.sim)
        values = {'object_state': t.root_state_tensor[t.object_indices[self.ids]],
                  'object_body_state': t.rigid_body_states[self.ids, self.body_ids],
                  'object_net_force': self.force[self.ids, self.body_ids],
                  'joint_pos': t.robot_dof_pos[self.ids], 'joint_vel': t.robot_dof_vel[self.ids]}
        if t.object_force_sensor_audit:
            t.gym.refresh_force_sensor_tensor(t.sim)
            values['object_constraint_sensor'] = self.sensors[self.sensor_ids]
        row = {name: value.detach().cpu().numpy().copy() for name, value in values.items()}
        row['policy_step'] = np.asarray(policy_step + 1)
        row['simulation_step_in_control'] = np.asarray(simulation_step + 1)
        self.frames.append(row)

    def save(self):
        arrays = {k: np.stack([row[k] for row in self.frames]) for k in self.frames[0]}
        np.savez_compressed(self.output / 'simulation_steps.npz', **arrays)
        finite = {k: bool(np.isfinite(v).all()) for k, v in arrays.items()}
        report = {**self.meta, 'samples': len(self.frames), 'finite': finite}
        (self.output / 'simulation_steps.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        assert all(finite.values())
