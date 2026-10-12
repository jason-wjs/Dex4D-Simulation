"""Iteration-boundary state for the original non-vision, position-driven teacher.

PhysX hidden solver/contact caches are not exposed by Gym and are not serialized.
"""
import copy

from isaacgym import gymapi, gymtorch
import torch


BUFFERS = '''root_state_tensor dof_state rigid_body_states goal_state cur_targets
prev_targets last_robot_dof_vel obs_buf states_buf obs_history_buf states_history_buf
rew_buf reset_buf progress_buf randomize_buf reset_goal_buf achieved_buf successes
current_successes consecutive_successes z_theta masked_keypoint_buf target_qpos
target_hand_pos target_hand_rot object_init_euler_xy object_init_z actions object_pos
object_rot rand_push_linvel rand_push_angvel goal_obj_dist'''.split()
SCALARS = '''id iteration common_step_counter random_time total_successes total_resets
last_step last_rand_step first_randomization too_far_reset_threshold
goal_reset_stable_ratio robot_dof_speed_scale'''.split()


def vec(value):
    return [value.x, value.y, value.z]


def body_state(prop):
    return {'mass': prop.mass, 'com': vec(prop.com), 'flags': int(prop.flags),
            'inertia': [vec(getattr(prop.inertia, axis)) for axis in 'xyz']}


def physical_state(task):
    actors = []
    for env in task.envs:
        row = {}
        for name in ['robot', 'object']:
            handle = task.gym.find_actor_handle(env, name)
            row[name] = {
                'bodies': [body_state(p) for p in task.gym.get_actor_rigid_body_properties(env, handle)],
                'friction': [p.friction for p in task.gym.get_actor_rigid_shape_properties(env, handle)]}
            if name == 'robot':
                row[name]['dofs'] = task.gym.get_actor_dof_properties(env, handle).copy()
        actors.append(row)
    return {'actors': actors, 'gravity': vec(task.gym.get_sim_params(task.sim).gravity)}


def check_task(task):
    assert type(task).__name__ == 'XArm6LeapHandAP2AP'
    assert not task.cfg['test'] and not task.use_osc_control and not task.goal_cond
    assert not task.enable_camera_sensors and task.actor_params_generator is None
    dr = task.randomization_params
    assert set(dr['sim_params']) == {'gravity'}
    assert set(dr['actor_params']) == {'robot', 'object'}
    for name, props in dr['actor_params'].items():
        expected = {'rigid_body_properties', 'rigid_shape_properties'}
        if name == 'robot':
            expected.add('dof_properties')
        assert set(props) == expected
        assert set(props['rigid_body_properties']) == {'mass'}
        assert set(props['rigid_shape_properties']) == {'friction'}
    for name in ['observations', 'actions']:
        assert dr[name]['operation'] == 'additive' and dr[name]['distribution'] == 'gaussian'


def capture(task):
    check_task(task)
    # Preserve only the original-property entries actually read by this recipe.
    original = {}
    if task.randomize:
        assert not task.first_randomization
        original = {
            'dof_properties': task.original_props['dof_properties'].copy(),
            'rigid_body_properties': [{'mass': p['mass']} for p in task.original_props['rigid_body_properties']],
            'rigid_shape_properties': [{'friction': p['friction']} for p in task.original_props['rigid_shape_properties']],
            'gravity': vec(task.original_props['sim_params']['gravity'])}
    return {'format': 'dex4d_ap2ap_task_v1', 'cfg': copy.deepcopy(task.cfg),
            'cases': copy.deepcopy(task.object_code_and_scale_str_for_envs),
            'keypoints': task.object_keypoint_buf.detach().cpu().clone(),
            'buffers': {name: getattr(task, name).detach().cpu().clone() for name in BUFFERS},
            'scalars': {name: copy.deepcopy(getattr(task, name)) for name in SCALARS},
            'physical': physical_state(task), 'original_properties': original,
            'noise': {name: {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                             for k, v in params.items() if k != 'noise_lambda'}
                      for name, params in task.dr_randomizations.items()},
            'frame_count': task.gym.get_frame_count(task.sim) + task.checkpoint_frame_offset}


def restore(task, state):
    check_task(task)
    assert state['format'] == 'dex4d_ap2ap_task_v1' and state['cfg'] == task.cfg
    assert state['cases'] == task.object_code_and_scale_str_for_envs
    assert torch.equal(state['keypoints'].cpu(), task.object_keypoint_buf.cpu())
    # Build the original noise closures, then replace every saved parameter and
    # physical value. Learner RNG restoration follows this initialization.
    if task.randomize:
        task.apply_randomizations(task.randomization_params)
        original = copy.deepcopy(state['original_properties'])
        gravity = original.pop('gravity')
        original['sim_params'] = {'gravity': gymapi.Vec3(*gravity)}
        task.original_props = original
        for name, params in state['noise'].items():
            noise_lambda = task.dr_randomizations[name]['noise_lambda']
            task.dr_randomizations[name] = {k: v.to(task.device).clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                                            for k, v in params.items()}
            task.dr_randomizations[name]['noise_lambda'] = noise_lambda
    for env, row in zip(task.envs, state['physical']['actors']):
        for name, saved in row.items():
            handle = task.gym.find_actor_handle(env, name)
            props = task.gym.get_actor_rigid_body_properties(env, handle)
            assert len(props) == len(saved['bodies'])
            for prop, body in zip(props, saved['bodies']):
                prop.mass, prop.flags = body['mass'], body['flags']
                prop.com = gymapi.Vec3(*body['com'])
                for axis, values in zip('xyz', body['inertia']):
                    setattr(prop.inertia, axis, gymapi.Vec3(*values))
            task.gym.set_actor_rigid_body_properties(env, handle, props, False)
            shapes = task.gym.get_actor_rigid_shape_properties(env, handle)
            assert len(shapes) == len(saved['friction'])
            for prop, friction in zip(shapes, saved['friction']):
                prop.friction = friction
            task.gym.set_actor_rigid_shape_properties(env, handle, shapes)
            if name == 'robot':
                task.gym.set_actor_dof_properties(env, handle, saved['dofs'])
    params = task.gym.get_sim_params(task.sim)
    params.gravity = gymapi.Vec3(*state['physical']['gravity'])
    task.gym.set_sim_params(task.sim, params)
    for name, value in state['buffers'].items():
        value = value.to(task.device)
        # Some task buffers are allocated only by the first control step.
        if hasattr(task, name):
            current = getattr(task, name)
            assert current.shape == value.shape and current.dtype == value.dtype, name
            current.copy_(value)
        else:
            setattr(task, name, value.clone())
    for name, value in state['scalars'].items():
        setattr(task, name, copy.deepcopy(value))
    task.checkpoint_frame_offset = state['frame_count'] - task.gym.get_frame_count(task.sim)
    assert task.gym.set_dof_state_tensor(task.sim, gymtorch.unwrap_tensor(task.dof_state))
    assert task.gym.set_actor_root_state_tensor(task.sim, gymtorch.unwrap_tensor(task.root_state_tensor))
    assert task.gym.set_dof_position_target_tensor(task.sim, gymtorch.unwrap_tensor(task.cur_targets))
    task.extras = {name: getattr(task, name) for name in ['successes', 'current_successes', 'consecutive_successes']}
